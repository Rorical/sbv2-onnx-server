use std::{
    collections::{HashMap, VecDeque},
    fs::{self, File},
    io::copy,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array1, Array2, Array3, Axis, CowArray};
use ort::{
    ExecutionProvider, GraphOptimizationLevel, SessionBuilder, environment::Environment,
    session::Session, tensor::OrtOwnedTensor, value::Value,
};
use reqwest::blocking::Client;
use tokenizers::{Encoding, Tokenizer};
#[cfg(any(feature = "cuda", feature = "coreml", feature = "rocm"))]
use tracing::info;

const CHINESE_BERT_REPO: &str = "tsukumijima/chinese-roberta-wwm-ext-large-onnx";
const REQUIRED_FILES: &[&str] = &[
    "model_fp16.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "added_tokens.json",
];
const ASSIST_CACHE_CAPACITY: usize = 8;

pub struct BertExtractor {
    session: Session,
    tokenizer: Tokenizer,
    assist_cache: Mutex<AssistCache>,
}

impl BertExtractor {
    pub fn new(env: &Arc<Environment>, model_dir: &Path) -> Result<Self> {
        ensure_bert_assets(model_dir)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            )
        })?;

        let model_path = locate_model_file(model_dir)?;
        let session = new_bert_session(env, &model_path).with_context(|| {
            format!("failed to load ONNX BERT model at {}", model_path.display())
        })?;

        Ok(Self {
            session,
            tokenizer,
            assist_cache: Mutex::new(AssistCache::new(ASSIST_CACHE_CAPACITY)),
        })
    }

    pub fn extract(
        &self,
        text: &str,
        word2ph: &[usize],
        assist_text: Option<(&str, f32)>,
    ) -> Result<Array2<f32>> {
        let (features, encoding) = self.forward(text)?;
        let aligned_word2ph = align_word2ph(text, word2ph, &encoding)
            .context("failed to align word2ph with BERT tokens")?;
        if features.shape()[0] != aligned_word2ph.len() {
            bail!(
                "word2ph length {} does not match BERT sequence length {}",
                aligned_word2ph.len(),
                features.shape()[0]
            );
        }

        let style_mean = match assist_text {
            Some((assist, weight)) if weight > 0.0 => {
                let trimmed = assist.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    let mean = self.cached_style_mean(trimmed)?;
                    Some((mean, weight))
                }
            }
            _ => None,
        };

        let hidden = features.shape()[1];
        let total_frames: usize = aligned_word2ph.iter().sum();
        let mut result = Array2::<f32>::zeros((hidden, total_frames));
        let mut frame_index = 0usize;

        for (idx, &repeat) in aligned_word2ph.iter().enumerate() {
            let mut base = features.row(idx).to_owned();
            if let Some((ref mean, weight)) = style_mean {
                let blend = 1.0 - weight;
                for (dst, &m) in base.iter_mut().zip(mean.iter()) {
                    *dst = *dst * blend + m * weight;
                }
            }
            for _ in 0..repeat {
                result.column_mut(frame_index).assign(&base);
                frame_index += 1;
            }
        }

        Ok(result)
    }

    fn cached_style_mean(&self, text: &str) -> Result<Arc<Array1<f32>>> {
        {
            let mut cache = self
                .assist_cache
                .lock()
                .expect("assist cache mutex poisoned");
            if let Some(mean) = cache.get(text) {
                return Ok(mean);
            }
        }

        let (features, _) = self.forward(text)?;
        let mean = features
            .mean_axis(Axis(0))
            .context("empty assist feature")?;
        let mean = Arc::new(mean);

        let mut cache = self
            .assist_cache
            .lock()
            .expect("assist cache mutex poisoned");
        if let Some(existing) = cache.get(text) {
            return Ok(existing);
        }
        cache.insert(text.to_string(), mean.clone());
        Ok(mean)
    }

    fn forward(&self, text: &str) -> Result<(Array2<f32>, Encoding)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("failed to tokenize '{text}': {e}"))?;
        let seq_len = encoding.len();
        if seq_len == 0 {
            bail!("tokenizer produced empty sequence for '{text}'");
        }
        let ids = to_i64_array(&encoding.get_ids());
        let type_ids = if encoding.get_type_ids().is_empty() {
            vec![0i64; seq_len]
        } else {
            to_i64_array(&encoding.get_type_ids())
        };
        let attention_mask = to_i64_array(&encoding.get_attention_mask());

        let input_ids_array =
            Array2::from_shape_vec((1, seq_len), ids).context("failed to reshape input_ids")?;
        let token_type_ids_array = Array2::from_shape_vec((1, seq_len), type_ids)
            .context("failed to reshape token_type_ids")?;
        let attention_array = Array2::from_shape_vec((1, seq_len), attention_mask)
            .context("failed to reshape attention_mask")?;

        let input_ids = CowArray::from(input_ids_array.view().into_dyn());
        let token_type_ids = CowArray::from(token_type_ids_array.view().into_dyn());
        let attention = CowArray::from(attention_array.view().into_dyn());

        let allocator = self.session.allocator();

        let mut ordered_inputs = Vec::new();
        for input in &self.session.inputs {
            let value = match input.name.as_str() {
                "input_ids" => Value::from_array(allocator, &input_ids)?,
                "token_type_ids" | "token_type_id" | "segment_ids" => {
                    Value::from_array(allocator, &token_type_ids)?
                }
                "attention_mask" | "attention_masks" => Value::from_array(allocator, &attention)?,
                other => bail!("unexpected BERT input '{}'", other),
            };
            ordered_inputs.push(value);
        }

        let outputs = self.session.run(ordered_inputs)?;
        let tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
        let array = tensor.view();
        let dims = array.shape();
        let features = match dims {
            [batch, seq_len, hidden] => {
                let data = array.iter().cloned().collect::<Vec<f32>>();
                let array = Array3::from_shape_vec((*batch, *seq_len, *hidden), data)
                    .context("failed to reshape BERT output")?;
                let view = array.index_axis(Axis(0), 0);
                view.to_owned()
            }
            [seq_len, hidden] => {
                let data = array.iter().cloned().collect::<Vec<f32>>();
                Array2::from_shape_vec((*seq_len, *hidden), data)
                    .context("failed to reshape 2D BERT output")?
            }
            other => bail!("unexpected BERT output dimensions: {:?}", other),
        };
        Ok((features, encoding))
    }
}

fn new_bert_session(env: &Arc<Environment>, model_path: &Path) -> Result<Session> {
    let mut session_builder = SessionBuilder::new(env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_parallel_execution(true)?;

    #[cfg(any(feature = "cuda", feature = "coreml", feature = "rocm"))]
    let mut providers: Vec<ExecutionProvider> = Vec::new();
    #[cfg(not(any(feature = "cuda", feature = "coreml", feature = "rocm")))]
    let providers: Vec<ExecutionProvider> = Vec::new();
    #[cfg(feature = "cuda")]
    {
        info!("Using CUDA for BERT");
        providers.push(ExecutionProvider::CUDA(Default::default()));
    }
    #[cfg(feature = "coreml")]
    {
        info!("Using CoreML for BERT");
        providers.push(ExecutionProvider::CoreML(Default::default()));
    }
    #[cfg(feature = "rocm")]
    {
        info!("Using ROCm for BERT");
        providers.push(ExecutionProvider::ROCm(Default::default()));
    }
    session_builder = session_builder.with_execution_providers(providers)?;
    Ok(session_builder.with_model_from_file(model_path)?)
}

fn locate_model_file(dir: &Path) -> Result<PathBuf> {
    let candidates = ["model_fp16.onnx", "model.onnx", "encoder_model.onnx"];
    for candidate in candidates {
        let path = dir.join(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    bail!("failed to locate ONNX model file in {}", dir.display())
}

fn to_i64_array(values: &[u32]) -> Vec<i64> {
    values.iter().map(|&v| v as i64).collect()
}

fn ensure_bert_assets(model_dir: &Path) -> Result<()> {
    let model_present = REQUIRED_FILES
        .iter()
        .all(|name| model_dir.join(name).exists());
    if model_present {
        return Ok(());
    }

    fs::create_dir_all(model_dir)
        .with_context(|| format!("failed to create {}", model_dir.display()))?;

    let client = Client::builder()
        .user_agent("sbv2-onnx-server/0.1")
        .build()
        .context("failed to build HTTP client")?;

    for file in REQUIRED_FILES {
        let destination = model_dir.join(file);
        if destination.exists() {
            continue;
        }

        let url = format!("https://huggingface.co/{CHINESE_BERT_REPO}/resolve/main/{file}");
        let mut response = client
            .get(&url)
            .send()
            .with_context(|| format!("failed to download {url}"))?
            .error_for_status()
            .with_context(|| format!("request failed {url}"))?;

        let mut out = File::create(&destination)
            .with_context(|| format!("failed to create {}", destination.display()))?;
        copy(&mut response, &mut out)
            .with_context(|| format!("failed to write {}", destination.display()))?;
    }

    Ok(())
}

struct AssistCache {
    entries: HashMap<String, Arc<Array1<f32>>>,
    order: VecDeque<String>,
    capacity: usize,
}

impl AssistCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    fn get(&mut self, key: &str) -> Option<Arc<Array1<f32>>> {
        let value = self.entries.get(key).cloned();
        if value.is_some() {
            self.touch(key);
        }
        value
    }

    fn insert(&mut self, key: String, value: Arc<Array1<f32>>) {
        self.entries.insert(key.clone(), value);
        self.touch(&key);
        self.evict();
    }

    fn touch(&mut self, key: &str) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(key.to_string());
    }

    fn evict(&mut self) {
        while self.entries.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
    }
}

fn align_word2ph(text: &str, word2ph: &[usize], encoding: &Encoding) -> Result<Vec<usize>> {
    if word2ph.is_empty() {
        bail!("word2ph is empty");
    }
    let offsets = encoding.get_offsets();
    if offsets.is_empty() {
        bail!("BERT encoding produced no offsets");
    }

    let leading = word2ph[0];
    let trailing = *word2ph.last().unwrap();

    let mut char_spans = Vec::new();
    let char_iter = text.char_indices();
    for (idx, (start, ch)) in char_iter.enumerate() {
        let end = start + ch.len_utf8();
        let count_idx = idx + 1;
        if count_idx >= word2ph.len() {
            bail!("word2ph length mismatch with text characters");
        }
        char_spans.push((start, end, word2ph[count_idx]));
    }

    if word2ph.len() != char_spans.len() + 2 {
        bail!("word2ph length does not equal text characters + 2");
    }

    let mut result = Vec::with_capacity(offsets.len());
    let mut char_index = 0usize;
    for (token_idx, &(start, end)) in offsets.iter().enumerate() {
        if token_idx == 0 {
            result.push(leading);
            continue;
        }
        if token_idx == offsets.len() - 1 {
            result.push(trailing);
            continue;
        }

        if start == 0 && end == 0 {
            result.push(0);
            continue;
        }

        let mut total = 0usize;
        let mut temp_index = char_index;
        while temp_index < char_spans.len() {
            let (c_start, c_end, count) = char_spans[temp_index];
            if c_end <= start {
                temp_index += 1;
                continue;
            }
            if c_start >= end {
                break;
            }
            total += count;
            temp_index += 1;
        }
        char_index = temp_index;
        result.push(total);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::sync::Arc;

    #[test]
    fn assist_cache_drops_least_recent_entry() {
        let mut cache = AssistCache::new(2);
        let first = Arc::new(Array1::from_vec(vec![0.0]));
        let second = Arc::new(Array1::from_vec(vec![1.0]));
        let third = Arc::new(Array1::from_vec(vec![2.0]));

        cache.insert("first".into(), first);
        cache.insert("second".into(), second);
        cache.get("first");
        cache.insert("third".into(), third);

        assert!(cache.entries.contains_key("first"));
        assert!(cache.entries.contains_key("third"));
        assert!(!cache.entries.contains_key("second"));
    }
}
