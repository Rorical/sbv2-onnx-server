use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array1, Array2, Array3, Axis, CowArray, arr0};
use ndarray_npy::ReadNpyExt;
use ort::{
    GraphOptimizationLevel, SessionBuilder, environment::Environment, session::Session,
    value::Value,
};

use crate::{
    config::HyperParameters,
    constants::{
        DEFAULT_ASSIST_TEXT_WEIGHT, DEFAULT_LENGTH, DEFAULT_NOISE, DEFAULT_NOISEW,
        DEFAULT_SDP_RATIO, DEFAULT_STYLE,
    },
    nlp::{
        LANGUAGE_ID_MAP, LANGUAGE_TONE_START_MAP, SYMBOL_ID_MAP,
        bert::BertExtractor,
        chinese::{g2p, normalizer},
    },
};

pub struct TtsProject {
    hps: HyperParameters,
    style_vectors: Array2<f32>,
    style2id: HashMap<String, usize>,
    spk2id: HashMap<String, usize>,
    onnx_session: Session,
    bert: BertExtractor,
    default_style_id: usize,
    default_speaker_id: usize,
}

pub struct InferenceResult {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
}

pub struct InferenceRequest<'a> {
    pub text: &'a str,
    pub speaker: Option<&'a str>,
    pub style: Option<&'a str>,
    pub style_weight: f32,
    pub sdp_ratio: f32,
    pub noise: f32,
    pub noise_w: f32,
    pub length_scale: f32,
    pub assist_text: Option<&'a str>,
    pub assist_weight: f32,
}

impl<'a> InferenceRequest<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            speaker: None,
            style: None,
            style_weight: 1.0,
            sdp_ratio: DEFAULT_SDP_RATIO,
            noise: DEFAULT_NOISE,
            noise_w: DEFAULT_NOISEW,
            length_scale: DEFAULT_LENGTH,
            assist_text: None,
            assist_weight: DEFAULT_ASSIST_TEXT_WEIGHT,
        }
    }
}

impl TtsProject {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        model_path: &Path,
        config_path: &Path,
        style_vec_path: &Path,
        bert_root: &Path,
    ) -> Result<Self> {
        if !model_path.exists() {
            bail!("TTS ONNX model not found at {}", model_path.display());
        }
        if !config_path.exists() {
            bail!("config.json not found at {}", config_path.display());
        }
        if !style_vec_path.exists() {
            bail!(
                "style_vectors.npy not found at {}",
                style_vec_path.display()
            );
        }

        let hps = HyperParameters::load_from_file(config_path)?;

        let style_file = File::open(style_vec_path).with_context(|| {
            format!(
                "failed to open style vectors at {}",
                style_vec_path.display()
            )
        })?;
        let style_vectors: Array2<f32> =
            Array2::read_npy(style_file).context("failed to read style_vectors.npy")?;
        let num_styles = style_vectors.nrows();
        if num_styles == 0 {
            bail!("style_vectors.npy is empty");
        }

        let style2id: HashMap<String, usize> = if hps.data.style2id.is_empty() {
            (0..num_styles).map(|idx| (idx.to_string(), idx)).collect()
        } else {
            hps.data
                .style2id
                .iter()
                .map(|(k, v)| (k.clone(), (*v).min(num_styles - 1)))
                .collect()
        };
        let spk2id: HashMap<String, usize> = hps
            .data
            .spk2id
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        let default_style_id = style2id
            .get(DEFAULT_STYLE)
            .copied()
            .unwrap_or(0)
            .min(num_styles - 1);
        let default_speaker_id = spk2id.values().copied().min().unwrap_or(0);

        let env = Environment::builder()
            .with_name("sbv2-tts")
            .build()
            .context("failed to initialize ONNX Runtime environment")?
            .into_arc();

        let session = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_path)
            .context("failed to load TTS ONNX model")?;

        let bert_dir = resolve_bert_dir(bert_root);
        let bert = BertExtractor::new(&bert_dir)
            .with_context(|| format!("failed to initialize BERT at {}", bert_dir.display()))?;

        Ok(Self {
            hps,
            style_vectors,
            style2id,
            spk2id,
            onnx_session: session,
            bert,
            default_style_id,
            default_speaker_id,
        })
    }

    pub fn infer_chinese(&self, request: InferenceRequest<'_>) -> Result<InferenceResult> {
        let normalized = normalizer::normalize_text(request.text);
        let (phones, tones, mut word2ph) = g2p::g2p(&normalized)?;

        let language_id = *LANGUAGE_ID_MAP
            .get("ZH")
            .ok_or_else(|| anyhow!("language id for ZH not found"))?
            as i64;
        let tone_start = *LANGUAGE_TONE_START_MAP
            .get("ZH")
            .ok_or_else(|| anyhow!("tone start for ZH not found"))? as i32;

        let mut phone_ids = Vec::with_capacity(phones.len());
        for phone in &phones {
            let id = SYMBOL_ID_MAP
                .get(phone.as_str())
                .copied()
                .ok_or_else(|| anyhow!("unknown phone symbol '{phone}'"))?;
            phone_ids.push(id as i64);
        }

        let mut tone_ids: Vec<i64> = tones
            .iter()
            .map(|&tone| (tone_start + tone) as i64)
            .collect();
        let mut lang_ids = vec![language_id; phone_ids.len()];

        if self.hps.data.add_blank {
            phone_ids = intersperse(&phone_ids, 0);
            tone_ids = intersperse(&tone_ids, 0);
            lang_ids = intersperse(&lang_ids, language_id);
            for val in &mut word2ph {
                *val *= 2;
            }
            if let Some(first) = word2ph.first_mut() {
                *first += 1;
            }
        }

        let bert_features = self.bert.extract(
            &normalized,
            &word2ph,
            request
                .assist_text
                .map(|text| (text, request.assist_weight)),
        )?;
        let bert_batch = bert_features.insert_axis(Axis(0)).to_owned();

        let hidden = bert_batch.shape()[1];
        let frames = bert_batch.shape()[2];

        let ja_bert = Array3::<f32>::zeros((1, hidden, frames));
        let en_bert = Array3::<f32>::zeros((1, hidden, frames));

        let phones_len = phone_ids.len();

        let x_tst = CowArray::from(
            Array2::from_shape_vec((1, phones_len), phone_ids)
                .context("failed to build phones array")?
                .into_dyn(),
        );
        let tones_arr = CowArray::from(
            Array2::from_shape_vec((1, phones_len), tone_ids)
                .context("failed to build tones array")?
                .into_dyn(),
        );
        let lang_arr = CowArray::from(
            Array2::from_shape_vec((1, phones_len), lang_ids)
                .context("failed to build language ids array")?
                .into_dyn(),
        );
        let x_tst_lengths = CowArray::from(Array1::from_vec(vec![phones_len as i64]).into_dyn());

        let speaker_id = match request.speaker {
            Some(name) => self
                .spk2id
                .get(name)
                .copied()
                .ok_or_else(|| anyhow!("speaker '{name}' not found in config"))?,
            None => self.default_speaker_id,
        };
        let sid_tensor = CowArray::from(Array1::from_vec(vec![speaker_id as i64]).into_dyn());

        let style_vector = self.make_style_vector(request.style, request.style_weight)?;
        let style_tensor = CowArray::from(style_vector.insert_axis(Axis(0)).into_dyn());

        let bert_tensor = CowArray::from(bert_batch.into_dyn());
        let ja_tensor = CowArray::from(ja_bert.into_dyn());
        let en_tensor = CowArray::from(en_bert.into_dyn());

        let length_scale = CowArray::from(arr0(request.length_scale).into_dyn());
        let sdp_ratio = CowArray::from(arr0(request.sdp_ratio).into_dyn());
        let noise = CowArray::from(arr0(request.noise).into_dyn());
        let noise_w = CowArray::from(arr0(request.noise_w).into_dyn());

        let allocator = self.onnx_session.allocator();
        let inputs = vec![
            Value::from_array(allocator, &x_tst)?,
            Value::from_array(allocator, &x_tst_lengths)?,
            Value::from_array(allocator, &sid_tensor)?,
            Value::from_array(allocator, &tones_arr)?,
            Value::from_array(allocator, &lang_arr)?,
            Value::from_array(allocator, &bert_tensor)?,
            Value::from_array(allocator, &ja_tensor)?,
            Value::from_array(allocator, &en_tensor)?,
            Value::from_array(allocator, &style_tensor)?,
            Value::from_array(allocator, &length_scale)?,
            Value::from_array(allocator, &sdp_ratio)?,
            Value::from_array(allocator, &noise)?,
            Value::from_array(allocator, &noise_w)?,
        ];

        let outputs = self.onnx_session.run(inputs)?;
        let tensor = outputs[0].try_extract::<f32>()?;
        let waveform = tensor.view().iter().cloned().collect::<Vec<f32>>();

        Ok(InferenceResult {
            audio: waveform,
            sample_rate: self.hps.data.sampling_rate,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.hps.data.sampling_rate
    }

    pub fn available_speakers(&self) -> Vec<String> {
        let mut entries: Vec<_> = self
            .spk2id
            .iter()
            .map(|(name, &id)| (id, name.as_str()))
            .collect();
        entries.sort_by_key(|(id, _)| *id);
        entries
            .into_iter()
            .map(|(_, name)| name.to_string())
            .collect()
    }

    pub fn available_styles(&self) -> Vec<String> {
        let mut entries: Vec<_> = self
            .style2id
            .iter()
            .map(|(name, &id)| (id, name.as_str()))
            .collect();
        entries.sort_by_key(|(id, _)| *id);
        entries
            .into_iter()
            .map(|(_, name)| name.to_string())
            .collect()
    }

    pub fn default_style_name(&self) -> Option<&str> {
        self.style2id
            .iter()
            .find(|(_, id)| **id == self.default_style_id)
            .map(|(name, _)| name.as_str())
    }

    pub fn default_speaker_name(&self) -> Option<&str> {
        self.spk2id
            .iter()
            .find(|(_, id)| **id == self.default_speaker_id)
            .map(|(name, _)| name.as_str())
    }

    pub fn style_id(&self, name: &str) -> Option<usize> {
        self.style2id.get(name).copied()
    }

    pub fn speaker_id(&self, name: &str) -> Option<usize> {
        self.spk2id.get(name).copied()
    }

    pub fn default_style_id(&self) -> usize {
        self.default_style_id
    }

    pub fn default_speaker_id(&self) -> usize {
        self.default_speaker_id
    }

    fn make_style_vector(&self, style_name: Option<&str>, weight: f32) -> Result<Array1<f32>> {
        let style_id = match style_name {
            Some(name) => *self
                .style2id
                .get(name)
                .ok_or_else(|| anyhow!("style '{name}' not found"))?,
            None => self.default_style_id,
        };
        if style_id >= self.style_vectors.nrows() {
            bail!("style id {style_id} out of range");
        }
        let mean = self.style_vectors.row(0);
        let target = self.style_vectors.row(style_id);
        let vec = &mean + (&target - &mean) * weight;
        Ok(vec.to_owned())
    }
}

fn intersperse(values: &[i64], blank: i64) -> Vec<i64> {
    let mut result = Vec::with_capacity(values.len() * 2 + 1);
    for value in values {
        result.push(blank);
        result.push(*value);
    }
    result.push(blank);
    result
}

fn resolve_bert_dir(root: &Path) -> PathBuf {
    if root.join("model_fp16.onnx").exists() {
        root.to_path_buf()
    } else {
        root.join("chinese-roberta-wwm-ext-large-onnx")
    }
}
