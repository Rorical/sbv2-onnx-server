use std::{sync::Arc, time::Instant};

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;

use crate::{
    audio,
    model::{InferenceRequest, TtsProject},
};

#[derive(Clone)]
pub struct ChineseSynthesizer {
    project: Arc<TtsProject>,
}

pub struct ChineseSynthesisInput {
    pub text: String,
    pub speaker: Option<String>,
    pub style: Option<String>,
    pub style_weight: Option<f32>,
    pub sdp_ratio: Option<f32>,
    pub noise: Option<f32>,
    pub noise_w: Option<f32>,
    pub length_scale: Option<f32>,
    pub assist_text: Option<String>,
    pub assist_weight: Option<f32>,
}

impl ChineseSynthesisInput {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            speaker: None,
            style: None,
            style_weight: None,
            sdp_ratio: None,
            noise: None,
            noise_w: None,
            length_scale: None,
            assist_text: None,
            assist_weight: None,
        }
    }
}

pub struct SynthesisTimings {
    pub total_ms: u128,
}

pub struct SynthesisResult {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    pub wav: Vec<u8>,
    pub timings: SynthesisTimings,
}

impl SynthesisResult {
    pub fn wav_base64(&self) -> String {
        BASE64_STANDARD.encode(&self.wav)
    }
}

impl ChineseSynthesizer {
    pub fn new(project: Arc<TtsProject>) -> Self {
        Self { project }
    }

    pub fn project(&self) -> &Arc<TtsProject> {
        &self.project
    }

    pub fn synthesize(&self, input: &ChineseSynthesisInput) -> Result<SynthesisResult> {
        if input.text.trim().is_empty() {
            bail!("text input must not be empty");
        }

        let request = self.build_request(input)?;
        let start = Instant::now();
        let mut result = self
            .project
            .infer_chinese(request)
            .context("failed to run TTS inference")?;
        let inference_elapsed = start.elapsed();

        audio::normalize_peak(&mut result.audio);
        let wav = audio::pcm_to_wav(&result.audio, result.sample_rate)
            .context("failed to encode WAV output")?;

        Ok(SynthesisResult {
            pcm: result.audio,
            sample_rate: result.sample_rate,
            wav,
            timings: SynthesisTimings {
                total_ms: inference_elapsed.as_millis(),
            },
        })
    }

    fn build_request<'a>(
        &'a self,
        input: &'a ChineseSynthesisInput,
    ) -> Result<InferenceRequest<'a>> {
        let mut request = InferenceRequest::new(&input.text);

        if let Some(ref speaker) = input.speaker {
            if self.project.speaker_id(speaker).is_none() {
                bail!("speaker '{}' is not available", speaker);
            }
            request.speaker = Some(speaker.as_str());
        }

        if let Some(ref style) = input.style {
            if self.project.style_id(style).is_none() {
                bail!("style '{}' is not available", style);
            }
            request.style = Some(style.as_str());
        }

        if let Some(weight) = input.style_weight {
            if !(0.0..=1.0).contains(&weight) {
                bail!("style_weight must be within [0.0, 1.0]");
            }
            request.style_weight = weight;
        }

        if let Some(sdp_ratio) = input.sdp_ratio {
            request.sdp_ratio = sdp_ratio.clamp(0.0, 1.0);
        }

        if let Some(noise) = input.noise {
            request.noise = noise.max(0.0);
        }

        if let Some(noise_w) = input.noise_w {
            request.noise_w = noise_w.max(0.0);
        }

        if let Some(length_scale) = input.length_scale {
            if length_scale <= 0.0 {
                bail!("length_scale must be positive");
            }
            request.length_scale = length_scale;
        }

        if let Some(ref assist) = input.assist_text {
            request.assist_text = Some(assist.as_str());
        }

        if let Some(weight) = input.assist_weight {
            if !(0.0..=1.0).contains(&weight) {
                bail!("assist_weight must be within [0.0, 1.0]");
            }
            request.assist_weight = weight;
        }

        Ok(request)
    }
}
