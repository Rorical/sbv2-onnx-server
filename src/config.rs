use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct HyperParameters {
    #[serde(default)]
    pub model_name: String,
    pub version: String,
    pub data: HyperParametersData,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HyperParametersData {
    #[serde(default)]
    pub use_jp_extra: bool,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: u32,
    #[serde(default = "default_add_blank")]
    pub add_blank: bool,
    #[serde(default)]
    pub cleaned_text: bool,
    #[serde(default)]
    pub spk2id: BTreeMap<String, usize>,
    #[serde(default)]
    pub num_styles: usize,
    #[serde(default)]
    pub style2id: BTreeMap<String, usize>,
}

const fn default_sampling_rate() -> u32 {
    44100
}

const fn default_add_blank() -> bool {
    true
}

impl HyperParameters {
    pub fn load_from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let buf = fs::read_to_string(&path).with_context(|| {
            format!(
                "failed to read hyper-parameters from {}",
                path.as_ref().display()
            )
        })?;
        let mut hps: HyperParameters = serde_json::from_str(&buf).with_context(|| {
            format!(
                "failed to parse hyper-parameters JSON at {}",
                path.as_ref().display()
            )
        })?;
        if hps.data.num_styles == 0 {
            hps.data.num_styles = hps.data.style2id.len().max(1);
        }
        if hps.data.style2id.is_empty() {
            hps.data
                .style2id
                .extend((0..hps.data.num_styles).map(|i| (i.to_string(), i)));
        }
        Ok(hps)
    }
}
