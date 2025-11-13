use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Language {
    Zh,
    Jp,
    En,
}

impl Language {
    pub fn as_code(&self) -> &'static str {
        match self {
            Language::Zh => "ZH",
            Language::Jp => "JP",
            Language::En => "EN",
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_code())
    }
}

impl FromStr for Language {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "ZH" => Ok(Language::Zh),
            "JP" => Ok(Language::Jp),
            "EN" => Ok(Language::En),
            _ => Err(anyhow::anyhow!("unsupported language code {s}")),
        }
    }
}

pub const DEFAULT_BERT_SUBDIR_ZH: &str = "chinese-roberta-wwm-ext-large-onnx";
pub const DEFAULT_STYLE: &str = "Neutral";
pub const DEFAULT_STYLE_WEIGHT: f32 = 1.0;
pub const DEFAULT_SDP_RATIO: f32 = 0.2;
pub const DEFAULT_NOISE: f32 = 0.6;
pub const DEFAULT_NOISEW: f32 = 0.8;
pub const DEFAULT_LENGTH: f32 = 1.0;
pub const DEFAULT_ASSIST_TEXT_WEIGHT: f32 = 1.0;
