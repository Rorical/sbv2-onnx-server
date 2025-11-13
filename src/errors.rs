use thiserror::Error;

#[derive(Error, Debug)]
pub enum TtsError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ONNX runtime error: {0}")]
    Ort(#[from] ort::OrtError),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("unsupported language")]
    UnsupportedLanguage,
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, TtsError>;
