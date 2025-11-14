use std::{net::SocketAddr, sync::Arc};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tracing::info;

use crate::{
    audio,
    inference::{ChineseSynthesisInput, ChineseSynthesizer},
    model::TtsProject,
};

#[derive(Clone)]
struct AppState {
    synthesizer: ChineseSynthesizer,
    index_html: &'static str,
}

#[derive(Debug, Deserialize)]
struct SpeechRequest {
    model: String,
    input: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    style: Option<String>,
    #[serde(default)]
    style_weight: Option<f32>,
    #[serde(default)]
    noise: Option<f32>,
    #[serde(default)]
    noise_w: Option<f32>,
    #[serde(default)]
    sdp_ratio: Option<f32>,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    length_scale: Option<f32>,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    #[serde(default)]
    audio_format: Option<AudioFormat>,
    #[serde(default)]
    assist_text: Option<String>,
    #[serde(default)]
    assist_weight: Option<f32>,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum AudioFormat {
    #[default]
    Wav,
    Mp3,
}

impl AudioFormat {
    fn as_str(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum ResponseFormat {
    #[serde(alias = "b64_json", alias = "base64")]
    B64Json,
}

impl Default for ResponseFormat {
    fn default() -> Self {
        ResponseFormat::B64Json
    }
}

#[derive(Serialize)]
struct SpeechResponse {
    model: String,
    voice: Option<String>,
    style: Option<String>,
    audio_base64: String,
    audio_format: &'static str,
    sample_rate: u32,
    duration_ms: u128,
}

#[derive(Serialize)]
struct ApiErrorBody {
    message: String,
}

type ApiResult<T> = std::result::Result<T, ApiError>;

#[derive(Serialize)]
struct MetadataResponse {
    voices: Vec<String>,
    styles: Vec<String>,
    sample_rate: u32,
}

pub async fn serve(addr: SocketAddr, project: TtsProject) -> Result<()> {
    let project = Arc::new(project);
    let synthesizer = ChineseSynthesizer::new(project.clone());
    static INDEX_HTML: &str = include_str!("templates/index.html");
    let state = AppState {
        synthesizer,
        index_html: INDEX_HTML,
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/healthz", get(health))
        .route("/v1/metadata", get(metadata))
        .route("/v1/audio/speech", post(create_speech))
        .with_state(state);

    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind HTTP listener on {addr}"))?;
    info!("listening on http://{}", listener.local_addr()?);

    axum::serve(listener, app.into_make_service())
        .await
        .context("HTTP server terminated unexpectedly")
}

async fn health() -> &'static str {
    "ok"
}

async fn index(State(state): State<AppState>) -> Html<&'static str> {
    Html(state.index_html)
}

async fn create_speech(
    State(state): State<AppState>,
    Json(payload): Json<SpeechRequest>,
) -> ApiResult<Json<SpeechResponse>> {
    let SpeechRequest {
        model,
        input,
        voice,
        style,
        style_weight,
        noise,
        noise_w,
        sdp_ratio,
        speed,
        length_scale,
        response_format,
        audio_format,
        assist_text,
        assist_weight,
    } = payload;

    let format = audio_format.unwrap_or_default();

    let response_format = response_format.unwrap_or_default();
    if !matches!(response_format, ResponseFormat::B64Json) {
        return Err(ApiError::bad_request(
            "only b64_json response_format is supported",
        ));
    }

    if input.trim().is_empty() {
        return Err(ApiError::bad_request("input text must not be empty"));
    }

    let mut synth_input = ChineseSynthesisInput::new(input);
    synth_input.speaker = voice.clone();
    synth_input.style = style.clone();
    synth_input.style_weight = style_weight;
    synth_input.noise = noise;
    synth_input.noise_w = noise_w;
    synth_input.sdp_ratio = sdp_ratio;
    synth_input.assist_text = assist_text;
    synth_input.assist_weight = assist_weight;

    if let Some(ls) = length_scale {
        synth_input.length_scale = Some(ls);
    } else if let Some(speed) = speed {
        if speed <= 0.0 {
            return Err(ApiError::bad_request("speed must be greater than 0"));
        }
        synth_input.length_scale = Some(1.0 / speed);
    }

    let synthesizer = state.synthesizer.clone();
    let result = tokio::task::spawn_blocking(move || synthesizer.synthesize(&synth_input))
        .await
        .map_err(|err| ApiError::internal(format!("inference task panicked: {err}")))?
        .map_err(|err| {
            tracing::error!("TTS inference failed: {err:?}");
            ApiError::from_anyhow(err)
        })?;

    let resolved_style = style.clone().or_else(|| {
        state
            .synthesizer
            .project()
            .default_style_name()
            .map(str::to_string)
    });
    let resolved_voice = voice.clone().or_else(|| {
        state
            .synthesizer
            .project()
            .default_speaker_name()
            .map(str::to_string)
    });

    let encode_result = match format {
        AudioFormat::Wav => Ok(result.wav_base64()),
        AudioFormat::Mp3 => audio::pcm_to_mp3(&result.pcm, result.sample_rate)
            .map(|bytes| BASE64_STANDARD.encode(bytes))
            .map_err(|err| {
                tracing::error!("MP3 encoding failed: {err:?}");
                ApiError::internal(format!("failed to encode MP3: {err}"))
            }),
    }?;

    let response = SpeechResponse {
        model,
        voice: resolved_voice,
        style: resolved_style,
        audio_base64: encode_result,
        audio_format: format.as_str(),
        sample_rate: result.sample_rate,
        duration_ms: result.timings.total_ms,
    };

    Ok(Json(response))
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    fn from_anyhow(err: anyhow::Error) -> Self {
        Self::internal(err.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = Json(ApiErrorBody {
            message: self.message,
        });
        (self.status, body).into_response()
    }
}

async fn metadata(State(state): State<AppState>) -> Json<MetadataResponse> {
    let project = state.synthesizer.project();
    Json(MetadataResponse {
        voices: project.available_speakers(),
        styles: project.available_styles(),
        sample_rate: project.sample_rate(),
    })
}
