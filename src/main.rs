mod audio;
mod config;
mod constants;
mod errors;
mod inference;
mod model;
mod nlp;
mod server;

use std::{net::SocketAddr, path::PathBuf};

use anyhow::Context;
use clap::Parser;
use tokio::runtime::Builder;
use tracing_subscriber::{EnvFilter, fmt};

use crate::{model::TtsProject, server::serve};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to Style-Bert-VITS2 ONNX model (.onnx)
    #[arg(long)]
    model: PathBuf,

    /// Path to config.json for the ONNX model
    #[arg(long)]
    config: PathBuf,

    /// Path to style_vectors.npy
    #[arg(long = "style-vectors")]
    style_vectors: PathBuf,

    /// Root directory for ONNX BERT models (expects chinese-roberta-wwm-ext-large-onnx)
    #[arg(long = "bert-root")]
    bert_root: PathBuf,

    /// Address to bind the HTTP server to
    #[arg(long, default_value = "0.0.0.0:8080")]
    listen: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    fmt().with_env_filter(env_filter).init();

    let project = TtsProject::load(
        &args.model,
        &args.config,
        &args.style_vectors,
        &args.bert_root,
    )
    .context("failed to initialise TTS project")?;

    let listen: SocketAddr = args.listen.parse().context("invalid listen address")?;

    let runtime = Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build tokio runtime")?;

    runtime
        .block_on(async { serve(listen, project).await })
        .context("server terminated unexpectedly")
}
