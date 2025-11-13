# sbv2-onnx-server

A high-performance, OpenAI-compatible inference server for the Style-BERT-VITS2 text-to-speech model, implemented in Rust. This server is specifically designed and optimized for **Chinese language** synthesis.

This project provides a Rust-based server for models trained with or converted by the [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) project. While the original project offers a comprehensive Python suite for training and experimentation, this server focuses on providing a robust, high-performance inference endpoint for deployment.

## Features

-   **High Performance**: Built with Rust, Axum, and Tokio for asynchronous, multi-threaded request handling.
-   **ONNX Runtime**: Utilizes the ONNX runtime for efficient, cross-platform model inference.
-   **OpenAI-Compatible API**: Implements an API that mirrors OpenAI's audio generation endpoints for seamless integration with existing clients and tools.
-   **Dedicated Chinese NLP Pipeline**: Includes a sophisticated pre-processing pipeline tailored for Chinese text.

## Getting Started

To run the server, you must provide paths to a `Style-Bert-VITS2` model (in ONNX format) and its associated configuration files.

### Prerequisites

-   A trained `Style-Bert-VITS2` model, converted to the ONNX format.
-   The corresponding `config.json` file for your model.
-   The `style_vectors.npy` file.
-   The `chinese-roberta-wwm-ext-large-onnx` BERT model files.

### Usage

Launch the server from your terminal, providing the paths to the required model assets.

```bash
./sbv2-onnx-server \
    --model /path/to/your/model.onnx \
    --config /path/to/your/config.json \
    --style-vectors /path/to/your/style_vectors.npy \
    --bert-root /path/to/bert/model/directory \
    --listen 0.0.0.0:8080
```

#### Command-Line Arguments

-   `--model`: Path to the main Style-Bert-VITS2 `.onnx` model file.
-   `--config`: Path to the `config.json` associated with the model.
-   `--style-vectors`: Path to the `style_vectors.npy` file containing voice style information.
-   `--bert-root`: Path to the root directory containing the ONNX BERT model assets.
-   `--listen`: The address and port for the server to bind to. (Default: `0.0.0.0:8080`)

If the BERT model is not found in the directory specified by `--bert-root`, the server will automatically attempt to download it from Hugging Face.

## Web UI for Testing

The server includes a simple web page for quick testing. Once the server is running, open your web browser and navigate to the root URL (e.g., `http://localhost:8080`) to access it.

## API Reference

The server provides an OpenAI-compatible endpoint at `/v1/audio/speech`. You can use any standard OpenAI client library or a tool like `curl` to send requests.

### Example Request

```bash
curl --location 'http://localhost:8080/v1/audio/speech' \
--header 'Content-Type: application/json' \
--data '{
    "model": "Style-Bert-VITS2",
    "input": "你好，世界！",
    "voice": "default"
}'
```

The synthesized audio will be returned in the response body.

## Acknowledgements

This project would not be possible without the foundational work done by the creators and contributors of the [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) repository.
