# sbv2-onnx-server

A high-performance, OpenAI-compatible inference server for the Style-BERT-VITS2 text-to-speech model, implemented in Rust. This server is specifically designed and optimized for **Chinese language** synthesis.

This project provides a Rust-based server for models trained with or converted by the [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) project. While the original project offers a comprehensive Python suite for training and experimentation, this server focuses on providing a robust, high-performance inference endpoint for deployment.

## Features

-   **High Performance**: Built with Rust, Axum, and Tokio for asynchronous, multi-threaded request handling.
-   **ONNX Runtime**: Utilizes the ONNX runtime for efficient, cross-platform model inference.
-   **OpenAI-Compatible API**: Implements an API that mirrors OpenAI's audio generation endpoints for seamless integration with existing clients and tools.
-   **Dedicated Chinese NLP Pipeline**: Includes a sophisticated pre-processing pipeline tailored for Chinese text.
-   **Flexible Audio Output**: Return audio as WAV or MP3 by toggling the `audio_format` request parameter.

## Hardware Acceleration

This server supports GPU acceleration via the ONNX Runtime's execution providers. It will automatically detect and use the following providers in order of preference:

1.  **CUDA** (for NVIDIA GPUs)
2.  **CoreML** (for Apple Silicon)
3.  **ROCm** (for AMD GPUs)

If no compatible GPU is found, the server will fall back to using the CPU. To use GPU acceleration, you must have the appropriate drivers and toolkits installed on your system (e.g., the CUDA Toolkit for NVIDIA GPUs).

## Docker Builds for Hardware Acceleration

This project includes a multi-stage `Dockerfile` that can build runtime images for different hardware targets. We provide `docker-compose` files for convenience.

### CPU (Default)

The standard `docker-compose.yml` builds and runs the CPU-only version of the server. This is the easiest way to get started and does not require any special hardware.

```bash
docker-compose up --build
```

### NVIDIA GPU (CUDA)

If you have an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, you can build and run the CUDA-enabled image using the `docker-compose.cuda.yml` file.

```bash
docker-compose -f docker-compose.cuda.yml up --build
```

This will build the image with CUDA support and automatically make the GPU available to the container.

### AMD GPU (ROCm)

A build stage for ROCm is included in the `Dockerfile`. To build and run it, you can create a `docker-compose.rocm.yml` file and configure it to pass the ROCm device to the container (e.g., `/dev/kfd`, `/dev/dri`).

## Getting Started

To run the server, you must provide paths to a `Style-Bert-VITS2` model (in ONNX format) and its associated configuration files.

### Prerequisites

-   A trained `Style-Bert-VITS2` model, converted to the ONNX format.
-   The corresponding `config.json` file for your model.
-   The `style_vectors.npy` file.
-   The `chinese-roberta-wwm-ext-large-onnx` BERT model files.
-   (Optional for MP3 output) [LAME](https://lame.sourceforge.io/) installed on the host so `libmp3lame` is available (`brew install lame` on macOS, `apt install libmp3lame-dev` on Debian/Ubuntu), and build the server with `--features mp3`.

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
