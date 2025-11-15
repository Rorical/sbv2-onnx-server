# Stage 1: Common builder
FROM rust:1-bookworm AS builder

# Build argument to control features, e.g., "cuda" or "rocm"
ARG CARGO_FEATURES=""

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y protobuf-compiler libmp3lame-dev pkg-config && rm -rf /var/lib/apt/lists/*
COPY . .

# Build with specified features. If CARGO_FEATURES is empty, this uses the default.
RUN cargo build --release --features "$CARGO_FEATURES" \
    && ./scripts/bundle_ort_libs.sh target/release

# --- Final Images ---

# Stage 2: CPU-only final image
FROM debian:bookworm-slim AS cpu

WORKDIR /app
RUN apt-get update && apt-get install -y ca-certificates libmp3lame0 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/target/release/sbv2_onnx_server .
COPY --from=builder /usr/src/app/target/release/libonnxruntime* .
COPY resources ./resources
EXPOSE 8080
CMD ["./sbv2_onnx_server"]

# Stage 3: CUDA-enabled final image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS cuda

WORKDIR /app
RUN apt-get update && apt-get install -y ca-certificates libmp3lame0 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/target/release/sbv2_onnx_server .
COPY --from=builder /usr/src/app/target/release/libonnxruntime* .
COPY resources ./resources
EXPOSE 8080
CMD ["./sbv2_onnx_server"]

# Stage 4: ROCm-enabled final image
FROM rocm/dev-ubuntu-22.04:5.7-complete AS rocm

WORKDIR /app
# This base image is Ubuntu 22.04 based
RUN apt-get update && apt-get install -y ca-certificates libmp3lame0 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/target/release/sbv2_onnx_server .
COPY --from=builder /usr/src/app/target/release/libonnxruntime* .
COPY resources ./resources
EXPOSE 8080
CMD ["./sbv2_onnx_server"]
