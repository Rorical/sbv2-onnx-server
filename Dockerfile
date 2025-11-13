# Dockerfile for sbv2-onnx-server

# Stage 1: Build the application in a container with the Rust toolchain
FROM rust:1-bookworm as builder

WORKDIR /usr/src/app

# Install build dependencies.
# protobuf-compiler is needed for some crates.
RUN apt-get update && apt-get install -y protobuf-compiler

# Copy the source code
COPY . .

# Build the application in release mode
# This will create a statically linked binary if possible.
RUN cargo build --release

# Stage 2: Create the final, minimal image
FROM debian:bookworm-slim

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/app/target/release/sbv2_onnx_server .

# Expose the port the server listens on
EXPOSE 8080

# Define the default command to run the server.
# Model paths and other arguments should be provided when running the container.
CMD ["./sbv2_onnx_server"]
