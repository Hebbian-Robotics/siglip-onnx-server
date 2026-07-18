# syntax=docker/dockerfile:1.7

FROM rust:1.96-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends mold=1.10.* \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY .cargo/config.toml ./.cargo/config.toml
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN --mount=type=cache,id=siglip-onnx-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    cargo build --release --locked

FROM debian:bookworm-slim

LABEL org.opencontainers.image.source="https://github.com/Hebbian-Robotics/siglip-onnx-server" \
      org.opencontainers.image.description="CPU-only SigLIP 2 text embeddings with Rust and ONNX Runtime" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates=20230311* \
        curl=7.88.* \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home siglip

WORKDIR /app
COPY --from=builder /src/target/release/siglip-onnx-server /app/siglip-onnx-server

# The deployment mounts or bakes the model manifest, tokenizer, ONNX graph,
# and libonnxruntime. Startup fails if any configured artifact is absent.
USER siglip
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=3 \
    CMD ["curl", "--fail", "--silent", "--show-error", "http://127.0.0.1:8000/health"]
ENTRYPOINT ["/app/siglip-onnx-server"]
