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

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates=20230311* \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home siglip

WORKDIR /app
COPY --from=builder /src/target/release/siglip-onnx-server /app/siglip-onnx-server

# The deployment mounts or bakes the model manifest, tokenizer, ONNX graph,
# and libonnxruntime. Startup fails if any configured artifact is absent.
USER siglip
EXPOSE 8000
ENTRYPOINT ["/app/siglip-onnx-server"]
