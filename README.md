# SigLIP ONNX Server

A CPU-only, text-only SigLIP embedding server built with Rust and ONNX Runtime.
It is intended for live text queries when GPU capacity is scarce; image
embedding and index ingestion remain GPU workloads.

The HTTP surface is deliberately small:

- `GET /health`
- `POST /v1/embeddings`

There are no image or legacy embedding routes.

## API

`POST /v1/embeddings` accepts a single string, a string batch, one token-ID
sequence, or a batch of token-ID sequences. The optional `model` must match the
served model, and `dimensions` may only equal its native embedding dimension.

```json
{
  "model": "organization/siglip-model",
  "input": ["a red block", "a robot arm"],
  "encoding_format": "base64"
}
```

`encoding_format` defaults to `float`. `base64` encodes each vector as
little-endian `f32` bytes. Responses preserve input order and use the familiar
OpenAI `object`, `data`, `model`, and `usage` envelope.

## Artifact contract

The server never downloads mutable model or runtime assets. It requires four
local files at startup:

```text
SIGLIP_MODEL_MANIFEST_PATH=/models/manifest.json
SIGLIP_ONNX_MODEL_PATH=/models/text_model.onnx
SIGLIP_TOKENIZER_PATH=/models/tokenizer.json
SIGLIP_ORT_LIBRARY_PATH=/opt/onnxruntime/lib/libonnxruntime.so
```

The manifest is a language-neutral JSON contract owned by the deployment that
produces the model artifacts:

```json
{
  "repository": "organization/siglip-model",
  "revision": "0123456789abcdef0123456789abcdef01234567",
  "embedding_dimension": 1152,
  "text_max_token_length": 64
}
```

Additional manifest fields are accepted so one canonical file can be shared by
multiple services. `revision` must be a full 40-character hexadecimal commit
SHA; mutable tags and branches fail closed.

Use an ONNX Runtime 1.24.x CPU shared library. The pinned Rust binding targets
the ONNX Runtime 1.24 API. The graph must expose `input_ids` and
`attention_mask` inputs plus a `text_embeds` output. A deliberately different
output name can be selected with `SIGLIP_ONNX_OUTPUT_NAME`.

At startup the server:

1. validates the model manifest and every local artifact path;
2. configures fixed-length tokenizer padding and truncation;
3. validates the ONNX graph inputs and output;
4. runs a real inference probe before binding the HTTP listener.

Every returned vector is checked for the declared dimension and finite values,
then L2-normalized.

## Run it

Install `mold` and use the standard Rust workflow:

```bash
cargo run --release
```

Or build the self-contained server image from this repository:

```bash
docker build -t siglip-onnx-server .
docker run --rm --env-file .env \
  -p 8000:8000 \
  -v /local/models:/models:ro \
  -v /local/onnxruntime:/opt/onnxruntime:ro \
  siglip-onnx-server
```

Copy [`.env.example`](.env.example) to `.env` and adjust its mounted paths.

## CPU concurrency

The process owns a bounded pool of independent ONNX Runtime sessions. Requests
run concurrently across sessions, while each session uses its own
intra-operation thread pool. Tune these values against the physical cores and
memory bandwidth of the target machine:

```text
SIGLIP_ORT_SESSION_COUNT=3
SIGLIP_ORT_INTRA_OP_THREADS=4
SIGLIP_MAX_TEXT_BATCH_SIZE=32
SIGLIP_MAX_PENDING_TEXT_REQUESTS=32
```

Keep sessions multiplied by threads near the physical-core budget as a starting
point, then measure. When active and pending capacity is full, the server returns
`429 Too Many Requests` with `Retry-After: 1` instead of building an unbounded
blocking queue. Sessions share ONNX Runtime's prepacked weights.

## Validate model parity

CPU and GPU embeddings are compatible only when the model revision, tokenizer,
fixed token length, graph semantics, vector dimension, and normalization agree.
Before routing production queries:

1. compare Rust and GPU tokenizer IDs and attention masks across ordinary,
   empty, long, Unicode, and punctuation-heavy queries;
2. compare CPU and GPU vectors using component error and cosine similarity;
3. query both vector sets against the same GPU-built image index and measure
   top-k overlap;
4. benchmark cold start, latency percentiles, throughput, and overload behavior
   at the intended session/thread settings;
5. canary only live text queries and keep image ingestion on GPU.

See [CONTRIBUTING.md](CONTRIBUTING.md) for local quality checks.

## License

Licensed under the [Apache License 2.0](LICENSE).
