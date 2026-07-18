# SigLIP 2 ONNX Server

A CPU-only, text-only SigLIP 2 embedding server built with Rust and ONNX
Runtime. It is intended for live text queries when GPU capacity is scarce; image
embedding and index ingestion remain GPU workloads.

It exposes an OpenAI-compatible embeddings endpoint, so existing clients that
support a custom OpenAI base URL can use it without a SigLIP 2-specific wrapper.
The HTTP surface is deliberately small:

- `GET /health`
- `POST /v1/embeddings`

There are no image or legacy embedding routes.

## Why this exists

SigLIP 2 places images and text in the same embedding space. Building a video
or image index is a throughput-heavy batch workload: decode the media, run the
image tower on GPUs, and persist those vectors in a vector index. Searching
the finished index is different. Each request only needs one small text
embedding before nearest-neighbor search, so reserving a GPU for that query
path is often unnecessary.

```text
+----------------------------+       +----------------------------+
| Offline indexing           |       | Online search              |
|                            |       |                            |
| images/video               |       | text query                 |
|      |                     |       |      |                     |
|      v                     |       |      v                     |
| GPU SigLIP 2 image tower   |       | CPU SigLIP 2 text tower    |
|      |                     |       |      |                     |
|      v                     |       |      v                     |
| image embeddings           |       | text embedding             |
|      |                     |       |      |                     |
|      v                     |       |      v                     |
| vector index               |<------+ nearest-neighbor search    |
+----------------------------+       +----------------------------+
```

This server owns the second path. It lets a system keep GPU ingestion for high
throughput while moving live text queries onto ordinary CPU instances, which
are generally easier to provision, scale, and keep available when GPU capacity
is constrained. It is especially useful as a capacity fallback or as the
steady-state query service after GPU indexing completes.

The CPU and GPU paths must use the same SigLIP 2 checkpoint contract: model
revision, tokenizer, maximum token length, projection dimension, and
normalization. A text vector from a different contract is not compatible with
the existing image index. Original SigLIP and SigLIP 2 checkpoints are not
interchangeable embedding spaces.

## Why Rust and ONNX Runtime

The common serving path for vision-language models carries Python, PyTorch, and
a general-purpose GPU inference server. Those are excellent tools for model
development and high-throughput image ingestion, but they are more machinery
than this steady-state query path needs.

ONNX Runtime executes the exported text graph with an optimized CPU backend.
Rust owns the smaller serving layer around it: tokenization, bounded admission
and session pooling, the HTTP contract, and strict response validation. The
result is one predictable service process without a Python environment or the
training framework in production. Rust does not replace the tensor runtime or
make the model mathematics faster by itself; ONNX Runtime performs that work.

The tradeoff is a stricter artifact boundary. Exported graph inputs and outputs,
tokenizer behavior, fixed token length, model revision, and normalization must
be proven equivalent to the GPU implementation. This design is a good fit for
a pinned, narrow production query service; Python remains the easier choice
while experimenting with models or changing their execution contract often.

## OpenAI-compatible usage

Point the client at this server's `/v1` base URL and request the exact model ID
reported by `GET /health`:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/v1/embeddings \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "organization/siglip-model",
    "input": ["a red block", "a robot arm"]
  }'
```

For example, with the OpenAI Python client:

```python
from openai import OpenAI

embedding_client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-used",
)
response = embedding_client.embeddings.create(
    model="organization/siglip-model",
    input=["a red block", "a robot arm"],
)
embeddings = [item.embedding for item in response.data]
```

The server does not validate `Authorization`; `api_key="not-used"` only
satisfies clients that require a value. Protect non-loopback deployments with
network controls. This is embeddings-endpoint compatibility, not an
implementation of the rest of the OpenAI API.

### Request contract

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

The same image is published publicly at
`ghcr.io/hebbian-robotics/siglip-onnx-server`. Use `latest` for a quick local
evaluation, and pin a source-commit tag or image digest for production. The
checked-in [`.env.example`](.env.example) demonstrates an immutable tag.

Copy [`.env.example`](.env.example) to `.env` and adjust its mounted paths.

## Production C4 deployment

The checked-in [Compose deployment](deploy/compose.yaml) replaces ad-hoc
`docker run` processes. It uses durable host paths, read-only mounts and root
filesystem, bounded concurrency, a startup-aware health check, and
`restart: unless-stopped` so the service returns after Docker or the VM
restarts.

On the C4 host, install the model and runtime assets outside `/tmp`, then start
the service from a checkout of this repository:

```bash
sudo install -d -m 0755 \
  /var/lib/siglip-onnx/model \
  /var/lib/siglip-onnx/onnxruntime/lib
cp .env.example .env
# Set SIGLIP_ONNX_IMAGE to the published immutable image and verify the paths.
sudo docker compose --env-file .env -f deploy/compose.yaml pull
sudo docker compose --env-file .env -f deploy/compose.yaml up -d
sudo docker compose --env-file .env -f deploy/compose.yaml ps
```

The default example listens on the VM's VPC interface so Cloud Run Direct VPC
egress can reach it. Restrict TCP port 8000 with the VPC firewall; the server
does not provide application-layer authentication. Bind
`SIGLIP_LISTEN_ADDRESS=127.0.0.1` for local-only testing.

Before changing query traffic, verify `/health` reports the exact model,
revision, and dimension expected by the GPU-built indexes. Then deploy serve
with the C4's internal URL and `SQUASH_SIGLIP_TEXT_PROTOCOL=openai`. Rollback is
the legacy GPU URL plus `SQUASH_SIGLIP_TEXT_PROTOCOL=legacy`; ingestion is not
part of this switch.

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

Licensed under the [MIT License](LICENSE).
