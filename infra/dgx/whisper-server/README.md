# DGX whisper-server — openai-whisper as a service (#953)

FastAPI wrapper around `openai-whisper` (OpenAI's first-party Python lib)
running on the DGX GB10 GPU. Exposes an OpenAI-compatible HTTP API:

| Verb | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/audio/transcriptions` | Multipart audio file → JSON `{"text": "..."}` |
| `GET` | `/health` | Liveness — 200 once the model is loaded |
| `GET` | `/v1/models` | OpenAI-style envelope |

## Why this exists (the short version)

The DGX has two whisper services running side-by-side on different ports:

| Port | Stack | Code provenance |
| --- | --- | --- |
| 8000 | `speaches` → `faster-whisper` → `ctranslate2` | Community implementations of OpenAI's MODEL |
| **8002 (this)** | **`openai-whisper`** | **OpenAI's own first-party library** |

The split exists because — surfaced by #948 — the speaches-bundled
ctranslate2 is NOT OpenAI's inference code, just a fast reimplementation
that loads the same model weights. Until issue #952 validates that
faster-whisper matches openai-whisper's WER on real podcasts, we want a
first-party path available. This service is that path.

Consumers point at whichever URL matches the trust/speed tradeoff they
want. After #952's verdict, one service becomes the default and the
other gets removed.

## Why not just use the cloud OpenAI Whisper API?

- Cost: $0.006/min audio. At 100 episodes × 90 min/month = $54/month —
  cheap but non-zero. This service is $0 marginal (we already own the
  DGX).
- Privacy: audio stays on the tailnet rather than going to OpenAI.
- We get to validate the speed tradeoff vs the cloud API as part of #929.

## Configuration

Environment variables (defaults match the deploy.py compose):

- `WHISPER_MODEL` (default: `large-v3`) — any model openai-whisper supports
  (`tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v2`, `large-v3`).
- `WHISPER_DEVICE` (default: `cuda`) — set to `cpu` for testing without GPU.
- `WHISPER_CACHE_DIR` (default: `/root/.cache/whisper`) — model download
  cache. Mounted from `/opt/llm-models/whisper-cache` on the DGX so the
  ~3 GB `large-v3` weights persist across container restarts.
- `UVICORN_HOST` (default: `0.0.0.0`)
- `UVICORN_PORT` (default: `8002`)
- `LOG_LEVEL` (default: `INFO`)

## Deploy

The `Dockerfile` here is what the pyinfra recipe builds and runs. The
service install (compose file + env injection + restart policy) lives
in `infra/dgx/converge/deploy.py`. Don't run this Dockerfile by hand in
prod.

For local development:

```bash
docker build -t podcast-whisper:dev infra/dgx/whisper-server/
docker run --rm --gpus all \
    -p 8002:8002 \
    -e WHISPER_MODEL=small.en \
    podcast-whisper:dev
```

## Validating the service

After `make dgx-deploy`:

```bash
# 1. Health check — service is up + model loaded
curl -s http://dgx-llm-1.tail6d0ed4.ts.net:8002/health
# Expected: {"status":"ok","model":"large-v3","device":"cuda"}

# 2. Transcribe a real audio file
curl -s -X POST http://dgx-llm-1.tail6d0ed4.ts.net:8002/v1/audio/transcriptions \
    -F "file=@some_episode.mp3" \
    -F "model=large-v3" \
    -F "response_format=json"
# Expected: {"text": "Hello and welcome..."}

# 3. End-to-end pipeline using this service (sets the URL the provider talks to)
WHISPER_DGX_URL=http://dgx-llm-1.tail6d0ed4.ts.net:8002/v1/audio/transcriptions \
    python -m podcast_scraper.cli --config config/profiles/cloud_with_dgx_balanced.yaml \
    --rss <feed_url> --output-dir /tmp/test-output
```

## Operational notes

- **First request after deploy is slow** — the `large-v3` model is ~3 GB
  and downloads on first startup. The HF cache mount means subsequent
  container restarts skip this.
- **Single-threaded by design** — openai-whisper's `model.transcribe()`
  holds the GIL and the CUDA context for the duration of a call.
  Concurrent requests serialize via the model lock; use the consumer-side
  single-flight pattern in `TailnetDgxWhisperTranscriptionProvider`
  (#946) when more than one client may call at once.
- **Expected throughput on GB10 with `large-v3`**:
  - 5-min podcast: ~10-20 s wall-time
  - 30-min podcast: ~60-120 s
  - 90-min podcast: ~3-7 min
  - These are ~2-4× faster than the same model on CPU and ~3-5× SLOWER
    than the cloud OpenAI Whisper API (which uses heavy internal
    batching we don't replicate here).
  - These are also ~2-4× SLOWER than `speaches` (faster-whisper) on
    port 8000 — that's the speed tradeoff for first-party code.

## References

- This issue: #953
- The "is faster-whisper actually equivalent?" validation: #952
- speaches/faster-whisper service: `infra/dgx/converge/deploy.py`
  plus `infra/dgx/speaches-gb10/`
- Consumer-side resilience: #946 (`src/podcast_scraper/providers/tailnet_dgx/whisper_provider.py`)
- Sibling DGX services: `infra/dgx/pyannote-server/` (#926)
