# DGX pyannote diarization service (#926)

FastAPI wrapper around `pyannote.audio.Pipeline` for the prod DGX deployment.
Same pattern as Speaches (#814) — Docker-based, OpenAI-compatible health
probe, env_file injection from the operator's `~/.env`.

## Endpoints

| Verb | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness — 200 once the model is loaded |
| `GET` | `/v1/models` | OpenAI-style envelope (used by `check_pyannote_diarize_health`) |
| `POST` | `/v1/diarize` | Multipart audio file + optional speaker hints → JSON segments |

### `POST /v1/diarize`

Form fields:

- `file` (required): audio file (wav, mp3, flac, m4a, ...)
- `num_speakers` (optional int): exact speaker count if known
- `min_speakers` (default `2`): lower bound
- `max_speakers` (default `20`): upper bound

Response:

```json
{
  "model_name": "pyannote/speaker-diarization-3.1",
  "num_speakers": 2,
  "segments": [
    {"start": 0.0,  "end": 4.5,  "speaker": "SPEAKER_00"},
    {"start": 4.5,  "end": 11.2, "speaker": "SPEAKER_01"},
    ...
  ]
}
```

## Deploy

The Dockerfile here is what the pyinfra recipe builds and runs. The actual
service install happens through `infra/dgx/converge/deploy.py` — that's
where the compose file, env_file references, port mappings, and systemd
restart policy live. Don't run this Dockerfile by hand in prod.

For local development:

```bash
docker build -t podcast-pyannote:dev infra/dgx/pyannote-server/
docker run --rm --gpus all \
    -p 8001:8001 \
    -e HF_TOKEN=hf_... \
    podcast-pyannote:dev
```

## Why pyannote and not WhisperX

We currently use Speaches (`#814`) for transcription, which doesn't bundle
diarization. WhisperX would combine both behind a single endpoint, but
replacing the working Speaches install with WhisperX is a larger swap with
unknown migration cost. This service keeps the working Speaches setup in
place and adds diarization as a separate, deletable thing. Future
architectural simplification stays open.

## References

- Issue: #926
- Whisper-on-DGX precedent: #814 → `infra/dgx/converge/deploy.py`
- Pipeline-side provider: `src/podcast_scraper/providers/tailnet_dgx/diarization_provider.py`
- Operational hardening (sibling): #910
