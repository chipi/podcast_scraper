# DGX MOSS-Transcribe-Diarize service (#1177)

Serves `OpenMOSS-Team/MOSS-Transcribe-Diarize` on `:8004` — a 0.9B Apache-2.0 model that emits
transcript + speaker labels + timestamps in a single pass. Sits alongside faster-whisper (`:8000`)
and pyannote (`:8001`). Reached over the tailnet by the pipeline's `moss` transcription/diarization
providers.

## Endpoints

- `POST /v1/transcribe` — multipart `file` → `{text, segments[], speakers[], num_speakers}`.
  Segments carry `start`/`end`/`speaker`/`text`, so the transcription provider and the diarization
  provider each read what they need from one inference (repeat calls hit a small audio-digest cache).
- `GET /health`, `GET /v1/models` — for `gpu-mode-swap.sh` / provider preflight.

## Long audio

The service does **not** chunk. The pipeline chunks upstream (`episode_processor` + `AudioChunker`,
`transcription_max_chunk_duration_seconds` = 1500s, #1174), so every request is a ≤25-min window
that fits the model's 128k context in one pass. A single MOSS pass truncates past ~30 min.

## Runtime notes (#1174 bake-off findings)

- **Base image `nvcr.io/nvidia/vllm:26.05-py3`** — the one base verified on GB10 (sm_121): torch
  2.12 (CUDA 13) + transformers 5.6 (MOSS needs ≥5.0). Its cuFFT is a **stub**, which breaks
  whisper's GPU mel-STFT; `app.py` forces feature extraction onto CPU (cheap) to dodge it.
- **Inference** uses the upstream `moss_transcribe_diarize` package helpers
  (`build_transcription_messages` / `generate_transcription` / `parse_transcript`) — the model's real
  API, verified during the bake-off (the earlier hand-rolled `_infer` + `sats.py` were wrong and are
  gone).
- **Quality/speed** (`EVAL_MOSS_BAKEOFF_2026_07.md`): transcription beats faster-whisper on real prod
  audio (WER 5.2% vs 8.5% vs Deepgram silver); diarization loses to pyannote on real audio; ~2.1–3.8×
  realtime in bare transformers (no speed win vs large-v3's 7.8×).

## Build / run

```bash
docker build -t podcast-moss:0.1.0 infra/dgx/moss-server/
docker run --rm --gpus all --ipc=host -p 8004:8004 \
    -v /opt/llm-models:/opt/llm-models --env-file ~/.env podcast-moss:0.1.0
```

Env: `MOSS_MODEL` (default the HF id), `MOSS_MODEL_REVISION` (default `main`; pin a SHA for prod),
`MOSS_DEVICE` (`auto`), `MOSS_MAX_NEW_TOKENS` (16384, sized for a 25-min window).

## Follow-ups (not yet done)

- **converge wiring** — a `deploy.py` block + `/opt/moss-server/docker-compose.yml` (mirror
  pyannote) so prod deploys reproducibly; `gpu-mode-swap.sh` `:8004` registration (homelab repo).
- **observability parity** — sentry-sdk + prometheus `/metrics` like the pyannote service.
