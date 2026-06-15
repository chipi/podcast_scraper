"""DGX-hosted OpenAI Whisper transcription service (#952 prerequisite).

FastAPI wrapper around ``openai-whisper`` (OpenAI's first-party Python lib,
``pip install openai-whisper``) that runs Whisper on the DGX GPU and exposes
the OpenAI-compatible HTTP API:

    POST /v1/audio/transcriptions  — multipart audio → JSON {"text": "..."}
    GET  /health                   — 200 when model is loaded
    GET  /v1/models                — OpenAI-style envelope

Why this service exists (vs the existing speaches/faster-whisper service
on port 8000):

The speaches container ships ``faster-whisper`` via ``ctranslate2`` —
community reimplementations that load OpenAI's MODEL but run inference
through different code. Until issue #952 validates that
faster-whisper's WER matches openai-whisper on real podcast content,
we want a first-party path available. This service is that path. It
runs alongside speaches (different port) so consumers can pick the
trust/speed tradeoff:

    - port 8000 (speaches/faster-whisper) — fast, community-implemented
    - port 8002 (this service / openai-whisper) — slower, OpenAI's own code

When #952 lands its verdict, one of the two services becomes the new
default and the other gets removed.

Configuration (env vars; defaults match the deploy.py compose):

    WHISPER_MODEL    (default: ``large-v3``)
    WHISPER_DEVICE   (default: ``cuda``)
    UVICORN_HOST     (default: ``0.0.0.0``)
    UVICORN_PORT     (default: ``8002``)
    LOG_LEVEL        (default: ``INFO``)

The model is loaded ONCE at startup. First request waits for warm-up;
all subsequent requests reuse the cached model.

Run locally for testing:

    uvicorn app:app --host 0.0.0.0 --port 8002

Run via Docker (production path):

    docker compose up -d   # see /opt/whisper-server/docker-compose.yml
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

logger = logging.getLogger("whisper-server")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Sentry SDK (#996 follow-up; mirrors the pyannote-server pattern from #942)
# — captures DGX-side errors the client-side dgx_fallback breadcrumb can't see
# (compat bugs at startup, in-handler exceptions before fallback triggers).
# No-op when SENTRY_DSN is unset so dev/local runs aren't affected. The
# before_send filter drops the boot-time 503 "whisper model not yet loaded"
# health-check noise.
_SENTRY_DSN = os.environ.get("SENTRY_DSN", "").strip()
if _SENTRY_DSN:
    import sentry_sdk

    sentry_sdk.init(
        dsn=_SENTRY_DSN,
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.01")),
        environment=os.environ.get("SENTRY_ENVIRONMENT", "dgx-prod"),
        server_name=os.environ.get("SENTRY_SERVER_NAME", "dgx-llm-1.tail6d0ed4.ts.net"),
        release=os.environ.get("SERVICE_VERSION", "dev"),
        before_send=lambda event, hint: (
            None if "whisper model not yet loaded" in str(event.get("message", "")) else event
        ),
    )
    sentry_sdk.set_tag("service", "whisper-server")
    sentry_sdk.set_tag("dgx_host", os.environ.get("DGX_HOST_TAG", "spark-2c14"))
    sentry_sdk.set_tag("gpu", "GB10")
    logger.info("Sentry SDK initialized for whisper-server")
else:
    logger.info("SENTRY_DSN not set — running without Sentry integration")


_MODEL: Any = None
_MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the openai-whisper model once at startup."""
    global _MODEL

    logger.info("Loading openai-whisper model %s on %s", _MODEL_NAME, _DEVICE)

    # openai-whisper handles download + cache. The HF_HOME bind mount means
    # the model is shared with the speaches cache (no double download).
    import whisper

    cache_dir = os.environ.get("WHISPER_CACHE_DIR", "/root/.cache/whisper")
    _MODEL = whisper.load_model(_MODEL_NAME, device=_DEVICE, download_root=cache_dir)
    logger.info("openai-whisper %s loaded on %s; service ready", _MODEL_NAME, _DEVICE)

    yield

    logger.info("whisper-server shutting down")


app = FastAPI(
    title="DGX openai-whisper transcription service (parallel to speaches)",
    version="0.1.0",
    lifespan=lifespan,
)

# Prometheus instrumentation (#996 follow-up; mirrors pyannote-server #943).
# Exposes /metrics with request rate / latency histo / status classifier.
# Pair with the dgx-whisper-app scrape in compose/grafana-agent.yaml to
# correlate whisper-side state (queue depth, per-request latency,
# connection-reset rate) with any catastrophic-tail measurement going
# forward. The 2026-06-15 #996 sweep had no whisper-side telemetry —
# this closes that observability gap.
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/metrics"],
    ).instrument(app).expose(app, include_in_schema=False, tags=["observability"])
    logger.info("Prometheus instrumentation enabled at /metrics")
except ImportError:
    logger.warning(
        "prometheus_fastapi_instrumentator not installed — /metrics endpoint disabled. "
        "Add to requirements/Dockerfile to enable scrape."
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness — returns 200 once the model is loaded."""
    if _MODEL is None:
        raise HTTPException(503, "whisper model not yet loaded")
    return {"status": "ok", "model": _MODEL_NAME, "device": _DEVICE}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """OpenAI-style envelope so the existing health-check helper pattern works."""
    if _MODEL is None:
        raise HTTPException(503, "whisper model not yet loaded")
    return {
        "object": "list",
        "data": [
            {
                "id": _MODEL_NAME,
                "object": "model",
                "owned_by": "openai",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav / mp3 / flac / m4a)"),
    model: Optional[str] = Form(
        None,
        description=(
            "Ignored — this server is pinned to the model named at startup "
            "via WHISPER_MODEL. The OpenAI cloud API requires the param so "
            "the field is accepted for client compatibility."
        ),
    ),
    response_format: str = Form(
        "json",
        description="One of: json, text. ``verbose_json`` returns segments + words.",
    ),
    language: Optional[str] = Form(None, description="ISO language code; auto-detect if omitted."),
    temperature: Optional[float] = Form(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Single-value sampling temperature. **OMIT for podcast-length audio.** "
            "Setting a scalar (incl. 0.0) DISABLES openai-whisper's built-in "
            "temperature fallback schedule (0.0 → 1.0). The fallback is what "
            "detects autoregressive runaway via compression_ratio + logprob "
            "thresholds and re-runs the failing segment at a higher temperature. "
            "Without the schedule, long audio reliably hits decoder loops "
            "(verified empirically: a 5-min v2 episode produces 5× extra hyp "
            "words with temperature=0.0 vs the default schedule). Pass a "
            "scalar only when the caller specifically wants deterministic decode "
            "and accepts the runaway risk."
        ),
    ),
) -> Any:
    """Transcribe the uploaded audio file using openai-whisper."""
    if _MODEL is None:
        raise HTTPException(503, "whisper model not yet loaded")
    if response_format not in {"json", "text", "verbose_json"}:
        raise HTTPException(
            400,
            f"response_format must be one of json / text / verbose_json, got {response_format!r}",
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    # fp16=True is required for CUDA path; fp16=False on CPU. We're
    # GPU-only here so True is correct. If the operator runs this
    # on CPU for testing, openai-whisper will warn and fall back.
    use_fp16 = _DEVICE == "cuda"
    try:
        transcribe_kwargs: dict[str, Any] = {"fp16": use_fp16}
        # Only set temperature when the caller explicitly asked for one;
        # otherwise let openai-whisper apply its default fallback schedule
        # (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) which is what saves long-audio
        # transcription from autoregressive runaway. See Form() docstring.
        if temperature is not None:
            transcribe_kwargs["temperature"] = temperature
        if language is not None:
            transcribe_kwargs["language"] = language

        result = _MODEL.transcribe(tmp_path, **transcribe_kwargs)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    text = result.get("text", "").strip()

    if response_format == "text":
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(text)
    if response_format == "verbose_json":
        # OpenAI's verbose_json shape: text + segments[].{start, end, text, ...}
        # openai-whisper's result["segments"] is already very close — just
        # rename `text` and pass through start/end + word_timestamps if any.
        return {
            "task": "transcribe",
            "language": result.get("language", language or "en"),
            "duration": result.get("duration"),
            "text": text,
            "segments": result.get("segments", []),
        }
    # Default: OpenAI-style json envelope
    return {"text": text}
