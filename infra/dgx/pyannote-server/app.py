"""DGX-hosted pyannote.audio diarization service (#926).

FastAPI wrapper around ``pyannote.audio.Pipeline`` that runs the diarization
model on the DGX GPU and exposes the same shape laptop-side
``TailnetDgxDiarizationProvider`` expects:

    POST /v1/diarize  — multipart file + optional speaker hints → JSON segments
    GET  /health      — 200 when pipeline is loaded
    GET  /v1/models   — OpenAI-style envelope (so the existing health helper
                        pattern in ``check_pyannote_diarize_health`` works)

Configuration (env vars; defaults match the deploy.py compose):

    PYANNOTE_MODEL          (default: ``pyannote/speaker-diarization-3.1``)
    PYANNOTE_DEVICE         (default: ``cuda``)
    HF_TOKEN                (required; pyannote models are gated)
    UVICORN_HOST            (default: ``0.0.0.0``)
    UVICORN_PORT            (default: ``8001``)
    LOG_LEVEL               (default: ``INFO``)

The model is loaded ONCE at startup. First request waits for warm-up; all
subsequent requests reuse the cached pipeline.

Run locally for testing:

    uvicorn app:app --host 0.0.0.0 --port 8001

Run via Docker (production path):

    docker compose up -d   # see /opt/pyannote-server/docker-compose.yml
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

logger = logging.getLogger("pyannote-server")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Sentry SDK (#942) — DGX-side errors that the client-side dgx_fallback
# breadcrumb can't see (compat bugs at startup, in-handler exceptions
# that turn into 500s before the fallback path triggers).
# No-op when SENTRY_DSN is unset so dev/local runs aren't affected.
_SENTRY_DSN = os.environ.get("SENTRY_DSN", "").strip()
if _SENTRY_DSN:
    import sentry_sdk

    sentry_sdk.init(
        dsn=_SENTRY_DSN,
        # Keep traces sample low — Sentry free tier is 10k transactions/mo,
        # we have ~5 services × ~1k requests/day = 150k/mo of raw traffic;
        # 0.01 keeps us well under budget while still surfacing slow paths.
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.01")),
        environment=os.environ.get("SENTRY_ENVIRONMENT", "dgx-prod"),
        server_name=os.environ.get("SENTRY_SERVER_NAME", "dgx-llm-1.tail6d0ed4.ts.net"),
        release=os.environ.get("SERVICE_VERSION", "dev"),
        # Drop expected 503s (model still loading at boot) — they're
        # health-check noise, not real errors. Anything else propagates.
        before_send=lambda event, hint: (
            None if "pyannote pipeline not yet loaded" in str(event.get("message", "")) else event
        ),
    )
    sentry_sdk.set_tag("service", "pyannote-server")
    sentry_sdk.set_tag("dgx_host", os.environ.get("DGX_HOST_TAG", "spark-2c14"))
    sentry_sdk.set_tag("gpu", "GB10")
    logger.info("Sentry SDK initialized for pyannote-server")
else:
    logger.info("SENTRY_DSN not set — running without Sentry integration")


_PIPELINE: Any = None
_MODEL_NAME = os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
_DEVICE = os.environ.get("PYANNOTE_DEVICE", "cuda")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the pyannote pipeline once at startup."""
    global _PIPELINE
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required (pyannote models are gated). "
            "Set it in /home/markodragoljevic/.env on DGX."
        )

    logger.info("Loading pyannote model %s on %s", _MODEL_NAME, _DEVICE)
    import torch

    # PyTorch 2.7 flipped `torch.load(..., weights_only=True)` to the default.
    # pyannote 3.x's checkpoints contain multiple non-tensor globals
    # (TorchVersion, numpy dtypes, omegaconf nodes, lightning hparams, etc.)
    # that aren't on the safe-globals allowlist — allowlisting them one by
    # one is brittle and the list drifts per pyannote release.
    #
    # Wrap torch.load so the weights_only default flips back to False ONLY
    # when callers don't pass it explicitly. We trust pyannote's checkpoints
    # because they come from the official HuggingFace pyannote repo over
    # HTTPS, downloaded via huggingface_hub with HF_TOKEN auth — the same
    # supply chain as every other HF model in this pipeline.
    #
    # When we bump to pyannote 4, checkpoints get re-serialized cleanly and
    # this wrapper can be removed.
    _orig_torch_load = torch.load

    def _torch_load_unsafe_for_pyannote(*args, **kwargs):
        # FORCE weights_only=False even if the caller passed True explicitly
        # — pytorch-lightning's load_from_checkpoint internally calls
        # `torch.load(..., weights_only=True)` since lightning 2.4+, and a
        # default-only setdefault leaves that call un-patched.
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_unsafe_for_pyannote

    # NVIDIA's NGC torch ships as version `2.7.0a0+79aa17489c.nv25.4`, which
    # is not a valid SemVer string. speechbrain (pyannote's transitive dep)
    # uses the `semver` library — not the looser `packaging` library — to
    # parse `torch.__version__` at startup, and chokes with
    # `ValueError: 2.7.0a0+... is not valid SemVer string`. Strip down to
    # the major.minor.patch portion so semver.parse() succeeds; the actual
    # ABI/CUDA behavior is unchanged (we just lie about the version string).
    _semver_safe_version = "2.7.0"
    if not torch.__version__.startswith(_semver_safe_version):
        logger.warning(
            "Unexpected torch version %r — semver patch may be incorrect",
            torch.__version__,
        )
    logger.info(
        "Patching torch.__version__ %r → %r for semver compat",
        torch.__version__,
        _semver_safe_version,
    )
    torch.__version__ = _semver_safe_version  # type: ignore[assignment]

    from pyannote.audio import Pipeline

    # Don't pass the token as a kwarg — different (pyannote, huggingface_hub)
    # version combinations expose it as `token=` (pyannote 4 / HF 1.x) or
    # `use_auth_token=` (pyannote 3 / HF <1.x). HF Hub reads HF_TOKEN from
    # the environment when no token is passed explicitly, and the env_file
    # already injects HF_TOKEN, so we get auth for free.
    pipeline = Pipeline.from_pretrained(_MODEL_NAME)
    if pipeline is None:
        raise RuntimeError(
            f"Pipeline.from_pretrained returned None for {_MODEL_NAME!r} — "
            "wrong model name, missing HF auth, or model not downloaded."
        )
    pipeline.to(torch.device(_DEVICE))
    _PIPELINE = pipeline
    logger.info("pyannote model loaded; service ready")

    yield

    logger.info("pyannote-server shutting down")


app = FastAPI(
    title="DGX pyannote diarization service (#926)",
    version="0.1.0",
    lifespan=lifespan,
)

# Prometheus instrumentation (#943) — request rate / latency histo /
# status codes at /metrics. The cardinality-control concern: don't
# include high-cardinality URL params (we don't have any here — paths
# are static), and don't expose default-collector metrics on `name`/
# `handler` labels beyond the route name (Instrumentator already groups
# /v1/diarize as one bucket, which is what we want).
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        # excluded_handlers: don't trace the metrics endpoint itself —
        # otherwise every scrape shows up in the histogram and skews p95.
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
    if _PIPELINE is None:
        raise HTTPException(503, "pyannote pipeline not yet loaded")
    return {"status": "ok", "model": _MODEL_NAME}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """OpenAI-style envelope so the existing health-check helper pattern works."""
    if _PIPELINE is None:
        raise HTTPException(503, "pyannote pipeline not yet loaded")
    return {
        "object": "list",
        "data": [
            {
                "id": _MODEL_NAME,
                "object": "model",
                "owned_by": "pyannote",
            }
        ],
    }


@app.post("/v1/diarize")
async def diarize(
    file: UploadFile = File(..., description="Audio file (wav / mp3 / flac / m4a)"),
    num_speakers: Optional[int] = Form(None, description="Exact speaker count if known."),
    min_speakers: int = Form(2, ge=1, description="Lower bound on speaker count."),
    max_speakers: int = Form(20, ge=1, description="Upper bound on speaker count."),
) -> dict[str, Any]:
    """Diarize the uploaded audio file and return speaker turns."""
    if _PIPELINE is None:
        raise HTTPException(503, "pyannote pipeline not yet loaded")
    if num_speakers is not None and num_speakers < 1:
        raise HTTPException(400, f"num_speakers must be >= 1, got {num_speakers}")
    if min_speakers < 1 or min_speakers > max_speakers:
        raise HTTPException(
            400,
            f"invalid speaker bounds: min={min_speakers}, max={max_speakers}",
        )

    # Persist to a tempfile because torchaudio loads off a path.
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(tmp_path)
        params: dict[str, int] = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        else:
            params["min_speakers"] = min_speakers
            params["max_speakers"] = max_speakers

        result = _PIPELINE(
            {"waveform": waveform, "sample_rate": int(sample_rate)},
            **params,
        )
        # pyannote 4.x: DiarizeOutput wrapper; 3.x: Annotation directly.
        annotation = getattr(result, "speaker_diarization", result)
        segments = [
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not segments:
        raise HTTPException(422, "pyannote produced no speaker segments for this audio")

    return {
        "model_name": _MODEL_NAME,
        "num_speakers": len({s["speaker"] for s in segments}),
        "segments": segments,
    }
