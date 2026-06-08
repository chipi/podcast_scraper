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
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(_MODEL_NAME, token=hf_token)
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
