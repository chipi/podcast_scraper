"""DGX MOSS-Transcribe-Diarize service (#1177).

Serves ``OpenMOSS-Team/MOSS-Transcribe-Diarize`` — a 0.9B Apache-2.0 model that emits transcript,
speaker labels and timestamps in a **single** autoregressive pass (SATS). It sits alongside the
services we already run: faster-whisper on :8000, pyannote on :8001, this on :8004.

Two things drive the design:

**transformers, not vLLM/SGLang.** The team recommends SGLang Omni and vLLM ships an aarch64/cu130
nightly wheel, but neither is verified on GB10 (sm_121). At 0.9B, bare transformers is fast enough
that taking that risk on the first pass would be trading a known quantity for an unknown one. If
the model earns its place, revisit.

**One inference, both stages.** The pipeline asks for transcription and diarization separately
(the Deepgram shape — two calls, for stage independence). Rather than run the model twice, the
response carries *both*: segments with speaker labels. Each provider takes what it needs. A repeat
call for the same audio hits the small in-process cache below, so the second stage is free.

Environment:
    MOSS_MODEL      (default: OpenMOSS-Team/MOSS-Transcribe-Diarize)
    MOSS_DEVICE     (default: cuda)
    UVICORN_PORT    (default: 8004)
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moss-server")

_MODEL_NAME = os.environ.get("MOSS_MODEL", "OpenMOSS-Team/MOSS-Transcribe-Diarize")
_DEVICE = os.environ.get("MOSS_DEVICE", "cuda")

_model: Any = None
_processor: Any = None

# The pipeline calls transcription and diarization as separate stages. Caching the last few
# inferences by audio digest means the second stage costs nothing, without the providers having to
# thread a response across the provider-interface boundary.
_CACHE_SIZE = 4
_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _model, _processor
    from transformers import AutoModelForCausalLM, AutoProcessor

    logger.info("Loading %s on %s", _MODEL_NAME, _DEVICE)
    started = time.time()
    _processor = AutoProcessor.from_pretrained(_MODEL_NAME, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME, trust_remote_code=True, torch_dtype="bfloat16"
    ).to(_DEVICE)
    _model.eval()
    logger.info("Loaded in %.1fs", time.time() - started)
    yield
    _model = None
    _processor = None


app = FastAPI(title="MOSS-Transcribe-Diarize", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok" if _model is not None else "loading", "model": _MODEL_NAME}


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    """OpenAI-shaped, so `gpu-mode-swap.sh prod` and the provider preflight can check it like the
    other services."""
    return {
        "object": "list",
        "data": [{"id": _MODEL_NAME, "object": "model", "owned_by": "openmoss"}],
    }


def _infer(audio_path: str) -> List[Dict[str, Any]]:
    """Run MOSS once and parse its SATS stream into segments."""
    from sats import parse_sats  # vendored next to this file at build time

    inputs = _processor(audio=audio_path, return_tensors="pt").to(_DEVICE)
    output = _model.generate(**inputs, max_new_tokens=8192)
    raw = _processor.batch_decode(output, skip_special_tokens=True)[0]
    return parse_sats(raw)


@app.post("/v1/transcribe")
async def transcribe(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Transcribe + diarize in one pass.

    Returns segments carrying BOTH text and speaker, so the transcription provider and the
    diarization provider can each read what they need from the same inference.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="model still loading")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty audio")

    digest = hashlib.sha256(payload).hexdigest()
    if digest in _cache:
        _cache.move_to_end(digest)
        logger.info("cache hit (%s) — second stage is free", digest[:12])
        segments = _cache[digest]
    else:
        suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        try:
            started = time.time()
            segments = _infer(tmp_path)
            logger.info("inferred %d segments in %.1fs", len(segments), time.time() - started)
        finally:
            os.unlink(tmp_path)
        _cache[digest] = segments
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)

    speakers: List[str] = []
    for seg in segments:
        label = str(seg.get("speaker") or "")
        if label and label not in speakers:
            speakers.append(label)

    return {
        "model": _MODEL_NAME,
        "text": " ".join(str(s["text"]) for s in segments),
        "segments": segments,
        "speakers": speakers,
        "num_speakers": len(speakers),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("UVICORN_PORT", "8004")))
