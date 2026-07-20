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

# Error reporting -> GlitchTip (self-hosted, Sentry-SDK/DSN compatible). No-op
# without SENTRY_DSN. before_send scrubs secrets before events leave (GlitchTip
# stores what you send). Traces off — perf tracing goes to VictoriaTraces, not here.
try:
    import sentry_sdk

    # Prefer GLITCHTIP_DSN (self-hosted) over the legacy SENTRY_DSN (Cloud); both
    # come from the container env (env_file), so no compose var-substitution needed.
    _sentry_dsn = (os.environ.get("GLITCHTIP_DSN") or os.environ.get("SENTRY_DSN", "")).strip()
    if _sentry_dsn:

        def _sentry_scrub(event: dict, _hint: object) -> dict:
            try:
                for _section in ("extra", "contexts"):
                    _data = event.get(_section) or {}
                    if isinstance(_data, dict):
                        for _k in list(_data):
                            if isinstance(_k, str) and any(
                                _s in _k.lower()
                                for _s in (
                                    "token",
                                    "secret",
                                    "password",
                                    "api_key",
                                    "authorization",
                                )
                            ):
                                _data[_k] = "[redacted]"
            except Exception:  # noqa: BLE001
                pass
            return event

        sentry_sdk.init(
            dsn=_sentry_dsn,
            environment=os.environ.get("SENTRY_ENVIRONMENT", os.environ.get("APP_ENV", "prod")),
            release=os.environ.get("APP_RELEASE") or None,
            send_default_pii=False,
            traces_sample_rate=0.0,
            before_send=_sentry_scrub,
        )
        sentry_sdk.set_tag("component", "moss")
        logger.info("Sentry/GlitchTip error reporting enabled")
except Exception:  # noqa: BLE001 - telemetry must never break startup
    logger.debug("sentry init skipped", exc_info=True)

_MODEL_NAME = os.environ.get("MOSS_MODEL", "OpenMOSS-Team/MOSS-Transcribe-Diarize")
# Pin the Hub revision (a tag/branch/commit) so a silent upstream change can't alter what we serve.
# Default ``main``; set MOSS_MODEL_REVISION to a commit SHA for a hard pin.
_MODEL_REVISION = os.environ.get("MOSS_MODEL_REVISION", "main")
_DEVICE = os.environ.get("MOSS_DEVICE", "auto")
# The pipeline chunks long audio UPSTREAM (episode_processor + AudioChunker, #1174), so each
# request is a <=25 min window that fits the 128k context in one pass; size generation for that.
_MAX_NEW_TOKENS = int(os.environ.get("MOSS_MAX_NEW_TOKENS", "16384"))

_model: Any = None
_processor: Any = None
_torch_device: Any = None
_torch_dtype: Any = None

# The pipeline calls transcription and diarization as separate stages. Caching the last few
# inferences by audio digest means the second stage costs nothing, without the providers having to
# thread a response across the provider-interface boundary.
_CACHE_SIZE = 4
_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _model, _processor, _torch_device, _torch_dtype
    import moss_transcribe_diarize.inference_utils as _iu
    import torch
    from moss_transcribe_diarize.inference_utils import resolve_device
    from transformers import AutoModelForCausalLM, AutoProcessor

    # The NGC vLLM base image ships a STUB cuFFT, so whisper's GPU mel-spectrogram (torch.stft)
    # fails with "cuFFT error 50". Force feature extraction onto CPU (device=None -> the moss
    # processor skips the cuda audio_kwargs); mel-extraction is cheap even for a 25 min window,
    # and the model still runs on GPU.
    _orig_prepare = _iu.prepare_inputs

    def _cpu_audio_prepare(processor: Any, messages: Any, **kw: Any) -> Any:
        kw["device"] = None
        return _orig_prepare(processor, messages, **kw)

    _iu.prepare_inputs = _cpu_audio_prepare

    _torch_device = resolve_device(_DEVICE)
    _torch_dtype = torch.bfloat16 if _torch_device.type == "cuda" else torch.float32
    logger.info("Loading %s on %s (%s)", _MODEL_NAME, _torch_device, _torch_dtype)
    started = time.time()
    _processor = AutoProcessor.from_pretrained(
        _MODEL_NAME, revision=_MODEL_REVISION, trust_remote_code=True, fix_mistral_regex=True
    )
    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME, revision=_MODEL_REVISION, trust_remote_code=True, dtype="auto"
    )
    _model = _model.to(dtype=_torch_dtype).to(_torch_device)
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
    """Run MOSS once and return segments (start/end/speaker/text).

    Uses the upstream package's helpers (verified against the model's real API during the #1174
    bake-off): ``build_transcription_messages`` renders the chat/audio prompt,
    ``generate_transcription`` runs the pass, ``parse_transcript`` splits the SATS stream.
    """
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
    )

    messages = build_transcription_messages(audio_path)
    result = generate_transcription(
        _model,
        _processor,
        messages,
        max_new_tokens=_MAX_NEW_TOKENS,
        do_sample=False,
        device=_torch_device,
        dtype=_torch_dtype,
    )
    return [
        {
            "start": float(s.start),
            "end": float(s.end),
            "speaker": str(s.speaker),
            "text": str(s.text),
        }
        for s in parse_transcript(result["text"])
    ]


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


# Prometheus instrumentation — request rate / latency histo / status codes at
# /metrics (mirrors pyannote-server + whisper-server). Paths are static (no
# high-cardinality URL params), so default grouping is fine. Scraped by the DGX
# Alloy collector (job=moss). No-op if the instrumentator isn't installed.
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
        "Add to the Dockerfile to enable scrape."
    )


if __name__ == "__main__":
    import uvicorn

    # Binds all interfaces by design: this runs inside the DGX container and is reached over the
    # tailnet by the pipeline; the host is env-overridable for a narrower bind. nosec B104 —
    # intentional service bind, not an accidental exposure.
    _host = os.environ.get("UVICORN_HOST", "0.0.0.0")  # nosec B104
    uvicorn.run(app, host=_host, port=int(os.environ.get("UVICORN_PORT", "8004")))
