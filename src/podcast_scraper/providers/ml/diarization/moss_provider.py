"""MOSS diarization provider — the speaker half of the joint model (#1177).

MOSS emits transcript + speakers + timestamps in one pass, so this provider and the MOSS
*transcription* provider read the same inference. The service caches its last few results by audio
digest, so whichever stage runs second pays nothing. That preserves the stage independence the
pipeline is built on (the shape Deepgram established — two calls, no response threaded across the
provider-interface boundary) without running a model twice.

The speaker labels MOSS returns (``S01``, ``S02``, …) are **anonymous and relative** — exactly the
semantics of pyannote's ``SPEAKER_00``. They are normalized to ``SPEAKER_NN`` here so every
downstream consumer (roster resolution, the #1167 placeholder guard, host/guest attribution) keeps
working unchanged, whichever diarizer produced them.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

from .... import config
from ...resilience.sockets import hardened_http_client
from .base import DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8004
_DEFAULT_TIMEOUT_SEC = 1800.0


def _normalize_speaker(label: str) -> str:
    """``S01`` -> ``SPEAKER_01``.

    Downstream code — the roster, the ``SPEAKER_NN`` placeholder predicate (#1167), the host/guest
    rules — keys off pyannote's naming. Normalizing here means MOSS is a drop-in for pyannote and
    nothing downstream has to learn a second dialect.
    """
    text = str(label or "").strip()
    if text.upper().startswith("S") and text[1:].isdigit():
        return f"SPEAKER_{int(text[1:]):02d}"
    return text or "SPEAKER_00"


class MossDiarizationProvider:
    """Diarize on the DGX MOSS service."""

    name = "moss"

    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self._host = (getattr(cfg, "dgx_tailnet_host", "") or "").strip()
        self._port = int(getattr(cfg, "moss_port", None) or _DEFAULT_PORT)
        self._model = (
            getattr(cfg, "moss_model", None) or "OpenMOSS-Team/MOSS-Transcribe-Diarize"
        ).strip()
        self._timeout = float(
            getattr(cfg, "moss_request_timeout_sec", None) or _DEFAULT_TIMEOUT_SEC
        )
        self._initialized = False

    def initialize(self) -> None:
        if not self._host:
            raise ValueError("dgx_tailnet_host is required for the moss diarization provider")
        self._initialized = True

    def cleanup(self) -> None:
        self._initialized = False

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        """Speaker turns from MOSS.

        ``num_speakers`` / ``min_speakers`` / ``max_speakers`` are accepted for interface
        compatibility but **ignored**: MOSS decides the speaker count autoregressively and exposes
        no knob to constrain it. Callers that rely on bounding the count (pyannote honours these)
        will not get that behaviour here — worth knowing when comparing DER against a diarizer that
        was given the true count.
        """
        if not self._initialized:
            self.initialize()
        path = os.fspath(audio_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(audio_path)

        url = f"http://{self._host}:{self._port}/v1/transcribe"
        started = time.perf_counter()
        with open(path, "rb") as fh:
            files = {"file": (os.path.basename(path), fh, "application/octet-stream")}
            with hardened_http_client(self._timeout) as client:
                response = client.post(url, files=files)
        response.raise_for_status()
        payload = response.json()
        elapsed = time.perf_counter() - started

        raw_segments = payload.get("segments") if isinstance(payload, dict) else None
        segments: List[DiarizationSegment] = []
        for seg in raw_segments or []:
            if not isinstance(seg, dict):
                continue
            try:
                start, end = float(seg["start"]), float(seg["end"])
            except (KeyError, TypeError, ValueError):
                continue
            segments.append(
                DiarizationSegment(
                    start=start, end=end, speaker=_normalize_speaker(str(seg.get("speaker", "")))
                )
            )

        voices = {s.speaker for s in segments}
        logger.info(
            "MOSS diarized %s in %.1fs (%d turns, %d voices)",
            os.path.basename(path),
            elapsed,
            len(segments),
            len(voices),
        )
        return DiarizationResult(
            segments=segments,
            num_speakers=len(voices),
            model_name=str((payload or {}).get("model") or self._model),
        )
