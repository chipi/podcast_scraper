"""MOSS transcription provider — talks to the DGX MOSS service (#1177).

MOSS is a *joint* model: one pass yields transcript, speakers and timestamps. This provider takes
the transcript half; :mod:`podcast_scraper.providers.ml.diarization.moss_provider` takes the
speaker half. They call the same service endpoint, which caches its last few inferences by audio
digest, so the second stage costs nothing — the pipeline keeps its stage independence (the shape
Deepgram already established) without paying to run the model twice.

**Known limit, deliberately not hidden:** MOSS emits *segment*-level timestamps only. There is no
word-level output. That is a precision trade against faster-whisper (p95 ~0.26 s vs ~1.6 s), not a
return of the #1173 drift — which came from silence removal, not from timestamp granularity.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ... import config
from ..resilience.sockets import hardened_http_client

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8004
_DEFAULT_TIMEOUT_SEC = 1800.0


def moss_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


class MossTranscriptionProvider:
    """Transcribe on the DGX MOSS service."""

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
            raise ValueError("dgx_tailnet_host is required for the moss provider")
        self._initialized = True

    def cleanup(self) -> None:
        self._initialized = False

    def _call(self, audio_path: str) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        path = os.fspath(audio_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(audio_path)

        url = f"{moss_base_url(self._host, self._port)}/v1/transcribe"
        with open(path, "rb") as fh:
            files = {"file": (os.path.basename(path), fh, "application/octet-stream")}
            with hardened_http_client(self._timeout) as client:
                response = client.post(url, files=files)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"MOSS returned {type(payload).__name__}, expected an object")
        return payload

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        return str(self._call(audio_path).get("text") or "").strip()

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: Optional[str] = None,
        pipeline_metrics: Any = None,
        episode_duration_seconds: Optional[int] = None,
        call_metrics: Any = None,
    ) -> Tuple[Dict[str, object], float]:
        """Return ``({text, segments}, elapsed_seconds)``.

        Segments carry ``speaker`` as well as ``start``/``end``/``text`` — harmless to the
        transcription path, and it is what lets the diarization stage reuse this same inference.
        """
        started = time.perf_counter()
        payload = self._call(audio_path)
        elapsed = time.perf_counter() - started

        raw_segments = payload.get("segments")
        segments: List[dict] = (
            [s for s in raw_segments if isinstance(s, dict)]
            if isinstance(raw_segments, list)
            else []
        )
        text = str(payload.get("text") or "").strip()
        if not text and not segments:
            raise ValueError("MOSS returned neither text nor segments")

        logger.info(
            "MOSS transcribed %s in %.1fs (%d segments, %s speakers)",
            os.path.basename(str(audio_path)),
            elapsed,
            len(segments),
            payload.get("num_speakers", "?"),
        )
        return ({"text": text, "segments": segments}, elapsed)
