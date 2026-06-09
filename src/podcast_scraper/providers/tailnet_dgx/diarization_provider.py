"""DGX-hosted pyannote diarization with mandatory local fallback (#926).

Mirrors the architecture of ``TailnetDgxWhisperTranscriptionProvider`` (#814):
the laptop / prod VPS uploads an audio file to a pyannote service running on
DGX, gets back diarization segments, and falls back to running pyannote
in-process if DGX is unreachable. Local fallback (not cloud) because there
is no reasonably-priced cloud diarize service to bill against.

Service contract (DGX-side, deploy.py / infra/dgx/pyannote-server/app.py):

- ``POST /v1/diarize`` — multipart form ``file=<audio>`` plus optional
  ``num_speakers`` / ``min_speakers`` / ``max_speakers`` form fields. Returns
  ``{"model_name": str, "num_speakers": int, "segments": [{"start", "end",
  "speaker"}, ...]}``.
- ``GET /health`` — 200 when pipeline is loaded.
- ``GET /v1/models`` — OpenAI-style envelope so the existing
  ``check_*_health`` helpers can probe it consistently.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from ... import config
from ...providers.ml.diarization.base import (
    DiarizationProvider,
    DiarizationResult,
    DiarizationSegment,
)
from ...utils.log_redaction import format_exception_for_log
from .health import check_pyannote_diarize_health, dgx_diarize_base_url
from .telemetry import emit_dgx_fallback_breadcrumb

logger = logging.getLogger(__name__)

_RETRY_BACKOFF_SEC = 5.0


class TailnetDgxDiarizationProvider:
    """Diarize on DGX pyannote service; fall back to local pyannote on failure.

    Construction is cheap — the actual local fallback provider is built lazily
    so DGX-only paths don't pay the in-process pyannote model load cost.
    """

    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self._host = (cfg.dgx_tailnet_host or "").strip()
        self._port = int(getattr(cfg, "dgx_diarize_port", None) or 8001)
        self._model = (
            getattr(cfg, "dgx_diarize_model", None) or "pyannote/speaker-diarization-3.1"
        ).strip()
        self._timeout_sec = float(cfg.dgx_request_timeout_sec or 300.0)
        self._fallback: Optional[DiarizationProvider] = None
        self._initialized = False

    def initialize(self) -> None:
        """Validate host config; defer local-fallback build until first failure."""
        if self._initialized:
            return
        if not self._host:
            raise ValueError("dgx_tailnet_host is required for diarization_provider=tailnet_dgx")
        self._initialized = True

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        """Diarize against DGX; fall back to in-process pyannote on failure."""
        if not self._initialized:
            self.initialize()

        # Health-check matches the prefix portion of the model id so we don't
        # fail on minor naming drift between client config and server install.
        require_model_substring = self._model.rsplit("/", 1)[-1]

        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                if check_pyannote_diarize_health(
                    self._host,
                    port=self._port,
                    require_model_substring=require_model_substring,
                ):
                    return self._diarize_dgx(
                        audio_path,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "DGX diarize attempt %s failed: %s",
                    attempt + 1,
                    format_exception_for_log(exc),
                )
            if attempt == 0:
                time.sleep(_RETRY_BACKOFF_SEC)

        reason = format_exception_for_log(last_err) if last_err else "health_check_failed"
        emit_dgx_fallback_breadcrumb(
            stage="diarization",
            model=self._model,
            failure_reason=reason,
        )
        logger.warning(
            "Falling back from DGX diarize to in-process pyannote (%s)",
            reason,
        )
        return self._get_local_fallback().diarize(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    def _diarize_dgx(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int,
    ) -> DiarizationResult:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx required for tailnet_dgx diarize") from exc

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(audio_path)

        base = dgx_diarize_base_url(self._host, self._port)
        url = f"{base}/v1/diarize"
        with path.open("rb") as audio_file:
            files = {"file": (path.name, audio_file, "application/octet-stream")}
            data: dict[str, Any] = {
                "min_speakers": str(min_speakers),
                "max_speakers": str(max_speakers),
            }
            if num_speakers is not None:
                data["num_speakers"] = str(num_speakers)
            with httpx.Client(timeout=self._timeout_sec) as client:
                resp = client.post(url, data=data, files=files)
        resp.raise_for_status()
        payload = resp.json()
        raw_segments = payload.get("segments") or []
        segments = [
            DiarizationSegment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                speaker=str(s.get("speaker", "SPEAKER_UNKNOWN")),
            )
            for s in raw_segments
            if isinstance(s, dict)
        ]
        if not segments:
            raise ValueError("empty diarization result from DGX pyannote service")
        return DiarizationResult(
            segments=segments,
            num_speakers=int(payload.get("num_speakers") or len({s.speaker for s in segments})),
            model_name=str(payload.get("model_name") or self._model),
        )

    def _get_local_fallback(self) -> DiarizationProvider:
        """Build the in-process pyannote provider lazily on first failure."""
        if self._fallback is None:
            from ...providers.ml.diarization.factory import (
                create_local_pyannote_provider,
            )

            logger.info("Building local pyannote diarize fallback on first DGX failure")
            self._fallback = create_local_pyannote_provider(self.cfg)
        return self._fallback
