"""DGX-hosted Whisper via Ollama with mandatory cloud fallback (ADR-096)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast, List, Optional

from ... import config
from ...transcription.base import TranscriptionProvider
from ...transcription.factory import create_transcription_provider
from ...utils.log_redaction import format_exception_for_log
from .health import check_ollama_health, dgx_ollama_base_url
from .telemetry import emit_dgx_fallback_breadcrumb

logger = logging.getLogger(__name__)

_RETRY_BACKOFF_SEC = 5.0


class TailnetDgxWhisperTranscriptionProvider:
    """Transcribe on DGX Ollama; fall back to configured cloud provider on failure."""

    def __init__(self, cfg: config.Config) -> None:
        """Store config and DGX connection parameters."""
        self.cfg = cfg
        self._host = (cfg.dgx_tailnet_host or "").strip()
        self._port = int(cfg.dgx_ollama_port or 11434)
        self._model = (cfg.dgx_whisper_model or "whisper-large-v3").strip()
        self._timeout_sec = float(cfg.dgx_request_timeout_sec or 300.0)
        self._fallback_name = (cfg.transcription_fallback_provider or "openai").strip()
        self._fallback: Optional[TranscriptionProvider] = None
        self._initialized = False

    def initialize(self) -> None:
        """Load fallback transcription provider (cloud) per ADR-096."""
        if self._initialized:
            return
        if not self._host:
            raise ValueError("dgx_tailnet_host is required for tailnet_dgx_whisper")
        fb_data = self.cfg.model_dump()
        fb_data["transcription_provider"] = self._fallback_name
        fb_cfg = config.Config.model_validate(fb_data)
        self._fallback = create_transcription_provider(fb_cfg)
        self._fallback.initialize()
        self._initialized = True

    def _ensure_init(self) -> None:
        if not self._initialized:
            self.initialize()

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Return transcript text, using DGX or fallback provider."""
        self._ensure_init()
        text, _segments, _dur = self._transcribe_with_fallback(audio_path, language)
        return text

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> tuple[dict[str, object], float]:
        """Return transcript dict with segments and elapsed seconds."""
        self._ensure_init()
        text, segments, duration = self._transcribe_with_fallback(audio_path, language)
        return (
            {
                "text": text,
                "segments": segments,
                "language": language or "en",
            },
            duration,
        )

    def _transcribe_with_fallback(
        self,
        audio_path: str,
        language: str | None,
    ) -> tuple[str, list[dict[str, object]], float]:
        assert self._fallback is not None
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                if check_ollama_health(
                    self._host,
                    port=self._port,
                    require_model_substring=self._model.split(":")[0],
                ):
                    return self._transcribe_ollama(audio_path, language)
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "DGX Whisper attempt %s failed: %s",
                    attempt + 1,
                    format_exception_for_log(exc),
                )
            if attempt == 0:
                time.sleep(_RETRY_BACKOFF_SEC)

        reason = format_exception_for_log(last_err) if last_err else "health_check_failed"
        emit_dgx_fallback_breadcrumb(
            stage="transcription",
            model=self._model,
            failure_reason=reason,
        )
        logger.warning(
            "Falling back from DGX Whisper to %s (%s)",
            self._fallback_name,
            reason,
        )
        result = self._fallback.transcribe_with_segments(audio_path, language)
        raw_segments = result[0].get("segments") or []
        segments = cast(List[dict[str, object]], raw_segments)
        return (
            str(result[0].get("text", "")),
            segments,
            float(result[1]),
        )

    def cleanup(self) -> None:
        """Release fallback provider resources."""
        if self._fallback is not None:
            self._fallback.cleanup()

    def _transcribe_ollama(
        self,
        audio_path: str,
        language: str | None,
    ) -> tuple[str, list[dict[str, object]], float]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx required for tailnet_dgx_whisper") from exc

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(audio_path)

        base = dgx_ollama_base_url(self._host, self._port)
        url = f"{base}/api/transcribe"
        started = time.perf_counter()
        with path.open("rb") as audio_file:
            files = {"file": (path.name, audio_file, "application/octet-stream")}
            data: dict[str, Any] = {"model": self._model}
            if language:
                data["language"] = language
            with httpx.Client(timeout=self._timeout_sec) as client:
                resp = client.post(url, data=data, files=files)
        resp.raise_for_status()
        payload = resp.json()
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("empty transcription from DGX Ollama")
        duration = float(time.perf_counter() - started)
        segments: list[dict[str, object]] = []
        if isinstance(payload.get("segments"), list):
            segments = [s for s in payload["segments"] if isinstance(s, dict)]
        return text, segments, duration
