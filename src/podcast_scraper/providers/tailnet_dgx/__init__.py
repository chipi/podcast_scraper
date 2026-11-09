"""Tailnet-routed DGX providers (RFC-089 (not in this repository) /
ADR-096 (not in this repository) / #814 / #926)."""

from typing import Any

from .health import (
    check_faster_whisper_health,
    check_ollama_health,
    check_pyannote_diarize_health,
)
from .telemetry import emit_dgx_fallback_breadcrumb
from .whisper_provider import TailnetDgxWhisperTranscriptionProvider


def __getattr__(name: str) -> Any:
    # The diarization provider belongs to the speaker stage (E04 S2): importing this package for
    # the Whisper provider must not load it.
    if name == "TailnetDgxDiarizationProvider":
        from .diarization_provider import TailnetDgxDiarizationProvider

        return TailnetDgxDiarizationProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TailnetDgxDiarizationProvider",
    "TailnetDgxWhisperTranscriptionProvider",
    "check_faster_whisper_health",
    "check_ollama_health",
    "check_pyannote_diarize_health",
    "emit_dgx_fallback_breadcrumb",
]
