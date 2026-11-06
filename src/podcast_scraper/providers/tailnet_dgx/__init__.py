"""Tailnet-routed DGX providers (RFC-089 (not in this repository) /
ADR-096 (not in this repository) / #814 / #926)."""

from .diarization_provider import TailnetDgxDiarizationProvider
from .health import (
    check_faster_whisper_health,
    check_ollama_health,
    check_pyannote_diarize_health,
)
from .telemetry import emit_dgx_fallback_breadcrumb
from .whisper_provider import TailnetDgxWhisperTranscriptionProvider

__all__ = [
    "TailnetDgxDiarizationProvider",
    "TailnetDgxWhisperTranscriptionProvider",
    "check_faster_whisper_health",
    "check_ollama_health",
    "check_pyannote_diarize_health",
    "emit_dgx_fallback_breadcrumb",
]
