"""Tailnet-routed DGX providers (RFC-089 / ADR-096 / #814)."""

from .health import check_faster_whisper_health, check_ollama_health
from .telemetry import emit_dgx_fallback_breadcrumb
from .whisper_provider import TailnetDgxWhisperTranscriptionProvider

__all__ = [
    "TailnetDgxWhisperTranscriptionProvider",
    "check_faster_whisper_health",
    "check_ollama_health",
    "emit_dgx_fallback_breadcrumb",
]
