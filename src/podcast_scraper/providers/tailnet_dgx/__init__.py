"""Tailnet-routed DGX providers (RFC-089 / ADR-096)."""

from .health import check_ollama_health
from .telemetry import emit_dgx_fallback_breadcrumb
from .whisper_provider import TailnetDgxWhisperTranscriptionProvider

__all__ = [
    "TailnetDgxWhisperTranscriptionProvider",
    "check_ollama_health",
    "emit_dgx_fallback_breadcrumb",
]
