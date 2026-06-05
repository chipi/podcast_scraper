"""Detect OpenAI / Gemini-style Whisper API payload and file-size limit errors (GitHub #557)."""

from __future__ import annotations

from typing import Any, Optional, Union

# OpenAI gpt-4o-transcribe family hard duration cap (seconds).
OPENAI_GPT4O_TRANSCRIBE_MAX_DURATION_SECONDS = 1400.0


def transcription_max_chunk_duration_seconds(cfg: Any) -> Optional[float]:
    """Return max single-request audio duration when provider enforces a cap."""
    provider = str(getattr(cfg, "transcription_provider", None) or "").lower()
    if provider != "openai":
        return None
    model = str(getattr(cfg, "openai_transcription_model", None) or "whisper-1")
    if model in ("gpt-4o-transcribe", "gpt-4o-mini-transcribe"):
        return OPENAI_GPT4O_TRANSCRIBE_MAX_DURATION_SECONDS
    return None


def is_provider_audio_payload_limit_error(exc_or_message: Union[BaseException, str]) -> bool:
    """Return True when the error is a known API payload / file size limit (not a generic bug).

    Covers HTTP 413, OpenAI ``maximum content size`` wording, and legacy validation strings
    that included ``exceeds`` / ``limit`` (pre-#557 substring checks).

    Args:
        exc_or_message: Exception from a provider or a raw message string.

    Returns:
        True if this should be treated as a policy / limit breach (typically skip episode).
    """
    if isinstance(exc_or_message, str):
        text = exc_or_message
    else:
        text = str(exc_or_message)
    lowered = text.lower()
    if "413" in lowered:
        return True
    if "payload too large" in lowered:
        return True
    if "maximum content size" in lowered:
        return True
    if "content size limit" in lowered:
        return True
    if "26214400" in lowered:
        return True
    if "exceeds" in lowered and "limit" in lowered:
        return True
    return False
