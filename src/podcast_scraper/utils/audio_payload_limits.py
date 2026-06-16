"""Detect OpenAI / Gemini-style Whisper API payload and file-size limit errors (GitHub #557)."""

from __future__ import annotations

from typing import Any, Optional, Union

# OpenAI gpt-4o-transcribe family hard duration cap (seconds).
OPENAI_GPT4O_TRANSCRIBE_MAX_DURATION_SECONDS = 1400.0

# Voxtral chat-style models (e.g. voxtral-mini-latest) degrade into a repetition
# loop on audio over ~30 min (mistral_provider.transcribe warns and recommends
# <25-min windows or a dedicated "…-transcribe…" model). Cap loop-prone models at
# 25 min so long episodes are chunked by duration; transcribe-dedicated models that
# handle long-form natively are exempt (see transcription_max_chunk_duration_seconds).
MISTRAL_VOXTRAL_LOOP_MAX_DURATION_SECONDS = 1500.0

# Conservative single-request audio byte cap (OpenAI's 25 MiB). Used as the default
# for every API transcription provider when sizing audio chunks.
_DEFAULT_MAX_AUDIO_BYTES = 25 * 1024 * 1024

# Deepgram pre-recorded API accepts up to 2 GB per request and handles long audio
# natively in a single call. Sending whole episodes (instead of 25 MiB chunks) is
# what makes server-side diarization coherent: Deepgram numbers speakers PER
# request, so chunking fragments speaker ids across chunk boundaries (chunk-2's
# "speaker 0" need not be chunk-1's). One request → one consistent speaker space
# (D3). The few episodes still over 2 GB fall back to chunked transcription with a
# warning that diarization speaker ids are chunk-local.
_DEEPGRAM_MAX_AUDIO_BYTES = 2 * 1024 * 1024 * 1024

# Gemini sends audio inline (genai_types.Part.from_bytes — no Files API upload path
# in gemini_provider). The inline / in-request payload cap was raised from 20 MiB to
# 100 MB on 2026-01-12 (ai.google.dev / blog.google). Inline data is base64-encoded
# in the request (~1.34x), so the raw audio must stay well under 100 MB: 64 MiB raw
# ≈ 86 MB encoded, leaving headroom for the prompt and thinking budget. Larger than
# the OpenAI default → fewer chunk seams → more coherent long-episode transcripts.
_GEMINI_MAX_AUDIO_BYTES = 64 * 1024 * 1024

# Mistral Voxtral transcription API accepts up to 500 MB per request (Transcribe-2:
# 1 GB). The real constraint for the chat-style models is duration, not bytes (see
# MISTRAL_VOXTRAL_LOOP_MAX_DURATION_SECONDS), so this just stops byte-based chunking
# from firing prematurely; the duration governor handles long-audio quality.
_MISTRAL_MAX_AUDIO_BYTES = 500 * 1024 * 1024

# Per-provider overrides go here as each API's real limit is confirmed. Keep the
# conservative default rather than an unverified higher value — an over-small cap
# just chunks more (works); an over-large cap risks an oversize upload that fails.
_PROVIDER_MAX_AUDIO_BYTES: dict[str, int] = {
    "openai": _DEFAULT_MAX_AUDIO_BYTES,
    "deepgram": _DEEPGRAM_MAX_AUDIO_BYTES,
    "gemini": _GEMINI_MAX_AUDIO_BYTES,
    "mistral": _MISTRAL_MAX_AUDIO_BYTES,
}


def transcription_max_bytes(cfg: Any) -> int:
    """Max single-request audio bytes for the configured transcription provider.

    Replaces the previous hard-coded OpenAI assumption with a per-provider lookup;
    unconfirmed providers fall back to the conservative OpenAI cap.
    """
    provider = str(getattr(cfg, "transcription_provider", None) or "").lower()
    return _PROVIDER_MAX_AUDIO_BYTES.get(provider, _DEFAULT_MAX_AUDIO_BYTES)


def transcription_max_chunk_duration_seconds(cfg: Any) -> Optional[float]:
    """Return max single-request audio duration when provider/model enforces a cap.

    Covers both hard API duration limits (OpenAI gpt-4o-transcribe) and quality
    ceilings that behave like one (Voxtral chat-style models loop on long audio).
    """
    provider = str(getattr(cfg, "transcription_provider", None) or "").lower()
    if provider == "openai":
        model = str(getattr(cfg, "openai_transcription_model", None) or "whisper-1")
        if model in ("gpt-4o-transcribe", "gpt-4o-mini-transcribe"):
            return OPENAI_GPT4O_TRANSCRIBE_MAX_DURATION_SECONDS
        return None
    if provider == "mistral":
        model = str(getattr(cfg, "mistral_transcription_model", None) or "voxtral-mini-latest")
        # Dedicated transcription models ("…-transcribe…", e.g. voxtral-mini-transcribe-2)
        # handle long-form natively; only the chat-style models loop on long audio.
        if "transcribe" in model.lower():
            return None
        return MISTRAL_VOXTRAL_LOOP_MAX_DURATION_SECONDS
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
