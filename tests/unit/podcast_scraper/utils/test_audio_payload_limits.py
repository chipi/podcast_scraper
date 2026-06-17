"""Unit tests for provider audio payload limit detection (GitHub #557)."""

from types import SimpleNamespace

import pytest

from podcast_scraper.exceptions import ProviderRuntimeError
from podcast_scraper.utils.audio_payload_limits import (
    is_provider_audio_payload_limit_error,
    MISTRAL_VOXTRAL_LOOP_MAX_DURATION_SECONDS,
    OPENAI_GPT4O_TRANSCRIBE_MAX_DURATION_SECONDS,
    transcription_max_bytes,
    transcription_max_chunk_duration_seconds,
)

pytestmark = [pytest.mark.unit, pytest.mark.module_utils]

_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HTTP 413", True),
        ("payload too large", True),
        ("Maximum content size", True),
        ("content size limit exceeded", True),
        ("size 26214400 bytes", True),
        ("file exceeds the 25mb limit", True),
        ("rate limited", False),
        ("exceeds budget", False),
    ],
)
def test_is_provider_audio_payload_limit_message(text: str, expected: bool) -> None:
    assert is_provider_audio_payload_limit_error(text) is expected


def test_is_provider_audio_payload_limit_exception() -> None:
    exc = ProviderRuntimeError(
        message="Error code: 413 - Maximum content size limit",
        provider="OpenAIProvider/Transcription",
    )
    assert is_provider_audio_payload_limit_error(exc) is True


@pytest.mark.parametrize(
    ("provider", "expected_bytes"),
    [
        ("openai", 25 * _MIB),
        ("deepgram", 2 * _GIB),
        ("gemini", 64 * _MIB),  # F1: inline limit raised to 100 MB (2026-01-12)
        ("mistral", 500 * _MIB),  # F1: Voxtral API accepts up to 500 MB
        ("whisper", 25 * _MIB),  # unconfirmed → conservative OpenAI default
        ("", 25 * _MIB),  # unset provider → default
    ],
)
def test_transcription_max_bytes_per_provider(provider: str, expected_bytes: int) -> None:
    cfg = SimpleNamespace(transcription_provider=provider)
    assert transcription_max_bytes(cfg) == expected_bytes


def test_gemini_cap_exceeds_default_but_stays_under_inline_limit() -> None:
    """Gemini gets more headroom than OpenAI yet leaves base64 room under 100 MB."""
    cfg = SimpleNamespace(transcription_provider="gemini")
    cap = transcription_max_bytes(cfg)
    assert cap > 25 * _MIB
    # base64 inflates ~1.34x; encoded request must stay under the 100 MB inline cap.
    assert cap * 4 / 3 < 100 * 1000 * 1000


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "whisper-1", None),
        ("openai", "gpt-4o-transcribe", OPENAI_GPT4O_TRANSCRIBE_MAX_DURATION_SECONDS),
        ("openai", "gpt-4o-mini-transcribe", OPENAI_GPT4O_TRANSCRIBE_MAX_DURATION_SECONDS),
        # Voxtral chat-style models loop on long audio → duration-capped.
        ("mistral", "voxtral-mini-latest", MISTRAL_VOXTRAL_LOOP_MAX_DURATION_SECONDS),
        ("mistral", "voxtral-small-24b", MISTRAL_VOXTRAL_LOOP_MAX_DURATION_SECONDS),
        # Dedicated transcription models handle long-form natively → no cap.
        ("mistral", "voxtral-mini-transcribe-2", None),
        ("deepgram", None, None),
        ("gemini", None, None),
    ],
)
def test_transcription_max_chunk_duration_seconds(provider, model, expected) -> None:
    cfg = SimpleNamespace(
        transcription_provider=provider,
        openai_transcription_model=model,
        mistral_transcription_model=model,
    )
    assert transcription_max_chunk_duration_seconds(cfg) == expected


def test_mistral_voxtral_exposes_both_byte_and_duration_caps() -> None:
    """The chunker reads both caps; for loop-prone Voxtral the duration governs, not bytes."""
    cfg = SimpleNamespace(
        transcription_provider="mistral",
        mistral_transcription_model="voxtral-mini-latest",
    )
    assert transcription_max_bytes(cfg) == 500 * _MIB
    duration = transcription_max_chunk_duration_seconds(cfg)
    assert duration == MISTRAL_VOXTRAL_LOOP_MAX_DURATION_SECONDS
    # ~25 min at a typical podcast bitrate is far under the 500 MB byte cap, so duration is
    # the binding constraint that triggers chunking — exactly the intended #964 F1 behaviour.
