"""Unit tests for provider audio payload limit detection (GitHub #557)."""

import pytest

from podcast_scraper.exceptions import ProviderRuntimeError
from podcast_scraper.utils.audio_payload_limits import is_provider_audio_payload_limit_error

pytestmark = [pytest.mark.unit, pytest.mark.module_utils]


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
