"""#913: OpenAI Whisper is eligible for the local pyannote diarization pass.

OpenAI ``verbose_json`` returns timestamped Whisper-format segments and the audio
is downloaded locally (for the API upload), so pyannote can run its second pass and
align speaker turns — exactly like ``whisper`` / ``tailnet_dgx_whisper``. These
guard that ``diarize`` / ``screenplay`` are NOT coerced off for ``openai`` while
still being coerced off for the plain-text cloud providers (gemini / mistral).
"""

from __future__ import annotations

import pytest

from podcast_scraper import config as config_mod
from podcast_scraper.config import (
    _DIARIZATION_ELIGIBLE_TRANSCRIPTION_PROVIDERS,
    Config,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_coerce_logs():
    # The coercion INFO log is one-shot per process; reset so ordering doesn't matter.
    config_mod.reset_screenplay_issue_562_gates()
    yield
    config_mod.reset_screenplay_issue_562_gates()


def test_openai_in_eligibility_set():
    assert "openai" in _DIARIZATION_ELIGIBLE_TRANSCRIPTION_PROVIDERS


def test_openai_diarize_off_by_default_optin():
    """#913: openai is eligible but opt-in — an unset ``diarize`` stays OFF so existing
    openai runs don't silently start a pyannote pass."""
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "openai",
            "openai_api_key": "sk-test",
            # diarize intentionally unset
        }
    )
    assert cfg.diarize is False


def test_whisper_diarize_on_by_default():
    """Local Whisper keeps the diarize-on-by-default behavior (not opt-in)."""
    cfg = Config.model_validate(
        {"rss_url": "https://example.com/feed.xml", "transcription_provider": "whisper"}
    )
    assert cfg.diarize is True


def test_openai_diarize_not_coerced_off():
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "openai",
            "openai_api_key": "sk-test",
            "diarize": True,
        }
    )
    assert cfg.diarize is True
    # An eligible provider with diarize on also defaults screenplay on.
    assert cfg.screenplay is True


def test_openai_explicit_screenplay_survives():
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "openai",
            "openai_api_key": "sk-test",
            "diarize": True,
            "screenplay": True,
        }
    )
    assert cfg.diarize is True
    assert cfg.screenplay is True


@pytest.mark.parametrize(
    "provider,key_field,key",
    [
        ("gemini", "gemini_api_key", "g-test"),
        ("mistral", "mistral_api_key", "m-test"),
    ],
)
def test_plaintext_cloud_providers_raise_on_explicit_diarize_true(
    provider, key_field, key, monkeypatch
):
    """Gemini / Mistral emit plain text (no segments). diarize=true here is a
    misconfig → strict-mode raises (2026-06-15 diarize-everywhere change)."""
    monkeypatch.delenv("PODCAST_SCRAPER_DIARIZE_LAX", raising=False)
    with pytest.raises(ValueError, match=r"diarize=true"):
        Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": provider,
                key_field: key,
                "diarize": True,
            }
        )
    assert provider not in _DIARIZATION_ELIGIBLE_TRANSCRIPTION_PROVIDERS


@pytest.mark.parametrize(
    "provider,key_field,key",
    [
        ("gemini", "gemini_api_key", "g-test"),
        ("mistral", "mistral_api_key", "m-test"),
    ],
)
def test_plaintext_cloud_providers_coerce_with_lax_env(provider, key_field, key, monkeypatch):
    """Escape hatch ``PODCAST_SCRAPER_DIARIZE_LAX=1`` restores the
    pre-2026-06-15 silent-coerce behavior for CI / migration runs."""
    monkeypatch.setenv("PODCAST_SCRAPER_DIARIZE_LAX", "1")
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": provider,
            key_field: key,
            "diarize": True,
        }
    )
    assert cfg.diarize is False
