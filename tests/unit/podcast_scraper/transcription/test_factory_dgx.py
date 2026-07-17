"""Transcription factory tests for tailnet_dgx_whisper (ADR-096)."""

from __future__ import annotations

import pytest

from podcast_scraper import Config
from podcast_scraper.transcription.factory import create_transcription_provider


def _dgx_cfg() -> Config:
    return Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_provider": "openai",
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "openai_api_key": "sk-test",
        }
    )


def test_factory_wraps_dgx_primary_in_a_fallback_chain() -> None:
    """RFC-106 (#1198): a DGX-primary cfg with a fallback is wrapped in a FallbackChain — the DGX
    provider is the primary tier and the (legacy singular) fallback becomes the second tier."""
    from podcast_scraper.providers.openai.openai_provider import OpenAIProvider
    from podcast_scraper.providers.resilience.fallback import (
        FallbackChainTranscriptionProvider,
    )
    from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
        TailnetDgxWhisperTranscriptionProvider,
    )

    provider = create_transcription_provider(_dgx_cfg())
    assert isinstance(provider, FallbackChainTranscriptionProvider)
    assert provider._names == ["tailnet_dgx_whisper", "openai"]
    # Tiers are built lazily (builders, not instances). Build each via the chain to check its type.
    assert isinstance(provider._ensure_tier(0), TailnetDgxWhisperTranscriptionProvider)
    assert isinstance(provider._ensure_tier(1), OpenAIProvider)


def test_factory_wraps_moss_primary_with_full_ladder() -> None:
    """The MOSS gap #1174 opened: a moss primary now carries a chain when the profile emits one."""
    from podcast_scraper.providers.moss.moss_provider import MossTranscriptionProvider
    from podcast_scraper.providers.resilience.fallback import (
        FallbackChainTranscriptionProvider,
    )

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "moss",
            "transcription_fallback_providers": ["tailnet_dgx_whisper", "openai"],
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "openai_api_key": "sk-test",
        }
    )
    provider = create_transcription_provider(cfg)
    assert isinstance(provider, FallbackChainTranscriptionProvider)
    assert provider._names == ["moss", "tailnet_dgx_whisper", "openai"]
    assert isinstance(provider._ensure_tier(0), MossTranscriptionProvider)


def test_chain_builds_when_a_fallback_tier_key_is_absent() -> None:
    """Regression (hardening): tiers are built LAZILY, so a healthy MOSS primary must construct its
    chain even when the openai fallback tier has no OPENAI_API_KEY — the key is only needed if that
    tier is actually reached. Eager construction used to crash the whole run here."""
    from podcast_scraper.providers.resilience.fallback import (
        FallbackChainTranscriptionProvider,
    )

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "moss",
            "transcription_fallback_providers": ["tailnet_dgx_whisper", "openai"],
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            # NO openai_api_key — the openai tier must not be constructed at factory time.
        }
    )
    provider = create_transcription_provider(cfg)  # must NOT raise
    assert isinstance(provider, FallbackChainTranscriptionProvider)
    assert provider._providers == [None, None, None]  # nothing built yet


def test_factory_leaves_a_no_fallback_provider_unwrapped() -> None:
    """A local-whisper cfg with no ladder is returned bare — no chain overhead."""
    from podcast_scraper.providers.resilience.fallback import (
        FallbackChainTranscriptionProvider,
    )

    cfg = Config.model_validate(
        {"rss_url": "https://example.com/feed.xml", "transcription_provider": "whisper"}
    )
    provider = create_transcription_provider(cfg)
    assert not isinstance(provider, FallbackChainTranscriptionProvider)


def test_factory_rejects_experiment_mode_for_dgx() -> None:
    from podcast_scraper.providers.params import TranscriptionParams

    params = TranscriptionParams(model_name="base.en", device="cpu", language="en")
    with pytest.raises(ValueError, match="not supported in experiment mode"):
        create_transcription_provider("tailnet_dgx_whisper", params)
