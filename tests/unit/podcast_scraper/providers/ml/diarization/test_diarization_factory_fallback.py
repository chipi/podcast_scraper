"""RFC-106 (#1198): the diarization factory wraps the primary + ladder in a FallbackChain.

Tier construction is patched so the test exercises the factory's wiring (does it build a chain, in
the right backend order?) without loading pyannote or hitting Deepgram.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from podcast_scraper import Config
from podcast_scraper.providers.ml.diarization import factory as diar_factory
from podcast_scraper.providers.resilience.fallback import FallbackChainDiarizationProvider


def _dgx_cfg_with_ladder() -> Config:
    return Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "diarization_provider": "tailnet_dgx",
            "diarize": True,
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "hf_token": "hf_test",
            "diarization_fallback_providers": ["local", "deepgram"],
        }
    )


def test_factory_wraps_dgx_diarization_in_a_chain() -> None:
    with patch.object(
        diar_factory, "_build_diarization_tier", side_effect=lambda cfg, backend: MagicMock()
    ) as build:
        provider = diar_factory.create_diarization_provider(_dgx_cfg_with_ladder())

    assert isinstance(provider, FallbackChainDiarizationProvider)
    assert provider._names == ["tailnet_dgx", "local", "deepgram"]
    # LAZY: only the primary (tier 0) is built at factory time (via chain.initialize()); the
    # fallback tiers are constructed on first use, not up-front.
    assert [c.args[1] for c in build.call_args_list] == ["tailnet_dgx"]
    assert provider._providers[1] is None and provider._providers[2] is None


def test_chain_builds_when_a_fallback_tier_credential_is_absent() -> None:
    """Regression (hardening #1198): a healthy DGX diarization primary must construct its chain even
    when the deepgram fallback tier has no DEEPGRAM_API_KEY — that tier is only built if reached.
    Eager construction (the diarization factory's deepgram branch raises at build time on a missing
    key) used to crash every episode despite a healthy primary."""
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "diarization_provider": "tailnet_dgx",
            "diarize": True,
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "hf_token": "hf_test",
            "diarization_fallback_providers": ["local", "deepgram"],
            # NO deepgram_api_key — the deepgram tier must not be constructed at factory time.
        }
    )
    provider = diar_factory.create_diarization_provider(cfg)  # must NOT raise
    assert isinstance(provider, FallbackChainDiarizationProvider)
    # local + deepgram deferred — no deepgram key needed to construct the chain.
    assert provider._providers[1] is None and provider._providers[2] is None


def test_factory_leaves_a_no_ladder_diarization_unwrapped() -> None:
    """A profile with no ladder is dispatched to the bare backend — no chain overhead."""
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "diarization_provider": "tailnet_dgx",
            "diarize": True,
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "hf_token": "hf_test",
        }
    )
    provider = diar_factory.create_diarization_provider(cfg)
    assert not isinstance(provider, FallbackChainDiarizationProvider)


def test_factory_reprocess_context_skips_fallback_chain_adr119() -> None:
    """ADR-122 (chunk 2): in reprocess run-context the self-hosted diarization provider must
    NEVER fall over to another model — a mixed-backend corpus is worse than a pause. Even with
    a declared fallback ladder, the factory returns the bare DGX provider; its ResiliencePolicy
    (hold-and-probe) is terminal. Mirrors the transcription factory's
    ``test_factory_reprocess_context_skips_fallback_chain_adr119`` (chunk 1)."""
    from podcast_scraper.providers.tailnet_dgx.diarization_provider import (
        TailnetDgxDiarizationProvider,
    )

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "diarization_provider": "tailnet_dgx",
            "diarize": True,
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "hf_token": "hf_test",
            "diarization_fallback_providers": ["local", "deepgram"],  # declared, but ignored
            "resilience_run_context": "reprocess",
        }
    )
    provider = diar_factory.create_diarization_provider(cfg)
    assert not isinstance(provider, FallbackChainDiarizationProvider)
    assert isinstance(provider, TailnetDgxDiarizationProvider)
