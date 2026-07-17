"""RFC-105 (#1198): the diarization factory wraps the primary + ladder in a FallbackChain.

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
    assert [name for name, _ in provider._tiers] == ["tailnet_dgx", "local", "deepgram"]
    # primary + two fallback tiers were each built once.
    assert [c.args[1] for c in build.call_args_list] == ["tailnet_dgx", "local", "deepgram"]


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
