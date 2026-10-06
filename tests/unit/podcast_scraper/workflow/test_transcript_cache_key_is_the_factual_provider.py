"""The transcript cache is keyed by the provider that produced the transcript — never the wrapper.

Operator (2026-10-06): the resilience strategy must have no effect on the cache key; only the
provider actually applied does. Before this, a failover chain named itself ``fallback_chain``
whatever tier ran, so the same DGX transcript got one key under ``failover`` (wrapped) and another
under ``hold`` (bare) — and a transcript from a fallback tier shared the key of DGX ones. Found by
the pipeline-check real A/B (#2287): main and the multilingual branch never shared a cache entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pytest

from podcast_scraper import config
from podcast_scraper.cache import transcript_cache
from podcast_scraper.providers.resilience.fallback import FallbackChainTranscriptionProvider
from podcast_scraper.workflow.episode_processor import _transcript_cache_identity

pytestmark = pytest.mark.unit


class _Provider:
    """A transcription provider stand-in: a name, a model, and an optional infra failure."""

    def __init__(self, name: str, model: str, *, fails: bool = False) -> None:
        self.name = name
        self.model = model
        self.fails = fails

    def initialize(self) -> None:
        return None

    def transcribe_with_segments(
        self, audio_path: str, language: Any = None, **_: Any
    ) -> Tuple[Dict[str, Any], float]:
        if self.fails:
            raise ConnectionError(f"{self.name} is down")  # infra: the chain cascades
        return {"text": f"from {self.name}", "segments": [], "model_used": self.model}, 0.1


def _cfg(tmp_path: Path) -> config.Config:
    return config.Config(rss="https://example.com/f.xml", output_dir=str(tmp_path))


def _builder(provider: _Provider) -> Callable[[], Any]:
    return lambda: provider


def _chain(*providers: _Provider) -> FallbackChainTranscriptionProvider:
    return FallbackChainTranscriptionProvider([(p.name, _builder(p)) for p in providers])


def test_the_strategy_has_no_effect_on_the_key(tmp_path: Path) -> None:
    """`failover` (wrapped) and `hold` (bare) key the same DGX transcript identically."""
    cfg = _cfg(tmp_path)
    dgx = _Provider("tailnet_dgx_whisper", "large-v3")
    chain = _chain(dgx, _Provider("deepgram", "nova-3"))
    chain.initialize()
    chain.transcribe_with_segments("a.mp3")

    bare = _transcript_cache_identity(dgx, cfg, produced=True)
    assert bare == ("tailnet_dgx_whisper", "large-v3")
    assert _transcript_cache_identity(chain, cfg, produced=False) == bare
    assert _transcript_cache_identity(chain, cfg, produced=True) == bare


def test_a_fallback_transcript_is_keyed_by_the_tier_that_made_it(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    chain = _chain(
        _Provider("tailnet_dgx_whisper", "large-v3", fails=True), _Provider("deepgram", "nova-3")
    )
    chain.initialize()
    chain.transcribe_with_segments("a.mp3")

    assert _transcript_cache_identity(chain, cfg, produced=True) == ("deepgram", "nova-3")
    # The lookup is still keyed on the primary: a later run with a healthy DGX misses the
    # fallback's entry and transcribes again with DGX.
    assert _transcript_cache_identity(chain, cfg, produced=False) == (
        "tailnet_dgx_whisper",
        "large-v3",
    )


def test_end_to_end_a_fallback_transcript_is_never_served_as_the_primarys(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cache = str(tmp_path / "cache")
    down = _chain(
        _Provider("tailnet_dgx_whisper", "large-v3", fails=True), _Provider("deepgram", "nova-3")
    )
    down.initialize()
    result, _ = down.transcribe_with_segments("a.mp3")
    name, model = _transcript_cache_identity(down, cfg, produced=True)
    transcript_cache.save_transcript_to_cache(
        "hash1", str(result["text"]), provider_name=name, model=model, cache_dir=cache
    )

    healthy_hold = _Provider("tailnet_dgx_whisper", "large-v3")  # the next run, `hold`, DGX up
    name, model = _transcript_cache_identity(healthy_hold, cfg, produced=False)
    assert (
        transcript_cache.get_cached_transcript("hash1", cache, provider_name=name, model=model)
        is None
    )


def test_end_to_end_one_dgx_transcript_serves_both_strategies(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cache = str(tmp_path / "cache")
    dgx = _Provider("tailnet_dgx_whisper", "large-v3")
    failover = _chain(dgx, _Provider("deepgram", "nova-3"))
    failover.initialize()
    result, _ = failover.transcribe_with_segments("a.mp3")
    name, model = _transcript_cache_identity(failover, cfg, produced=True)
    transcript_cache.save_transcript_to_cache(
        "hash1", str(result["text"]), provider_name=name, model=model, cache_dir=cache
    )

    name, model = _transcript_cache_identity(dgx, cfg, produced=False)  # `hold`: the bare provider
    assert (
        transcript_cache.get_cached_transcript("hash1", cache, provider_name=name, model=model)
        == "from tailnet_dgx_whisper"
    )


def test_a_plain_provider_is_keyed_exactly_as_before(tmp_path: Path) -> None:
    """No wrapper, no change: the identity is the provider's own name and model."""
    cfg = _cfg(tmp_path)
    plain = _Provider("whisper", "base.en")
    assert _transcript_cache_identity(plain, cfg, produced=False) == ("whisper", "base.en")
    assert _transcript_cache_identity(None, cfg, produced=False) == (None, None)
