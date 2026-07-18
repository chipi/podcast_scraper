"""End-to-end coverage for the RFC-106 (#1198) tiered fallback chains.

The chain wraps ordered transcription / diarization tiers and fails over from a broken primary to
the next tier on an *infrastructure* failure, re-raising a content failure or an exhausted chain,
and builds each tier lazily so a never-reached tier's credentials are never required. In production
the tiers are real DGX / cloud providers; the e2e suite runs under a network guard with no DGX, so
here the tiers are in-process stubs. That lets us exercise the chain's own behaviour — failover,
provenance, lazy construction, cleanup — deterministically and without a live service.
"""

from __future__ import annotations

from typing import Any, Callable, cast, List, Optional, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.resilience.fallback import (
    FallbackChainDiarizationProvider,
    FallbackChainTranscriptionProvider,
)

pytestmark = [pytest.mark.e2e]


class _StubTranscription:
    """A transcription tier that either returns a fixed transcript or raises on every call."""

    def __init__(
        self, *, text: Optional[str] = None, raises: Optional[BaseException] = None
    ) -> None:
        self._text = text
        self._raises = raises
        self.initialized = False
        self.cleaned = False

    def initialize(self) -> None:
        self.initialized = True

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: Optional[str] = None,
        *,
        pipeline_metrics: Any = None,
        episode_duration_seconds: Optional[int] = None,
        call_metrics: Any = None,
    ) -> Tuple[dict[str, object], float]:
        if self._raises is not None:
            raise self._raises
        return {
            "text": self._text,
            "segments": [{"start": 0.0, "end": 1.0, "text": self._text}],
        }, 0.01

    def cleanup(self) -> None:
        self.cleaned = True


class _StubDiarization:
    """A diarization tier that either returns a fixed result or raises on every call."""

    def __init__(self, *, raises: Optional[BaseException] = None, model_name: str = "") -> None:
        self._raises = raises
        self._model = model_name
        self.initialized = False
        self.cleaned = False

    def initialize(self) -> None:
        self.initialized = True

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        if self._raises is not None:
            raise self._raises
        return DiarizationResult(
            segments=[DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00")],
            num_speakers=1,
            model_name=self._model,
        )

    def cleanup(self) -> None:
        self.cleaned = True


# The chains duck-type their tiers (any object exposing the provider methods works); the stubs are
# structural stand-ins, so cast past the nominal Protocol type the constructors annotate.
def _t_chain(tiers: List[Tuple[str, Callable[[], Any]]]) -> FallbackChainTranscriptionProvider:
    return FallbackChainTranscriptionProvider(cast(Any, tiers))


def _d_chain(tiers: List[Tuple[str, Callable[[], Any]]]) -> FallbackChainDiarizationProvider:
    return FallbackChainDiarizationProvider(cast(Any, tiers))


# --------------------------------------------------------------------------- #
# transcription chain                                                          #
# --------------------------------------------------------------------------- #
def test_transcription_chain_fails_over_to_next_tier_on_infra_error() -> None:
    """A primary that raises an (infra-classified) error hands off to the next tier, and the winning
    fallback tier is credited in the provenance (#1046) since it did not attribute a model itself.
    """
    primary = _StubTranscription(raises=RuntimeError("dgx unreachable"))
    fallback = _StubTranscription(text="from the fallback tier")
    chain = _t_chain([("tailnet_dgx_whisper", lambda: primary), ("openai", lambda: fallback)])
    chain.initialize()  # eagerly builds + inits the primary only

    result, elapsed = chain.transcribe_with_segments("ep.wav")

    assert result["text"] == "from the fallback tier"
    assert result["model_used"] == "openai:default"
    assert elapsed >= 0
    assert primary.initialized and fallback.initialized


def test_transcription_chain_reraises_when_every_tier_fails() -> None:
    """When the last tier also fails, the chain is exhausted and the final exception propagates."""
    primary = _StubTranscription(raises=RuntimeError("primary infra"))
    last = _StubTranscription(raises=RuntimeError("last tier infra"))
    chain = _t_chain([("tailnet_dgx_whisper", lambda: primary), ("openai", lambda: last)])
    with pytest.raises(RuntimeError, match="last tier infra"):
        chain.transcribe_with_segments("ep.wav")


def test_transcription_chain_never_builds_an_unreached_tier() -> None:
    """A healthy primary means the fallback tier's builder is never invoked — the lazy contract that
    keeps a missing fallback credential from crashing a healthy run (#926)."""
    builds: List[str] = []

    def build_primary() -> _StubTranscription:
        builds.append("primary")
        return _StubTranscription(text="primary wins")

    def build_fallback() -> _StubTranscription:
        builds.append("fallback")
        raise AssertionError("the unreached fallback tier must never be built")

    chain = _t_chain([("tailnet_dgx_whisper", build_primary), ("openai", build_fallback)])
    result, _elapsed = chain.transcribe_with_segments("ep.wav")

    assert result["text"] == "primary wins"
    assert builds == ["primary"]


def test_transcription_chain_cleanup_releases_every_built_tier() -> None:
    primary = _StubTranscription(raises=RuntimeError("infra"))
    fallback = _StubTranscription(text="ok")
    chain = _t_chain([("tailnet_dgx_whisper", lambda: primary), ("openai", lambda: fallback)])
    chain.transcribe_with_segments("ep.wav")  # forces both tiers to build + init
    chain.cleanup()
    assert primary.cleaned and fallback.cleaned


# --------------------------------------------------------------------------- #
# diarization chain                                                            #
# --------------------------------------------------------------------------- #
def test_diarization_chain_fails_over_to_next_tier_on_infra_error() -> None:
    primary = _StubDiarization(raises=RuntimeError("pyannote host down"))
    fallback = _StubDiarization(model_name="deepgram-nova-3")
    chain = _d_chain(
        [
            ("pyannote_diarization_community1", lambda: primary),
            ("deepgram_diarization_nova3", lambda: fallback),
        ]
    )
    chain.initialize()

    result = chain.diarize("ep.wav")

    assert result.model_name == "deepgram-nova-3"
    assert result.num_speakers == 1
    assert [s.speaker for s in result.segments] == ["SPEAKER_00"]
    assert primary.initialized and fallback.initialized


def test_diarization_chain_reraises_when_every_tier_fails() -> None:
    primary = _StubDiarization(raises=RuntimeError("primary infra"))
    last = _StubDiarization(raises=RuntimeError("last tier infra"))
    chain = _d_chain(
        [
            ("pyannote_diarization_community1", lambda: primary),
            ("deepgram_diarization_nova3", lambda: last),
        ]
    )
    with pytest.raises(RuntimeError, match="last tier infra"):
        chain.diarize("ep.wav")


def test_diarization_chain_cleanup_releases_every_built_tier() -> None:
    primary = _StubDiarization(raises=RuntimeError("infra"))
    fallback = _StubDiarization(model_name="deepgram-nova-3")
    chain = _d_chain(
        [
            ("pyannote_diarization_community1", lambda: primary),
            ("deepgram_diarization_nova3", lambda: fallback),
        ]
    )
    chain.diarize("ep.wav")
    chain.cleanup()
    assert primary.cleaned and fallback.cleaned


# --------------------------------------------------------------------------- #
# factory wiring — the chains are assembled from a profile's declared ladder    #
# --------------------------------------------------------------------------- #
def test_transcription_factory_assembles_the_declared_ladder() -> None:
    """A DGX-primary cfg with a declared fallback ladder is wired into a chain by the factory, with
    every tier left lazy — assembly touches no network and needs no fallback credential."""
    from podcast_scraper import Config
    from podcast_scraper.transcription.factory import create_transcription_provider

    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_providers": ["openai"],
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "openai_api_key": "sk-test",
        }
    )
    provider = create_transcription_provider(cfg)
    assert isinstance(provider, FallbackChainTranscriptionProvider)
    assert provider._names == ["tailnet_dgx_whisper", "openai"]
    assert provider._providers == [None, None]  # tiers stay lazy — nothing built at assembly
