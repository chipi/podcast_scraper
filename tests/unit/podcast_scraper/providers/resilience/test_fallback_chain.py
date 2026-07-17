"""Unit tests for the RFC-105 (#1198) tiered transcription failover chain.

Covers the contract that matters: an infra failure cascades to the next tier, a content failure does
not, the last tier's error propagates once the chain is exhausted, fallback tiers initialize lazily,
and the winning tier is attributed in the result (#1046 provenance).
"""

from __future__ import annotations

from typing import Any, cast, List, Optional, Tuple

import httpx
import pytest

from podcast_scraper.providers.guardrails.exceptions import GuardrailViolation
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.resilience.fallback import (
    FallbackChainDiarizationProvider,
    FallbackChainTranscriptionProvider,
    is_infra_failure,
)


class _FakeTier:
    """A transcription provider stand-in that records init/cleanup and either returns or raises."""

    def __init__(
        self,
        result: Optional[dict[str, object]] = None,
        raises: Optional[BaseException] = None,
    ) -> None:
        self._result = result
        self._raises = raises
        self.initialized = 0
        self.cleaned = 0
        self.calls = 0

    def initialize(self) -> None:
        self.initialized += 1

    def cleanup(self) -> None:
        self.cleaned += 1

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        pipeline_metrics: Any | None = None,
        episode_duration_seconds: int | None = None,
        call_metrics: Any | None = None,
    ) -> Tuple[dict[str, object], float]:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        assert self._result is not None
        return self._result, 1.0


def _chain(tiers: List[Tuple[str, _FakeTier]]) -> FallbackChainTranscriptionProvider:
    # The chain takes (name, builder) — wrap each pre-made fake tier in a builder that returns it.
    # _FakeTier is a structural stand-in for TranscriptionProvider (duck-typed by the chain).
    built = [(name, (lambda inst=inst: inst)) for name, inst in tiers]
    return FallbackChainTranscriptionProvider(cast(Any, built))


# --- is_infra_failure classification -------------------------------------------------------------


def test_infra_failures_cascade() -> None:
    assert is_infra_failure(httpx.ConnectError("refused")) is True
    assert is_infra_failure(httpx.ReadTimeout("slow")) is True
    assert is_infra_failure(TimeoutError()) is True
    assert is_infra_failure(GuardrailViolation("whisper", "garbage_response", "WER=1.0")) is True
    assert is_infra_failure(RuntimeError("something odd")) is True  # default: cascade


def test_5xx_cascades_but_4xx_does_not() -> None:
    req = httpx.Request("POST", "http://x")
    for code in (500, 502, 503):
        resp = httpx.Response(code, request=req)
        assert is_infra_failure(httpx.HTTPStatusError("x", request=req, response=resp)) is True
    for code in (400, 401, 404, 422):
        resp = httpx.Response(code, request=req)
        assert is_infra_failure(httpx.HTTPStatusError("x", request=req, response=resp)) is False
    # 408 (Request Timeout) and 429 (Too Many Requests) are transient/pressure, not content faults
    # -> cascade.
    for code in (408, 429):
        resp = httpx.Response(code, request=req)
        assert is_infra_failure(httpx.HTTPStatusError("x", request=req, response=resp)) is True


# --- chain behaviour -----------------------------------------------------------------------------


def test_primary_success_never_touches_fallbacks() -> None:
    primary = _FakeTier(result={"text": "primary", "segments": []})
    fb = _FakeTier(result={"text": "fb", "segments": []})
    chain = _chain([("moss", primary), ("openai", fb)])
    chain.initialize()

    result, _elapsed = chain.transcribe_with_segments("a.mp3")
    assert result["text"] == "primary"
    assert fb.calls == 0
    assert fb.initialized == 0  # lazy: an unused fallback is never initialized


def test_infra_failure_advances_to_next_tier() -> None:
    primary = _FakeTier(raises=httpx.ConnectError("dgx down"))
    fb = _FakeTier(result={"text": "cloud", "segments": []})
    chain = _chain([("moss", primary), ("openai", fb)])
    chain.initialize()

    result, _elapsed = chain.transcribe_with_segments("a.mp3")
    assert result["text"] == "cloud"
    assert primary.calls == 1
    assert fb.calls == 1
    assert fb.initialized == 1  # initialized lazily, right before use


def test_content_failure_does_not_cascade() -> None:
    """A deterministic 4xx is the request's fault; every tier would reject it, so stop and raise."""
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(400, request=req)
    primary = _FakeTier(raises=httpx.HTTPStatusError("bad", request=req, response=resp))
    fb = _FakeTier(result={"text": "cloud", "segments": []})
    chain = _chain([("moss", primary), ("openai", fb)])
    chain.initialize()

    with pytest.raises(httpx.HTTPStatusError):
        chain.transcribe_with_segments("a.mp3")
    assert fb.calls == 0  # did not waste the next tier on a content failure


def test_exhausted_chain_raises_last_error() -> None:
    primary = _FakeTier(raises=httpx.ConnectError("dgx down"))
    fb = _FakeTier(raises=httpx.ReadTimeout("cloud slow"))
    chain = _chain([("moss", primary), ("openai", fb)])
    chain.initialize()

    with pytest.raises(httpx.ReadTimeout):
        chain.transcribe_with_segments("a.mp3")
    assert primary.calls == 1 and fb.calls == 1


def test_fallback_winner_is_attributed_when_no_model_used() -> None:
    """A fallback tier that wins and did not attribute a model is stamped with its tier name, so a
    run never credits the primary's model for a call it never made (#1046)."""
    primary = _FakeTier(raises=httpx.ConnectError("dgx down"))
    fb = _FakeTier(result={"text": "cloud", "segments": []})  # no model_used
    chain = _chain([("moss", primary), ("openai", fb)])
    chain.initialize()

    result, _elapsed = chain.transcribe_with_segments("a.mp3")
    assert result["model_used"] == "openai:default"


def test_fallback_winner_keeps_its_own_model_used() -> None:
    primary = _FakeTier(raises=httpx.ConnectError("dgx down"))
    fb = _FakeTier(result={"text": "cloud", "segments": [], "model_used": "whisper-1"})
    chain = _chain([("moss", primary), ("openai", fb)])
    chain.initialize()

    result, _elapsed = chain.transcribe_with_segments("a.mp3")
    assert result["model_used"] == "whisper-1"  # not overwritten


def test_primary_winner_is_not_stamped() -> None:
    primary = _FakeTier(result={"text": "primary", "segments": []})
    chain = _chain([("moss", primary), ("openai", _FakeTier())])
    chain.initialize()

    result, _elapsed = chain.transcribe_with_segments("a.mp3")
    assert "model_used" not in result  # tier 0 winning is not annotated


def test_cleanup_releases_only_initialized_tiers() -> None:
    primary = _FakeTier(result={"text": "primary", "segments": []})
    fb = _FakeTier(result={"text": "fb", "segments": []})
    chain = _chain([("moss", primary), ("openai", fb)])
    chain.initialize()
    chain.transcribe_with_segments("a.mp3")  # primary wins; fb stays uninitialized

    chain.cleanup()
    assert primary.cleaned == 1
    assert fb.cleaned == 0  # never initialized, so nothing to release


def test_empty_chain_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one tier"):
        FallbackChainTranscriptionProvider([])


# --- diarization chain ---------------------------------------------------------------------------


class _FakeDiarTier:
    def __init__(
        self,
        model_name: str = "",
        raises: Optional[BaseException] = None,
    ) -> None:
        self._model_name = model_name
        self._raises = raises
        self.initialized = 0
        self.calls = 0

    def initialize(self) -> None:
        self.initialized += 1

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return DiarizationResult(
            segments=[DiarizationSegment(0.0, 1.0, "SPEAKER_00")],
            num_speakers=1,
            model_name=self._model_name,
        )


def _diar_chain(tiers: List[Tuple[str, _FakeDiarTier]]) -> FallbackChainDiarizationProvider:
    built = [(name, (lambda inst=inst: inst)) for name, inst in tiers]
    return FallbackChainDiarizationProvider(cast(Any, built))


def test_diar_primary_success_never_touches_fallbacks() -> None:
    primary = _FakeDiarTier(model_name="dgx")
    fb = _FakeDiarTier(model_name="local")
    chain = _diar_chain([("tailnet_dgx", primary), ("local", fb)])
    chain.initialize()

    result = chain.diarize("a.wav", num_speakers=2)
    assert result.model_name == "dgx"
    assert fb.calls == 0 and fb.initialized == 0  # lazy: unused tier untouched


def test_diar_infra_failure_advances_to_next_tier() -> None:
    primary = _FakeDiarTier(raises=httpx.ConnectError("dgx down"))
    fb = _FakeDiarTier(model_name="local")
    chain = _diar_chain([("tailnet_dgx", primary), ("local", fb)])
    chain.initialize()

    result = chain.diarize("a.wav")
    assert result.model_name == "local"
    assert primary.calls == 1 and fb.calls == 1 and fb.initialized == 1


def test_diar_content_failure_does_not_cascade() -> None:
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(400, request=req)
    primary = _FakeDiarTier(raises=httpx.HTTPStatusError("bad", request=req, response=resp))
    fb = _FakeDiarTier(model_name="local")
    chain = _diar_chain([("tailnet_dgx", primary), ("local", fb)])
    chain.initialize()

    with pytest.raises(httpx.HTTPStatusError):
        chain.diarize("a.wav")
    assert fb.calls == 0


def test_diar_exhausted_chain_raises_last_error() -> None:
    primary = _FakeDiarTier(raises=httpx.ConnectError("dgx down"))
    local = _FakeDiarTier(raises=RuntimeError("no pyannote installed"))
    deepgram = _FakeDiarTier(raises=httpx.ReadTimeout("cloud slow"))
    chain = _diar_chain([("tailnet_dgx", primary), ("local", local), ("deepgram", deepgram)])
    chain.initialize()

    with pytest.raises(httpx.ReadTimeout):
        chain.diarize("a.wav")
    assert primary.calls == 1 and local.calls == 1 and deepgram.calls == 1


def test_diar_full_ladder_dgx_then_local_then_deepgram() -> None:
    """The prod ladder: DGX down, local pyannote can't run here, deepgram serves."""
    primary = _FakeDiarTier(raises=httpx.ConnectError("dgx down"))
    local = _FakeDiarTier(raises=RuntimeError("no pyannote installed on this VPS"))
    deepgram = _FakeDiarTier(model_name="nova-3")
    chain = _diar_chain([("tailnet_dgx", primary), ("local", local), ("deepgram", deepgram)])
    chain.initialize()

    result = chain.diarize("a.wav")
    assert result.model_name == "nova-3"
    assert deepgram.calls == 1
