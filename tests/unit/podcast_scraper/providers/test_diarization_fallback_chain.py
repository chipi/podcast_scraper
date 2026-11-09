"""``FallbackChainDiarizationProvider`` after its move to ``providers.ml.diarization.fallback``
(E04 S2): lazy tiers, advancing on infra failure, re-raising at the end, best-effort cleanup."""

from __future__ import annotations

from typing import Any, cast, List

import pytest

from podcast_scraper.providers.ml.diarization.fallback import FallbackChainDiarizationProvider

pytestmark = pytest.mark.unit


class _Tier:
    def __init__(self, name: str, log: List[str], *, fail: BaseException | None = None) -> None:
        self.name, self.log, self.fail = name, log, fail

    def initialize(self) -> None:
        self.log.append(f"init:{self.name}")

    def diarize(self, audio_path: str, **kw: Any) -> str:
        self.log.append(f"diarize:{self.name}")
        if self.fail is not None:
            raise self.fail
        return f"result:{self.name}"

    def cleanup(self) -> None:
        self.log.append(f"cleanup:{self.name}")
        if self.name == "boom":
            raise RuntimeError("cleanup failed")


def _chain(*tiers: Any) -> FallbackChainDiarizationProvider:
    """A chain over duck-typed fake tiers (the chain only calls initialize/diarize/cleanup)."""
    return FallbackChainDiarizationProvider(cast(Any, list(tiers)))


def test_an_empty_chain_is_refused() -> None:
    with pytest.raises(ValueError):
        FallbackChainDiarizationProvider([])


def test_only_the_primary_is_built_until_a_fallback_is_needed() -> None:
    log: List[str] = []
    chain = _chain(("a", lambda: _Tier("a", log)), ("b", lambda: _Tier("b", log)))
    chain.initialize()
    assert chain.diarize("x.wav") == "result:a"
    assert log == ["init:a", "diarize:a"]


def test_an_infra_failure_advances_and_the_last_failure_is_raised() -> None:
    log: List[str] = []
    chain = _chain(
        ("a", lambda: _Tier("a", log, fail=ConnectionError("down"))), ("b", lambda: _Tier("b", log))
    )
    assert chain.diarize("x.wav", num_speakers=2) == "result:b"
    assert log == ["init:a", "diarize:a", "init:b", "diarize:b"]

    last = _chain(("a", lambda: _Tier("a", [], fail=TimeoutError("t"))))
    with pytest.raises(TimeoutError):
        last.diarize("x.wav")


def test_cleanup_releases_every_built_tier_even_when_one_raises() -> None:
    log: List[str] = []
    chain = _chain(
        ("boom", lambda: _Tier("boom", log, fail=ConnectionError("down"))),
        ("b", lambda: _Tier("b", log)),
        ("never", lambda: _Tier("never", log)),
    )
    chain.diarize("x.wav")
    chain.cleanup()
    assert log[-2:] == ["cleanup:boom", "cleanup:b"]
    assert "init:never" not in log
