"""The value gate trims filler — and must never empty an episode.

Every silent-failure bug in this codebase has the same shape: a stage fails, swallows it, and the
episode lands looking successful with its content gone. The gate is a new place for that to
happen, so the fail-open paths are tested harder than the happy path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

from podcast_scraper.gi.value_gate import apply_value_gate

SPECS = [
    ("OpenAI renegotiated its Microsoft deal, removing revenue sharing.", "fact"),
    ("AI is advancing rapidly.", "unknown"),
    ("Amazon invested $50 billion and will sell OpenAI models via Bedrock.", "fact"),
    ("The hosts discuss the news.", "unknown"),
]


class _Provider:
    def __init__(self, tiers: Any = None, exc: Exception | None = None) -> None:
        self._tiers = tiers
        self._exc = exc
        self.calls = 0

    def classify_insights(self, insights: List[str]) -> Any:
        self.calls += 1
        if self._exc:
            raise self._exc
        return self._tiers


def _cfg(**kw):
    base = {"gi_value_gate_enabled": True, "gi_value_gate_min_tier": 2}
    base.update(kw)
    return SimpleNamespace(**base)


def test_drops_filler_and_keeps_substance() -> None:
    provider = _Provider(tiers=[3, 1, 3, 0])
    kept = apply_value_gate(SPECS, provider=provider, cfg=_cfg())
    assert [t for t, _ in kept] == [SPECS[0][0], SPECS[2][0]]


def test_disabled_by_default_is_a_no_op() -> None:
    provider = _Provider(tiers=[0, 0, 0, 0])
    kept = apply_value_gate(SPECS, provider=provider, cfg=SimpleNamespace())
    assert kept == SPECS
    assert provider.calls == 0, "gate must not call the provider when disabled"


def test_min_tier_3_keeps_only_core() -> None:
    provider = _Provider(tiers=[3, 1, 2, 0])
    kept = apply_value_gate(SPECS, provider=provider, cfg=_cfg(gi_value_gate_min_tier=3))
    assert [t for t, _ in kept] == [SPECS[0][0]]


# --- fail-open: the gate must never destroy an episode -------------------------------------


def test_provider_exception_keeps_every_insight() -> None:
    provider = _Provider(exc=RuntimeError("gate model down"))
    kept = apply_value_gate(SPECS, provider=provider, cfg=_cfg())
    assert kept == SPECS


def test_wrong_length_response_keeps_every_insight() -> None:
    """A misaligned tier list would silently drop the wrong insights. Keep them all instead."""
    provider = _Provider(tiers=[3, 1])  # 2 tiers for 4 insights
    kept = apply_value_gate(SPECS, provider=provider, cfg=_cfg())
    assert kept == SPECS


def test_rejecting_everything_keeps_every_insight() -> None:
    """An episode where nothing clears the bar means a broken gate, not a worthless episode."""
    provider = _Provider(tiers=[0, 0, 0, 0])
    kept = apply_value_gate(SPECS, provider=provider, cfg=_cfg())
    assert kept == SPECS, "the gate must never emit an empty episode"


def test_provider_without_classify_is_skipped() -> None:
    kept = apply_value_gate(SPECS, provider=object(), cfg=_cfg())
    assert kept == SPECS


def test_unparsable_tier_keeps_the_insight() -> None:
    provider = _Provider(tiers=[3, "junk", 3, 0])
    kept = apply_value_gate(SPECS, provider=provider, cfg=_cfg())
    assert SPECS[1] in kept, "an unreadable tier must not silently discard real content"


def test_empty_input_short_circuits() -> None:
    provider = _Provider(tiers=[])
    assert apply_value_gate([], provider=provider, cfg=_cfg()) == []
    assert provider.calls == 0


def test_metrics_record_what_was_dropped() -> None:
    metrics = SimpleNamespace()
    provider = _Provider(tiers=[3, 1, 3, 0])
    apply_value_gate(SPECS, provider=provider, cfg=_cfg(), pipeline_metrics=metrics)
    assert getattr(metrics, "gi_insights_dropped_by_value_gate") == 2
    assert getattr(metrics, "gi_value_gate_calls") == 1


@pytest.mark.parametrize("bad", [None, "nope", 42])
def test_non_list_response_keeps_every_insight(bad) -> None:
    provider = _Provider(tiers=bad)
    assert apply_value_gate(SPECS, provider=provider, cfg=_cfg()) == SPECS
