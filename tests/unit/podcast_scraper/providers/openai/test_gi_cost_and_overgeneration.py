"""GI cost is emitted once per call, and over-generation is counted (#1891).

Three defects found on prod 2026-08-31, all of which stayed invisible for a full night of
production because nothing FAILED — episodes finished green with ok=1, $0 cost and no
fallbacks while the model returned up to 14x the requested insight count.

1. A truncated GI call emitted TWO llm_cost events (guardrail branch, then fall-through),
   so every spend figure was inflated by exactly the truncated calls — wrong in the
   direction that hides the problem. Measured: 35,601 real output tokens reported as 65,601.
2. GI_INSIGHT_TOKENS_EACH was 150 against a measured 12.8-20.6 tokens per insight.
3. Over-generation had a WARNING and no counter, so nobody could aggregate it.
"""

from __future__ import annotations

import pytest

from podcast_scraper.config_constants import (
    GI_INSIGHT_TOKENS_EACH,
    GI_INSIGHT_TOKENS_FLOOR,
)
from podcast_scraper.providers.openai.openai_provider import _bump_metric


class _Metrics:
    """Stand-in for the pipeline metrics object (plain attribute counters)."""


def test_bump_creates_then_increments():
    m = _Metrics()
    _bump_metric(m, "gi_insight_overgeneration_events")
    _bump_metric(m, "gi_insight_overgeneration_events")
    assert m.gi_insight_overgeneration_events == 2


def test_bump_accepts_an_amount():
    m = _Metrics()
    _bump_metric(m, "gi_insight_overgenerated_total", 395)
    assert m.gi_insight_overgenerated_total == 395


@pytest.mark.parametrize("metrics", [None])
def test_bump_is_none_safe(metrics):
    """Telemetry must never break a run that otherwise succeeded."""
    _bump_metric(metrics, "anything")


def test_bump_swallows_a_hostile_metrics_object():
    class Frozen:
        __slots__ = ()

    _bump_metric(Frozen(), "nope")  # must not raise


def test_zero_amount_is_a_noop():
    m = _Metrics()
    _bump_metric(m, "x", 0)
    assert not hasattr(m, "x")


class TestInsightTokenBudget:
    def test_budget_is_sized_from_the_measured_per_insight_cost(self):
        """12.8-20.6 tokens per insight measured on prod; 150 provisioned 7-11x that.

        50 is the smallest value above every well-behaved call observed (2814 tokens for a
        ceiling of 60 == 47 per insight), so good calls are untouched and a runaway is cut.
        """
        assert GI_INSIGHT_TOKENS_EACH == 50

    def test_a_ceiling_of_60_now_budgets_3000_not_9000(self):
        assert max(GI_INSIGHT_TOKENS_FLOOR, 60 * GI_INSIGHT_TOKENS_EACH) == 3000

    def test_the_floor_still_protects_small_requests(self):
        """A 3-insight request must still get room to answer, not 150 tokens."""
        assert max(GI_INSIGHT_TOKENS_FLOOR, 3 * GI_INSIGHT_TOKENS_EACH) == GI_INSIGHT_TOKENS_FLOOR

    def test_budget_stays_above_the_widest_well_behaved_call_seen(self):
        """Guards against tuning this so low that healthy calls start truncating.

        Widest well-behaved GI call measured on prod was 2814 output tokens (ceiling 100).
        """
        assert max(GI_INSIGHT_TOKENS_FLOOR, 100 * GI_INSIGHT_TOKENS_EACH) >= 2814
