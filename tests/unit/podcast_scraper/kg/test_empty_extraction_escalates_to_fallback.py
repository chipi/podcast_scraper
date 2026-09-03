"""A provider that returns nothing must still reach the fallback chain.

THE BLOCKER, measured on a real run whose primary vLLM endpoint was unreachable:

    llm_kg_calls            0      kg_provider_extractions  0
    kg_extractions_no_llm   5      kg_failures              0
    gi_insights_total       0      gi_failures              0
    llm_summary_fallback_active_count  1

Zero successful extractions, zero recorded failures, ~106s per episode burned in retry backoff —
and a healthy ollama tier sitting unused, while the SUMMARY stage failed over to it correctly.

The asymmetry is the whole bug. ``OpenAICompatibleProvider.extract_kg_graph`` ends in
``except Exception: return None``, so a dead endpoint is indistinguishable from "the model found
no topics". ``FallbackAwareSummarizationProvider._wrap_call`` walks the chain only on an
EXCEPTION, so a returned ``None`` sails straight through it. Summarization raises, so it failed
over; extraction swallows, so it did not.

Downstream this is not a missing-topics problem: ``build_artifact`` then substituted the episode's
summary BULLETS as Topic nodes, producing eight fabricated sentence-topics per episode with zero
Insight, Person or Organization nodes — and nothing anywhere saying extraction had failed.

``call_via_fallback`` exists for exactly this class of failure (its docstring: "failures the
exception contract cannot see", #1878). These tests pin that it is used, and that a plain provider
is unaffected.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from podcast_scraper.kg.pipeline import _try_provider_extraction

pytestmark = pytest.mark.unit

_GOOD = {"topics": [{"label": "ai regulation"}], "entities": []}


class _PlainProvider:
    """No fallback chain — the shape a bare provider has."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls = 0

    def extract_kg_graph(self, *_a: Any, **_kw: Any) -> Any:
        self.calls += 1
        return self.result


class _FallbackAwareProvider(_PlainProvider):
    """Mirrors FallbackAwareSummarizationProvider's surface for this one method."""

    def __init__(self, result: Any, fallback_result: Any) -> None:
        super().__init__(result)
        self.fallback_result = fallback_result
        self.fallback_calls = 0

    def call_via_fallback(self, method_name: str, *_a: Any, **_kw: Any) -> Any:
        assert method_name == "extract_kg_graph"
        self.fallback_calls += 1
        if isinstance(self.fallback_result, Exception):
            raise self.fallback_result
        return self.fallback_result


def test_an_empty_primary_escalates_to_the_chain() -> None:
    """THE regression: None from the primary must not end the attempt."""
    provider = _FallbackAwareProvider(result=None, fallback_result=_GOOD)
    partial = _try_provider_extraction("transcript", "T", None, provider, None)
    assert provider.fallback_calls == 1, (
        "the fallback chain was never tried — a dead endpoint looks identical to an empty "
        "result on this provider, and the summary stage failed over while this one did not"
    )
    assert partial == _GOOD


def test_a_topic_less_but_non_none_result_also_escalates() -> None:
    """``{"topics": [], "entities": []}`` is the same failure wearing a different shape."""
    provider = _FallbackAwareProvider(result={"topics": [], "entities": []}, fallback_result=_GOOD)
    assert _try_provider_extraction("transcript", "T", None, provider, None) == _GOOD
    assert provider.fallback_calls == 1


def test_an_exception_from_the_primary_also_escalates() -> None:
    """Both failure shapes converge on the same escalation."""

    class _Raising(_FallbackAwareProvider):
        def extract_kg_graph(self, *_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("connection refused")

    provider = _Raising(result=None, fallback_result=_GOOD)
    assert _try_provider_extraction("transcript", "T", None, provider, None) == _GOOD
    assert provider.fallback_calls == 1


def test_a_successful_primary_never_touches_the_chain() -> None:
    """The mirror. Escalating on success would double every extraction's cost."""
    provider = _FallbackAwareProvider(result=_GOOD, fallback_result={"topics": [], "entities": []})
    assert _try_provider_extraction("transcript", "T", None, provider, None) == _GOOD
    assert provider.fallback_calls == 0


def test_a_plain_provider_is_unaffected() -> None:
    """No chain configured → behave exactly as before, and never raise."""
    provider = _PlainProvider(result=None)
    assert _try_provider_extraction("transcript", "T", None, provider, None) is None
    assert provider.calls == 1


def test_an_exhausted_chain_is_not_an_error(caplog: pytest.LogCaptureFixture) -> None:
    """Every tier failing is a normal outcome — the caller handles "no partial" honestly."""
    provider = _FallbackAwareProvider(result=None, fallback_result=RuntimeError("all tiers dead"))
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.kg.pipeline"):
        assert _try_provider_extraction("transcript", "T", None, provider, None) is None
    assert any("fallback chain could not extract" in str(r.msg) for r in caplog.records)


def test_the_escalation_is_announced(caplog: pytest.LogCaptureFixture) -> None:
    """Silence is what let 48 fabricated topics ship across six episodes."""
    provider = _FallbackAwareProvider(result=None, fallback_result=_GOOD)
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.kg.pipeline"):
        _try_provider_extraction("transcript", "T", None, provider, None)
    assert any("escalating to the fallback chain" in str(r.msg) for r in caplog.records)
