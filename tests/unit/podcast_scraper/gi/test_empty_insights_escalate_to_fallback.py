"""Insights: the same swallowed-failure defect as KG, one stage over.

``OpenAICompatibleProvider.generate_insights`` ends in ``except Exception: return []``. Only
``GuardrailViolation`` is re-raised — the comment there says so explicitly ("GI is fail-up.
Propagate so FallbackAware can route… Do NOT mask as empty insights") — but a TRANSPORT error is
masked exactly that way. The failover wrapper walks the chain only on an exception, so the empty
list sails through and a healthy tier is never tried.

Measured on the run whose primary vLLM was idle: ``gi_insights_total=0`` beside ``gi_failures=0``,
while summarization — which raises — failed over correctly on the very same provider. An episode
shipped with no insights because a socket was shut, and it recorded itself as a clean run.

An episode with genuinely nothing to say costs one extra call on the fallback tier. That is the
right trade, and it is logged either way.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from podcast_scraper.gi.pipeline import _retry_insights_on_fallback_chain

pytestmark = pytest.mark.unit

_INSIGHTS = ["Rates stay higher for longer.", "Supply chains are re-shoring."]


class _Plain:
    """No chain — a bare provider."""


class _WithChain:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls = 0

    def call_via_fallback(self, method_name: str, *_a: Any, **_kw: Any) -> Any:
        assert method_name == "generate_insights"
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_the_chain_is_tried_and_its_answer_used() -> None:
    provider = _WithChain(_INSIGHTS)
    assert _retry_insights_on_fallback_chain(provider, "t", "T", 5, None) == _INSIGHTS
    assert provider.calls == 1


def test_a_provider_without_a_chain_is_untouched() -> None:
    """Behave exactly as before, and never raise."""
    assert _retry_insights_on_fallback_chain(_Plain(), "t", "T", 5, None) is None


def test_an_exhausted_chain_is_not_an_error(caplog: pytest.LogCaptureFixture) -> None:
    provider = _WithChain(RuntimeError("all tiers dead"))
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.gi.pipeline"):
        assert _retry_insights_on_fallback_chain(provider, "t", "T", 5, None) is None
    assert any("fallback chain produced no insights" in str(r.msg) for r in caplog.records)


def test_an_empty_chain_answer_is_not_treated_as_success() -> None:
    """The fallback returning [] is still "no insights" — do not dress it up as a result."""
    assert _retry_insights_on_fallback_chain(_WithChain([]), "t", "T", 5, None) is None


def test_a_non_list_answer_is_rejected() -> None:
    assert _retry_insights_on_fallback_chain(_WithChain("nope"), "t", "T", 5, None) is None


def test_the_escalation_is_announced(caplog: pytest.LogCaptureFixture) -> None:
    """Silence is what let episodes ship with zero insights and zero recorded failures."""
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.gi.pipeline"):
        _retry_insights_on_fallback_chain(_WithChain(_INSIGHTS), "t", "T", 5, None)
    assert any("escalating to the fallback chain" in str(r.msg) for r in caplog.records)
