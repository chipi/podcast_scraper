"""Every provider's spend must reach the run budget, exactly once, whatever the path.

``record_provider_call_cost`` is the one function all eight provider namespaces route through
(gemini, deepgram, grok, anthropic, ollama, mistral, openai, and ml/diarization), which is why the
budget counts there rather than by summing named fields on ``Metrics`` — the old approach, which
silently omitted ``diarization_cost_usd`` and would omit any future paid stage too.

But that function can legitimately execute more than once for a single call: the grounding path
re-enters it with the cost it already set, and a backfill can raise a None cost to a real one
afterwards. These tests pin both, because getting it wrong is expensive in BOTH directions —
double-counting aborts a run that was well inside its cap, and over-latching lets real spend go
uncounted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from podcast_scraper.utils.provider_metrics import ProviderCallMetrics, record_provider_call_cost
from podcast_scraper.workflow.run_budget import get_run_budget, reset_run_budget


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_run_budget(cap_usd=5.0, action="abort")
    yield
    reset_run_budget()


def _cfg():
    return SimpleNamespace(
        cost_soft_cap_usd_per_run=5.0,
        cost_soft_cap_action="abort",
        llm_cost_events_jsonl=None,
    )


def _record(cm, cost, capability="transcription", provider="deepgram", model="nova-3"):
    record_provider_call_cost(
        cm,
        cost,
        cfg=_cfg(),
        provider_type=provider,
        capability=capability,
        model=model,
        audio_minutes=10.0,
    )


def test_a_priced_call_reaches_the_run_budget() -> None:
    _record(ProviderCallMetrics(), 0.43)
    assert get_run_budget().spent_usd == pytest.approx(0.43)


def test_re_entry_with_the_SAME_cost_does_not_double_count() -> None:
    """The grounding path calls this again passing ``call_metrics.estimated_cost`` back in."""
    cm = ProviderCallMetrics()
    _record(cm, 0.43)
    _record(cm, cm.estimated_cost)
    _record(cm, cm.estimated_cost)
    assert get_run_budget().spent_usd == pytest.approx(0.43), "one call must cost the budget once"


def test_a_call_that_was_unpriceable_still_counts_once_a_price_is_known() -> None:
    cm = ProviderCallMetrics()
    _record(cm, None, provider="", model="")  # unpriceable: nothing to count yet
    assert get_run_budget().spent_usd == 0.0
    cm.estimated_cost = None  # a later pass resolves a real price
    _record(cm, 0.25)
    assert get_run_budget().spent_usd == pytest.approx(0.25)


def test_a_cost_REVISED_UPWARD_adds_only_the_difference() -> None:
    """Why the latch stores a running total rather than a yes/no flag.

    A boolean latch would count the first $0.10 and silently ignore the correction to $0.40 —
    understating spend by 75% for that call. No caller in this repo revises a cost upward today
    (``apply_estimated_cost_if_missing`` early-returns once a cost is set, and the grounding path
    re-passes the identical value), so this pins the FUNCTION's contract rather than an existing
    path: whatever the final agreed cost of a call, the budget must end up holding exactly that,
    once.
    """
    cm = ProviderCallMetrics()
    _record(cm, 0.10)
    assert get_run_budget().spent_usd == pytest.approx(0.10)
    _record(cm, 0.40)
    assert get_run_budget().spent_usd == pytest.approx(0.40), "not 0.10, and not 0.50"


def test_independent_calls_each_count() -> None:
    for _ in range(4):
        _record(ProviderCallMetrics(), 0.5)
    assert get_run_budget().spent_usd == pytest.approx(2.0)


def test_a_zero_cost_local_provider_adds_nothing() -> None:
    _record(ProviderCallMetrics(), 0.0, provider="ollama", model="llama3")
    assert get_run_budget().spent_usd == 0.0


def test_DIARIZATION_spend_is_counted_though_it_is_not_an_llm_field() -> None:
    """The exact gap in the old seven-``llm_*``-field sum.

    ``diarization_cost_usd`` is accumulated on Metrics and was absent from that sum, so cloud
    diarization was invisible to the cap. Counting at the choke point makes the field name
    irrelevant.
    """
    _record(ProviderCallMetrics(), 1.1, capability="diarization")
    assert get_run_budget().spent_usd == pytest.approx(1.1)


def test_accounting_never_breaks_a_provider_call(monkeypatch) -> None:
    """Cost bookkeeping is on the hot path; a failure there must not fail the transcription.

    Exercises the guard directly by making the ledger lookup itself explode, rather than by
    breaking the dataclass — a subclass with a read-only property fails at CONSTRUCTION, before
    the code under test is even reached, so it would prove nothing about the guard.
    """
    import podcast_scraper.workflow.run_budget as rb

    def boom():
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(rb, "get_run_budget", boom)
    cm = ProviderCallMetrics()
    _record(cm, 0.4)  # must not raise
    assert cm.estimated_cost == 0.4, "the provider call's own bookkeeping still completed"


def test_spend_accumulates_toward_the_cap_across_many_calls() -> None:
    """What the incident needed and did not have: ASR calls adding up to a refusal."""
    budget = get_run_budget()
    for _ in range(11):
        _record(ProviderCallMetrics(), 0.5)  # 11 x $0.50 = $5.50 against a $5.00 cap
    assert budget.spent_usd == pytest.approx(5.5)
    assert budget.remaining_usd == 0.0
    assert budget.check_and_reserve(0.01) is False
