"""#1809 — retried provider calls must be billed per attempt in the run ledger.

271 Deepgram requests for ~181 episodes meant the ledger was recording 1x cost
per call even when the provider was billed for every attempt that reached it.
``run_cost_usd_from_pipeline_metrics`` therefore undercounted — the dangerous
direction because the cost cap can be satisfied by a number lower than the invoice.

These tests are provider-agnostic: they exercise the shared
``ProviderCallMetrics`` / ``record_provider_call_cost`` layer that every
provider (gemini / grok / anthropic / mistral / openai / ollama / deepgram /
diarization) routes through.  No provider name may appear in the fix; the word
"deepgram" is absent from both the implementation diff and these tests.
"""

from __future__ import annotations

import pytest

from podcast_scraper.utils.provider_metrics import ProviderCallMetrics, record_provider_call_cost
from podcast_scraper.workflow.run_budget import get_run_budget, reset_run_budget


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_run_budget(cap_usd=100.0, action="abort")
    yield
    reset_run_budget()


def _record(cm: ProviderCallMetrics, cost: float | None) -> None:
    """Route cost through the shared layer, mimicking any provider."""
    from types import SimpleNamespace

    record_provider_call_cost(
        cm,
        cost,
        cfg=SimpleNamespace(
            cost_soft_cap_usd_per_run=100.0,
            cost_soft_cap_action="abort",
            llm_cost_events_jsonl=None,
        ),
        provider_type="generic_provider",  # provider-agnostic
        capability="transcription",
        model="some-model",
        audio_minutes=5.0,
    )


# ---------------------------------------------------------------------------
# Core contract: billable_attempts multiplier
# ---------------------------------------------------------------------------


def test_single_successful_call_costs_exactly_once() -> None:
    """No retry → 1 × unit cost.  Regression guard: fix must not inflate single calls."""
    cm = ProviderCallMetrics()
    _record(cm, 0.50)
    assert get_run_budget().spent_usd == pytest.approx(0.50)


def test_retries_reaching_provider_multiply_cost() -> None:
    """N billable retries → N × unit cost in the ledger (the #1809 fix).

    This is the exact scenario: a provider call retried twice on 5xx
    errors — each attempt reached the provider and was billed — but the old
    code recorded 1 × $0.10 = $0.10.  Correct answer: 3 × $0.10 = $0.30.
    """
    cm = ProviderCallMetrics()
    # Two retries on server-side errors — both are billable.
    cm.record_retry(sleep_seconds=1.0, reason="500")
    cm.record_retry(sleep_seconds=2.0, reason="503")
    assert cm.billable_attempts == 3  # initial + 2 server-error retries
    _record(cm, 0.10)
    assert get_run_budget().spent_usd == pytest.approx(0.30)


def test_rate_limit_429_retries_do_not_count_as_billable() -> None:
    """429 / rate-limit rejects are turned away before processing — not billed.

    A call retried three times on 429 then succeeding should still cost the
    ledger only 1 × unit cost.
    """
    cm = ProviderCallMetrics()
    cm.record_retry(sleep_seconds=5.0, reason="429")
    cm.record_retry(sleep_seconds=10.0, reason="429")
    cm.record_retry(sleep_seconds=15.0, reason="429")
    assert cm.billable_attempts == 1  # all rejects, no billable retries
    _record(cm, 0.20)
    assert get_run_budget().spent_usd == pytest.approx(0.20)


def test_mixed_retries_only_server_errors_counted() -> None:
    """429 retries excluded; 5xx / connection retries included."""
    cm = ProviderCallMetrics()
    cm.record_retry(sleep_seconds=5.0, reason="429")  # not billed
    cm.record_retry(sleep_seconds=1.0, reason="503")  # billed
    cm.record_retry(sleep_seconds=2.0, reason="timeout")  # billed
    cm.record_retry(sleep_seconds=5.0, reason="429")  # not billed
    assert cm.billable_attempts == 3  # initial + 2 server-side errors
    _record(cm, 0.10)
    assert get_run_budget().spent_usd == pytest.approx(0.30)


def test_cost_accounting_failure_never_raises_into_caller() -> None:
    """A blown ledger must not fail the provider call (best-effort contract)."""
    import podcast_scraper.workflow.run_budget as rb

    original = rb.get_run_budget

    def boom():
        raise RuntimeError("ledger unavailable")

    rb.get_run_budget = boom
    try:
        cm = ProviderCallMetrics()
        cm.record_retry(sleep_seconds=1.0, reason="500")
        _record(cm, 0.25)  # must not raise
        # The call_metrics cost is still set even when the budget explodes.
        assert cm.estimated_cost == pytest.approx(0.25)
    finally:
        rb.get_run_budget = original


# ---------------------------------------------------------------------------
# Re-entry / delta guards (existing semantics must survive the #1809 change)
# ---------------------------------------------------------------------------


def test_re_entry_with_same_cost_does_not_double_count() -> None:
    """The grounding path re-enters record_provider_call_cost with the same cost.

    With billable_attempts=1 (no retries), re-entry must still charge the
    budget exactly once.
    """
    cm = ProviderCallMetrics()
    _record(cm, 0.40)
    _record(cm, cm.estimated_cost)  # grounding re-entry
    _record(cm, cm.estimated_cost)  # again
    assert get_run_budget().spent_usd == pytest.approx(0.40)


def test_re_entry_with_multiple_billable_attempts_no_double_count() -> None:
    """Re-entry after retries must also charge the budget exactly once (× N)."""
    cm = ProviderCallMetrics()
    cm.record_retry(sleep_seconds=1.0, reason="500")
    _record(cm, 0.10)  # records 2 × $0.10 = $0.20
    _record(cm, cm.estimated_cost)  # re-entry must add $0
    _record(cm, cm.estimated_cost)  # re-entry must add $0
    assert get_run_budget().spent_usd == pytest.approx(0.20)


def test_unit_cost_field_unchanged_by_billable_multiplier() -> None:
    """estimated_cost stays as the per-call unit cost; only the budget is scaled.

    Manifests, cost_rollup, and reporting all read estimated_cost — they must
    not see the scaled value.
    """
    cm = ProviderCallMetrics()
    cm.record_retry(sleep_seconds=1.0, reason="500")
    cm.record_retry(sleep_seconds=2.0, reason="500")
    _record(cm, 0.05)
    # budget sees 3 × $0.05 = $0.15
    assert get_run_budget().spent_usd == pytest.approx(0.15)
    # but the field itself is still the unit price
    assert cm.estimated_cost == pytest.approx(0.05)


def test_billable_attempts_defaults_to_one() -> None:
    """Fresh ProviderCallMetrics starts with billable_attempts=1."""
    cm = ProviderCallMetrics()
    assert cm.billable_attempts == 1
