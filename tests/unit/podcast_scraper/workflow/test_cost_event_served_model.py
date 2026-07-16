"""emit_llm_cost_event must catch a silent model substitution and record served-model provenance.

Guard 2/3 of the sonnet-4-6 work: providers now pass the model the API reported serving. When it
disagrees with what we requested on a cloud provider, that is a substitution — the emitter logs a
loud ``llm_model_substitution`` event, and it does so even for a zero-cost call (below the billing
early-return), so a free/local-billed substitution is still caught. On billable events the served id
is recorded for provenance and offline reconciliation.
"""

from __future__ import annotations

import json
import logging

import pytest

from podcast_scraper.workflow.cost_monitoring import emit_llm_cost_event

pytestmark = pytest.mark.unit


class _Cfg:
    output_dir = None
    rss_url = None
    jsonl_metrics_echo_stdout = False


def _substitution_events(caplog) -> list:
    out = []
    for rec in caplog.records:
        try:
            payload = json.loads(rec.getMessage())
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("event_type") == "llm_model_substitution":
            out.append(payload)
    return out


def test_a_substitution_is_logged_loudly(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        emit_llm_cost_event(
            _Cfg(),
            provider="anthropic",
            stage="gi",
            model="claude-sonnet-4-6",
            estimated_cost_usd=0.01,
            served_model="claude-sonnet-5",
        )
    subs = _substitution_events(caplog)
    assert len(subs) == 1, "a served≠requested cloud model must emit exactly one substitution event"
    assert subs[0]["requested_model"] == "claude-sonnet-4-6"
    assert subs[0]["served_model"] == "claude-sonnet-5"


def test_substitution_is_caught_even_at_zero_cost(caplog) -> None:
    """OpenAI logs $0 on the eval path; the check must fire BEFORE the cost early-return."""
    with caplog.at_level(logging.WARNING):
        emit_llm_cost_event(
            _Cfg(),
            provider="openai",
            stage="gi",
            model="gpt-5.4-mini",
            estimated_cost_usd=0.0,  # early-returns for cost — but the check runs first
            served_model="gpt-4o",
        )
    assert len(_substitution_events(caplog)) == 1, "0-cost must not skip the substitution check"


def test_a_matching_or_dated_served_model_is_not_flagged(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        emit_llm_cost_event(
            _Cfg(),
            provider="anthropic",
            stage="gi",
            model="claude-sonnet-4-5",
            estimated_cost_usd=0.01,
            served_model="claude-sonnet-4-5-20250219",  # dated pin of the same family
        )
    assert _substitution_events(caplog) == [], "a dated pin of the requested family is not a swap"


def test_provenance_served_model_is_on_the_billable_event(caplog) -> None:
    with caplog.at_level(logging.INFO):
        emit_llm_cost_event(
            _Cfg(),
            provider="deepseek",
            stage="gi",
            model="deepseek-v4-flash",
            estimated_cost_usd=0.02,
            served_model="deepseek-v4-flash",
        )
    cost_events = [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith("{") and '"llm_cost"' in r.getMessage()
    ]
    assert cost_events and cost_events[0]["served_model"] == "deepseek-v4-flash"
