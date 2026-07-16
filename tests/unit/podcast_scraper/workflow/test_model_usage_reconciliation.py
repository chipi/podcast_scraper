"""Offline reconciliation: catch a served≠requested model drift from a finished run's logs.

Guard 5 — the automated version of the manual dashboard check that caught the sonnet-4-6 incident.
Given the JSON telemetry a run emits, it must surface every (provider, requested→served) mismatch
and stay silent on a clean run.
"""

from __future__ import annotations

import pytest

from podcast_scraper.workflow.model_usage_reconciliation import (
    reconcile_events,
    reconcile_run_log,
)

pytestmark = pytest.mark.unit


def test_a_clean_run_reconciles_empty() -> None:
    events = [
        {
            "event_type": "llm_cost",
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "served_model": "deepseek-v4-flash",
            "stage": "gi",
        },
        {
            "event_type": "llm_cost",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "served_model": "claude-sonnet-4-5-20250219",
            "stage": "gi",
        },  # dated pin = match
    ]
    assert reconcile_events(events) == []


def test_a_substitution_event_is_surfaced_and_counted() -> None:
    events = [
        {
            "event_type": "llm_model_substitution",
            "provider": "anthropic",
            "requested_model": "claude-sonnet-4-6",
            "served_model": "claude-sonnet-5",
            "stage": "gi",
        },
        {
            "event_type": "llm_model_substitution",
            "provider": "anthropic",
            "requested_model": "claude-sonnet-4-6",
            "served_model": "claude-sonnet-5",
            "stage": "cleaning",
        },
    ]
    out = reconcile_events(events)
    assert len(out) == 1
    m = out[0]
    assert m.provider == "anthropic"
    assert m.requested_model == "claude-sonnet-4-6"
    assert m.served_model == "claude-sonnet-5"
    assert m.call_count == 2
    assert set(m.stages) == {"gi", "cleaning"}


def test_a_cost_event_whose_served_model_drifts_is_caught() -> None:
    """Even with no explicit substitution event, a cost event's served≠requested is reconciled."""
    events = [
        {
            "event_type": "llm_cost",
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "served_model": "gpt-4o",
            "stage": "gi",
        },
    ]
    out = reconcile_events(events)
    assert len(out) == 1 and out[0].served_model == "gpt-4o"


def test_reconcile_run_log_parses_prefixed_json_lines(tmp_path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "2026-07-15 10:00:00,000 WARNING podcast_scraper.workflow.cost_monitoring: "
        '{"event_type": "llm_model_substitution", "provider": "anthropic", '
        '"requested_model": "claude-sonnet-4-6", "served_model": "claude-sonnet-5", '
        '"stage": "gi"}\n'
        "2026-07-15 10:00:01,000 INFO something else not json\n"
        "2026-07-15 10:00:02,000 INFO podcast_scraper.workflow.cost_monitoring: "
        '{"event_type": "llm_cost", "provider": "deepseek", "model": "deepseek-v4-flash", '
        '"served_model": "deepseek-v4-flash", "estimated_cost_usd": 0.01, "stage": "gi"}\n',
        encoding="utf-8",
    )
    out = reconcile_run_log(log)
    assert len(out) == 1
    assert out[0].requested_model == "claude-sonnet-4-6"
    assert reconcile_run_log(tmp_path / "missing.log") == []
