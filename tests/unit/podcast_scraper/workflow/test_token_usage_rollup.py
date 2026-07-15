"""Token/cost rollup: slice telemetry by any dimension, with NO double/over/under counting.

The operator's requirement: solid results, attributable to model / operation / request / episode /
run. These tests pin the slicing AND the counting integrity — a doubly-logged request must count
once, and the grand total must equal the exact (de-duplicated) sum of the events.
"""

from __future__ import annotations

import pytest

from podcast_scraper.workflow.token_usage_rollup import rollup_events, rollup_run_log

pytestmark = pytest.mark.unit


def _ev(**kw):
    base = {
        "event_type": "llm_cost",
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "operation": "gi",
        "episode_id": "ep1",
        "run_id": "run1",
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "estimated_cost_usd": 0.01,
        "request_id": None,
    }
    base.update(kw)
    return base


def test_group_by_model_sums_tokens_and_cost() -> None:
    events = [
        _ev(model="gpt-5.4-mini", request_id="a", estimated_cost_usd=0.01, prompt_tokens=1000),
        _ev(model="gpt-5.4-mini", request_id="b", estimated_cost_usd=0.02, prompt_tokens=2000),
        _ev(model="gpt-5.4-nano", request_id="c", estimated_cost_usd=0.001, prompt_tokens=500),
    ]
    r = rollup_events(events, group_by=("model",))
    assert r.total.calls == 3
    assert r.total.input_tokens == 3500
    assert round(r.total.estimated_cost_usd, 6) == 0.031
    by = {k[0]: v for k, v in r.groups.items()}
    assert by["gpt-5.4-mini"].calls == 2 and by["gpt-5.4-mini"].input_tokens == 3000
    assert by["gpt-5.4-nano"].calls == 1


def test_a_doubly_logged_request_counts_ONCE() -> None:
    """COUNTING INTEGRITY. A request_id appearing twice (flushed/re-read log) must not inflate."""
    events = [
        _ev(request_id="dup", estimated_cost_usd=0.05, prompt_tokens=1000),
        _ev(request_id="dup", estimated_cost_usd=0.05, prompt_tokens=1000),  # same request, again
        _ev(request_id="other", estimated_cost_usd=0.02, prompt_tokens=500),
    ]
    r = rollup_events(events, group_by=("provider",))
    assert r.total.calls == 2, "the duplicate request_id must be collapsed"
    assert r.total.input_tokens == 1500
    assert round(r.total.estimated_cost_usd, 6) == 0.07


def test_events_without_request_id_each_count() -> None:
    """A provider that returns no id can't be de-duped onto anything — each call counts once."""
    events = [_ev(request_id=None), _ev(request_id=None), _ev(request_id=None)]
    r = rollup_events(events, group_by=("provider",))
    assert r.total.calls == 3


def test_slice_by_episode_and_operation() -> None:
    events = [
        _ev(request_id="1", episode_id="ep1", operation="gi", estimated_cost_usd=0.01),
        _ev(request_id="2", episode_id="ep1", operation="cleaning", estimated_cost_usd=0.002),
        _ev(request_id="3", episode_id="ep2", operation="gi", estimated_cost_usd=0.03),
    ]
    by_ep = {k[0]: v for k, v in rollup_events(events, group_by=("episode_id",)).groups.items()}
    assert round(by_ep["ep1"].estimated_cost_usd, 6) == 0.012
    assert round(by_ep["ep2"].estimated_cost_usd, 6) == 0.03
    by_op = {k[0]: v for k, v in rollup_events(events, group_by=("operation",)).groups.items()}
    assert by_op["gi"].calls == 2 and by_op["cleaning"].calls == 1


def test_cached_tokens_and_guardrail_are_tallied() -> None:
    events = [
        _ev(request_id="1", cached_input_tokens=800, cache_write_tokens=100),
        _ev(request_id="2", cached_input_tokens=600, triggered_guardrail=True),
    ]
    r = rollup_events(events, group_by=("provider",))
    assert r.total.cached_input_tokens == 1400
    assert r.total.cache_write_tokens == 100
    assert r.total.guardrail_calls == 1


def test_non_llm_cost_events_are_ignored() -> None:
    events = [
        {"event_type": "llm_model_substitution", "provider": "openai"},
        _ev(request_id="1"),
        {"event_type": "something_else"},
    ]
    assert rollup_events(events, group_by=("provider",)).total.calls == 1


def test_rollup_run_log_parses_prefixed_lines(tmp_path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        '2026-07-15 10:00:00 INFO x: {"event_type": "llm_cost", "provider": "deepseek", '
        '"model": "deepseek-v4-flash", "request_id": "r1", "prompt_tokens": 5000, '
        '"completion_tokens": 200, "cached_input_tokens": 4000, "estimated_cost_usd": 0.0012}\n'
        "2026-07-15 10:00:01 INFO x: not json here\n",
        encoding="utf-8",
    )
    r = rollup_run_log(log, group_by=("model",))
    assert r.total.calls == 1 and r.total.cached_input_tokens == 4000
    assert rollup_run_log(tmp_path / "nope.log").total.calls == 0
