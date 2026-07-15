"""The operator usage snapshot behind /api/usage + the MCP prod_usage tool.

Rolls up ``llm_cost`` telemetry from disk, sliced by dimension, self-contained (no Loki/Langfuse).
Pins: it finds events under a corpus, validates group_by, de-dups, and flags an uninstrumented run.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.utils.usage_status import usage_rollup_snapshot

pytestmark = pytest.mark.unit


def _write_log(path, events) -> None:
    lines = []
    for ev in events:
        lines.append(
            f"2026-07-15 10:00:00 INFO podcast_scraper.workflow.cost_monitoring: {json.dumps(ev)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ev(**kw):
    base = {
        "event_type": "llm_cost",
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "operation": "gi",
        "stage": "gi",
        "episode_id": "ep1",
        "run_id": "run1",
        "request_id": None,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "cached_input_tokens": 500,
        "estimated_cost_usd": 0.01,
    }
    base.update(kw)
    return base


def test_rolls_up_events_found_under_a_corpus_dir(tmp_path) -> None:
    run = tmp_path / "feeds" / "show" / "run_1"
    run.mkdir(parents=True)
    _write_log(
        run / "run.log",
        [
            _ev(request_id="a", model="gpt-5.4-mini", estimated_cost_usd=0.01),
            _ev(request_id="b", model="gpt-5.4-nano", estimated_cost_usd=0.002),
        ],
    )
    snap = usage_rollup_snapshot(tmp_path, group_by=("model",))
    assert snap["total"]["calls"] == 2
    assert snap["total"]["cached_input_tokens"] == 1000
    assert snap["uninstrumented"] is False
    assert any(f.endswith("run.log") for f in snap["source_files"])
    models = {g["model"] for g in snap["groups"]}
    assert models == {"gpt-5.4-mini", "gpt-5.4-nano"}


def test_dedups_by_request_id_across_discovered_files(tmp_path) -> None:
    (tmp_path / "run_1").mkdir()
    (tmp_path / "run_2").mkdir()
    _write_log(tmp_path / "run_1" / "run.log", [_ev(request_id="same", estimated_cost_usd=0.05)])
    _write_log(tmp_path / "run_2" / "run.log", [_ev(request_id="same", estimated_cost_usd=0.05)])
    snap = usage_rollup_snapshot(tmp_path, group_by=("provider",))
    assert snap["total"]["calls"] == 1, "same request_id in two files counts once"


def test_invalid_group_by_falls_back_to_provider_model(tmp_path) -> None:
    (tmp_path / "run_1").mkdir()
    _write_log(tmp_path / "run_1" / "run.log", [_ev(request_id="a")])
    snap = usage_rollup_snapshot(tmp_path, group_by=("not_a_dim",))
    assert snap["group_by"] == ["provider", "model"]
    assert "dimensions" in snap and "episode_id" in snap["dimensions"]


def test_a_single_log_file_is_accepted(tmp_path) -> None:
    log = tmp_path / "run.log"
    _write_log(log, [_ev(request_id="a"), _ev(request_id="b")])
    snap = usage_rollup_snapshot(log, group_by=("operation",))
    assert snap["total"]["calls"] == 2


def test_uninstrumented_flag_when_files_have_no_events(tmp_path) -> None:
    run = tmp_path / "run_1"
    run.mkdir()
    (run / "run.log").write_text("2026-07-15 INFO nothing to see here\n", encoding="utf-8")
    snap = usage_rollup_snapshot(tmp_path, group_by=("provider",))
    assert snap["uninstrumented"] is True and snap["total"]["calls"] == 0


def test_missing_source_is_empty_not_an_error(tmp_path) -> None:
    snap = usage_rollup_snapshot(tmp_path / "nope", group_by=("provider",))
    assert snap["total"]["calls"] == 0 and snap["uninstrumented"] is False
