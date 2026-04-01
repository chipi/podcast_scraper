"""Tests for merging local nightly metrics run bundles into history-nightly.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DASH = Path(__file__).resolve().parents[4] / "scripts" / "dashboard"


@pytest.fixture()
def merge_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "merge_nightly_metrics_runs_to_history",
        _DASH / "merge_nightly_metrics_runs_to_history.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def metrics_jsonl_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "metrics_jsonl",
        _DASH / "metrics_jsonl.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_list_run_bundles_skips_bad_names_and_sorts(tmp_path: Path, merge_module) -> None:
    p200 = tmp_path / "run-200" / "latest-nightly.json"
    p200.parent.mkdir(parents=True)
    p200.write_text("{}", encoding="utf-8")
    p100 = tmp_path / "run-100" / "latest-nightly.json"
    p100.parent.mkdir(parents=True)
    p100.write_text("{}", encoding="utf-8")
    (tmp_path / "run-foo").mkdir()

    pairs = merge_module.list_run_bundles(tmp_path)
    assert [p[0] for p in pairs] == [100, 200]


def test_load_snapshots_ordered_oldest_first(tmp_path: Path, merge_module) -> None:
    (tmp_path / "run-10" / "latest-nightly.json").parent.mkdir(parents=True)
    (tmp_path / "run-10" / "latest-nightly.json").write_text(
        json.dumps({"commit": "aaa", "metrics": {"runtime": {"total": 1.0}}}),
        encoding="utf-8",
    )
    (tmp_path / "run-20" / "latest-nightly.json").parent.mkdir(parents=True)
    (tmp_path / "run-20" / "latest-nightly.json").write_text(
        json.dumps({"commit": "bbb", "metrics": {"runtime": {"total": 2.0}}}),
        encoding="utf-8",
    )
    rows = merge_module.load_snapshots_ordered(tmp_path)
    assert [r["commit"] for r in rows] == ["aaa", "bbb"]


def test_write_and_copy_latest(tmp_path: Path, merge_module, metrics_jsonl_module) -> None:
    runs = tmp_path / "runs"
    (runs / "run-1" / "latest-nightly.json").parent.mkdir(parents=True)
    (runs / "run-1" / "latest-nightly.json").write_text(
        json.dumps({"timestamp": "2026-01-01T00:00:00Z", "commit": "a", "metrics": {}}),
        encoding="utf-8",
    )
    (runs / "run-2" / "latest-nightly.json").parent.mkdir(parents=True)
    (runs / "run-2" / "latest-nightly.json").write_text(
        json.dumps({"timestamp": "2026-01-02T00:00:00Z", "commit": "b", "metrics": {}}),
        encoding="utf-8",
    )
    hist = tmp_path / "history-nightly.jsonl"
    latest = tmp_path / "latest-nightly.json"
    merge_module.write_history_jsonl(merge_module.load_snapshots_ordered(runs), hist)
    assert merge_module.copy_latest_from_newest_run(runs, latest)
    loaded = metrics_jsonl_module.load_metrics_history(hist)
    assert len(loaded) == 2
    assert json.loads(latest.read_text(encoding="utf-8"))["commit"] == "b"


def test_skips_invalid_json(tmp_path: Path, merge_module, metrics_jsonl_module) -> None:
    runs = tmp_path / "runs"
    (runs / "run-1" / "latest-nightly.json").parent.mkdir(parents=True)
    (runs / "run-1" / "latest-nightly.json").write_text("not json", encoding="utf-8")
    (runs / "run-2" / "latest-nightly.json").parent.mkdir(parents=True)
    (runs / "run-2" / "latest-nightly.json").write_text(
        json.dumps({"commit": "ok"}), encoding="utf-8"
    )
    out = tmp_path / "h.jsonl"
    merge_module.write_history_jsonl(merge_module.load_snapshots_ordered(runs), out)
    loaded = metrics_jsonl_module.load_metrics_history(out)
    assert len(loaded) == 1
    assert loaded[0]["commit"] == "ok"
