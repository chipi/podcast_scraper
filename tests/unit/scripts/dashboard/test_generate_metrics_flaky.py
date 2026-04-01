"""Tests for flaky test detection from pytest-json-report + rerunfailures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_DASH = Path(__file__).resolve().parents[4] / "scripts" / "dashboard"


def _load_metrics():
    spec = importlib.util.spec_from_file_location(
        "generate_metrics",
        _DASH / "generate_metrics.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_passed_after_rerun_detected() -> None:
    mod = _load_metrics()
    item = {
        "nodeid": "t.py::test_x",
        "outcome": "rerun",
        "call": {"outcome": "passed", "duration": 0.05},
    }
    assert mod.pytest_json_test_passed_after_rerun(item) is True


def test_clean_pass_not_flaky() -> None:
    mod = _load_metrics()
    item = {
        "nodeid": "t.py::test_x",
        "outcome": "passed",
        "call": {"outcome": "passed"},
    }
    assert mod.pytest_json_test_passed_after_rerun(item) is False


def test_failed_after_reruns_not_flaky() -> None:
    mod = _load_metrics()
    item = {
        "nodeid": "t.py::test_x",
        "outcome": "failed",
        "call": {"outcome": "failed"},
    }
    assert mod.pytest_json_test_passed_after_rerun(item) is False


def test_legacy_rerun_true_flag() -> None:
    mod = _load_metrics()
    item = {
        "nodeid": "t.py::test_x",
        "outcome": "passed",
        "rerun": True,
        "call": {"outcome": "passed"},
    }
    assert mod.pytest_json_test_passed_after_rerun(item) is True


def test_extract_test_metrics_from_reports_dir_keeps_flaky_when_merged_is_clean_pass(
    tmp_path: Path,
) -> None:
    """CI often has both merged pytest.json and shards; merge must not drop rerun rows."""
    mod = _load_metrics()
    merged_only = {
        "nodeid": "x.py::flaky",
        "outcome": "passed",
        "call": {"outcome": "passed", "duration": 0.01},
    }
    shard_rerun = {
        "nodeid": "x.py::flaky",
        "outcome": "rerun",
        "call": {"outcome": "passed", "duration": 0.02},
    }
    (tmp_path / "pytest.json").write_text(
        json.dumps(
            {
                "summary": {"total": 1, "passed": 1, "failed": 0, "skipped": 0},
                "tests": [merged_only],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "pytest-unit.json").write_text(
        json.dumps(
            {
                "summary": {"total": 1, "passed": 1, "failed": 0, "skipped": 0},
                "tests": [shard_rerun],
            }
        ),
        encoding="utf-8",
    )
    m = mod.extract_test_metrics_from_reports_dir(tmp_path)
    assert m["flaky"] == 1
    assert m["flaky_tests"][0]["name"] == "x.py::flaky"


def test_extract_test_metrics_counts_flaky(tmp_path: Path) -> None:
    mod = _load_metrics()
    report = {
        "summary": {"total": 2, "passed": 2, "failed": 0, "skipped": 0},
        "tests": [
            {
                "nodeid": "a.py::one",
                "outcome": "passed",
                "call": {"outcome": "passed", "duration": 0.01},
            },
            {
                "nodeid": "b.py::two",
                "outcome": "rerun",
                "call": {"outcome": "passed", "duration": 0.02},
            },
        ],
    }
    p = tmp_path / "pytest.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    m = mod.extract_test_metrics(p)
    assert m["flaky"] == 1
    assert len(m["flaky_tests"]) == 1
    assert m["flaky_tests"][0]["name"] == "b.py::two"
    assert m["flaky_tests"][0]["duration"] == 0.02
