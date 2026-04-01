"""Tests for slowest-test duration extraction from pytest-json-report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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


def test_duration_from_stages_when_no_top_level() -> None:
    mod = _load_metrics()
    item = {
        "nodeid": "t.py::a",
        "setup": {"duration": 0.01},
        "call": {"duration": 2.5},
        "teardown": {"duration": 0.02},
    }
    assert mod.pytest_json_test_duration_seconds(item) == pytest.approx(2.53)


def test_duration_prefers_positive_top_level() -> None:
    mod = _load_metrics()
    item = {
        "nodeid": "t.py::b",
        "duration": 9.0,
        "call": {"duration": 1.0},
    }
    assert mod.pytest_json_test_duration_seconds(item) == 9.0


def test_duration_coerces_string_in_call_stage() -> None:
    mod = _load_metrics()
    item = {"nodeid": "t.py::x", "call": {"duration": "2.5"}}
    assert mod.pytest_json_test_duration_seconds(item) == pytest.approx(2.5)


def test_extract_slowest_prefers_shard_json_over_merged(tmp_path: Path) -> None:
    """CI merged pytest.json may lack timings; per-job shards should drive slowest."""
    mod = _load_metrics()
    merged = {
        "summary": {"total": 2},
        "tests": [
            {"nodeid": "a.py::one", "call": {"duration": 0}},
            {"nodeid": "a.py::two", "call": {"duration": 0}},
        ],
    }
    shard = {
        "summary": {"total": 1},
        "tests": [
            {
                "nodeid": "slow.py::heavy",
                "setup": {"duration": 0.01},
                "call": {"duration": 9.0},
                "teardown": {"duration": 0.01},
            }
        ],
    }
    (tmp_path / "pytest.json").write_text(json.dumps(merged), encoding="utf-8")
    (tmp_path / "pytest-unit.json").write_text(json.dumps(shard), encoding="utf-8")
    slow = mod.extract_slowest_tests(tmp_path, top_n=5)
    assert len(slow) == 1
    assert slow[0]["name"] == "slow.py::heavy"
    assert slow[0]["duration"] == pytest.approx(9.02)


def test_extract_slowest_merges_junit_when_json_sparse_so_top_n_filled(tmp_path: Path) -> None:
    """Sparse JSON (few timed tests) must not block JUnit; top N should use both sources."""
    mod = _load_metrics()
    json_tests = []
    for i in range(5):
        json_tests.append(
            {
                "nodeid": f"short.py::t{i}",
                "call": {"duration": float(i + 1)},
            }
        )
    (tmp_path / "pytest.json").write_text(
        json.dumps({"summary": {"total": 5}, "tests": json_tests}),
        encoding="utf-8",
    )
    cases = "\n".join(
        f'    <testcase classname="pkg" name="u{i}" time="{float(50 + i)}"/>' for i in range(12)
    )
    junit = f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="12" failures="0" time="600">
{cases}
  </testsuite>
</testsuites>
"""
    (tmp_path / "junit-unit.xml").write_text(junit, encoding="utf-8")
    slow = mod.extract_slowest_tests(tmp_path, top_n=10)
    assert len(slow) == 10
    assert slow[0]["duration"] == pytest.approx(61.0)
    assert slow[0]["name"] == "pkg::u11"
    assert slow[9]["duration"] == pytest.approx(52.0)


def test_extract_slowest_falls_back_to_junit_glob_when_json_has_no_timings(
    tmp_path: Path,
) -> None:
    """CI: xdist JSON may lack per-test durations; junit*.xml still has testcase time=."""
    mod = _load_metrics()
    merged = {
        "summary": {"total": 1},
        "tests": [{"nodeid": "a.py::one", "call": {"duration": 0}}],
    }
    (tmp_path / "pytest.json").write_text(json.dumps(merged), encoding="utf-8")
    junit = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="1" failures="0" time="1.0">
    <testcase classname="slow_mod" name="test_heavy" time="12.5"/>
  </testsuite>
</testsuites>
"""
    (tmp_path / "junit-unit.xml").write_text(junit, encoding="utf-8")
    slow = mod.extract_slowest_tests(tmp_path, top_n=5)
    assert len(slow) == 1
    assert slow[0]["name"] == "slow_mod::test_heavy"
    assert slow[0]["duration"] == pytest.approx(12.5)


def test_extract_slowest_tests_uses_stages(tmp_path: Path) -> None:
    mod = _load_metrics()
    tests = []
    for i in range(12):
        tests.append(
            {
                "nodeid": f"mod.py::test_{i}",
                "setup": {"duration": 0.001},
                "call": {"duration": float(i)},
                "teardown": {"duration": 0.001},
            }
        )
    p = tmp_path / "pytest.json"
    p.write_text(
        json.dumps({"summary": {"total": 12}, "tests": tests}),
        encoding="utf-8",
    )
    slow = mod.extract_slowest_tests(tmp_path, top_n=10)
    assert len(slow) == 10
    assert slow[0]["name"] == "mod.py::test_11"
    assert slow[0]["duration"] == pytest.approx(11.002)
    assert slow[9]["name"] == "mod.py::test_2"
