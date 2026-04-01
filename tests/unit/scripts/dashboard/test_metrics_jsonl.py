"""Tests for metrics JSONL parsing (dashboard history)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DASH = Path(__file__).resolve().parents[4] / "scripts" / "dashboard"


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


def test_load_strict_jsonl(tmp_path: Path, metrics_jsonl_module) -> None:
    p = tmp_path / "h.jsonl"
    r1 = {"timestamp": "2026-01-01T00:00:00Z", "metrics": {"runtime": {"total": 1.0}}}
    r2 = {"timestamp": "2026-01-02T00:00:00Z", "metrics": {"runtime": {"total": 2.0}}}
    p.write_text(
        "\n".join(json.dumps(x, separators=(",", ":")) for x in (r1, r2)) + "\n",
        encoding="utf-8",
    )
    got = metrics_jsonl_module.load_metrics_history(p)
    assert len(got) == 2
    assert got[0]["metrics"]["runtime"]["total"] == 1.0
    assert got[1]["metrics"]["runtime"]["total"] == 2.0


def test_recover_pretty_printed_two_objects(tmp_path: Path, metrics_jsonl_module) -> None:
    """Legacy CI appended multi-line JSON; recover full objects."""
    p = tmp_path / "h.jsonl"
    obj_a = {"timestamp": "a", "metrics": {"x": 1}}
    obj_b = {"timestamp": "b", "metrics": {"x": 2}}
    blob = json.dumps(obj_a, indent=2) + "\n" + json.dumps(obj_b, indent=2)
    p.write_text(blob, encoding="utf-8")
    got = metrics_jsonl_module.load_metrics_history(p)
    assert len(got) == 2
    assert got[0]["timestamp"] == "a"
    assert got[1]["timestamp"] == "b"


def test_dump_compact_roundtrip(metrics_jsonl_module) -> None:
    d = {"a": 1, "b": "x"}
    line = metrics_jsonl_module.dump_compact_line(d)
    assert "\n" not in line
    assert json.loads(line) == d
