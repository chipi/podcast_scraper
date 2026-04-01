"""Tests for consolidate_dashboard_data (dashboard-data.json bundle)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DASH = Path(__file__).resolve().parents[4] / "scripts" / "dashboard"


@pytest.fixture()
def consolidate_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "consolidate_dashboard_data",
        _DASH / "consolidate_dashboard_data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_build_bundle_merges_ci_and_nightly(tmp_path: Path, consolidate_mod) -> None:
    (tmp_path / "latest-ci.json").write_text(
        json.dumps({"commit": "aaa", "metrics": {"runtime": {"total": 1.0}}}),
        encoding="utf-8",
    )
    (tmp_path / "history-ci.jsonl").write_text(
        json.dumps({"commit": "old", "metrics": {}}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "latest-nightly.json").write_text(
        json.dumps({"commit": "bbb", "metrics": {"runtime": {"total": 2.0}}}),
        encoding="utf-8",
    )
    (tmp_path / "history-nightly.jsonl").write_text("", encoding="utf-8")

    b = consolidate_mod.build_bundle(tmp_path, strict=False)
    assert b["version"] == 1
    assert b["ci"]["latest"]["commit"] == "aaa"
    assert len(b["ci"]["history"]) == 1
    assert b["nightly"]["latest"]["commit"] == "bbb"
    assert b["nightly"]["history"] == []


def test_strict_pretty_jsonl_exits(tmp_path: Path, consolidate_mod) -> None:
    """One pretty-printed object spanning many lines → at most 1 parsed record."""
    (tmp_path / "latest-ci.json").write_text("{}", encoding="utf-8")
    obj = {"commit": "x", "metrics": {"a": 1, "b": 2, "c": {"d": 3, "e": 4, "f": 5}}}
    pretty = json.dumps(obj, indent=2)
    assert sum(1 for line in pretty.splitlines() if line.strip()) >= 8
    (tmp_path / "history-ci.jsonl").write_text(pretty + "\n", encoding="utf-8")
    (tmp_path / "latest-nightly.json").write_text("{}", encoding="utf-8")
    (tmp_path / "history-nightly.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as ei:
        consolidate_mod.build_bundle(tmp_path, strict=True)
    assert ei.value.code == 2


def test_non_strict_warns_but_builds(tmp_path: Path, consolidate_mod, capsys) -> None:
    (tmp_path / "latest-ci.json").write_text("{}", encoding="utf-8")
    obj = {"commit": "x", "metrics": {"a": 1, "b": 2, "c": {"d": 3, "e": 4, "f": 5}}}
    pretty = json.dumps(obj, indent=2)
    (tmp_path / "history-ci.jsonl").write_text(pretty + "\n", encoding="utf-8")
    (tmp_path / "latest-nightly.json").write_text("{}", encoding="utf-8")
    (tmp_path / "history-nightly.jsonl").write_text("", encoding="utf-8")

    b = consolidate_mod.build_bundle(tmp_path, strict=False)
    assert len(b["warnings"]) >= 1
    err = capsys.readouterr().err
    assert "pretty-printed" in err or "jsonl" in err.lower()
