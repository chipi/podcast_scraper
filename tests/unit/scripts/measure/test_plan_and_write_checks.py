"""scripts/measure/plan_diff_ro.py and write_digest_ro.py: read-only pre-apply checks (C5)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[4]


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "measure" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pd = _load("plan_diff_ro")
wd = _load("write_digest_ro")


def _tree(root: Path) -> Dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_the_plan_report_is_the_hooks_own_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mig = SimpleNamespace(id="0099_x")
    gate = [{"meta": "a", "counts": {"n": 1}, "inputs": {"f": "1"}}]
    prod = [
        {"meta": "a", "counts": {"n": 1}, "inputs": {"f": "2"}},
        {"meta": "b", "counts": {}, "inputs": {}},
    ]
    monkeypatch.setattr(pd, "get_migrations", lambda: [mig])
    monkeypatch.setattr(
        pd.oh, "load_outcome", lambda mid: ({"plan": gate, "code_digest": "d"}, "sha")
    )
    monkeypatch.setattr(pd.oh, "episode_plan", lambda m, root: prod)
    monkeypatch.setattr(pd.oh, "migration_digest", lambda m: "d")
    rep = pd.plan_report("0099_x", tmp_path)
    assert (rep["clone_episodes"], rep["prod_episodes"], rep["digest_matches"]) == (1, 2, True)
    assert rep["diff_kinds"] == {"differs": 1, "prod_only": 1}
    assert {e["meta"]: e["kind"] for e in rep["entries"]} == {"a": "differs", "b": "prod_only"}
    out = tmp_path / "out.json"
    before = _tree(tmp_path)
    assert pd.main(["--migration", "0099_x", "--corpus", str(tmp_path), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["entries"] == rep["entries"]
    assert {k: v for k, v in _tree(tmp_path).items() if k != "out.json"} == before


class _Ep:
    def __init__(self, root: Path, name: str, payload: Any, text: str) -> None:
        self.json_files = {root / f"{name}.json": payload}
        self.text_files = {root / f"{name}.txt": text}


def _mig(root: Path, payload: Any, text: str) -> Any:
    eps: List[_Ep] = [_Ep(root, "e1", payload, text)]
    mig = SimpleNamespace(id="0099_x", _scan=lambda r: (eps, [], {}))
    mig._reviewed = lambda: ({}, [])
    return mig


def test_the_write_digest_changes_with_any_written_byte_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _tree(tmp_path)
    monkeypatch.setattr(wd, "get_migrations", lambda: [_mig(tmp_path, {"a": 1}, "x")])
    one = wd.write_digest("0099_x", tmp_path)
    assert (one["episodes"], one["files"], one["refused"]) == (1, 2, 0)
    assert wd.write_digest("0099_x", tmp_path)["writes_sha256"] == one["writes_sha256"]
    monkeypatch.setattr(wd, "get_migrations", lambda: [_mig(tmp_path, {"a": 2}, "x")])
    assert wd.write_digest("0099_x", tmp_path)["writes_sha256"] != one["writes_sha256"]
    monkeypatch.setattr(wd, "get_migrations", lambda: [_mig(tmp_path, {"a": 1}, "y")])
    assert wd.write_digest("0099_x", tmp_path)["writes_sha256"] != one["writes_sha256"]
    assert _tree(tmp_path) == before


def test_a_migration_without_a_scan_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(wd, "get_migrations", lambda: [SimpleNamespace(id="0099_x")])
    with pytest.raises(SystemExit, match="no _scan"):
        wd.write_digest("0099_x", tmp_path)
