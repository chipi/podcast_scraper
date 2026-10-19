"""build_outcome.py: the shipped outcome carries the record's verdict and its conditions."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[4]
GATE = {
    "snapshot": "snap",
    "applied_before": ["0001"],
    "code_digest": "dg",
    "code": {"dirty": False, "sha": "abc"},
    "verdicts_sha256": None,
    "scoreboard_before": {},
    "scoreboard_after": {},
    "episode_deltas": {},
    "plan": [],
    "plan_sha256": "p",
    "computed_class": "full",
}


@pytest.fixture
def bo(tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "build_outcome", REPO / "scripts/measure/build_outcome.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config/corpus_change_gate.json", tmp_path / "config")
    monkeypatch.setattr(mod, "REPO", tmp_path)
    return mod


def _write(root: Path, step8: str, gate: dict = GATE) -> None:
    cfg = json.loads((root / "config/corpus_change_gate.json").read_text())
    d = root / cfg["records_dir"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "0099_x.gate.json").write_text(json.dumps(gate))
    (d / "0099_x.md").write_text(
        f"---\nid: 0099_x\nclass: full\n---\n\n# x\n\n## 8. Verdict\n\n{step8}\n"
    )


def test_an_inline_condition_is_carried(bo, tmp_path: Path) -> None:
    """Hook condition C8: 0031 B.6's condition was written inline and was dropped."""
    _write(
        tmp_path,
        "Verdict: GO WITH CONDITIONS (r, B.6). Condition: ci-fast EXIT=0. Open, unchanged: x.",
    )
    out = bo.build("0099_x")
    assert out["verdict"] == "GO WITH CONDITIONS" and out["conditions"] == ["ci-fast EXIT=0."]


def test_bullet_conditions_are_carried(bo, tmp_path: Path) -> None:
    _write(tmp_path, "Verdict: GO WITH CONDITIONS (r)\n\n- **C1** one.\n- **C2** two.\n")
    assert bo.build("0099_x")["conditions"] == ["- **C1** one.", "- **C2** two."]


def test_go_with_conditions_and_none_found_is_refused(bo, tmp_path: Path) -> None:
    _write(tmp_path, "Verdict: GO WITH CONDITIONS (r), with the conditions in another file.")
    with pytest.raises(SystemExit, match="no condition was found"):
        bo.build("0099_x")


def test_a_plain_go_needs_no_condition_and_the_last_verdict_counts(bo, tmp_path: Path) -> None:
    _write(tmp_path, "Verdict: NO-GO (r, A)\n\nVerdict: GO (r, B)")
    out = bo.build("0099_x")
    assert out["verdict"] == "GO" and out["conditions"] == []
    _write(tmp_path, "Verdict: GO (r, A)\n\nVerdict: NO-GO (r, B)")
    with pytest.raises(SystemExit, match="not GO"):
        bo.build("0099_x")


def test_a_gate_report_on_uncommitted_code_is_refused(bo, tmp_path: Path) -> None:
    _write(tmp_path, "Verdict: GO (r)", {**GATE, "code": {"dirty": True, "sha": "abc"}})
    with pytest.raises(SystemExit, match="uncommitted"):
        bo.build("0099_x")


def test_the_refusal_reads_the_last_verdict_not_an_earlier_one(bo, tmp_path: Path) -> None:
    """The hook's final re-check: a GWC-with-none refusal must judge the LAST verdict."""
    _write(
        tmp_path, "Verdict: GO (r, A)\n\nVerdict: GO WITH CONDITIONS (r, B), conditions elsewhere."
    )
    with pytest.raises(SystemExit, match="no condition was found"):
        bo.build("0099_x")
    _write(
        tmp_path, "Verdict: GO WITH CONDITIONS (r, A), conditions elsewhere.\n\nVerdict: GO (r, B)"
    )
    assert bo.build("0099_x")["verdict"] == "GO"
