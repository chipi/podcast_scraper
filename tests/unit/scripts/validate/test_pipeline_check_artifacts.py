"""pipeline-check real mode (#2287): artifact comparison and the LLM noise band.

Real runs need the DGX, so these tests stand in a copied fixture run for each side: identical
copies must compare clean, and each kind of change must surface in the right place — an allowed
field or file must not, anything else must, and an LLM count outside the base's own spread must be
flagged.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "validate"))

from pipeline_check import artifacts  # noqa: E402

pytestmark = pytest.mark.unit

_FIXTURE_RUN = next((REPO_ROOT / "tests/fixtures/app-validation-corpus/v3/feeds/p01").glob("run_*"))
_ALLOWED = {
    "metadata.json": ["episode.language"],
    "manifest.json": ["stages.translation"],
    "new_files": ["*.turns.json"],
}


def _run(tmp_path: Path, name: str) -> Path:
    out = tmp_path / name / "out" / "feeds" / "p01"
    shutil.copytree(_FIXTURE_RUN, out / _FIXTURE_RUN.name)
    return tmp_path / name / "out"


def _meta(run: Path) -> Path:
    return next(run.rglob("metadata/p01_e01.metadata.json"))


def _edit(path: Path, fn) -> None:  # type: ignore[no-untyped-def]
    data = json.loads(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_identical_runs_compare_clean(tmp_path: Path) -> None:
    a, b = artifacts.collect(_run(tmp_path, "a")), artifacts.collect(_run(tmp_path, "b"))
    assert a["episodes"], "the fixture run must yield episodes"
    assert artifacts.compare_deterministic(a, b, _ALLOWED, ["summary"]) == []


def test_a_changed_metadata_field_is_reported(tmp_path: Path) -> None:
    a_dir, b_dir = _run(tmp_path, "a"), _run(tmp_path, "b")
    _edit(_meta(b_dir), lambda d: d["episode"].__setitem__("title", "Something else"))
    diffs = artifacts.compare_deterministic(
        artifacts.collect(a_dir), artifacts.collect(b_dir), _ALLOWED, ["summary"]
    )
    assert any("episode.title" in d for d in diffs)


def test_an_allowed_field_and_an_llm_field_are_not(tmp_path: Path) -> None:
    a_dir, b_dir = _run(tmp_path, "a"), _run(tmp_path, "b")
    _edit(_meta(b_dir), lambda d: d["episode"].__setitem__("language", "en"))
    _edit(_meta(b_dir), lambda d: d["summary"].__setitem__("bullets", ["different words"]))
    assert (
        artifacts.compare_deterministic(
            artifacts.collect(a_dir), artifacts.collect(b_dir), _ALLOWED, ["summary"]
        )
        == []
    )


def test_new_files_are_allowed_only_by_pattern(tmp_path: Path) -> None:
    a_dir, b_dir = _run(tmp_path, "a"), _run(tmp_path, "b")
    transcripts = next(b_dir.rglob("transcripts"))
    (transcripts / "p01_e01.turns.json").write_text("{}", encoding="utf-8")
    (transcripts / "p01_e01.unexpected.json").write_text("{}", encoding="utf-8")
    diffs = artifacts.compare_deterministic(
        artifacts.collect(a_dir), artifacts.collect(b_dir), _ALLOWED, []
    )
    assert [d for d in diffs if d.startswith("new file")] == [
        d for d in diffs if "unexpected" in d
    ] and any("unexpected" in d for d in diffs)


def test_llm_counts_inside_the_base_spread_are_ok_and_outside_are_flagged(tmp_path: Path) -> None:
    base = artifacts.collect(_run(tmp_path, "a"))
    ep = next(iter(base["episodes"]))
    b1 = json.loads(json.dumps(base))
    b2 = json.loads(json.dumps(base))
    b1["episodes"][ep]["llm"]["insights"] = 20
    b2["episodes"][ep]["llm"]["insights"] = 24
    near = json.loads(json.dumps(base))
    near["episodes"][ep]["llm"]["insights"] = 23
    far = json.loads(json.dumps(base))
    far["episodes"][ep]["llm"]["insights"] = 40

    def ok(cand) -> bool:  # type: ignore[no-untyped-def]
        row = next(r for r in artifacts.llm_rows([b1, b2], cand) if r["metric"] == f"{ep} insights")
        return bool(row["ok"])

    assert ok(near) is True
    assert ok(far) is False
