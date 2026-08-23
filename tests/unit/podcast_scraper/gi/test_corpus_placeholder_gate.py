"""The placeholder gate must be usable by an operator and must fail loudly.

Detection for the #1655 repair existed since #19 but had no caller outside its own unit test —
no CLI, no Make target — so "are there still placeholders in the corpus?", the repair's exit
criterion, could not be answered with shipped code. ``check_corpus_for_placeholders`` is that
entrypoint, wired to ``make corpus-placeholder-check``.

It runs twice in the repair: before, to produce the work-list, and after, to prove the work
landed. The after-run is the one whose failure must be impossible to miss, which is why a find
is a non-zero exit rather than a printed warning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from podcast_scraper.gi.corpus import (
    check_corpus_for_placeholders,
    LEGACY_PLACEHOLDER_INSIGHT_TEXT,
)


def _artifact(episode_id: str, insight_texts: List[str]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = [{"id": episode_id, "type": "Episode", "properties": {}}]
    for i, text in enumerate(insight_texts):
        nodes.append({"id": f"{episode_id}:i{i}", "type": "Insight", "properties": {"text": text}})
    return {"episode_id": episode_id, "nodes": nodes, "edges": []}


def _write(root: Path, name: str, doc: Dict[str, Any]) -> Path:
    path = root / "feeds" / name / f"{name}.gi.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_a_clean_corpus_passes(tmp_path):
    _write(tmp_path, "ep1", _artifact("ep1", ["A real insight.", "Another one."]))
    _write(tmp_path, "ep2", _artifact("ep2", ["Something genuinely extracted."]))

    ok, report = check_corpus_for_placeholders(tmp_path)

    assert ok is True
    assert "VERDICT: PASS" in report
    assert "legacy placeholders found : 0" in report


def test_a_placeholder_fails_the_gate_and_is_named(tmp_path):
    """The point of the gate: the operator gets the work-list, not just a count."""
    _write(tmp_path, "ep1", _artifact("ep1", ["A real insight."]))
    _write(tmp_path, "bad", _artifact("ep-broken", [LEGACY_PLACEHOLDER_INSIGHT_TEXT]))

    ok, report = check_corpus_for_placeholders(tmp_path)

    assert ok is False, "a surviving placeholder must be a non-zero exit, not a warning"
    assert "VERDICT: FAIL" in report
    assert "legacy placeholders found : 1" in report
    assert "ep-broken" in report, "the episode must be named so the repair can target it"


def test_the_work_list_is_not_truncated(tmp_path):
    """A head -N here would silently under-report the repair; there is no second source."""
    for i in range(25):
        _write(tmp_path, f"bad{i}", _artifact(f"ep-{i}", [LEGACY_PLACEHOLDER_INSIGHT_TEXT]))

    ok, report = check_corpus_for_placeholders(tmp_path)

    assert ok is False
    for i in range(25):
        assert f"ep-{i}" in report, f"ep-{i} missing from the work-list"


def test_a_genuine_single_insight_episode_is_not_swept_up(tmp_path):
    """Re-deriving these replaces FAILURES. Redoing work that succeeded is the opposite."""
    _write(tmp_path, "ep1", _artifact("ep1", ["The one thing this episode actually said."]))

    ok, report = check_corpus_for_placeholders(tmp_path)

    assert ok is True, report


def test_an_empty_corpus_passes_but_says_so(tmp_path):
    """Zero artifacts scanned is a PASS by arithmetic — it must not read as a clean bill."""
    ok, report = check_corpus_for_placeholders(tmp_path)

    assert ok is True
    assert (
        "zero artifacts scanned" in report.lower()
    ), "an empty/wrong CORPUS_DIR must be distinguishable from a genuinely repaired corpus"


def test_a_clean_non_empty_corpus_points_at_the_cleanup(tmp_path):
    """Zero placeholders is the trigger to delete the forensic constant — say so there."""
    _write(tmp_path, "ep1", _artifact("ep1", ["Real."]))

    _ok, report = check_corpus_for_placeholders(tmp_path)

    assert "can now be deleted" in report


def test_a_malformed_artifact_does_not_take_the_scan_down(tmp_path):
    """One corrupt file must not hide the placeholders in the other 677 episodes."""
    _write(tmp_path, "ep1", _artifact("ep1", ["Real."]))
    bad = tmp_path / "feeds" / "corrupt" / "corrupt.gi.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not valid json", encoding="utf-8")
    _write(tmp_path, "stubbed", _artifact("ep-stub", [LEGACY_PLACEHOLDER_INSIGHT_TEXT]))

    ok, report = check_corpus_for_placeholders(tmp_path)

    assert ok is False
    assert "ep-stub" in report, "a corrupt neighbour must not mask a real find"
