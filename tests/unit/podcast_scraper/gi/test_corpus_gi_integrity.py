"""The repair's exit criterion must assert insights EXIST, not that a bad string is absent.

WHY THIS FILE EXISTS — a failure of the gate it replaces, demonstrated 2026-08-16
``check_corpus_for_placeholders`` scans for the literal "Summary insight (stub)." across
``rglob("*.gi.json")``. Rehearsing the corpus repair, the placeholder artifact was deleted and
the pipeline failed to regenerate it. The old gate reported::

    legacy placeholders found : 0
    VERDICT: PASS

for a corpus where that episode had NO GI AT ALL. An operator whose repair is ``rm`` gets a green
light and a corpus with holes — strictly worse than the defect the gate was built to catch,
because it converts a visible problem into an invisible one.

The rule these tests encode: start from EPISODES (metadata), not from artifacts that happen to
exist. A missing artifact is invisible to a glob over artifacts, and that invisibility was the
whole bug.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.gi.corpus import LEGACY_PLACEHOLDER_INSIGHT_TEXT
from podcast_scraper.gi.integrity import assess_gi_integrity, check_corpus_gi_integrity

pytestmark = [pytest.mark.unit]


def _write_episode(
    corpus: Path,
    *,
    feed: str = "feed_a",
    run: str = "run_20260815-120000",
    episode_id: str = "ep-1",
    name: str = "0001 - Episode",
    insight_texts: List[str] | None = None,
    declare_gi: bool = True,
    write_artifact: bool = True,
) -> Path:
    """Create one episode's metadata (+ optional gi.json) in corpus layout."""
    meta_dir = corpus / "feeds" / feed / run / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    gi_rel = f"metadata/{name}.gi.json"
    meta: Dict[str, Any] = {
        "episode": {"episode_id": episode_id, "title": name},
        "content": {"transcript_file_path": f"transcripts/{name}.txt"},
    }
    if declare_gi:
        meta["grounded_insights"] = {
            "artifact_path": gi_rel,
            "insight_count": len(insight_texts or []),
            "schema_version": "3.1",
        }
    (meta_dir / f"{name}.metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    artifact = meta_dir / f"{name}.gi.json"
    if declare_gi and write_artifact:
        nodes: List[Dict[str, Any]] = [{"id": episode_id, "type": "Episode", "properties": {}}]
        for i, text in enumerate(insight_texts or []):
            nodes.append(
                {"id": f"{episode_id}:i{i}", "type": "Insight", "properties": {"text": text}}
            )
        artifact.write_text(
            json.dumps({"episode_id": episode_id, "nodes": nodes, "edges": []}), encoding="utf-8"
        )
    return artifact


def test_a_healthy_corpus_passes(tmp_path):
    _write_episode(tmp_path, episode_id="ep-1", insight_texts=["A real claim.", "Another."])
    _write_episode(tmp_path, episode_id="ep-2", name="0002 - Second", insight_texts=["Real."])

    ok, report = check_corpus_gi_integrity(tmp_path)

    assert ok, report
    assert "VERDICT: PASS" in report
    assert "episodes with healthy GI    : 2" in report


def test_a_declared_but_missing_artifact_FAILS(tmp_path):
    """THE regression. The old gate passed this exact corpus."""
    _write_episode(tmp_path, episode_id="ep-1", insight_texts=["Real."])
    _write_episode(
        tmp_path,
        episode_id="ep-gone",
        name="0002 - Amputated",
        insight_texts=["was here"],
        write_artifact=False,
    )

    ok, report = check_corpus_gi_integrity(tmp_path)

    assert ok is False, "metadata claims GI that is not on disk — this must never pass"
    assert "declared but MISSING      : 1" in report
    assert "ep-gone" in report


def test_a_surviving_placeholder_FAILS(tmp_path):
    _write_episode(tmp_path, episode_id="ep-1", insight_texts=["Real."])
    _write_episode(
        tmp_path,
        episode_id="ep-stub",
        name="0002 - Stubbed",
        insight_texts=[LEGACY_PLACEHOLDER_INSIGHT_TEXT],
    )

    ok, report = check_corpus_gi_integrity(tmp_path)

    assert ok is False
    assert "legacy placeholders       : 1" in report
    assert "ep-stub" in report


def test_one_episode_with_two_artifacts_FAILS(tmp_path):
    """The supersede hazard — observed for real in the acceptance corpus, not hypothetical.

    A pipeline re-run writes its output into a FRESH ``run_<ts>/`` dir and leaves the previous
    artifact on disk. Two subsystems then disagree about which is canonical:
    ``_scan_corpus_metadata_index`` is first-writer-wins (OLDEST), while search's
    ``merged_episode_gi_paths`` takes the NEWEST.
    """
    _write_episode(
        tmp_path, run="run_20260815-120000", episode_id="ep-dup", insight_texts=["First pass."]
    )
    _write_episode(
        tmp_path, run="run_20260816-090000", episode_id="ep-dup", insight_texts=["Second pass."]
    )

    ok, report = check_corpus_gi_integrity(tmp_path)

    assert ok is False, "one episode resolving to two artifacts must fail"
    assert "duplicate episode_id      : 1" in report
    assert "ep-dup" in report


def test_a_zero_insight_artifact_PASSES_but_is_counted(tmp_path):
    """Post-#1657 "nothing extracted means nothing returned" is LEGAL — and must stay visible.

    If the repair turned 112 placeholders into 112 empty artifacts it would have re-derived
    nothing, while satisfying a gate that only asks "no placeholders?". The count is the guard.
    """
    _write_episode(tmp_path, episode_id="ep-1", insight_texts=["Real."])
    _write_episode(tmp_path, episode_id="ep-empty", name="0002 - Empty", insight_texts=[])

    ok, report = check_corpus_gi_integrity(tmp_path)

    assert ok is True, "an honestly empty artifact is a legal state"
    assert "zero-insight artifacts    : 1" in report
    assert "ep-empty" in report, "it must be listed, not merely counted"


def test_an_unreadable_artifact_FAILS(tmp_path):
    artifact = _write_episode(tmp_path, episode_id="ep-bad", insight_texts=["Real."])
    artifact.write_text("{ not valid json", encoding="utf-8")

    ok, report = check_corpus_gi_integrity(tmp_path)

    assert ok is False
    assert "unreadable artifact       : 1" in report


def test_an_episode_with_no_gi_block_is_not_a_failure(tmp_path):
    """GI simply never ran for it. Legal, but reported so a clean verdict cannot hide it."""
    _write_episode(tmp_path, episode_id="ep-1", insight_texts=["Real."])
    _write_episode(tmp_path, episode_id="ep-nogi", name="0002 - No GI", declare_gi=False)

    ok, report = check_corpus_gi_integrity(tmp_path)

    assert ok is True
    assert "episodes with no GI block : 1" in report


def test_unreadable_metadata_is_reported_not_silently_skipped(tmp_path):
    _write_episode(tmp_path, episode_id="ep-1", insight_texts=["Real."])
    bad = (
        tmp_path / "feeds" / "feed_a" / "run_20260815-120000" / "metadata" / "broken.metadata.json"
    )
    bad.write_text("{ nope", encoding="utf-8")

    ok, report = check_corpus_gi_integrity(tmp_path)

    assert "unreadable metadata       : 1" in report
    assert ok is True, "a corrupt metadata file is reported, but does not by itself fail the gate"


def test_an_empty_corpus_does_not_read_as_a_clean_bill(tmp_path):
    ok, report = check_corpus_gi_integrity(tmp_path)

    assert ok is True
    assert "zero metadata files scanned" in report


def test_a_corpus_where_nothing_is_healthy_says_so(tmp_path):
    """Passing with zero healthy episodes is arithmetic, not success."""
    _write_episode(tmp_path, episode_id="ep-nogi", declare_gi=False)

    _ok, report = check_corpus_gi_integrity(tmp_path)

    assert "no episode has healthy GI" in report


def test_the_raw_assessment_exposes_every_bucket(tmp_path):
    """The report is for humans; the dict is what a repair tool drives from."""
    _write_episode(tmp_path, episode_id="ep-ok", insight_texts=["Real."])
    _write_episode(
        tmp_path,
        episode_id="ep-stub",
        name="0002 - Stub",
        insight_texts=[LEGACY_PLACEHOLDER_INSIGHT_TEXT],
    )

    r = assess_gi_integrity(tmp_path)

    assert {e["episode_id"] for e in r["legacy_placeholders"]} == {"ep-stub"}
    assert {e["episode_id"] for e in r["healthy"]} == {"ep-ok"}
    assert r["metadata_scanned"] == 2
