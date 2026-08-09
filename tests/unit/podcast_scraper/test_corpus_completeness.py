"""Corpus-completeness + index-staleness gates (#1494 / #1497 → #16).

Verifies the preventative guard catches exactly the shapes the two incidents produced:
a stale/absent LanceDB index, and a corpus missing typed edges / enrichments / diarization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.corpus_completeness import (
    assess_completeness,
    assess_index_staleness,
    check_corpus,
    format_report,
    LANCE_SCHEMA_VERSION,
)

pytestmark = pytest.mark.unit


def _write_index(root: Path, schema_version: int | None) -> None:
    idx = root / "search" / "lance_index"
    idx.mkdir(parents=True, exist_ok=True)
    meta = {} if schema_version is None else {"schema_version": schema_version}
    (idx / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_episode(root: Path, eid: str, edge_types: list[str]) -> None:
    run = root / "feeds" / "feedA" / "run_1" / "metadata"
    run.mkdir(parents=True, exist_ok=True)
    edges = [{"type": t, "from": f"a:{i}", "to": f"b:{i}"} for i, t in enumerate(edge_types)]
    (run / f"{eid}.gi.json").write_text(
        json.dumps({"episode": {"episode_id": eid}, "edges": edges}), encoding="utf-8"
    )


def _write_enrichments(root: Path) -> None:
    enr = root / "feeds" / "feedA" / "run_1" / "enrichments"
    enr.mkdir(parents=True, exist_ok=True)
    (enr / "insight_density.json").write_text("{}", encoding="utf-8")


def _write_topic_clusters(root: Path) -> None:
    idx = root / "search"
    idx.mkdir(parents=True, exist_ok=True)
    (idx / "topic_clusters.json").write_text(json.dumps({"clusters": [{"id": "tc1"}]}), "utf-8")


def _complete_corpus(root: Path) -> Path:
    """A fully-populated corpus: fresh index, typed edges, diarization, enrichments, clusters."""
    _write_index(root, LANCE_SCHEMA_VERSION)
    _write_episode(root, "e1", ["HAS_EPISODE", "MENTIONS_PERSON", "MENTIONS_ORG", "SPOKEN_BY"])
    _write_enrichments(root)
    _write_topic_clusters(root)
    return root


# ---------------- index staleness (#1494) ----------------


def test_index_fresh_is_ok(tmp_path: Path):
    _write_index(tmp_path, LANCE_SCHEMA_VERSION)
    s = assess_index_staleness(tmp_path)
    assert s.present and not s.stale and s.ok
    assert s.served_version == LANCE_SCHEMA_VERSION


def test_index_old_schema_is_stale(tmp_path: Path):
    _write_index(tmp_path, 1)  # the FAISS-era v1 on v3 code — the #1494 incident
    s = assess_index_staleness(tmp_path)
    assert s.present and s.stale and not s.ok
    assert "stale" in s.reason()


def test_index_absent_is_not_ok(tmp_path: Path):
    s = assess_index_staleness(tmp_path)
    assert not s.present and not s.ok
    assert "no LanceDB index" in s.reason()


# ---------------- edge / stage completeness (#1497) ----------------


def test_complete_corpus_passes(tmp_path: Path):
    _complete_corpus(tmp_path)
    report = assess_completeness(tmp_path)
    assert report.ok
    assert report.missing_hard == []
    assert report.missing_soft == []
    assert report.has_enrichments
    assert report.episodes_scanned == 1


def test_missing_has_episode_fails_hard(tmp_path: Path):
    _write_index(tmp_path, LANCE_SCHEMA_VERSION)
    _write_episode(tmp_path, "e1", ["MENTIONS_PERSON", "SPOKEN_BY"])  # no HAS_EPISODE
    _write_enrichments(tmp_path)
    report = assess_completeness(tmp_path)
    assert not report.ok
    stages = [m.stage for m in report.missing_hard]
    assert any("HAS_EPISODE" in s for s in stages)
    assert "show_episodes" in {t for m in report.missing_hard for t in m.kills}


def test_only_generic_mentions_fails_typed_drift(tmp_path: Path):
    """The exact prod drift: generic MENTIONS present but no typed MENTIONS_PERSON/ORG."""
    _write_index(tmp_path, LANCE_SCHEMA_VERSION)
    _write_episode(tmp_path, "e1", ["HAS_EPISODE", "MENTIONS", "SPOKEN_BY"])
    _write_enrichments(tmp_path)
    report = assess_completeness(tmp_path)
    assert not report.ok
    assert any("MENTIONS_PERSON/ORG" in m.stage for m in report.missing_hard)


def test_missing_enrichments_fails(tmp_path: Path):
    _write_index(tmp_path, LANCE_SCHEMA_VERSION)
    _write_episode(tmp_path, "e1", ["HAS_EPISODE", "MENTIONS_PERSON", "SPOKEN_BY"])
    # no enrichments/ dir
    report = assess_completeness(tmp_path)
    assert not report.ok
    assert not report.has_enrichments


def test_missing_diarization_is_soft_only(tmp_path: Path):
    """Diarization is optional (bridge-only audio) → warn, don't fail the gate."""
    _write_index(tmp_path, LANCE_SCHEMA_VERSION)
    _write_episode(tmp_path, "e1", ["HAS_EPISODE", "MENTIONS_PERSON"])  # no SPOKEN_BY
    _write_enrichments(tmp_path)
    _write_topic_clusters(tmp_path)
    report = assess_completeness(tmp_path)
    assert report.ok  # hard checks pass → gate passes
    assert any("SPOKEN_BY" in m.stage for m in report.missing_soft)


def test_missing_topic_clusters_fails_when_index_present(tmp_path: Path):
    """#14 cutover: index present but search/topic_clusters.json absent → 404 smoke → HARD fail."""
    _write_index(tmp_path, LANCE_SCHEMA_VERSION)
    _write_episode(tmp_path, "e1", ["HAS_EPISODE", "MENTIONS_PERSON", "SPOKEN_BY"])
    _write_enrichments(tmp_path)
    # no topic_clusters.json
    report = assess_completeness(tmp_path)
    assert not report.ok
    assert not report.has_topic_clusters
    assert "topic_clusters.json" in format_report(report)

    _write_topic_clusters(tmp_path)  # once present → passes
    report2 = assess_completeness(tmp_path)
    assert report2.ok and report2.has_topic_clusters


def test_stale_index_fails_even_if_edges_complete(tmp_path: Path):
    _write_index(tmp_path, 1)  # stale
    _write_episode(tmp_path, "e1", ["HAS_EPISODE", "MENTIONS_PERSON", "SPOKEN_BY"])
    _write_enrichments(tmp_path)
    report = assess_completeness(tmp_path)
    assert not report.ok
    assert report.index.stale


# ---------------- CLI-level verdict ----------------


def test_check_corpus_pass_and_fail(tmp_path: Path):
    complete = tmp_path / "good"
    complete.mkdir()
    _complete_corpus(complete)
    ok, text = check_corpus(complete)
    assert ok and "VERDICT: PASS" in text

    bad = tmp_path / "bad"
    bad.mkdir()
    _write_index(bad, 1)  # stale + no edges + no enrichments
    ok2, text2 = check_corpus(bad)
    assert not ok2 and "VERDICT: FAIL" in text2
