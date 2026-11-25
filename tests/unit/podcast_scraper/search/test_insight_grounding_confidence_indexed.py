"""#2521: an insight row stores the insight's grounding confidence, not the constant 0.0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper import stage_hooks
from podcast_scraper.search import indexer, two_tier_indexer as tti

pytestmark = pytest.mark.unit


def _gi(path: Path, insights: dict) -> None:
    nodes = [
        {"id": nid, "type": "Insight", "properties": props, "text": "t"}
        for nid, props in insights.items()
    ]
    path.write_text(
        json.dumps({"schema_version": "4.0", "item_id": "ep1", "nodes": nodes, "edges": []}),
        encoding="utf-8",
    )


def _collect(tmp_path: Path, monkeypatch, insights: dict) -> dict:
    (tmp_path / "metadata").mkdir()
    meta_path = tmp_path / "metadata" / "ep1.metadata.json"
    _gi(tmp_path / "metadata" / "ep1.gi.json", insights)
    doc = {
        "episode": {"episode_id": "ep1"},
        "feed": {"feed_id": "feedkey"},
        "grounded_insights": {"artifact_path": "metadata/ep1.gi.json"},
    }
    hook_rows = [
        (
            f"insight:{nid}",
            "text",
            {"doc_type": "insight", "episode_id": "ep1", "feed_id": "feedkey", "source_id": nid},
        )
        for nid in insights
    ]
    monkeypatch.setattr(stage_hooks, "index_rows", lambda *a, **k: list(hook_rows))
    rows = indexer._collect_docs_for_episode(
        tmp_path,
        meta_path,
        doc,
        target_tokens=200,
        overlap_tokens=20,
        metadata_relative_path="metadata/ep1.metadata.json",
    )
    return {r[0]: r[2] for r in rows if r[2].get("doc_type") == "insight"}


def test_insight_meta_carries_the_gi_grounding_confidence(tmp_path, monkeypatch) -> None:
    metas = _collect(tmp_path, monkeypatch, {"i1": {"grounding_confidence": 0.91}, "i2": {}})
    assert metas["insight:i1"]["grounding_confidence"] == 0.91
    assert "grounding_confidence" not in metas["insight:i2"]


def test_insight_document_confidence_is_the_grounding_confidence() -> None:
    assert tti._grounding_confidence({"grounding_confidence": 0.91}) == 0.91
    assert tti._grounding_confidence({}) == 0.0
    assert tti._grounding_confidence({"grounding_confidence": None}) == 0.0


def test_an_episode_that_only_gained_a_confidence_keeps_its_fingerprint() -> None:
    # 3,549 of prod's 4,089 GI files carry grounding_confidence (2026-11-25): hashing it would
    # re-embed nearly the whole corpus on the first index after a deploy. The operator held that
    # back; rows written for any other reason still store the confidence.
    base = [("insight:i1", "text", {"doc_type": "insight"})]
    fp = tti._episode_fingerprint(base, "m")
    for conf in (0.9, 0.1, None):
        graded = [("insight:i1", "text", {"doc_type": "insight", "grounding_confidence": conf})]
        assert tti._episode_fingerprint(graded, "m") == fp
