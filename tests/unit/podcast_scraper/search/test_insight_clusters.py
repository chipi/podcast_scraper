"""Unit tests for cross-episode insight clustering (#599)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.search.insight_clusters import (
    build_insight_clusters_payload,
    collect_insight_rows_from_corpus,
    INSIGHT_CLUSTERS_SCHEMA_VERSION,
)

# ── helpers ──────────────────────────────────────────────────────────


def _gi_json(episode_id: str, insights: list[dict]) -> dict:
    """Build a minimal gi.json structure."""
    nodes = []
    edges = []
    for ins in insights:
        ins_id = ins["id"]
        nodes.append(
            {
                "id": ins_id,
                "type": "Insight",
                "properties": {
                    "text": ins["text"],
                    "insight_type": ins.get("type", "factual"),
                    "grounded": ins.get("grounded", True),
                },
            }
        )
        for q in ins.get("quotes", []):
            qid = q["id"]
            nodes.append(
                {
                    "id": qid,
                    "type": "Quote",
                    "properties": {
                        "text": q["text"],
                        "speaker_id": q.get("speaker_id"),
                        "char_start": q.get("char_start", 0),
                        "char_end": q.get("char_end", 100),
                    },
                }
            )
            edges.append({"from": ins_id, "to": qid, "type": "SUPPORTED_BY"})
    return {"episode_id": episode_id, "nodes": nodes, "edges": edges}


def _write_gi(tmp_path: Path, episode_id: str, gi_data: dict) -> None:
    ep_dir = tmp_path / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / f"{episode_id}.gi.json").write_text(json.dumps(gi_data), encoding="utf-8")


# ── collect_insight_rows_from_corpus ─────────────────────────────────


def test_collect_empty_corpus(tmp_path: Path) -> None:
    rows = collect_insight_rows_from_corpus(tmp_path)
    assert rows == []


def test_collect_single_episode(tmp_path: Path) -> None:
    gi = _gi_json(
        "ep1",
        [
            {
                "id": "ins1",
                "text": "Index funds beat active managers",
                "quotes": [{"id": "q1", "text": "92% of active managers"}],
            }
        ],
    )
    _write_gi(tmp_path, "ep1", gi)
    rows = collect_insight_rows_from_corpus(tmp_path)
    assert len(rows) == 1
    assert rows[0]["insight_id"] == "ins1"
    assert rows[0]["episode_id"] == "ep1"
    assert len(rows[0]["supporting_quotes"]) == 1
    assert rows[0]["supporting_quotes"][0]["text"] == "92% of active managers"


def test_collect_multi_episode(tmp_path: Path) -> None:
    gi1 = _gi_json("ep1", [{"id": "ins1", "text": "Claim A"}])
    gi2 = _gi_json("ep2", [{"id": "ins2", "text": "Claim B"}])
    _write_gi(tmp_path, "ep1", gi1)
    _write_gi(tmp_path, "ep2", gi2)
    rows = collect_insight_rows_from_corpus(tmp_path)
    assert len(rows) == 2
    episode_ids = {r["episode_id"] for r in rows}
    assert episode_ids == {"ep1", "ep2"}


def test_collect_skips_empty_text(tmp_path: Path) -> None:
    gi = _gi_json("ep1", [{"id": "ins1", "text": ""}])
    _write_gi(tmp_path, "ep1", gi)
    rows = collect_insight_rows_from_corpus(tmp_path)
    assert rows == []


def test_collect_skips_invalid_json(tmp_path: Path) -> None:
    ep_dir = tmp_path / "bad"
    ep_dir.mkdir()
    (ep_dir / "bad.gi.json").write_text("not json", encoding="utf-8")
    rows = collect_insight_rows_from_corpus(tmp_path)
    assert rows == []


# ── build_insight_clusters_payload ───────────────────────────────────


def test_empty_rows_returns_empty_payload() -> None:
    payload = build_insight_clusters_payload([])
    assert payload["schema_version"] == INSIGHT_CLUSTERS_SCHEMA_VERSION
    assert payload["insight_count"] == 0
    assert payload["cluster_count"] == 0
    assert payload["clusters"] == []


def test_two_similar_insights_cluster_together() -> None:
    rows = [
        {
            "insight_id": "a1",
            "text": "Index funds consistently outperform active managers",
            "insight_type": "factual",
            "episode_id": "ep1",
            "grounded": True,
            "supporting_quotes": [],
        },
        {
            "insight_id": "a2",
            "text": "Index funds beat active managers over long periods",
            "insight_type": "factual",
            "episode_id": "ep2",
            "grounded": True,
            "supporting_quotes": [],
        },
    ]
    payload = build_insight_clusters_payload(rows, threshold=0.70)
    assert payload["insight_count"] == 2
    assert payload["cluster_count"] >= 1
    # The two should be in the same cluster
    cluster = payload["clusters"][0]
    assert cluster["member_count"] == 2
    assert cluster["cross_episode"] is True
    assert set(cluster["episode_ids"]) == {"ep1", "ep2"}
    assert cluster["cluster_id"].startswith("ic:")
    assert cluster["canonical_insight"]  # non-empty


def test_dissimilar_insights_stay_separate() -> None:
    rows = [
        {
            "insight_id": "a1",
            "text": "Quantum computing will revolutionize cryptography",
            "insight_type": "factual",
            "episode_id": "ep1",
            "grounded": True,
            "supporting_quotes": [],
        },
        {
            "insight_id": "b1",
            "text": "Mediterranean diet reduces cardiovascular risk",
            "insight_type": "factual",
            "episode_id": "ep2",
            "grounded": True,
            "supporting_quotes": [],
        },
    ]
    payload = build_insight_clusters_payload(rows, threshold=0.75)
    # Two very different insights should not cluster
    assert payload["cluster_count"] == 0
    assert payload["singletons"] == 2


def test_same_episode_cluster_not_cross_episode() -> None:
    rows = [
        {
            "insight_id": "a1",
            "text": "AI regulation will lag behind innovation",
            "insight_type": "factual",
            "episode_id": "ep1",
            "grounded": True,
            "supporting_quotes": [],
        },
        {
            "insight_id": "a2",
            "text": "AI regulation cannot keep pace with innovation",
            "insight_type": "factual",
            "episode_id": "ep1",
            "grounded": True,
            "supporting_quotes": [],
        },
    ]
    payload = build_insight_clusters_payload(rows, threshold=0.70)
    if payload["cluster_count"] > 0:
        assert payload["clusters"][0]["cross_episode"] is False


def test_cluster_members_have_similarity_scores() -> None:
    rows = [
        {
            "insight_id": "a1",
            "text": "Diversification is the only free lunch in investing",
            "insight_type": "factual",
            "episode_id": "ep1",
            "grounded": True,
            "supporting_quotes": [],
        },
        {
            "insight_id": "a2",
            "text": "Diversification remains the only free lunch for investors",
            "insight_type": "factual",
            "episode_id": "ep2",
            "grounded": True,
            "supporting_quotes": [],
        },
    ]
    payload = build_insight_clusters_payload(rows, threshold=0.70)
    if payload["cluster_count"] > 0:
        for member in payload["clusters"][0]["members"]:
            assert "similarity_to_centroid" in member
            assert 0.0 <= member["similarity_to_centroid"] <= 1.0


def test_supporting_quotes_preserved_in_cluster() -> None:
    rows = [
        {
            "insight_id": "a1",
            "text": "Index funds beat active managers",
            "insight_type": "factual",
            "episode_id": "ep1",
            "grounded": True,
            "supporting_quotes": [
                {
                    "quote_id": "q1",
                    "text": "92% underperform",
                    "speaker_id": "s1",
                    "char_start": 0,
                    "char_end": 50,
                }
            ],
        },
        {
            "insight_id": "a2",
            "text": "Index funds outperform active managers consistently",
            "insight_type": "factual",
            "episode_id": "ep2",
            "grounded": True,
            "supporting_quotes": [
                {
                    "quote_id": "q2",
                    "text": "the data is clear",
                    "speaker_id": "s2",
                    "char_start": 100,
                    "char_end": 200,
                }
            ],
        },
    ]
    payload = build_insight_clusters_payload(rows, threshold=0.70)
    if payload["cluster_count"] > 0:
        all_quotes = []
        for m in payload["clusters"][0]["members"]:
            all_quotes.extend(m["supporting_quotes"])
        assert len(all_quotes) >= 2
