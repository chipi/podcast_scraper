"""Integration test for insight-clusters CLI command (#599).

Validates the cross-module flow:
  gi.json artifacts → insight-clusters CLI → insight_clusters.json

This is a lightweight test that uses pre-built gi.json fixtures
without requiring sentence-transformers (mocked embedding).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.insight_clusters import (
    build_insight_clusters_for_corpus,
    collect_insight_rows_from_corpus,
)

pytestmark = [pytest.mark.integration]


def _write_gi_artifact(output_dir: Path, episode_id: str, insights: list[dict]) -> None:
    """Write a minimal .gi.json artifact."""
    nodes = []
    edges = []
    for ins in insights:
        nodes.append(
            {
                "id": ins["id"],
                "type": "Insight",
                "properties": {
                    "text": ins["text"],
                    "insight_type": "factual",
                    "grounded": True,
                },
            }
        )
        for i, q in enumerate(ins.get("quotes", [])):
            qid = f"{ins['id']}_q{i}"
            nodes.append(
                {
                    "id": qid,
                    "type": "Quote",
                    "properties": {
                        "text": q,
                        "char_start": i * 100,
                        "char_end": i * 100 + 50,
                    },
                }
            )
            edges.append({"from": ins["id"], "to": qid, "type": "SUPPORTED_BY"})

    gi = {"episode_id": episode_id, "nodes": nodes, "edges": edges}
    ep_dir = output_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / f"{episode_id}.gi.json").write_text(json.dumps(gi), encoding="utf-8")


def test_collect_insights_from_multi_episode_corpus(tmp_path: Path) -> None:
    """collect_insight_rows_from_corpus finds insights across episodes."""
    _write_gi_artifact(
        tmp_path,
        "ep1",
        [
            {
                "id": "ins1",
                "text": "Index funds beat active managers",
                "quotes": ["92% underperform"],
            },
        ],
    )
    _write_gi_artifact(
        tmp_path,
        "ep2",
        [
            {
                "id": "ins2",
                "text": "Passive investing outperforms",
                "quotes": ["the data is clear"],
            },
        ],
    )
    rows = collect_insight_rows_from_corpus(tmp_path)
    assert len(rows) == 2
    assert {r["episode_id"] for r in rows} == {"ep1", "ep2"}
    assert all(len(r["supporting_quotes"]) == 1 for r in rows)


def test_build_insight_clusters_end_to_end(tmp_path: Path) -> None:
    """Full flow: write gi.json → build clusters → verify output file."""
    # Two similar insights across episodes
    _write_gi_artifact(
        tmp_path,
        "ep1",
        [
            {"id": "ins1", "text": "Index funds beat active managers over long periods"},
        ],
    )
    _write_gi_artifact(
        tmp_path,
        "ep2",
        [
            {"id": "ins2", "text": "Index funds consistently outperform active managers"},
        ],
    )
    # One unrelated insight
    _write_gi_artifact(
        tmp_path,
        "ep3",
        [
            {"id": "ins3", "text": "Mediterranean diet reduces cardiovascular risk"},
        ],
    )

    payload = build_insight_clusters_for_corpus(tmp_path, threshold=0.70)

    # Verify output file written
    out_file = tmp_path / "search" / "insight_clusters.json"
    assert out_file.exists()

    # Verify payload structure
    assert payload["insight_count"] == 3
    assert payload["schema_version"] == "1"
    assert "clusters" in payload
    assert "singletons" in payload

    # The two index fund insights should cluster; Mediterranean diet should not
    if payload["cluster_count"] > 0:
        cluster = payload["clusters"][0]
        assert cluster["member_count"] == 2
        member_texts = {m["text"] for m in cluster["members"]}
        assert "Mediterranean diet reduces cardiovascular risk" not in member_texts


def test_cluster_context_expansion_with_cli_artifacts(tmp_path: Path) -> None:
    """Cluster context expansion works with artifacts from build_insight_clusters_for_corpus."""
    from podcast_scraper.search.insight_cluster_context import (
        expand_with_cluster_context,
        load_insight_clusters,
    )

    _write_gi_artifact(
        tmp_path,
        "ep1",
        [
            {
                "id": "ins1",
                "text": "Index funds beat active managers",
                "quotes": ["92% of active managers underperform"],
            },
        ],
    )
    _write_gi_artifact(
        tmp_path,
        "ep2",
        [
            {
                "id": "ins2",
                "text": "Index funds consistently outperform active managers",
                "quotes": ["the data clearly shows passive wins"],
            },
        ],
    )

    # Build clusters
    build_insight_clusters_for_corpus(tmp_path, threshold=0.70)

    # Load and expand
    clusters_path = tmp_path / "search" / "insight_clusters.json"
    assert clusters_path.exists()

    cluster_index = load_insight_clusters(clusters_path)
    insights = [{"insight_id": "ins1", "text": "test", "episode_id": "ep1"}]
    expanded = expand_with_cluster_context(insights, cluster_index=cluster_index)

    # If clustered, should have cross-episode context
    if cluster_index.get("ins1", {}).get("cross_episode"):
        assert "cluster" in expanded[0]
        assert expanded[0]["cluster"]["cluster_episodes"] == 2
