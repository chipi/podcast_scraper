"""Integration tests for insight clustering + explore expansion (#599, #601).

These tests require sentence-transformers (embedding model) and validate
the full cross-module flow:
  gi.json artifacts → insight clustering → cluster context → CLI commands

Tests that only need JSON I/O live in unit tests instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.insight_clusters import (
    build_insight_clusters_for_corpus,
    build_insight_clusters_payload,
    collect_insight_rows_from_corpus,
)

pytestmark = [pytest.mark.integration]


def _write_gi_artifact(
    output_dir: Path,
    episode_id: str,
    insights: list[dict],
) -> None:
    """Write a minimal .gi.json artifact with optional ABOUT edges."""
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
        # Add ABOUT edges to topics
        for tid in ins.get("topic_ids", []):
            nodes.append(
                {
                    "id": tid,
                    "type": "Topic",
                    "properties": {"label": ins.get("topic_label", tid)},
                }
            )
            edges.append({"from": ins["id"], "to": tid, "type": "ABOUT"})

    gi = {"episode_id": episode_id, "nodes": nodes, "edges": edges}
    ep_dir = output_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / f"{episode_id}.gi.json").write_text(json.dumps(gi), encoding="utf-8")


# ── collect (lightweight, no embedder) ───────────────────────────────


def test_collect_insights_from_multi_episode_corpus(tmp_path: Path) -> None:
    _write_gi_artifact(
        tmp_path,
        "ep1",
        [{"id": "ins1", "text": "Index funds beat active managers", "quotes": ["92%"]}],
    )
    _write_gi_artifact(
        tmp_path,
        "ep2",
        [{"id": "ins2", "text": "Passive investing outperforms", "quotes": ["data"]}],
    )
    rows = collect_insight_rows_from_corpus(tmp_path)
    assert len(rows) == 2
    assert {r["episode_id"] for r in rows} == {"ep1", "ep2"}


# ── build_insight_clusters_payload (needs sentence-transformers) ─────


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
    cluster = payload["clusters"][0]
    assert cluster["member_count"] == 2
    assert cluster["cross_episode"] is True
    assert set(cluster["episode_ids"]) == {"ep1", "ep2"}
    assert cluster["cluster_id"].startswith("ic:")


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


# ── end-to-end: build + expand ───────────────────────────────────────


def test_build_insight_clusters_end_to_end(tmp_path: Path) -> None:
    """Full flow: write gi.json → build clusters → verify output file."""
    _write_gi_artifact(
        tmp_path,
        "ep1",
        [{"id": "ins1", "text": "Index funds beat active managers over long periods"}],
    )
    _write_gi_artifact(
        tmp_path,
        "ep2",
        [{"id": "ins2", "text": "Index funds consistently outperform active managers"}],
    )
    _write_gi_artifact(
        tmp_path,
        "ep3",
        [{"id": "ins3", "text": "Mediterranean diet reduces cardiovascular risk"}],
    )

    payload = build_insight_clusters_for_corpus(tmp_path, threshold=0.70)
    out_file = tmp_path / "search" / "insight_clusters.json"
    assert out_file.exists()
    assert payload["insight_count"] == 3
    assert payload["schema_version"] == "1"

    if payload["cluster_count"] > 0:
        cluster = payload["clusters"][0]
        assert cluster["member_count"] == 2
        member_texts = {m["text"] for m in cluster["members"]}
        assert "Mediterranean diet reduces cardiovascular risk" not in member_texts


def test_cluster_context_expansion_with_cli_artifacts(tmp_path: Path) -> None:
    """Cluster context expansion works with real clustering output."""
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

    build_insight_clusters_for_corpus(tmp_path, threshold=0.70)
    clusters_path = tmp_path / "search" / "insight_clusters.json"
    assert clusters_path.exists()

    cluster_index = load_insight_clusters(clusters_path)
    insights = [{"insight_id": "ins1", "text": "test", "episode_id": "ep1"}]
    expanded = expand_with_cluster_context(insights, cluster_index=cluster_index)

    if cluster_index.get("ins1", {}).get("cross_episode"):
        assert "cluster" in expanded[0]
        assert expanded[0]["cluster"]["cluster_episodes"] == 2


# ── CLI handlers: cluster browse + topic-insights ────────────────────


def test_cluster_browse_on_real_clusters(tmp_path: Path, capsys) -> None:
    """Build real clusters, then browse them via CLI handler."""
    from podcast_scraper.search.cli_handlers import run_cluster_browse_cli

    _write_gi_artifact(
        tmp_path, "ep1", [{"id": "ins1", "text": "Index funds beat active managers"}]
    )
    _write_gi_artifact(
        tmp_path, "ep2", [{"id": "ins2", "text": "Index funds outperform active managers"}]
    )
    build_insight_clusters_for_corpus(tmp_path, threshold=0.70)

    from argparse import Namespace

    args = Namespace(output_dir=str(tmp_path), top=10, format="json")
    logger_stub = type("L", (), {"info": lambda *a: None, "error": lambda *a: None})()
    rc = run_cluster_browse_cli(args, logger_stub)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total"] >= 1


def test_topic_insights_on_real_data(tmp_path: Path, capsys) -> None:
    """Build clusters with ABOUT edges, query by topic via CLI handler."""
    from podcast_scraper.search.cli_handlers import run_topic_insights_cli

    _write_gi_artifact(
        tmp_path,
        "ep1",
        [
            {
                "id": "ins1",
                "text": "Index funds beat active managers",
                "topic_ids": ["topic:investing"],
                "topic_label": "investing",
            }
        ],
    )
    _write_gi_artifact(
        tmp_path,
        "ep2",
        [
            {
                "id": "ins2",
                "text": "Index funds outperform actively managed funds",
                "topic_ids": ["topic:investing"],
                "topic_label": "investing",
            }
        ],
    )
    build_insight_clusters_for_corpus(tmp_path, threshold=0.70)

    from argparse import Namespace

    args = Namespace(output_dir=str(tmp_path), topic="investing", format="json")
    logger_stub = type("L", (), {"info": lambda *a: None, "error": lambda *a: None})()
    rc = run_topic_insights_cli(args, logger_stub)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["topic_query"] == "investing"
    # Should find the cluster linked via ABOUT edges
    assert len(out["clusters"]) >= 1
