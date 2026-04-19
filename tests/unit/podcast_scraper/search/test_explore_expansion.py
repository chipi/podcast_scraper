"""Unit tests for explore expansion features (#601 3b-3e)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from podcast_scraper.search.cli_handlers import (
    run_cluster_browse_cli,
    run_topic_insights_cli,
)

_logger = logging.getLogger(__name__)


def _make_ns(**kwargs):
    """Build a minimal argparse-like Namespace."""
    from argparse import Namespace

    return Namespace(**kwargs)


def _write_insight_clusters(tmp_path: Path, clusters: list) -> Path:
    search_dir = tmp_path / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "insight_count": sum(c["member_count"] for c in clusters),
        "cluster_count": len(clusters),
        "cross_episode_clusters": sum(1 for c in clusters if c.get("cross_episode")),
        "singletons": 0,
        "clusters": clusters,
    }
    path = search_dir / "insight_clusters.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_gi_artifact(tmp_path: Path, episode_id: str, insights_topics: list) -> None:
    """Write gi.json with insights, topics, and ABOUT edges."""
    nodes = []
    edges = []
    for item in insights_topics:
        ins_id = item["insight_id"]
        nodes.append(
            {
                "id": ins_id,
                "type": "Insight",
                "properties": {"text": item.get("text", ""), "grounded": True},
            }
        )
        for tid in item.get("topic_ids", []):
            nodes.append(
                {
                    "id": tid,
                    "type": "Topic",
                    "properties": {"label": item.get("topic_label", tid)},
                }
            )
            edges.append({"from": ins_id, "to": tid, "type": "ABOUT"})

    gi = {"episode_id": episode_id, "nodes": nodes, "edges": edges}
    ep_dir = tmp_path / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / f"{episode_id}.gi.json").write_text(json.dumps(gi), encoding="utf-8")


# ── cluster browse (#601 3b) ────────────────────────────────────────


def test_cluster_browse_no_file(tmp_path: Path) -> None:
    args = _make_ns(output_dir=str(tmp_path), top=10, format="pretty")
    rc = run_cluster_browse_cli(args, _logger)
    assert rc != 0


def test_cluster_browse_pretty(tmp_path: Path, capsys) -> None:
    clusters = [
        {
            "cluster_id": "ic:test-cluster",
            "canonical_insight": "Index funds beat active managers",
            "member_count": 3,
            "episode_count": 2,
            "cross_episode": True,
            "episode_ids": ["ep1", "ep2"],
            "members": [],
        },
        {
            "cluster_id": "ic:small-cluster",
            "canonical_insight": "AI will transform healthcare",
            "member_count": 2,
            "episode_count": 1,
            "cross_episode": False,
            "episode_ids": ["ep3"],
            "members": [],
        },
    ]
    _write_insight_clusters(tmp_path, clusters)
    args = _make_ns(output_dir=str(tmp_path), top=10, format="pretty")
    rc = run_cluster_browse_cli(args, _logger)
    assert rc == 0
    out = capsys.readouterr().out
    assert "ic:test-cluster" in out
    assert "3 insights" in out
    assert "[cross-episode]" in out
    assert "ic:small-cluster" in out


def test_cluster_browse_json(tmp_path: Path, capsys) -> None:
    clusters = [
        {
            "cluster_id": "ic:a",
            "canonical_insight": "Test",
            "member_count": 2,
            "episode_count": 1,
            "cross_episode": False,
            "episode_ids": ["ep1"],
            "members": [],
        }
    ]
    _write_insight_clusters(tmp_path, clusters)
    args = _make_ns(output_dir=str(tmp_path), top=5, format="json")
    rc = run_cluster_browse_cli(args, _logger)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 1
    assert len(out["clusters"]) == 1


def test_cluster_browse_top_limit(tmp_path: Path, capsys) -> None:
    clusters = [
        {
            "cluster_id": f"ic:c{i}",
            "canonical_insight": f"Insight {i}",
            "member_count": 10 - i,
            "episode_count": 1,
            "cross_episode": False,
            "episode_ids": ["ep1"],
            "members": [],
        }
        for i in range(5)
    ]
    _write_insight_clusters(tmp_path, clusters)
    args = _make_ns(output_dir=str(tmp_path), top=2, format="json")
    rc = run_cluster_browse_cli(args, _logger)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["clusters"]) == 2
    assert out["total"] == 5


# ── topic × insight matrix (#601 3d) ─────────────────────────────────


def test_topic_insights_no_clusters(tmp_path: Path) -> None:
    args = _make_ns(output_dir=str(tmp_path), topic="test", format="pretty")
    rc = run_topic_insights_cli(args, _logger)
    assert rc != 0  # No insight_clusters.json


def test_topic_insights_finds_match(tmp_path: Path, capsys) -> None:
    # Create gi.json with insight → topic ABOUT edge
    _write_gi_artifact(
        tmp_path,
        "ep1",
        [
            {
                "insight_id": "ins1",
                "text": "Index funds beat active managers",
                "topic_ids": ["topic:investing"],
                "topic_label": "investing",
            }
        ],
    )
    # Create insight clusters containing ins1
    clusters = [
        {
            "cluster_id": "ic:index-funds",
            "canonical_insight": "Index funds beat active managers",
            "member_count": 2,
            "episode_count": 2,
            "cross_episode": True,
            "episode_ids": ["ep1", "ep2"],
            "members": [
                {"insight_id": "ins1", "text": "Index funds beat active managers"},
                {"insight_id": "ins2", "text": "Passive investing wins"},
            ],
        }
    ]
    _write_insight_clusters(tmp_path, clusters)

    args = _make_ns(output_dir=str(tmp_path), topic="investing", format="json")
    rc = run_topic_insights_cli(args, _logger)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["clusters"]) == 1
    assert out["clusters"][0]["cluster_id"] == "ic:index-funds"


def test_topic_insights_no_match(tmp_path: Path, capsys) -> None:
    _write_gi_artifact(
        tmp_path,
        "ep1",
        [
            {
                "insight_id": "ins1",
                "text": "test",
                "topic_ids": ["topic:cooking"],
                "topic_label": "cooking",
            }
        ],
    )
    _write_insight_clusters(
        tmp_path,
        [
            {
                "cluster_id": "ic:test",
                "canonical_insight": "test",
                "member_count": 2,
                "episode_count": 1,
                "cross_episode": False,
                "episode_ids": ["ep1"],
                "members": [{"insight_id": "ins1", "text": "test"}],
            }
        ],
    )
    args = _make_ns(output_dir=str(tmp_path), topic="quantum", format="json")
    rc = run_topic_insights_cli(args, _logger)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["clusters"]) == 0
