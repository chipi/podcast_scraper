"""Unit tests for insight cluster context expansion (#601)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.search.insight_cluster_context import (
    expand_with_cluster_context,
    format_cluster_context,
    load_insight_clusters,
)

# ── helpers ──────────────────────────────────────────────────────────


def _clusters_payload() -> dict:
    """Minimal insight_clusters.json payload with 1 cross-episode cluster."""
    return {
        "schema_version": "1",
        "clusters": [
            {
                "cluster_id": "ic:index-funds-beat-active",
                "canonical_insight": "Index funds consistently outperform active managers",
                "member_count": 2,
                "episode_count": 2,
                "cross_episode": True,
                "members": [
                    {
                        "insight_id": "ins1",
                        "text": "Index funds beat active managers",
                        "episode_id": "ep1",
                        "supporting_quotes": [
                            {"text": "92% of active managers underperform", "speaker_id": "s1"}
                        ],
                    },
                    {
                        "insight_id": "ins2",
                        "text": "Index funds outperform actively managed funds",
                        "episode_id": "ep2",
                        "supporting_quotes": [
                            {"text": "the data clearly shows passive wins", "speaker_id": "s2"}
                        ],
                    },
                ],
            },
            {
                "cluster_id": "ic:singleton-cluster",
                "canonical_insight": "AI will transform healthcare",
                "member_count": 2,
                "episode_count": 1,
                "cross_episode": False,
                "members": [
                    {
                        "insight_id": "ins3",
                        "text": "AI will transform healthcare",
                        "episode_id": "ep3",
                        "supporting_quotes": [],
                    },
                    {
                        "insight_id": "ins4",
                        "text": "AI is transforming healthcare delivery",
                        "episode_id": "ep3",
                        "supporting_quotes": [],
                    },
                ],
            },
        ],
    }


# ── load_insight_clusters ────────────────────────────────────────────


def test_load_nonexistent_path(tmp_path: Path) -> None:
    result = load_insight_clusters(tmp_path / "missing.json")
    assert result == {}


def test_load_indexes_by_insight_id(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps(_clusters_payload()), encoding="utf-8")
    index = load_insight_clusters(path)
    assert "ins1" in index
    assert "ins2" in index
    assert index["ins1"]["cluster_id"] == "ic:index-funds-beat-active"
    assert index["ins1"]["cross_episode"] is True
    assert index["ins1"]["member_count"] == 2


# ── expand_with_cluster_context ──────────────────────────────────────


def test_expand_no_clusters_returns_unchanged() -> None:
    insights = [{"insight_id": "ins1", "text": "test"}]
    result = expand_with_cluster_context(insights)
    assert result == insights


def test_expand_with_empty_index() -> None:
    insights = [{"insight_id": "ins1", "text": "test"}]
    result = expand_with_cluster_context(insights, cluster_index={})
    assert result == insights


def test_expand_adds_cluster_context(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps(_clusters_payload()), encoding="utf-8")

    insights = [
        {"insight_id": "ins1", "text": "Index funds beat active managers", "episode_id": "ep1"}
    ]
    result = expand_with_cluster_context(insights, clusters_path=path)
    assert len(result) == 1
    assert "cluster" in result[0]
    cluster = result[0]["cluster"]
    assert cluster["cluster_id"] == "ic:index-funds-beat-active"
    assert cluster["cluster_size"] == 2
    assert cluster["cluster_episodes"] == 2


def test_expand_cross_episode_quotes_exclude_same_episode(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps(_clusters_payload()), encoding="utf-8")

    insights = [
        {"insight_id": "ins1", "text": "Index funds beat active managers", "episode_id": "ep1"}
    ]
    result = expand_with_cluster_context(insights, clusters_path=path)
    cross_quotes = result[0]["cluster"]["cross_episode_quotes"]
    # Should only have quotes from ep2, not ep1
    for cq in cross_quotes:
        assert cq["episode_id"] != "ep1"
    assert len(cross_quotes) >= 1
    assert cross_quotes[0]["episode_id"] == "ep2"


def test_expand_non_cross_episode_no_cluster_added(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps(_clusters_payload()), encoding="utf-8")

    # ins3 is in a non-cross-episode cluster
    insights = [{"insight_id": "ins3", "text": "AI will transform healthcare", "episode_id": "ep3"}]
    result = expand_with_cluster_context(insights, clusters_path=path)
    assert "cluster" not in result[0]


def test_expand_unknown_insight_passes_through(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps(_clusters_payload()), encoding="utf-8")

    insights = [{"insight_id": "unknown", "text": "Not in any cluster"}]
    result = expand_with_cluster_context(insights, clusters_path=path)
    assert "cluster" not in result[0]


def test_expand_uses_preloaded_index() -> None:
    # Build index manually
    cluster_index = {
        "ins1": {
            "cluster_id": "ic:test",
            "canonical_insight": "Test canonical",
            "member_count": 3,
            "episode_count": 2,
            "cross_episode": True,
            "all_members": [
                {
                    "insight_id": "ins1",
                    "text": "A",
                    "episode_id": "ep1",
                    "supporting_quotes": [{"text": "quote A"}],
                },
                {
                    "insight_id": "ins2",
                    "text": "B",
                    "episode_id": "ep2",
                    "supporting_quotes": [{"text": "quote B"}],
                },
            ],
        }
    }
    insights = [{"insight_id": "ins1", "text": "A", "episode_id": "ep1"}]
    result = expand_with_cluster_context(insights, cluster_index=cluster_index)
    assert result[0]["cluster"]["cluster_id"] == "ic:test"


# ── format_cluster_context ───────────────────────────────────────────


def test_format_no_cluster() -> None:
    assert format_cluster_context({"text": "test"}) == ""


def test_format_with_cluster() -> None:
    insight = {
        "text": "test",
        "cluster": {
            "cluster_id": "ic:test",
            "canonical_insight": "Test canonical insight text here",
            "cluster_size": 3,
            "cluster_episodes": 2,
            "cross_episode_quotes": [
                {"text": "a quote from another episode", "episode_id": "ep2", "speaker_id": "s1"}
            ],
        },
    }
    output = format_cluster_context(insight)
    assert "ic:test" in output
    assert "3 insights across 2 episodes" in output
    assert "Cross-episode evidence" in output
    assert "a quote from another episode" in output


def test_format_truncates_long_quote_list() -> None:
    quotes = [{"text": f"quote {i}", "episode_id": f"ep{i}", "speaker_id": None} for i in range(10)]
    insight = {
        "cluster": {
            "cluster_id": "ic:test",
            "canonical_insight": "Test",
            "cluster_size": 10,
            "cluster_episodes": 10,
            "cross_episode_quotes": quotes,
        },
    }
    output = format_cluster_context(insight)
    assert "+7 more" in output
