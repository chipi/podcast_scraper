"""Unit tests for CIL digest topic pills (bridge + topic clusters)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.server.cil_digest_topics import (
    build_cil_digest_topics_for_row,
    CIL_TOPIC_PILL_CAP,
    load_topic_cluster_index,
    row_has_cluster_topic,
    row_matches_library_topic_cluster_filter,
)
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow


def _row(
    *,
    bridge_rel: str = "ep.bridge.json",
    has_bridge: bool = True,
    episode_id: str | None = "e1",
) -> CatalogEpisodeRow:
    return CatalogEpisodeRow(
        metadata_relative_path="ep.metadata.json",
        feed_id="f",
        feed_title=None,
        episode_id=episode_id,
        episode_title="T",
        publish_date="2024-06-01",
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path="ep.gi.json",
        kg_relative_path="ep.kg.json",
        bridge_relative_path=bridge_rel,
        has_gi=True,
        has_kg=True,
        has_bridge=has_bridge,
    )


def test_load_topic_cluster_index_missing_file(tmp_path: Path) -> None:
    idx = load_topic_cluster_index(tmp_path)
    assert idx.cluster_for_topic("topic:any", "e1") == (False, None)


def test_cluster_for_topic_respects_episode_ids(tmp_path: Path) -> None:
    search = tmp_path / "search"
    search.mkdir()
    payload = {
        "schema_version": "2",
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ab",
                "canonical_label": "AB",
                "member_count": 2,
                "members": [
                    {"topic_id": "topic:a", "episode_ids": ["e1"]},
                    {"topic_id": "topic:b", "episode_ids": ["e2"]},
                ],
            }
        ],
    }
    (search / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")
    idx = load_topic_cluster_index(tmp_path)
    assert idx.cluster_for_topic("topic:a", "e1")[0] is True
    assert idx.cluster_for_topic("topic:a", "e2")[0] is False
    assert idx.cluster_for_topic("topic:b", "e1")[0] is False
    assert idx.cluster_for_topic("topic:b", "e2")[0] is True


def test_build_orders_cluster_topics_first(tmp_path: Path) -> None:
    bridge = {
        "identities": [
            {"id": "topic:zebra", "display_name": "Zebra"},
            {"id": "topic:apple", "display_name": "Apple"},
            {"id": "topic:alone", "display_name": "Alone"},
        ]
    }
    (tmp_path / "ep.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    search = tmp_path / "search"
    search.mkdir()
    clusters = {
        "schema_version": "2",
        "clusters": [
            {
                "graph_compound_parent_id": "tc:x",
                "canonical_label": "XA",
                "member_count": 2,
                "members": [
                    {"topic_id": "topic:apple"},
                    {"topic_id": "topic:other"},
                ],
            }
        ],
    }
    (search / "topic_clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    idx = load_topic_cluster_index(tmp_path)
    row = _row()
    pills = build_cil_digest_topics_for_row(tmp_path, row, idx)
    labels = [p.label for p in pills]
    assert labels[0] == "Apple"
    assert "Zebra" in labels
    assert "Alone" in labels
    assert labels.index("Apple") < labels.index("Zebra")
    assert labels.index("Apple") < labels.index("Alone")
    assert any(p.in_topic_cluster and p.topic_id == "topic:apple" for p in pills)
    assert not any(p.in_topic_cluster and p.topic_id == "topic:alone" for p in pills)


def test_library_topic_cluster_filter_ignores_global_member_rows(tmp_path: Path) -> None:
    """Library list filter does not match corpus-global cluster rows (no episode_ids)."""
    (tmp_path / "ep.bridge.json").write_text(
        json.dumps(
            {
                "identities": [
                    {"id": "topic:x", "display_name": "X"},
                ]
            }
        ),
        encoding="utf-8",
    )
    search = tmp_path / "search"
    search.mkdir()
    (search / "topic_clusters.json").write_text(
        json.dumps(
            {
                "schema_version": "2",
                "clusters": [
                    {
                        "graph_compound_parent_id": "tc:x",
                        "member_count": 2,
                        "members": [{"topic_id": "topic:x"}, {"topic_id": "topic:y"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    idx = load_topic_cluster_index(tmp_path)
    row = _row(episode_id="e1")
    assert row_matches_library_topic_cluster_filter(tmp_path, row, idx) is False
    pills = build_cil_digest_topics_for_row(tmp_path, row, idx)
    assert row_has_cluster_topic(pills) is True


def test_library_topic_cluster_filter_true_when_episode_on_member(tmp_path: Path) -> None:
    (tmp_path / "ep.bridge.json").write_text(
        json.dumps({"identities": [{"id": "topic:x", "display_name": "X"}]}),
        encoding="utf-8",
    )
    search = tmp_path / "search"
    search.mkdir()
    (search / "topic_clusters.json").write_text(
        json.dumps(
            {
                "schema_version": "2",
                "clusters": [
                    {
                        "graph_compound_parent_id": "tc:x",
                        "member_count": 2,
                        "members": [
                            {"topic_id": "topic:x", "episode_ids": ["e1"]},
                            {"topic_id": "topic:y", "episode_ids": ["e2"]},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    idx = load_topic_cluster_index(tmp_path)
    row = _row(episode_id="e1")
    assert row_matches_library_topic_cluster_filter(tmp_path, row, idx) is True


def test_row_has_cluster_topic(tmp_path: Path) -> None:
    (tmp_path / "ep.bridge.json").write_text(
        json.dumps(
            {
                "identities": [
                    {"id": "topic:x", "display_name": "X"},
                ]
            }
        ),
        encoding="utf-8",
    )
    search = tmp_path / "search"
    search.mkdir()
    (search / "topic_clusters.json").write_text(
        json.dumps(
            {
                "schema_version": "2",
                "clusters": [
                    {
                        "graph_compound_parent_id": "tc:x",
                        "member_count": 2,
                        "members": [{"topic_id": "topic:x"}, {"topic_id": "topic:y"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    idx = load_topic_cluster_index(tmp_path)
    pills = build_cil_digest_topics_for_row(tmp_path, _row(), idx)
    assert row_has_cluster_topic(pills) is True


def test_cap_respected(tmp_path: Path) -> None:
    identities = [
        {"id": f"topic:t{i}", "display_name": f"T{i}"} for i in range(CIL_TOPIC_PILL_CAP + 3)
    ]
    (tmp_path / "ep.bridge.json").write_text(
        json.dumps({"identities": identities}),
        encoding="utf-8",
    )
    idx = load_topic_cluster_index(tmp_path)
    pills = build_cil_digest_topics_for_row(tmp_path, _row(), idx)
    assert len(pills) == CIL_TOPIC_PILL_CAP
