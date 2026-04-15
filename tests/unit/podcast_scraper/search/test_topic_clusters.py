"""Unit tests for corpus topic clustering (RFC-075)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from podcast_scraper.search.topic_clusters import (
    build_topic_clusters_payload,
    cluster_indices_by_threshold,
    cosine_similarity_matrix,
    evaluate_validation_against_topics,
    load_topic_cluster_enrichment_map,
    load_validation_yaml,
    pick_centroid_closest_label,
    topic_cluster_enrichment_by_topic_id,
    TOPIC_CLUSTERS_SCHEMA_VERSION,
    topic_id_aliases_from_clusters_payload,
)


def test_cosine_similarity_matrix_identity() -> None:
    v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    s = cosine_similarity_matrix(v)
    assert s.shape == (2, 2)
    assert float(s[0, 1]) == pytest.approx(0.0)
    assert float(s[0, 0]) == pytest.approx(1.0)


def test_cluster_indices_by_threshold_two_groups() -> None:
    # Four unit directions in 4D: (0,1) similar, (2,3) similar, across groups orthogonal
    e0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e1 = np.array([0.99, 0.1414, 0.0, 0.0], dtype=np.float32)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    e3 = np.array([0.0, 0.0, 0.99, 0.1414], dtype=np.float32)
    e3 = e3 / np.linalg.norm(e3)
    mat = np.stack([e0, e1, e2, e3], axis=0)
    sim = cosine_similarity_matrix(mat)
    labels = cluster_indices_by_threshold(sim, 0.9)
    assert int(labels[0]) == int(labels[1])
    assert int(labels[2]) == int(labels[3])
    assert int(labels[0]) != int(labels[2])


def test_pick_centroid_closest_label() -> None:
    mat = np.eye(3, dtype=np.float32)
    # Members 0 and 2; centroid ~ (1/sqrt(2), 0, 1/sqrt(2)) — closest to 0 or 2 equally; pick one
    best = pick_centroid_closest_label([0, 1, 2], mat)
    assert best in (0, 1, 2)


def test_build_topic_clusters_payload_multi_and_singleton() -> None:
    from podcast_scraper.search.topic_clusters import TopicVectorRow

    # Two close + one far
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.99, 0.1414], dtype=np.float32)
    b = b / np.linalg.norm(b)
    c = np.array([0.0, 1.0], dtype=np.float32)
    rows = [
        TopicVectorRow("topic:a", "A", ["ep1"], a),
        TopicVectorRow("topic:b", "B", ["ep1"], b),
        TopicVectorRow("topic:c", "C", ["ep2"], c),
    ]
    payload = build_topic_clusters_payload(rows, threshold=0.9, embedding_model="test-model")
    assert payload["schema_version"] == TOPIC_CLUSTERS_SCHEMA_VERSION
    assert payload["topic_count"] == 3
    assert payload["cluster_count"] == 1
    assert payload["singletons"] == 1
    assert len(payload["clusters"]) == 1
    assert payload["clusters"][0]["member_count"] == 2
    c0 = payload["clusters"][0]
    assert "cil_alias_target_topic_id" in c0
    assert "graph_compound_parent_id" in c0
    assert str(c0["graph_compound_parent_id"]).startswith("tc:")


def test_topic_id_aliases_from_clusters_payload_skips_self_maps() -> None:
    payload = {
        "clusters": [
            {
                "canonical_topic_id": "topic:canon",
                "members": [
                    {"topic_id": "topic:a"},
                    {"topic_id": "topic:canon"},
                ],
            }
        ]
    }
    assert topic_id_aliases_from_clusters_payload(payload) == {"topic:a": "topic:canon"}


def test_topic_cluster_enrichment_by_topic_id_v2() -> None:
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:foo-bar",
                "canonical_label": "Foo Theme",
                "cil_alias_target_topic_id": "topic:canon",
                "members": [
                    {"topic_id": "topic:a"},
                    {"topic_id": "topic:b"},
                ],
            }
        ]
    }
    m = topic_cluster_enrichment_by_topic_id(payload)
    assert m["topic:a"]["graph_compound_parent_id"] == "tc:foo-bar"
    assert m["topic:a"]["canonical_label"] == "Foo Theme"
    assert m["topic:a"]["cil_alias_target_topic_id"] == "topic:canon"
    assert m["topic:b"]["graph_compound_parent_id"] == "tc:foo-bar"


def test_topic_cluster_enrichment_later_cluster_wins_duplicate_topic_id() -> None:
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:first",
                "canonical_label": "First",
                "members": [{"topic_id": "topic:x"}],
            },
            {
                "graph_compound_parent_id": "tc:second",
                "canonical_label": "Second",
                "members": [{"topic_id": "topic:x"}],
            },
        ]
    }
    m = topic_cluster_enrichment_by_topic_id(payload)
    assert m["topic:x"]["graph_compound_parent_id"] == "tc:second"


def test_load_topic_cluster_enrichment_map_missing_dir(tmp_path: Path) -> None:
    assert load_topic_cluster_enrichment_map(tmp_path) == {}


def test_load_topic_cluster_enrichment_map_reads_search_json(tmp_path: Path) -> None:
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / "topic_clusters.json").write_text(
        '{"clusters": [{"graph_compound_parent_id": "tc:z", "canonical_label": "Z", '
        '"members": [{"topic_id": "topic:leaf"}]}]}\n',
        encoding="utf-8",
    )
    m = load_topic_cluster_enrichment_map(tmp_path)
    assert m["topic:leaf"]["canonical_label"] == "Z"


def test_topic_id_aliases_from_clusters_payload_v2_keys() -> None:
    payload = {
        "clusters": [
            {
                "cil_alias_target_topic_id": "topic:canon",
                "members": [
                    {"topic_id": "topic:a"},
                    {"topic_id": "topic:canon"},
                ],
            }
        ]
    }
    assert topic_id_aliases_from_clusters_payload(payload) == {"topic:a": "topic:canon"}


def test_topic_id_aliases_from_clusters_payload_bad_cluster_shape() -> None:
    assert topic_id_aliases_from_clusters_payload({}) == {}
    assert topic_id_aliases_from_clusters_payload({"clusters": None}) == {}


def test_evaluate_validation_against_topics_pass() -> None:
    spec = {
        "expected_merge_pairs": [{"id": "m1", "topic_ids": ["topic:a", "topic:b"]}],
        "expected_distinct": [{"id": "d1", "topic_ids": ["topic:a", "topic:c"]}],
    }
    ids = ["topic:a", "topic:b", "topic:c"]
    # a,b cluster 0; c cluster 1
    labels = [0, 0, 1]
    ok, errors = evaluate_validation_against_topics(spec, ids, labels)
    assert ok
    assert errors == []


def test_evaluate_validation_against_topics_fail_same_cluster() -> None:
    spec = {
        "expected_distinct": [{"id": "d1", "topic_ids": ["topic:x", "topic:y"]}],
    }
    ids = ["topic:x", "topic:y"]
    labels = [0, 0]
    ok, errors = evaluate_validation_against_topics(spec, ids, labels)
    assert not ok
    assert errors


def test_evaluate_validation_missing_topic_reports_error() -> None:
    spec = {
        "expected_merge_pairs": [{"id": "m1", "topic_ids": ["topic:missing", "topic:a"]}],
    }
    ids = ["topic:a"]
    labels = [0]
    ok, errors = evaluate_validation_against_topics(spec, ids, labels)
    assert not ok
    assert any("missing" in e for e in errors)


def test_evaluate_merge_pair_mismatch_reports_error() -> None:
    spec = {
        "expected_merge_pairs": [{"id": "m1", "topic_ids": ["topic:x", "topic:y"]}],
    }
    ids = ["topic:x", "topic:y"]
    labels = [0, 1]
    ok, errors = evaluate_validation_against_topics(spec, ids, labels)
    assert not ok
    assert any("share a cluster" in e for e in errors)


def test_load_validation_yaml_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "v.yaml"
    p.write_text(
        "schema_version: '1'\nexpected_clusters: []\nexpected_distinct: []\n",
        encoding="utf-8",
    )
    data = load_validation_yaml(p)
    assert data["schema_version"] == "1"


def _topic_clusters_validation_fixture_path() -> Path:
    """RFC-075 example YAML under tests/fixtures (not config/ — no long-lived validation file)."""
    tests_root = Path(__file__).resolve().parents[3]
    return tests_root / "fixtures" / "search" / "topic_clusters_validation.example.yaml"


def test_fixture_topic_clusters_validation_yaml_loads() -> None:
    cfg_path = _topic_clusters_validation_fixture_path()
    assert cfg_path.is_file()
    data = load_validation_yaml(cfg_path)
    assert data.get("schema_version") == "1"
    assert len(data.get("expected_merge_pairs") or []) >= 1


def test_fixture_topic_clusters_validation_yaml_inline_episode_sources_shape() -> None:
    """Per-row episode_sources maps each topic_id in that row to a list of episode_id strings."""
    data = load_validation_yaml(_topic_clusters_validation_fixture_path())
    for key in ("expected_merge_pairs", "expected_distinct"):
        for g in data.get(key) or []:
            if not isinstance(g, dict):
                continue
            row_id = g.get("id", "?")
            tids = [
                t.strip() for t in (g.get("topic_ids") or []) if isinstance(t, str) and t.strip()
            ]
            eps = g.get("episode_sources")
            assert eps is not None, f"{key}[{row_id}]: episode_sources required (use [] per topic)"
            assert isinstance(eps, dict), f"{key}[{row_id}]: episode_sources must be a mapping"
            assert set(eps.keys()) == set(tids), (
                f"{key}[{row_id}]: episode_sources keys must match topic_ids; "
                f"got {sorted(eps.keys())} want {sorted(tids)}"
            )
            for tid, ep_list in eps.items():
                assert isinstance(ep_list, list), f"{key}[{row_id}] episode_sources[{tid!r}]"
                for e in ep_list:
                    assert isinstance(e, str), f"{key}[{row_id}] episode_sources[{tid!r}]"
