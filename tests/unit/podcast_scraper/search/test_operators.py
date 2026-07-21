"""Unit tests for search/operators.py — Search v3 §S4b.

Both operators run Python-side AFTER ``rrf_fuse``; these tests exercise the
pure post-processing (mock filesystem for consensus, in-memory hit dicts for
cluster). No LanceDB touch; no network. ``make lint-search-v3`` covers the
module's forbidden-import surface at the repo level.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.operators import (
    cluster_hits,
    consensus_pairs_for_hits,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def empty_corpus(tmp_path: Path) -> Path:
    """A corpus root with no enrichments — validates the graceful-degrade paths."""
    (tmp_path / "enrichments").mkdir()
    return tmp_path


def _make_hit(**meta) -> dict:
    return {"doc_id": meta.pop("doc_id", "d:x"), "metadata": meta}


# --------------------------------------------------------------------------
# cluster_hits
# --------------------------------------------------------------------------


class TestClusterHits:
    def test_empty_hits_returns_empty_list(self, empty_corpus: Path) -> None:
        assert cluster_hits([], empty_corpus) == []

    def test_kg_topic_hit_with_topic_cluster_metadata_groups_by_compound_id(
        self, empty_corpus: Path
    ) -> None:
        hits = [
            _make_hit(
                doc_type="kg_topic",
                source_id="topic:climate",
                topic_cluster={
                    "topic_cluster_compound_id": "tc:environment",
                    "label": "Environment",
                },
            ),
            _make_hit(
                doc_type="kg_topic",
                source_id="topic:policy",
                topic_cluster={
                    "topic_cluster_compound_id": "tc:environment",
                    "label": "Environment",
                },
            ),
        ]
        groups = cluster_hits(hits, empty_corpus)
        assert len(groups) == 1
        (g,) = groups
        assert g["cluster_id"] == "tc:environment"
        assert g["cluster_kind"] == "topic_cluster"
        assert g["label"] == "Environment"
        assert g["size"] == 2
        assert g["hit_indices"] == [0, 1]

    def test_hits_with_no_cluster_signal_land_in_ungrouped_bucket(self, empty_corpus: Path) -> None:
        hits = [
            _make_hit(doc_type="transcript", episode_id="ep-a"),
            _make_hit(doc_type="summary", episode_id="ep-b"),
        ]
        groups = cluster_hits(hits, empty_corpus)
        assert len(groups) == 1
        assert groups[0]["cluster_id"] is None
        assert groups[0]["cluster_kind"] == "ungrouped"
        assert groups[0]["size"] == 2
        assert groups[0]["hit_indices"] == [0, 1]

    def test_insight_hits_with_shared_about_topic_id_group_via_topic_fallback(
        self, empty_corpus: Path
    ) -> None:
        # No topic_cluster on hit metadata, no theme_clusters.json — fallback to
        # bare-topic groups. Two insights ABOUT the same topic collapse into one
        # single-topic group; a third insight ABOUT a different topic makes its own.
        hits = [
            _make_hit(doc_type="insight", about_topic_id="topic:llm", topic_label="LLMs"),
            _make_hit(doc_type="insight", about_topic_id="topic:llm"),
            _make_hit(doc_type="insight", about_topic_id="topic:bio"),
        ]
        groups = cluster_hits(hits, empty_corpus)
        # Ordered by size desc: topic:llm (2), topic:bio (1).
        assert [g["cluster_id"] for g in groups] == ["topic:llm", "topic:bio"]
        assert groups[0]["cluster_kind"] == "topic"
        # Label promoted from the first hit that carried topic_label.
        assert groups[0]["label"] == "LLMs"
        assert groups[0]["hit_indices"] == [0, 1]
        assert groups[1]["hit_indices"] == [2]

    def test_theme_cluster_map_takes_priority_over_bare_topic_fallback(
        self, tmp_path: Path
    ) -> None:
        # Wire a theme_clusters.json so topic:llm resolves to a theme cluster.
        enrich = tmp_path / "enrichments"
        enrich.mkdir()
        (enrich / "topic_theme_clusters.json").write_text(
            json.dumps(
                {
                    "clusters": [
                        {
                            "graph_compound_parent_id": "thc:ai",
                            "canonical_label": "Artificial intelligence",
                            "members": [{"topic_id": "topic:llm"}],
                        }
                    ]
                }
            )
        )
        hits = [
            _make_hit(doc_type="insight", about_topic_id="topic:llm"),
            _make_hit(doc_type="insight", about_topic_id="topic:llm"),
        ]
        groups = cluster_hits(hits, tmp_path)
        assert len(groups) == 1
        assert groups[0]["cluster_id"] == "thc:ai"
        assert groups[0]["cluster_kind"] == "theme_cluster"
        assert groups[0]["label"] == "Artificial intelligence"

    def test_groups_are_ordered_by_descending_size_then_ungrouped_last(
        self, empty_corpus: Path
    ) -> None:
        hits = [
            _make_hit(doc_type="insight", about_topic_id="topic:a"),
            _make_hit(doc_type="insight", about_topic_id="topic:a"),
            _make_hit(doc_type="insight", about_topic_id="topic:a"),
            _make_hit(doc_type="insight", about_topic_id="topic:b"),
            _make_hit(doc_type="insight", about_topic_id="topic:b"),
            _make_hit(doc_type="insight"),  # ungrouped
        ]
        groups = cluster_hits(hits, empty_corpus)
        assert [g["cluster_id"] for g in groups] == ["topic:a", "topic:b", None]
        assert [g["size"] for g in groups] == [3, 2, 1]


# --------------------------------------------------------------------------
# consensus_pairs_for_hits
# --------------------------------------------------------------------------


def _write_consensus(corpus_root: Path, pairs: list[dict]) -> None:
    enrich = corpus_root / "enrichments"
    enrich.mkdir(exist_ok=True)
    (enrich / "topic_consensus.json").write_text(
        json.dumps(
            {
                "derived": True,
                "enricher_id": "topic_consensus",
                "schema_version": "1.0",
                "data": {"consensus": pairs},
            }
        )
    )


class TestConsensusPairs:
    def test_missing_enrichment_file_returns_empty_list(self, empty_corpus: Path) -> None:
        # No topic_consensus.json under enrichments/.
        assert consensus_pairs_for_hits([_make_hit(doc_type="insight")], empty_corpus) == []

    def test_malformed_json_returns_empty_list_without_raising(self, tmp_path: Path) -> None:
        (tmp_path / "enrichments").mkdir()
        (tmp_path / "enrichments" / "topic_consensus.json").write_text("{not json")
        assert consensus_pairs_for_hits([], tmp_path) == []

    def test_pairs_filtered_to_topics_surfaced_by_kg_topic_hits(self, tmp_path: Path) -> None:
        _write_consensus(
            tmp_path,
            [
                {
                    "topic_id": "topic:climate",
                    "person_a_id": "person:alice",
                    "person_b_id": "person:bob",
                    "insight_a_id": "insight:a1",
                    "insight_b_id": "insight:b1",
                    "insight_a_text": "Alice",
                    "insight_b_text": "Bob",
                    "contradiction_score": 0.1,
                    "cosine_similarity": 0.82,
                },
                {
                    # Different topic — should be filtered out when hits only
                    # surface topic:climate.
                    "topic_id": "topic:markets",
                    "person_a_id": "person:carol",
                    "person_b_id": "person:dan",
                    "insight_a_id": "insight:a2",
                    "insight_b_id": "insight:b2",
                    "contradiction_score": 0.2,
                },
            ],
        )
        hits = [_make_hit(doc_type="kg_topic", source_id="topic:climate")]
        pairs = consensus_pairs_for_hits(hits, tmp_path)
        assert len(pairs) == 1
        assert pairs[0]["topic_id"] == "topic:climate"
        assert pairs[0]["person_a_id"] == "person:alice"
        assert pairs[0]["cosine_similarity"] == pytest.approx(0.82)

    def test_falls_back_to_referenced_topic_ids_when_no_kg_topic_hits(self, tmp_path: Path) -> None:
        _write_consensus(
            tmp_path,
            [
                {
                    "topic_id": "topic:bio",
                    "person_a_id": "person:alice",
                    "person_b_id": "person:bob",
                    "insight_a_id": "insight:a",
                    "insight_b_id": "insight:b",
                    "contradiction_score": 0.05,
                }
            ],
        )
        hits = [
            _make_hit(doc_type="insight", about_topic_id="topic:bio"),
            _make_hit(doc_type="insight", topic_ids=["topic:bio", "topic:other"]),
        ]
        pairs = consensus_pairs_for_hits(hits, tmp_path)
        assert len(pairs) == 1
        assert pairs[0]["topic_id"] == "topic:bio"

    def test_no_relevant_topics_degrades_to_strongest_pairs_first(self, tmp_path: Path) -> None:
        # Hits carry no topic signal at all — the operator degrades to a
        # "strongest N by (low contradiction, high cosine)" fallback so the
        # user still sees the corpus-wide top pairs instead of an empty state.
        _write_consensus(
            tmp_path,
            [
                {
                    "topic_id": "topic:x",
                    "person_a_id": "p:a",
                    "person_b_id": "p:b",
                    "insight_a_id": "i:1",
                    "insight_b_id": "i:2",
                    "contradiction_score": 0.4,
                    "cosine_similarity": 0.71,
                },
                {
                    "topic_id": "topic:y",
                    "person_a_id": "p:c",
                    "person_b_id": "p:d",
                    "insight_a_id": "i:3",
                    "insight_b_id": "i:4",
                    "contradiction_score": 0.1,  # strongest — should win first
                    "cosine_similarity": 0.9,
                },
            ],
        )
        pairs = consensus_pairs_for_hits(
            [_make_hit(doc_type="transcript")],
            tmp_path,
        )
        assert [p["topic_id"] for p in pairs] == ["topic:y", "topic:x"]

    def test_max_pairs_caps_output(self, tmp_path: Path) -> None:
        _write_consensus(
            tmp_path,
            [
                {
                    "topic_id": "topic:x",
                    "person_a_id": f"p:{i}",
                    "person_b_id": f"p:{i}b",
                    "insight_a_id": f"i:{i}",
                    "insight_b_id": f"i:{i}b",
                    "contradiction_score": 0.1,
                }
                for i in range(10)
            ],
        )
        hits = [_make_hit(doc_type="kg_topic", source_id="topic:x")]
        pairs = consensus_pairs_for_hits(hits, tmp_path, max_pairs=3)
        assert len(pairs) == 3

    def test_missing_optional_fields_downgrade_to_safe_defaults(self, tmp_path: Path) -> None:
        _write_consensus(
            tmp_path,
            [
                {
                    "topic_id": "topic:x",
                    "person_a_id": "p:a",
                    "person_b_id": "p:b",
                    "insight_a_id": "i:1",
                    "insight_b_id": "i:2",
                    # No contradiction_score → 0.0; no cosine → None; no labels → None.
                }
            ],
        )
        hits = [_make_hit(doc_type="kg_topic", source_id="topic:x")]
        pairs = consensus_pairs_for_hits(hits, tmp_path)
        assert len(pairs) == 1
        p = pairs[0]
        assert p["contradiction_score"] == 0.0
        assert p["cosine_similarity"] is None
        assert p["topic_label"] is None
        assert p["insight_a_text"] == ""
