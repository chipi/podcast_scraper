"""Unit tests for ``corpus_search`` (dedupe, filters, mocked LanceDB hybrid path)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.search.corpus_search import dedupe_kg_surface_rows, run_corpus_search
from podcast_scraper.search.protocol import SearchResult

pytestmark = [pytest.mark.unit]

_HYBRID = "podcast_scraper.search.corpus_search.hybrid_candidates"


def test_run_corpus_search_empty_query(tmp_path: Path) -> None:
    out = run_corpus_search(tmp_path, "   ")
    assert out.error == "empty_query"


def test_run_corpus_search_no_index(tmp_path: Path) -> None:
    # No LanceDB index on disk → hybrid_candidates returns None → no_index.
    out = run_corpus_search(tmp_path, "climate")
    assert out.error == "no_index"


def test_dedupe_kg_surface_rows_merges_same_surface_text() -> None:
    rows = [
        {
            "doc_id": "a",
            "score": 0.9,
            "metadata": {"doc_type": "kg_entity", "episode_id": "ep1"},
            "text": "Acme Corp",
        },
        {
            "doc_id": "b",
            "score": 0.7,
            "metadata": {"doc_type": "kg_entity", "episode_id": "ep2"},
            "text": "acme  corp",
        },
        {"doc_id": "c", "score": 0.5, "metadata": {"doc_type": "insight"}, "text": "x"},
    ]
    out = dedupe_kg_surface_rows(rows)
    assert len(out) == 2
    assert out[0]["doc_id"] == "a"
    wmeta = out[0]["metadata"]
    assert wmeta.get("kg_surface_match_count") == 2
    assert "ep1" in wmeta.get("kg_surface_episode_ids", [])
    assert "ep2" in wmeta.get("kg_surface_episode_ids", [])


def test_dedupe_kg_surface_rows_non_dict_metadata_still_keeps_row() -> None:
    rows = [
        {"doc_id": "x", "score": 1.0, "metadata": "bad", "text": "t"},
    ]
    out = dedupe_kg_surface_rows(rows)
    assert len(out) == 1


@patch(_HYBRID)
def test_run_corpus_search_success_mocked(mock_hybrid: object, tmp_path: Path) -> None:
    mock_hybrid.return_value = [
        SearchResult(
            doc_id="d1",
            score=0.99,
            metadata={"doc_type": "insight", "episode_id": "e1", "text": "hello"},
        ),
    ]
    out = run_corpus_search(tmp_path, "q", top_k=5, dedupe_kg_surfaces=False)
    assert out.error is None
    assert len(out.results) == 1
    assert out.results[0]["doc_id"] == "d1"
    assert out.lift_stats is not None
    assert "transcript_hits_returned" in out.lift_stats


@patch(_HYBRID)
def test_run_corpus_search_attaches_topic_cluster_metadata(
    mock_hybrid: object, tmp_path: Path
) -> None:
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / "topic_clusters.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "graph_compound_parent_id": "tc:cam",
                        "canonical_label": "Cam Label",
                        "members": [{"topic_id": "topic:leaf1"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    mock_hybrid.return_value = [
        SearchResult(
            doc_id="d1",
            score=0.99,
            metadata={
                "doc_type": "kg_topic",
                "episode_id": "e1",
                "source_id": "topic:leaf1",
                "text": "surface",
            },
        ),
    ]
    out = run_corpus_search(tmp_path, "q", top_k=5, dedupe_kg_surfaces=False)
    assert out.error is None
    assert len(out.results) == 1
    tc = out.results[0]["metadata"].get("topic_cluster")
    assert isinstance(tc, dict)
    assert tc.get("graph_compound_parent_id") == "tc:cam"
    assert tc.get("canonical_label") == "Cam Label"


@patch(_HYBRID)
def test_run_corpus_search_no_index_when_hybrid_none(mock_hybrid: object, tmp_path: Path) -> None:
    # hybrid_candidates returns None for a missing/unusable index → no_index (no FAISS fallback).
    mock_hybrid.return_value = None
    out = run_corpus_search(tmp_path, "q")
    assert out.error == "no_index"


@patch(_HYBRID)
def test_run_corpus_search_embed_failed_when_query_embedding_error(
    mock_hybrid: object, tmp_path: Path
) -> None:
    # A query-embedding failure (model missing/offline) is reported as embed_failed, NOT
    # no_index — re-indexing wouldn't help. Regression guard for the LanceDB-migration
    # taxonomy collapse (the index is fine; the query couldn't be embedded).
    from podcast_scraper.search.hybrid_search import QueryEmbeddingError

    mock_hybrid.side_effect = QueryEmbeddingError("model offline")
    out = run_corpus_search(tmp_path, "q")
    assert out.error == "embed_failed"
    assert "offline" in (out.detail or "")


@patch(_HYBRID)
def test_run_corpus_search_multi_doc_type_filter(mock_hybrid: object, tmp_path: Path) -> None:
    mock_hybrid.return_value = [
        SearchResult("a", 0.9, {"doc_type": "insight", "episode_id": "e"}),
        SearchResult("b", 0.8, {"doc_type": "quote", "episode_id": "e"}),
        SearchResult("c", 0.7, {"doc_type": "summary", "episode_id": "e"}),
    ]
    out = run_corpus_search(
        tmp_path,
        "q",
        doc_types=["insight", "quote"],
        top_k=5,
        dedupe_kg_surfaces=False,
    )
    assert out.error is None
    dts = {r["metadata"]["doc_type"] for r in out.results}
    assert dts <= {"insight", "quote"}
    assert "summary" not in dts


# --- storyline hits: the query-time join (#2114 / review 2026-09-17) ----------------------------


def _write_theme_artifact(root: Path, clusters: list[dict]) -> None:
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    (root / "enrichments" / "topic_theme_clusters.json").write_text(
        json.dumps({"data": {"clusters": clusters}}), encoding="utf-8"
    )


def _storyline_hit(source_id: str) -> SearchResult:
    """A storyline hit exactly as the index returns one — `source_id` and nothing else.

    This is the shape that matters: the aux schema has no label/size/anchor columns and
    ``hybrid_search._to_search_result`` rebuilds metadata from a fixed field list, so whatever the
    indexer wrote into the row's metadata dict never reaches a caller.
    """
    return SearchResult(
        doc_id=f"storyline:{source_id}",
        score=0.99,
        metadata={
            "doc_type": "storyline",
            "episode_id": None,
            "source_id": source_id,
            "text": "Managing risk across domains risk management",
        },
    )


def test_storyline_hit_is_joined_to_a_usable_label_and_anchor(tmp_path: Path) -> None:
    """A storyline hit must come back with a LABEL and a ROUTABLE id.

    Without this join the client had only `source_id`, so it rendered the raw ``thc:`` slug as the
    title and linked to a page that 404s — there is no storyline endpoint, the anchor topic's card
    IS the storyline. The index cannot supply either field, so the read path must.
    """
    _write_theme_artifact(
        tmp_path,
        [
            {
                "graph_compound_parent_id": "thc:managing-risk",
                "canonical_label": "Managing risk across domains",
                "member_count": 4,
                "members": [
                    {"topic_id": "topic:risk-management", "label": "risk management"},
                    {"topic_id": "topic:systems", "label": "systems thinking"},
                ],
            }
        ],
    )
    with patch(_HYBRID) as hybrid:
        hybrid.return_value = [_storyline_hit("thc:managing-risk")]
        out = run_corpus_search(tmp_path, "risk", top_k=5, dedupe_kg_surfaces=False)
    assert out.error is None
    assert len(out.results) == 1
    meta = out.results[0]["metadata"]
    assert meta["storyline_label"] == "Managing risk across domains"
    assert meta["storyline_size"] == 4
    assert meta["anchor_topic_id"] == "topic:risk-management"
    assert not str(meta["anchor_topic_id"]).startswith("thc:"), "must be routable, not a thc: id"


def test_storyline_hit_for_a_vanished_cluster_is_dropped(tmp_path: Path) -> None:
    """An orphaned row is DROPPED, not shown.

    ``thc:`` ids are label-derived, so relabelling or re-anchoring a cluster mints a new id and
    leaves the old row in the index until a build prunes it. A user must never be offered a
    storyline that no longer exists.
    """
    _write_theme_artifact(
        tmp_path,
        [
            {
                "graph_compound_parent_id": "thc:still-here",
                "canonical_label": "Still here",
                "member_count": 4,
                "members": [{"topic_id": "topic:a", "label": "a"}],
            }
        ],
    )
    with patch(_HYBRID) as hybrid:
        hybrid.return_value = [_storyline_hit("thc:long-gone")]
        out = run_corpus_search(tmp_path, "risk", top_k=5, dedupe_kg_surfaces=False)
    assert out.error is None
    assert out.results == [], "an orphaned storyline row was served to the client"


def test_storyline_join_leaves_other_doc_types_alone(tmp_path: Path) -> None:
    """A passage hit passes through untouched, and is never dropped by the storyline join."""
    _write_theme_artifact(tmp_path, [])
    with patch(_HYBRID) as hybrid:
        hybrid.return_value = [
            SearchResult(
                doc_id="d1",
                score=0.9,
                metadata={"doc_type": "insight", "episode_id": "e1", "text": "hello"},
            )
        ]
        out = run_corpus_search(tmp_path, "q", top_k=5, dedupe_kg_surfaces=False)
    assert [r["doc_id"] for r in out.results] == ["d1"]
