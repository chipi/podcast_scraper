"""Viewer API: GET /api/search.

Requires ``fastapi`` (``pip install -e '.[dev]'``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, Dict, List

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def test_search_no_corpus_path() -> None:
    app = create_app(None, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/search", params={"q": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["error"] == "no_corpus_path"
    assert body["results"] == []


def test_search_returns_results_when_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        output_dir: Path,
        query: str,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        assert output_dir == tmp_path.resolve()
        assert query == "climate"
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "insight:ep1:n1",
                    "score": 0.91,
                    "metadata": {"doc_type": "insight", "episode_id": "ep1"},
                    "text": "hello",
                }
            ],
            lift_stats={
                "transcript_hits_returned": 0,
                "lift_applied": 0,
            },
        )

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/search",
        params={"q": "climate", "path": str(tmp_path), "top_k": "5"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert len(body["results"]) == 1
    assert body["results"][0]["doc_id"] == "insight:ep1:n1"
    assert body["results"][0]["score"] == pytest.approx(0.91)
    # PRD-033 FR1.1/FR1.4: tier derived from doc_type, intent on the response.
    assert body["results"][0]["source_tier"] == "insight"
    assert body["query_type"] == "semantic"
    assert body.get("lift_stats") == {
        "transcript_hits_returned": 0,
        "lift_applied": 0,
    }


def test_search_type_query_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run(
        output_dir: Path,
        query: str,
        *,
        doc_types: List[str] | None = None,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        captured["doc_types"] = doc_types
        return CorpusSearchOutcome(results=[])

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/search",
        params=[
            ("q", "q"),
            ("path", str(tmp_path)),
            ("type", "insight,quote"),
            ("type", "summary"),
        ],
    )
    assert response.status_code == 200
    assert captured.get("doc_types") == ["insight", "quote", "summary"]


def test_search_dedupe_kg_surfaces_query_param(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: Dict[str, Any] = {}

    def fake_run(
        output_dir: Path,
        query: str,
        *,
        dedupe_kg_surfaces: bool = True,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        captured["dedupe_kg_surfaces"] = dedupe_kg_surfaces
        return CorpusSearchOutcome(results=[])

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r1 = client.get(
        "/api/search",
        params={"q": "q", "path": str(tmp_path)},
    )
    assert r1.status_code == 200
    assert captured.get("dedupe_kg_surfaces") is True

    r2 = client.get(
        "/api/search",
        params={"q": "q", "path": str(tmp_path), "dedupe_kg_surfaces": "false"},
    )
    assert r2.status_code == 200
    assert captured.get("dedupe_kg_surfaces") is False


def test_search_maps_outcome_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(error="no_index", detail="/tmp/x")

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/search",
        params={"q": "hello", "path": str(tmp_path)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"] == "no_index"
    assert body["detail"] == "/tmp/x"
    assert body["results"] == []
    assert body.get("lift_stats") is None


def test_search_tier_and_intent_for_entity_transcript_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "transcript:ep1:c3",
                    "score": 0.8,
                    "metadata": {"doc_type": "transcript", "episode_id": "ep1"},
                    "text": "raw chunk",
                },
                {
                    "doc_id": "kg_entity:person:jane",
                    "score": 0.7,
                    "metadata": {"doc_type": "kg_entity", "source_id": "person:jane"},
                    "text": "Jane Doe",
                },
            ]
        )

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    # "Jane Doe" trips the name regex → entity_lookup intent.
    body = client.get("/api/search", params={"q": "Jane Doe", "path": str(tmp_path)}).json()
    assert body["query_type"] == "entity_lookup"
    tiers = {h["doc_id"]: h["source_tier"] for h in body["results"]}
    assert tiers == {"transcript:ep1:c3": "segment", "kg_entity:person:jane": "aux"}


def test_search_lift_stats_reflects_transcript_and_lift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        output_dir: Path,
        query: str,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "t1",
                    "score": 0.5,
                    "metadata": {"doc_type": "transcript", "episode_id": "e1"},
                    "text": "chunk",
                    "lifted": {"insight": {"id": "i1"}},
                },
                {
                    "doc_id": "t2",
                    "score": 0.4,
                    "metadata": {"doc_type": "transcript", "episode_id": "e1"},
                    "text": "chunk2",
                },
            ],
            lift_stats={"transcript_hits_returned": 2, "lift_applied": 1},
        )

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/search",
        params={"q": "q", "path": str(tmp_path)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["lift_stats"] == {
        "transcript_hits_returned": 2,
        "lift_applied": 1,
    }


def test_search_lift_stats_invalid_dropped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        *_a: Any,
        **_k: Any,
    ) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[],
            lift_stats=cast(
                Dict[str, int],
                {"transcript_hits_returned": "bad", "lift_applied": 1},
            ),
        )

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/search",
        params={"q": "q", "path": str(tmp_path)},
    )
    assert response.status_code == 200
    assert response.json().get("lift_stats") is None


# ---------------------------------------------------------------------------
# RFC-088 chunk 5: enrich_results query param wires the QueryEnricher chain
# ---------------------------------------------------------------------------


def test_search_enrich_results_decorates_hits_with_related_topics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When enrich_results=true and topic_similarity.json exists, hits
    that carry a topic_id should gain a `query_enrichments.related_topics`
    annotation."""
    import json

    # Seed the chunk-3 enricher output the query enricher reads.
    out = tmp_path / "enrichments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "topic_similarity.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "enricher_id": "topic_similarity",
                "data": {
                    "topic_count": 1,
                    "topics": [
                        {
                            "topic_id": "topic:ai",
                            "topic_label": "AI",
                            "top_k": [
                                {"topic_id": "topic:ml", "similarity": 0.91},
                                {"topic_id": "topic:safety", "similarity": 0.83},
                            ],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "kg_topic:ai",
                    "score": 0.9,
                    "metadata": {"doc_type": "kg_topic", "topic_id": "topic:ai"},
                    "text": "AI",
                }
            ]
        )

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    body = client.get(
        "/api/search",
        params={"q": "ai", "path": str(tmp_path), "enrich_results": "true"},
    ).json()
    related = body["results"][0]["metadata"].get("query_enrichments", {}).get("related_topics")
    assert related is not None
    assert related[0]["topic_id"] == "topic:ml"


def test_search_enrich_results_false_leaves_hits_unmodified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default enrich_results=false skips the chain — no related_topics added."""

    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "kg_topic:ai",
                    "score": 0.9,
                    "metadata": {"doc_type": "kg_topic", "topic_id": "topic:ai"},
                    "text": "AI",
                }
            ]
        )

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    body = client.get(
        "/api/search",
        params={"q": "ai", "path": str(tmp_path)},
    ).json()
    assert "query_enrichments" not in body["results"][0]["metadata"]


def test_search_enrich_results_empty_hits_does_not_break_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """enrich_results=true with zero hits → the chain runs over an empty
    envelope and the response stays 200."""

    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(results=[])

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/search",
        params={"q": "x", "path": str(tmp_path), "enrich_results": "true"},
    )
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_search_enrich_results_passes_through_when_no_topic_similarity_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """enrich_results=true + no enrichments/topic_similarity.json on disk
    → response 200, hits unmodified (the chain logs an availability
    annotation and passes through)."""

    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "kg_topic:ai",
                    "score": 0.9,
                    "metadata": {"doc_type": "kg_topic", "topic_id": "topic:ai"},
                    "text": "AI",
                }
            ]
        )

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    body = client.get(
        "/api/search",
        params={"q": "ai", "path": str(tmp_path), "enrich_results": "true"},
    ).json()
    assert body["results"][0]["doc_id"] == "kg_topic:ai"
    assert "query_enrichments" not in body["results"][0]["metadata"]


def test_search_enrich_results_decorates_multiple_hits_with_different_topic_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple hits with distinct topic_ids each get their own
    related_topics decoration; a hit whose topic_id is not in the
    similarity envelope is left untouched."""
    import json as _json

    out = tmp_path / "enrichments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "topic_similarity.json").write_text(
        _json.dumps(
            {
                "schema_version": "1.0",
                "enricher_id": "topic_similarity",
                "data": {
                    "topic_count": 2,
                    "topics": [
                        {
                            "topic_id": "topic:ai",
                            "top_k": [{"topic_id": "topic:ml", "similarity": 0.9}],
                        },
                        {
                            "topic_id": "topic:climate",
                            "top_k": [{"topic_id": "topic:policy", "similarity": 0.8}],
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "h1",
                    "score": 0.9,
                    "metadata": {"doc_type": "kg_topic", "topic_id": "topic:ai"},
                    "text": "AI",
                },
                {
                    "doc_id": "h2",
                    "score": 0.8,
                    "metadata": {"doc_type": "kg_topic", "topic_id": "topic:climate"},
                    "text": "Climate",
                },
                {
                    "doc_id": "h3",
                    "score": 0.7,
                    "metadata": {"doc_type": "kg_topic", "topic_id": "topic:none"},
                    "text": "None",
                },
            ]
        )

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    body = client.get(
        "/api/search",
        params={"q": "x", "path": str(tmp_path), "enrich_results": "true"},
    ).json()
    results = body["results"]
    assert (
        results[0]["metadata"]["query_enrichments"]["related_topics"][0]["topic_id"] == "topic:ml"
    )
    assert (
        results[1]["metadata"]["query_enrichments"]["related_topics"][0]["topic_id"]
        == "topic:policy"
    )


# --------------------------------------------------------------------------
# Search v3 §S4b — result-set operators (cluster / consensus) on /api/search.
# --------------------------------------------------------------------------


def _install_fake_search(monkeypatch: pytest.MonkeyPatch, results: list[dict]) -> None:
    """Mock the retrieval path so operator tests can seed hit metadata directly."""

    def fake_run(
        output_dir: Path,
        query: str,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=results,
            lift_stats={"transcript_hits_returned": 0, "lift_applied": 0},
        )

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )


def test_search_no_operator_omits_operator_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backward compatibility — the operator fields are additive; when the
    caller doesn't ask for one, the response carries them as null so old
    clients ignore them without noticing."""
    _install_fake_search(
        monkeypatch,
        [{"doc_id": "d:1", "score": 0.5, "metadata": {"doc_type": "insight"}, "text": "x"}],
    )
    app = create_app(tmp_path, static_dir=False)
    body = TestClient(app).get("/api/search", params={"q": "x", "path": str(tmp_path)}).json()
    assert body["operator"] is None
    assert body["clusters"] is None
    assert body["consensus_pairs"] is None


def test_search_invalid_operator_is_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown operator names silently fall through (endpoint never breaks)."""
    _install_fake_search(
        monkeypatch,
        [{"doc_id": "d:1", "score": 0.5, "metadata": {"doc_type": "insight"}, "text": "x"}],
    )
    app = create_app(tmp_path, static_dir=False)
    body = (
        TestClient(app)
        .get("/api/search", params={"q": "x", "path": str(tmp_path), "operator": "bogus"})
        .json()
    )
    assert body["operator"] is None
    assert body["clusters"] is None


def test_search_operator_cluster_groups_by_topic_cluster_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``operator=cluster`` — server groups hits by ``metadata.topic_cluster``
    when present. Ungrouped hits land in a trailing bucket with cluster_id=null."""
    _install_fake_search(
        monkeypatch,
        [
            {
                "doc_id": "d:1",
                "score": 0.9,
                "metadata": {
                    "doc_type": "kg_topic",
                    "source_id": "topic:climate",
                    "topic_cluster": {
                        "topic_cluster_compound_id": "tc:env",
                        "label": "Environment",
                    },
                },
                "text": "a",
            },
            {
                "doc_id": "d:2",
                "score": 0.8,
                "metadata": {
                    "doc_type": "kg_topic",
                    "source_id": "topic:policy",
                    "topic_cluster": {
                        "topic_cluster_compound_id": "tc:env",
                        "label": "Environment",
                    },
                },
                "text": "b",
            },
            {
                "doc_id": "d:3",
                "score": 0.7,
                "metadata": {"doc_type": "transcript", "episode_id": "ep-x"},
                "text": "c",
            },
        ],
    )
    app = create_app(tmp_path, static_dir=False)
    body = (
        TestClient(app)
        .get(
            "/api/search",
            params={"q": "x", "path": str(tmp_path), "operator": "cluster"},
        )
        .json()
    )
    assert body["operator"] == "cluster"
    clusters = body["clusters"]
    assert clusters is not None and len(clusters) == 2
    assert clusters[0]["cluster_id"] == "tc:env"
    assert clusters[0]["cluster_kind"] == "topic_cluster"
    assert clusters[0]["size"] == 2
    assert clusters[0]["hit_indices"] == [0, 1]
    assert clusters[1]["cluster_id"] is None
    assert clusters[1]["cluster_kind"] == "ungrouped"


def test_search_operator_consensus_reads_topic_consensus_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``operator=consensus`` — server reads enrichments/topic_consensus.json
    and filters pairs to topics surfaced in the hit set."""
    import json as _json

    (tmp_path / "enrichments").mkdir()
    (tmp_path / "enrichments" / "topic_consensus.json").write_text(
        _json.dumps(
            {
                "derived": True,
                "enricher_id": "topic_consensus",
                "schema_version": "1.0",
                "data": {
                    "consensus": [
                        {
                            "topic_id": "topic:climate",
                            "person_a_id": "person:alice",
                            "person_b_id": "person:bob",
                            "insight_a_id": "i:a",
                            "insight_b_id": "i:b",
                            "contradiction_score": 0.1,
                            "cosine_similarity": 0.85,
                        },
                        {
                            "topic_id": "topic:unrelated",
                            "person_a_id": "person:c",
                            "person_b_id": "person:d",
                            "insight_a_id": "i:c",
                            "insight_b_id": "i:d",
                            "contradiction_score": 0.2,
                        },
                    ]
                },
            }
        )
    )
    _install_fake_search(
        monkeypatch,
        [
            {
                "doc_id": "d:1",
                "score": 0.9,
                "metadata": {"doc_type": "kg_topic", "source_id": "topic:climate"},
                "text": "x",
            },
        ],
    )
    app = create_app(tmp_path, static_dir=False)
    body = (
        TestClient(app)
        .get(
            "/api/search",
            params={"q": "x", "path": str(tmp_path), "operator": "consensus"},
        )
        .json()
    )
    assert body["operator"] == "consensus"
    pairs = body["consensus_pairs"]
    assert pairs is not None
    assert len(pairs) == 1
    assert pairs[0]["topic_id"] == "topic:climate"
    assert pairs[0]["person_a_id"] == "person:alice"


def test_search_operator_consensus_missing_enrichment_returns_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``enrichments/topic_consensus.json`` → operator still runs; pairs=[]."""
    _install_fake_search(
        monkeypatch,
        [{"doc_id": "d:1", "score": 0.5, "metadata": {"doc_type": "insight"}, "text": "x"}],
    )
    app = create_app(tmp_path, static_dir=False)
    body = (
        TestClient(app)
        .get(
            "/api/search",
            params={"q": "x", "path": str(tmp_path), "operator": "consensus"},
        )
        .json()
    )
    assert body["operator"] == "consensus"
    assert body["consensus_pairs"] == []
