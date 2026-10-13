"""Unit tests for the consumer search-outcome mapping (#1068).

Pure dict-in / model-out — no HTTP, no index.
"""

from __future__ import annotations

from podcast_scraper.server.app_search_view import build_search_response


def _outcome(results, *, error=None, detail=None, query_type="semantic", lift_stats=None) -> dict:
    return {
        "results": results,
        "error": error,
        "detail": detail,
        "query_type": query_type,
        "lift_stats": lift_stats,
    }


class TestBuildSearchResponse:
    def test_maps_hits_and_metadata(self) -> None:
        out = _outcome(
            [
                {
                    "doc_id": "insight:1",
                    "score": 0.9,
                    "metadata": {"episode_id": "ep1", "doc_type": "insight"},
                    "text": "hi",
                    "source_tier": "insight",
                    "supporting_quotes": None,
                    "lifted": None,
                }
            ],
            query_type="entity_lookup",
            lift_stats={"transcript_hits_returned": 2, "lift_applied": 1},
        )
        resp = build_search_response("q", out)
        assert resp.error is None
        assert resp.query == "q"
        assert resp.query_type == "entity_lookup"
        assert len(resp.results) == 1
        assert resp.results[0].doc_id == "insight:1"
        assert resp.results[0].source_tier == "insight"
        assert resp.lift_stats is not None
        assert resp.lift_stats.transcript_hits_returned == 2

    def test_error_passthrough(self) -> None:
        resp = build_search_response("q", _outcome([], error="no_index", detail="no index"))
        assert resp.error == "no_index"
        assert resp.detail == "no index"
        assert resp.results == []

    def test_tolerates_missing_optional_hit_fields(self) -> None:
        resp = build_search_response("q", _outcome([{"doc_id": "x", "metadata": {}, "text": "t"}]))
        assert resp.results[0].score == 0.0
        assert resp.results[0].source_tier == ""
