"""Map corpus-search outcomes to the consumer search response (#1068).

Pure helpers over the plain dict returned by ``structured_corpus_search`` — no HTTP,
no index. The consumer search routes (library-wide + episode-scoped) reuse the existing
hybrid retrieval (RFC-090) and these helpers; there is no request-time LLM (D6).
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.server.schemas import (
    CorpusSearchApiResponse,
    CorpusSearchLiftStatsModel,
    SearchHitModel,
)


def _lift_stats(raw: Any) -> CorpusSearchLiftStatsModel | None:
    if not isinstance(raw, dict):
        return None
    try:
        th = int(raw.get("transcript_hits_returned", 0))
        la = int(raw.get("lift_applied", 0))
    except (TypeError, ValueError):
        return None
    return CorpusSearchLiftStatsModel(transcript_hits_returned=max(0, th), lift_applied=max(0, la))


def build_search_response(query: str, outcome: dict[str, Any]) -> CorpusSearchApiResponse:
    """Map a ``structured_corpus_search`` outcome dict to the API response model."""
    error = outcome.get("error")
    if error:
        return CorpusSearchApiResponse(
            query=query, results=[], error=str(error), detail=outcome.get("detail")
        )
    hits = [
        SearchHitModel(
            doc_id=str(r.get("doc_id", "")),
            score=float(r.get("score") or 0.0),
            metadata=dict(r.get("metadata") or {}),
            text=str(r.get("text", "")),
            source_tier=str(r.get("source_tier") or ""),
            supporting_quotes=r.get("supporting_quotes"),
            lifted=r.get("lifted"),
        )
        for r in (outcome.get("results") or [])
    ]
    return CorpusSearchApiResponse(
        query=query,
        results=hits,
        query_type=str(outcome.get("query_type") or ""),
        lift_stats=_lift_stats(outcome.get("lift_stats")),
    )
