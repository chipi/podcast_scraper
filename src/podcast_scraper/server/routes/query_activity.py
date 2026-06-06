"""GET /api/corpus/query-activity — daily search volume (PRD-033 FR6.2, #888 follow-up).

Reads the append-only search-activity log (``search/query_log.jsonl``) written by
``/api/search`` and returns zero-filled daily counts for the Dashboard activity chart.
This is an honest *search-volume-over-time* signal (not the unattainable
"query volume by topic" — queries are not topic-tagged).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query, Request

from podcast_scraper.search.query_log import read_query_activity
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import QueryActivityBucket, QueryActivityResponse

router = APIRouter(tags=["dashboard"])


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    return fallback


@router.get("/corpus/query-activity", response_model=QueryActivityResponse)
async def query_activity(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    days: int = Query(default=30, ge=1, le=365),
) -> QueryActivityResponse:
    """Daily search counts for the last *days* days (zero-filled, oldest→newest)."""
    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    if root is None:
        return QueryActivityResponse(error="no_corpus_path")
    data = read_query_activity(root, days=days)
    buckets = [
        QueryActivityBucket(date=str(b["date"]), count=int(b["count"]))
        for b in data.get("buckets", [])
    ]
    return QueryActivityResponse(total=int(data.get("total", 0)), buckets=buckets)
