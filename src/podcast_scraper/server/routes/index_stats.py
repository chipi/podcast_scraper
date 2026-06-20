"""GET /api/index/stats — LanceDB vector index metrics + staleness (GitHub #507)."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Query, Request

from podcast_scraper.search.corpus_scope import normalize_feed_id
from podcast_scraper.server.index_rebuild import rebuild_status_snapshot
from podcast_scraper.server.index_staleness import compute_index_staleness
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import (
    IndexStatsBody,
    IndexStatsEnvelope,
    IndexTimeseriesMonth,
    IndexTimeseriesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["index"])


def _canonical_feeds_indexed(raw: list[str]) -> list[str]:
    """Deduplicate and sort feed ids the same way as the corpus catalog (strip / normalize)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        n = normalize_feed_id(item)
        if n is None:
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    out.sort()
    return out


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    return fallback


@router.get("/index/stats", response_model=IndexStatsEnvelope)
async def index_stats(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus output dir (contains search/). Omit to use server default output_dir.",
    ),
    embedding_model: str | None = Query(
        default=None,
        description=(
            "Optional embedding model id to compare with the on-disk index; "
            "defaults to Config.vector_embedding_model when omitted."
        ),
    ),
) -> IndexStatsEnvelope:
    """Return LanceDB index statistics when ``<corpus>/search/lance_index`` is loadable."""
    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    if root is None:
        return IndexStatsEnvelope(available=False, reason="no_corpus_path")

    rb_prog, rb_err = rebuild_status_snapshot(request.app, root)
    index_dir = (root / "search").resolve()
    lance_dir = index_dir / "lance_index"
    from podcast_scraper.search.lance_index_stats import read_lance_index_stats

    raw = read_lance_index_stats(lance_dir)
    if raw is None:
        st = compute_index_staleness(
            root,
            index_available=False,
            index_reason="no_index",
            index_last_updated=None,
            index_embedding_model=None,
            embedding_model_query=embedding_model,
        )
        return IndexStatsEnvelope(
            available=False,
            reason="no_index",
            index_path=str(index_dir),
            reindex_recommended=st.reindex_recommended,
            reindex_reasons=st.reindex_reasons,
            artifact_newest_mtime=st.artifact_newest_mtime,
            search_root_hints=st.search_root_hints,
            rebuild_in_progress=rb_prog,
            rebuild_last_error=rb_err,
        )

    body = IndexStatsBody(
        total_vectors=raw.total_vectors,
        doc_type_counts=dict(raw.doc_type_counts),
        feeds_indexed=_canonical_feeds_indexed(list(raw.feeds_indexed)),
        embedding_model=raw.embedding_model or "",
        embedding_dim=raw.embedding_dim or 0,
        last_updated=raw.last_updated or "",
        index_size_bytes=raw.index_size_bytes,
    )
    st = compute_index_staleness(
        root,
        index_available=True,
        index_reason=None,
        index_last_updated=raw.last_updated,
        index_embedding_model=raw.embedding_model,
        embedding_model_query=embedding_model,
        index_embedding_provider=raw.embedding_provider,
    )
    return IndexStatsEnvelope(
        available=True,
        index_path=str(index_dir),
        stats=body,
        reindex_recommended=st.reindex_recommended,
        reindex_reasons=st.reindex_reasons,
        artifact_newest_mtime=st.artifact_newest_mtime,
        search_root_hints=st.search_root_hints,
        rebuild_in_progress=rb_prog,
        rebuild_last_error=rb_err,
    )


@router.get("/index/timeseries", response_model=IndexTimeseriesResponse)
async def index_timeseries(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus output dir (contains search/). Omit to use server default output_dir.",
    ),
) -> IndexTimeseriesResponse:
    """Return indexed-document counts by publish month × doc_type.

    Reads each index row's ``publish_date`` + ``doc_type`` so the Index section
    can chart what the index covers over time, broken down by document type.
    Always 200; empty ``by_month`` when no corpus/index is loadable.
    """
    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    if root is None:
        return IndexTimeseriesResponse(available=False)

    lance_dir = (root / "search").resolve() / "lance_index"
    from podcast_scraper.search.lance_index_stats import read_lance_doc_type_by_month

    by_month = read_lance_doc_type_by_month(lance_dir)
    doc_types = sorted({dt for bucket in by_month.values() for dt in bucket})
    return IndexTimeseriesResponse(
        available=bool(by_month),
        by_month=[IndexTimeseriesMonth(month=m, doc_types=by_month[m]) for m in sorted(by_month)],
        doc_types=doc_types,
    )
