"""GET /api/corpus/* — Corpus Library catalog (RFC-067 Phases 1 & 3)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.search.corpus_similar import episode_scope_key, run_similar_episodes
from podcast_scraper.server.corpus_catalog import (
    aggregate_feeds,
    build_catalog_rows,
    catalog_row_for_metadata_path,
    CatalogEpisodeRow,
    decode_catalog_cursor,
    episode_list_summary_preview,
    episode_list_topics,
    feed_description_by_feed_id,
    feed_display_title_by_feed_id,
    feed_rss_url_by_feed_id,
    filter_rows,
    index_rows_by_feed_episode,
    resolve_feed_description,
    resolve_feed_display_title,
    resolve_feed_rss_url,
    slice_page,
)
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import (
    CorpusEpisodeDetailResponse,
    CorpusEpisodeListItem,
    CorpusEpisodesResponse,
    CorpusFeedItem,
    CorpusFeedsResponse,
    CorpusSimilarEpisodeItem,
    CorpusSimilarEpisodesResponse,
)
from podcast_scraper.utils.path_validation import safe_relpath_under_corpus_root

router = APIRouter(tags=["corpus"])


def _safe_metadata_path_str(base: Path, relpath: str) -> str:
    rel = relpath.strip().replace("\\", "/")
    if not rel or rel.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path.")
    safe = safe_relpath_under_corpus_root(base, rel)
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid path.")
    return safe


def _similar_items_to_response_models(
    items: list[dict[str, Any]],
    by_scope: dict[tuple[str, str], CatalogEpisodeRow],
) -> list[CorpusSimilarEpisodeItem]:
    out: list[CorpusSimilarEpisodeItem] = []
    for it in items:
        meta = dict(it.get("metadata") or {})
        key = episode_scope_key(meta)
        row = by_scope.get(key) if key else None
        fid = key[0] if key else ""
        eid = key[1] if key else None
        dt = it.get("doc_type")
        doc_type = dt if isinstance(dt, str) else None
        out.append(
            CorpusSimilarEpisodeItem(
                score=float(it.get("score", 0.0)),
                feed_id=fid,
                episode_id=eid,
                episode_title=row.episode_title if row else "",
                metadata_relative_path=row.metadata_relative_path if row else None,
                publish_date=row.publish_date if row else None,
                doc_type=doc_type,
                snippet=str(it.get("text") or ""),
                feed_image_url=row.feed_image_url if row else None,
                episode_image_url=row.episode_image_url if row else None,
                duration_seconds=row.duration_seconds if row else None,
                episode_number=row.episode_number if row else None,
                feed_image_local_relpath=row.feed_image_local_relpath if row else None,
                episode_image_local_relpath=row.episode_image_local_relpath if row else None,
            )
        )
    return out


def _metadata_suffix_ok(name: str) -> bool:
    lower = name.lower()
    return (
        lower.endswith(".metadata.json")
        or lower.endswith(
            ".metadata.yaml",
        )
        or lower.endswith(".metadata.yml")
    )


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    if fallback is not None:
        return Path(fallback).expanduser().resolve()
    raise HTTPException(
        status_code=400,
        detail=(
            "path query parameter is required when the server has no default output_dir "
            "(set PODCAST_SERVE_OUTPUT_DIR or pass output_dir to create_app)."
        ),
    )


@router.get("/corpus/feeds", response_model=CorpusFeedsResponse)
async def corpus_feeds(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus root (contains metadata/). Omit to use server default output_dir.",
    ),
) -> CorpusFeedsResponse:
    """List feeds and per-feed episode counts for the corpus library API."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    rows = build_catalog_rows(root)
    feeds_raw = aggregate_feeds(rows)
    feeds = [
        CorpusFeedItem(
            feed_id=str(f["feed_id"]),
            display_title=f["display_title"],
            episode_count=int(f["episode_count"]),
            image_url=f.get("image_url"),
            image_local_relpath=f.get("image_local_relpath"),
            rss_url=f.get("rss_url"),
            description=f.get("description"),
        )
        for f in feeds_raw
    ]
    return CorpusFeedsResponse(path=str(root), feeds=feeds)


@router.get("/corpus/episodes", response_model=CorpusEpisodesResponse)
async def corpus_episodes(
    request: Request,
    path: str | None = Query(default=None, description="Corpus root."),
    feed_id: str | None = Query(default=None, description="Exact feed_id filter."),
    q: str | None = Query(default=None, description="Case-insensitive substring on episode title."),
    topic_q: str | None = Query(
        default=None,
        description="Case-insensitive substring on summary title or any summary bullet.",
    ),
    since: str | None = Query(default=None, description="Publish date on/after YYYY-MM-DD."),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, description="Pagination cursor from previous page."),
) -> CorpusEpisodesResponse:
    """Paginated episode list with optional feed, title, topic, and date filters."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    rows = build_catalog_rows(root)
    titles_by_feed = feed_display_title_by_feed_id(rows)
    rss_by_feed = feed_rss_url_by_feed_id(rows)
    desc_by_feed = feed_description_by_feed_id(rows)
    filtered = filter_rows(rows, feed_id=feed_id, title_q=q, topic_q=topic_q, since=since)
    offset = decode_catalog_cursor(cursor)
    page_rows, next_cur = slice_page(filtered, offset, limit)
    items = [
        CorpusEpisodeListItem(
            metadata_relative_path=r.metadata_relative_path,
            feed_id=r.feed_id,
            feed_display_title=resolve_feed_display_title(r, titles_by_feed),
            feed_rss_url=resolve_feed_rss_url(r, rss_by_feed),
            feed_description=resolve_feed_description(r, desc_by_feed),
            topics=episode_list_topics(r.summary_bullets),
            summary_title=r.summary_title,
            summary_bullets_preview=list(r.summary_bullets[:4]),
            summary_preview=episode_list_summary_preview(r),
            episode_id=r.episode_id,
            episode_title=r.episode_title,
            publish_date=r.publish_date,
            feed_image_url=r.feed_image_url,
            episode_image_url=r.episode_image_url,
            duration_seconds=r.duration_seconds,
            episode_number=r.episode_number,
            feed_image_local_relpath=r.feed_image_local_relpath,
            episode_image_local_relpath=r.episode_image_local_relpath,
        )
        for r in page_rows
    ]
    feed_echo: str | None = None
    if feed_id is not None:
        feed_echo = feed_id.strip()
    return CorpusEpisodesResponse(
        path=str(root),
        feed_id=feed_echo,
        items=items,
        next_cursor=next_cur,
    )


@router.get("/corpus/episodes/detail", response_model=CorpusEpisodeDetailResponse)
async def corpus_episode_detail(
    request: Request,
    path: str | None = Query(default=None, description="Corpus root."),
    metadata_relpath: str = Query(
        ...,
        description="Metadata file path relative to corpus root (as listed in episodes).",
    ),
) -> CorpusEpisodeDetailResponse:
    """Return full episode metadata and summary fields for one catalog row."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    target = _safe_metadata_path_str(root, metadata_relpath)

    # codeql[py/path-injection] -- target from normpath+startswith in safe_relpath.
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="Metadata file not found.")
    # codeql[py/path-injection] -- target sanitized above.
    if not _metadata_suffix_ok(os.path.basename(target)):
        raise HTTPException(status_code=400, detail="Not an episode metadata file.")
    root_s = os.path.normpath(str(root.resolve()))
    rel_posix = os.path.relpath(target, root_s).replace("\\", "/")
    if rel_posix.startswith(".."):
        raise HTTPException(status_code=400, detail="Invalid path.")
    rows = build_catalog_rows(root)
    r = next((row for row in rows if row.metadata_relative_path == rel_posix), None)
    if r is None:
        raise HTTPException(status_code=404, detail="Metadata not in catalog scan.")
    rss_by_feed = feed_rss_url_by_feed_id(rows)
    desc_by_feed = feed_description_by_feed_id(rows)
    return CorpusEpisodeDetailResponse(
        path=str(root),
        metadata_relative_path=r.metadata_relative_path,
        feed_id=r.feed_id,
        feed_rss_url=resolve_feed_rss_url(r, rss_by_feed),
        feed_description=resolve_feed_description(r, desc_by_feed),
        episode_id=r.episode_id,
        episode_title=r.episode_title,
        publish_date=r.publish_date,
        summary_title=r.summary_title,
        summary_bullets=list(r.summary_bullets),
        summary_text=r.summary_text,
        gi_relative_path=r.gi_relative_path,
        kg_relative_path=r.kg_relative_path,
        has_gi=r.has_gi,
        has_kg=r.has_kg,
        feed_image_url=r.feed_image_url,
        episode_image_url=r.episode_image_url,
        duration_seconds=r.duration_seconds,
        episode_number=r.episode_number,
        feed_image_local_relpath=r.feed_image_local_relpath,
        episode_image_local_relpath=r.episode_image_local_relpath,
    )


@router.get("/corpus/episodes/similar", response_model=CorpusSimilarEpisodesResponse)
async def corpus_episodes_similar(
    request: Request,
    path: str | None = Query(default=None, description="Corpus root."),
    metadata_relpath: str = Query(
        ...,
        description="Source metadata path relative to corpus root (episodes list).",
    ),
    top_k: int = Query(default=8, ge=1, le=25, description="Max peer episodes after dedupe."),
) -> CorpusSimilarEpisodesResponse:
    """Semantic peers via FAISS (RFC-067 Phase 3); 200 with ``error`` when index missing."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    target = _safe_metadata_path_str(root, metadata_relpath)

    # codeql[py/path-injection] -- target from normpath+startswith in safe_relpath.
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="Metadata file not found.")
    # codeql[py/path-injection] -- target sanitized above.
    if not _metadata_suffix_ok(os.path.basename(target)):
        raise HTTPException(status_code=400, detail="Not an episode metadata file.")
    root_s = os.path.normpath(str(root.resolve()))
    rel_posix = os.path.relpath(target, root_s).replace("\\", "/")
    if rel_posix.startswith(".."):
        raise HTTPException(status_code=400, detail="Invalid path.")
    r = catalog_row_for_metadata_path(root, rel_posix)
    if r is None:
        raise HTTPException(status_code=404, detail="Metadata not in catalog scan.")

    catalog_rows = build_catalog_rows(root)
    by_scope = index_rows_by_feed_episode(catalog_rows)
    outcome = run_similar_episodes(
        root,
        summary_title=r.summary_title,
        summary_bullets=r.summary_bullets,
        episode_title=r.episode_title,
        source_feed_id=r.feed_id,
        source_episode_id=r.episode_id,
        top_k=top_k,
    )
    if outcome.error:
        return CorpusSimilarEpisodesResponse(
            path=str(root),
            source_metadata_relative_path=r.metadata_relative_path,
            query_used=outcome.query_used,
            items=[],
            error=outcome.error,
            detail=outcome.detail,
        )
    models = _similar_items_to_response_models(outcome.items, by_scope)
    return CorpusSimilarEpisodesResponse(
        path=str(root),
        source_metadata_relative_path=r.metadata_relative_path,
        query_used=outcome.query_used,
        items=models,
    )
