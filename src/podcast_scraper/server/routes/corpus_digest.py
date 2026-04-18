"""GET /api/corpus/digest — Corpus Digest (RFC-068)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.graph_id_utils import slugify_label, topic_node_id_from_slug
from podcast_scraper.search.corpus_search import run_corpus_search
from podcast_scraper.server.cil_digest_topics import (
    build_cil_digest_topics_for_row,
    load_topic_cluster_index,
)
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows,
    CatalogEpisodeRow,
    episode_list_summary_preview,
    feed_description_by_feed_id,
    feed_display_title_by_feed_id,
    feed_rss_url_by_feed_id,
    resolve_feed_description,
    resolve_feed_display_title,
    resolve_feed_rss_url,
)
from podcast_scraper.server.corpus_digest import (
    DIGEST_MAX_TOPICS_PER_REQUEST,
    digest_row_dict,
    DIGEST_TOPIC_SEARCH_TIMEOUT_SEC,
    DIGEST_TOPIC_SEARCH_TOP_K,
    diversify_digest_rows,
    episode_in_utc_window,
    filter_rows_in_window,
    load_digest_topics,
    lookup_scope_index,
    meta_episode_key,
    since_str_for_search,
    utc_bounds_for_window,
)
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import (
    CorpusDigestResponse,
    CorpusDigestRow,
    CorpusDigestTopicBand,
    CorpusDigestTopicHit,
)

router = APIRouter(tags=["corpus"])

_MAX_ROWS_FULL_DEFAULT = 36
_MAX_ROWS_FULL_ABS = 50
_MAX_ROWS_COMPACT_DEFAULT = 8
_MAX_ROWS_COMPACT_ABS = 8
_PER_FEED_FULL = 3
_PER_FEED_COMPACT = 2
_MAX_HITS_PER_TOPIC = 8


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


def _iso_z(dt: datetime) -> str:
    u = dt.astimezone(timezone.utc)
    s = u.isoformat()
    return s.replace("+00:00", "Z")


def _topic_band_for_query(
    root: Path,
    topic_id: str,
    label: str,
    query: str,
    *,
    start: datetime,
    end: datetime,
    since: str,
    scope: dict[tuple[str, str], CatalogEpisodeRow],
    feed_titles_by_feed_id: dict[str, str],
    feed_rss_urls_by_feed_id: dict[str, str],
    feed_descriptions_by_feed_id: dict[str, str],
) -> CorpusDigestTopicBand | None:
    """Run one semantic search; map hits to catalog rows inside the digest window."""

    def _call() -> object:
        return run_corpus_search(
            root,
            query,
            since=since,
            top_k=DIGEST_TOPIC_SEARCH_TOP_K,
            dedupe_kg_surfaces=False,
        )

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            outcome = fut.result(timeout=DIGEST_TOPIC_SEARCH_TIMEOUT_SEC)
        except FuturesTimeoutError:
            return None

    if outcome.error:
        return None

    best: dict[str, tuple[float, CorpusDigestTopicHit]] = {}
    for row in outcome.results:
        meta = dict(row.get("metadata") or {})
        key = meta_episode_key(meta)
        if key is None:
            continue
        catalog_row = scope.get(key)
        if catalog_row is None:
            continue
        if not episode_in_utc_window(catalog_row, start, end):
            continue
        score = float(row.get("score", 0.0))
        rel = catalog_row.metadata_relative_path
        prev = best.get(rel)
        if prev is not None and score <= prev[0]:
            continue
        fn = catalog_row.feed_id
        best[rel] = (
            score,
            CorpusDigestTopicHit(
                metadata_relative_path=rel,
                episode_title=catalog_row.episode_title,
                feed_id=fn,
                feed_display_title=resolve_feed_display_title(catalog_row, feed_titles_by_feed_id),
                feed_rss_url=resolve_feed_rss_url(catalog_row, feed_rss_urls_by_feed_id),
                feed_description=resolve_feed_description(
                    catalog_row, feed_descriptions_by_feed_id
                ),
                score=score,
                summary_preview=episode_list_summary_preview(catalog_row),
                episode_id=catalog_row.episode_id,
                publish_date=catalog_row.publish_date,
                gi_relative_path=catalog_row.gi_relative_path,
                kg_relative_path=catalog_row.kg_relative_path,
                has_gi=catalog_row.has_gi,
                has_kg=catalog_row.has_kg,
                feed_image_url=catalog_row.feed_image_url,
                episode_image_url=catalog_row.episode_image_url,
                duration_seconds=catalog_row.duration_seconds,
                episode_number=catalog_row.episode_number,
                feed_image_local_relpath=catalog_row.feed_image_local_relpath,
                episode_image_local_relpath=catalog_row.episode_image_local_relpath,
            ),
        )

    ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)
    hits = [h for _, h in ranked[:_MAX_HITS_PER_TOPIC]]
    if not hits:
        return None
    graph_topic_id = topic_node_id_from_slug(slugify_label(label))
    return CorpusDigestTopicBand(
        topic_id=topic_id,
        label=label,
        query=query,
        graph_topic_id=graph_topic_id,
        hits=hits,
    )


@router.get("/corpus/digest", response_model=CorpusDigestResponse)
async def corpus_digest(
    request: Request,
    path: str | None = Query(default=None, description="Corpus root."),
    window: Literal["all", "24h", "7d", "1mo", "since"] = Query(
        default="all",
        description=(
            "Time window (UTC). all = no lower bound (publish dates from 1970-01-01). "
            "Rolling: 24h, 7d. 1mo = previous calendar month. since = from YYYY-MM-DD. "
            "Ignored when compact=true (forces 24h)."
        ),
    ),
    since: str | None = Query(
        default=None,
        description="Required when window=since (YYYY-MM-DD).",
    ),
    compact: bool = Query(
        default=False,
        description="Library glance: force 24h, smaller cap, omit topics.",
    ),
    include_topics: bool = Query(
        default=True,
        description="Include semantic topic bands (Digest tab). Ignored when compact=true.",
    ),
    max_rows: int | None = Query(
        default=None,
        description="Optional row cap; server clamps to safe maximum.",
    ),
) -> CorpusDigestResponse:
    """Return digest rows and optional semantic topic bands for the corpus window."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    now_utc = datetime.now(timezone.utc)

    eff_window: Literal["all", "24h", "7d", "1mo", "since"] = "24h" if compact else window
    try:
        start, end = utc_bounds_for_window(eff_window, since=since, now_utc=now_utc)
    except ValueError as exc:
        if str(exc) == "since_required":
            raise HTTPException(
                status_code=400,
                detail="Query parameter 'since' (YYYY-MM-DD) is required when window=since.",
            ) from exc
        raise HTTPException(status_code=400, detail="Invalid window.") from exc

    if compact:
        default_cap = _MAX_ROWS_COMPACT_DEFAULT
        abs_cap = _MAX_ROWS_COMPACT_ABS
        per_feed = _PER_FEED_COMPACT
        want_topics = False
    else:
        default_cap = _MAX_ROWS_FULL_DEFAULT
        abs_cap = _MAX_ROWS_FULL_ABS
        per_feed = _PER_FEED_FULL
        want_topics = include_topics

    row_cap = default_cap if max_rows is None else int(max_rows)
    row_cap = max(1, min(row_cap, abs_cap))

    catalog = build_catalog_rows(root)
    in_window = filter_rows_in_window(catalog, start, end)
    picked = diversify_digest_rows(
        in_window,
        max_rows=row_cap,
        per_feed_cap=per_feed,
    )
    titles_by_feed = feed_display_title_by_feed_id(catalog)
    rss_by_feed = feed_rss_url_by_feed_id(catalog)
    desc_by_feed = feed_description_by_feed_id(catalog)
    cluster_index = None if compact else load_topic_cluster_index(root)
    row_models: list[CorpusDigestRow] = []
    for r in picked:
        cil_payload: list[dict[str, Any]] = []
        if cluster_index is not None:
            cil_payload = [
                p.model_dump() for p in build_cil_digest_topics_for_row(root, r, cluster_index)
            ]
        row_models.append(
            CorpusDigestRow(
                **digest_row_dict(
                    r,
                    feed_titles_by_feed_id=titles_by_feed,
                    feed_rss_urls_by_feed_id=rss_by_feed,
                    feed_descriptions_by_feed_id=desc_by_feed,
                    cil_digest_topics=cil_payload,
                )
            )
        )

    topics: list[CorpusDigestTopicBand] = []
    topics_reason: str | None = None

    if want_topics:
        scope = lookup_scope_index(catalog)
        since_s = since_str_for_search(start)
        topics_cfg = load_digest_topics()[:DIGEST_MAX_TOPICS_PER_REQUEST]

        probe = run_corpus_search(
            root,
            "digest",
            since=since_s,
            top_k=1,
            dedupe_kg_surfaces=False,
        )
        if probe.error == "no_index":
            topics_reason = "no_index"
        else:
            for t in topics_cfg:
                band = _topic_band_for_query(
                    root,
                    t["id"],
                    t["label"],
                    t["query"],
                    start=start,
                    end=end,
                    since=since_s,
                    scope=scope,
                    feed_titles_by_feed_id=titles_by_feed,
                    feed_rss_urls_by_feed_id=rss_by_feed,
                    feed_descriptions_by_feed_id=desc_by_feed,
                )
                if band is not None:
                    topics.append(band)

    return CorpusDigestResponse(
        path=str(root),
        window=eff_window,
        window_start_utc=_iso_z(start),
        window_end_utc=_iso_z(end),
        compact=compact,
        rows=row_models,
        topics=topics,
        topics_unavailable_reason=topics_reason,
    )
