"""Consumer Learning Platform episode routes (``/api/app/episodes/*``).

Read-only, slug-addressed surface for the end-user app (RFC-098 §4–§5; #1067/#1068/#1070).
Mounted under the ``/api/app`` namespace, separate from the operator routes; access
becomes auth-gated in later Epic-1 tasks (#1063/#1066).
Serves the single shared corpus at ``app.state.output_dir`` — there is no ``?path``
override (that is an operator concern; see the operator ``/api/corpus/*`` routes).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.search.capability import structured_corpus_search
from podcast_scraper.search.corpus_similar import episode_scope_key, run_similar_episodes
from podcast_scraper.search.query_log import append_query_event
from podcast_scraper.search.topic_clusters import consumer_topic_cluster_map
from podcast_scraper.server.app_artwork import artwork_url
from podcast_scraper.server.app_audio_bridge import resolve_audio
from podcast_scraper.server.app_content_source import (
    get_content_source,
    row_to_summary,
    transcript_corpus_relpath,
    transcript_relpath,
)
from podcast_scraper.server.app_corpus_access import corpus_root_or_503, load_json_artifact
from podcast_scraper.server.app_gi_view import insights_from_gi
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_search_view import build_search_response, filter_outcome_to_episode
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.corpus_catalog import (
    _load_metadata_doc,
    aggregate_feeds,
    build_catalog_rows_cumulative,
    CatalogEpisodeRow,
    index_rows_by_feed_episode,
)
from podcast_scraper.server.schemas import (
    AppEntitiesResponse,
    AppEpisodeDetail,
    AppEpisodesResponse,
    AppInsightsResponse,
    AppPodcastItem,
    AppPodcastsResponse,
    AudioSourceResponse,
    CorpusSearchApiResponse,
    SegmentsResponse,
)
from podcast_scraper.server.segments_view import (
    segments_relpaths_for_transcript,
    to_contract_segments,
)
from podcast_scraper.utils.path_validation import safe_relpath_under_corpus_root

logger = logging.getLogger(__name__)

router = APIRouter(tags=["app"])


def _resolve(request: Request, slug: str) -> tuple[Path, CatalogEpisodeRow]:
    """Resolve ``(corpus_root, row)`` for a slug, or 404 when the slug is unknown."""
    root = corpus_root_or_503(request)
    row = resolve_slug(root, slug)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown episode slug.")
    return root, row


def _content_block(root: Path, metadata_relpath: str) -> dict:
    """Load an episode's ``content`` metadata block (empty dict when absent)."""
    doc = _load_metadata_doc(root / metadata_relpath)
    content = doc.get("content") if isinstance(doc, dict) else None
    return content if isinstance(content, dict) else {}


def _episodes_page(
    request: Request,
    *,
    feed_id: str | None,
    status: str | None,
    page: int,
    page_size: int,
) -> AppEpisodesResponse:
    """Shared catalog-list builder for the global + per-podcast endpoints."""
    root = corpus_root_or_503(request)
    source = get_content_source(request.app.state, root)
    offset = (page - 1) * page_size
    result = source.list_episodes(feed_id=feed_id, status=status, offset=offset, limit=page_size)
    has_more = offset + len(result.items) < result.total
    return AppEpisodesResponse(
        items=result.items,
        page=page,
        page_size=page_size,
        total=result.total,
        has_more=has_more,
    )


@router.get("/episodes", response_model=AppEpisodesResponse)
async def episodes_list(
    request: Request,
    page: int = Query(default=1, ge=1, description="1-based page index."),
    page_size: int = Query(default=20, ge=1, le=100, description="Episodes per page."),
    status: str | None = Query(
        default=None, description="Optional status filter: 'ready' or 'pending'."
    ),
    feed_id: str | None = Query(default=None, description="Optional feed-id scope."),
) -> AppEpisodesResponse:
    """Catalog: episodes across the corpus, newest-first (PRD-038 FR1).

    Served via the pluggable ``ContentSource`` (local corpus for the MVP; #1069 extends
    it). Per-artifact depth counts are not computed here — see the detail endpoints.
    """
    return _episodes_page(request, feed_id=feed_id, status=status, page=page, page_size=page_size)


@router.get("/podcasts", response_model=AppPodcastsResponse)
async def podcasts_list(request: Request) -> AppPodcastsResponse:
    """Distinct shows in the corpus, for Home 'Your shows' (PRD-042 FR6)."""
    root = corpus_root_or_503(request)
    feeds = aggregate_feeds(build_catalog_rows_cumulative(root))
    items = [
        AppPodcastItem(
            feed_id=f["feed_id"],
            title=f.get("display_title"),
            artwork_url=artwork_url(f.get("image_local_relpath"), "thumb"),
            image_url=f.get("image_url"),
            description=f.get("description"),
            episode_count=int(f.get("episode_count", 0)),
        )
        for f in feeds
        if f.get("feed_id")
    ]
    return AppPodcastsResponse(items=items)


@router.get("/podcasts/{feed_id}/episodes", response_model=AppEpisodesResponse)
async def podcast_episodes_list(
    request: Request,
    feed_id: str,
    page: int = Query(default=1, ge=1, description="1-based page index."),
    page_size: int = Query(default=20, ge=1, le=100, description="Episodes per page."),
    status: str | None = Query(
        default=None, description="Optional status filter: 'ready' or 'pending'."
    ),
) -> AppEpisodesResponse:
    """Catalog: one podcast's episodes, newest-first (PRD-038 FR2)."""
    return _episodes_page(request, feed_id=feed_id, status=status, page=page, page_size=page_size)


@router.get("/episodes/{slug}", response_model=AppEpisodeDetail)
async def episode_detail(request: Request, slug: str) -> AppEpisodeDetail:
    """Consumer episode detail (metadata + summary + artifact-availability flags)."""
    root, row = _resolve(request, slug)
    transcript_rel = transcript_relpath(_content_block(root, row.metadata_relative_path))
    has_transcript = transcript_rel is not None
    has_summary = bool(row.summary_title or row.summary_bullets or row.summary_text)
    local_art = row.episode_image_local_relpath or row.feed_image_local_relpath
    return AppEpisodeDetail(
        slug=slug,
        title=row.episode_title,
        feed_id=row.feed_id,
        podcast_title=row.feed_title,
        publish_date=row.publish_date,
        duration_seconds=row.duration_seconds,
        episode_image_url=row.episode_image_url,
        feed_image_url=row.feed_image_url,
        artwork_url=artwork_url(local_art, "large"),
        summary_title=row.summary_title,
        summary_bullets=list(row.summary_bullets),
        summary_text=row.summary_text,
        has_transcript=has_transcript,
        has_summary=has_summary,
        has_gi=row.has_gi,
        has_kg=row.has_kg,
        has_bridge=row.has_bridge,
    )


@router.get("/episodes/{slug}/related", response_model=AppEpisodesResponse)
async def episode_related(
    request: Request,
    slug: str,
    top_k: int = Query(default=8, ge=1, le=25, description="Max 'more like this' peers."),
) -> AppEpisodesResponse:
    """ "More like this" — semantic peer episodes via the vector index (RFC-099; #1084 follow-up).

    Reuses the same similarity engine as the operator library, projected to the consumer
    card shape. Returns 200 with empty items when the index is unavailable (graceful, same as
    no-index search) so the panel section simply hides.
    """
    root, row = _resolve(request, slug)
    outcome = run_similar_episodes(
        root,
        summary_title=row.summary_title,
        summary_bullets=row.summary_bullets,
        episode_title=row.episode_title,
        source_feed_id=row.feed_id,
        source_episode_id=row.episode_id,
        top_k=top_k,
    )
    if outcome.error:
        return AppEpisodesResponse(items=[], page=1, page_size=top_k, total=0, has_more=False)

    by_scope = index_rows_by_feed_episode(build_catalog_rows_cumulative(root))
    items = []
    seen: set[str] = {row.metadata_relative_path}
    for it in outcome.items:
        key = episode_scope_key(dict(it.get("metadata") or {}))
        peer = by_scope.get(key) if key else None
        if peer is not None and peer.metadata_relative_path not in seen:
            seen.add(peer.metadata_relative_path)
            items.append(row_to_summary(root, peer))
    return AppEpisodesResponse(
        items=items, page=1, page_size=top_k, total=len(items), has_more=False
    )


@router.get("/episodes/{slug}/insights", response_model=AppInsightsResponse)
async def episode_insights(request: Request, slug: str) -> AppInsightsResponse:
    """Grounded GIL insights (with supporting quotes) for one episode.

    Returns an empty list (200) when the episode has no GI artifact — graceful
    degradation, not an error.
    """
    root, row = _resolve(request, slug)
    if not row.has_gi:
        return AppInsightsResponse(episode_slug=slug, insights=[])
    artifact = load_json_artifact(root, row.gi_relative_path)
    return AppInsightsResponse(episode_slug=slug, insights=insights_from_gi(artifact))


@router.get("/episodes/{slug}/entities", response_model=AppEntitiesResponse)
async def episode_entities(request: Request, slug: str) -> AppEntitiesResponse:
    """KG persons, organisations, and topics for one episode (empty when no KG)."""
    root, row = _resolve(request, slug)
    if not row.has_kg:
        return AppEntitiesResponse(episode_slug=slug)
    persons, orgs, topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
    # Cluster-first grouping (RFC-102 / PRD-043 FR1): attach corpus topic-cluster identity to each
    # topic (no-op when search/topic_clusters.json is absent → flat list, today's behaviour).
    cluster_map = consumer_topic_cluster_map(root)
    if cluster_map:
        topics = [
            t.model_copy(update=cluster_map[t.id]) if t.id in cluster_map else t for t in topics
        ]
    return AppEntitiesResponse(episode_slug=slug, persons=persons, orgs=orgs, topics=topics)


@router.get("/episodes/{slug}/segments", response_model=SegmentsResponse)
async def episode_segments(request: Request, slug: str) -> SegmentsResponse:
    """Serve the transcript ``segments.json`` contract for one episode (by slug)."""
    root, row = _resolve(request, slug)
    transcript_rel = transcript_relpath(_content_block(root, row.metadata_relative_path))
    if transcript_rel is None:
        raise HTTPException(status_code=404, detail="Transcript not available for this episode.")

    # transcript_file_path is relative to the run dir (parent of the metadata dir); resolve it
    # to a corpus-root-relative path before deriving the adjacent segments file.
    transcript_corpus_rel = transcript_corpus_relpath(row.metadata_relative_path, transcript_rel)
    for candidate in segments_relpaths_for_transcript(transcript_corpus_rel):
        safe = safe_relpath_under_corpus_root(root, candidate)
        if not safe:
            continue
        seg_path = root / safe
        if seg_path.is_file():
            try:
                raw = json.loads(seg_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Unreadable segments file %s: %s", seg_path, exc)
                raise HTTPException(status_code=500, detail="Segments file unreadable.") from exc
            return SegmentsResponse(episode_slug=slug, segments=to_contract_segments(raw))

    raise HTTPException(
        status_code=404, detail="Transcript segments not available for this episode."
    )


@router.get("/episodes/{slug}/audio-source", response_model=AudioSourceResponse)
async def episode_audio_source(
    request: Request,
    slug: str,
    validate: bool = Query(
        default=False,
        description="HEAD-validate the origin URL and resolve redirects (adds a network call).",
    ),
) -> AudioSourceResponse:
    """Resolve the origin enclosure URL the client plays directly (bridge, never rehost).

    Reads the already-persisted ``content.media_url`` (RFC-100; G5 resolved); ``strategy``
    is always ``direct``. With ``validate=true`` a HEAD follows redirects and reports the
    resolved final URL + reachability + content-length. The no-store proxy stays deferred.
    """
    root, row = _resolve(request, slug)
    content = _content_block(root, row.metadata_relative_path)
    media_url = content.get("media_url")
    if not isinstance(media_url, str) or not media_url.strip():
        raise HTTPException(status_code=404, detail="No origin audio URL for this episode.")
    url = media_url.strip()

    mime_raw = content.get("media_type")
    mime = mime_raw if isinstance(mime_raw, str) and mime_raw.strip() else None
    media_id = content.get("media_id")

    resolved_url: str | None = None
    verified: bool | None = None
    content_length: int | None = None
    if validate:
        resolution = resolve_audio(url)
        verified = resolution.verified
        resolved_url = resolution.final_url
        content_length = resolution.content_length
        if mime is None and resolution.content_type:
            mime = resolution.content_type

    return AudioSourceResponse(
        episode_slug=slug,
        url=url,
        mime=mime,
        duration_seconds=row.duration_seconds,
        media_id=media_id if isinstance(media_id, str) and media_id.strip() else None,
        strategy="direct",
        resolved_url=resolved_url,
        verified=verified,
        content_length=content_length,
    )


@router.get("/episodes/{slug}/search", response_model=CorpusSearchApiResponse)
async def episode_search(
    request: Request,
    slug: str,
    q: str = Query(min_length=1, description="Natural-language query."),
    top_k: int = Query(default=10, ge=1, le=100),
) -> CorpusSearchApiResponse:
    """Grounded search within one episode (extractive grounded passages; no LLM, D6).

    Over-fetches by feed, then narrows to this episode — the retrieval layer has no
    episode filter yet, so we scope ``metadata.episode_id`` client-side.
    """
    root, row = _resolve(request, slug)
    internal_k = min(100, max(top_k, top_k * 5))
    outcome = structured_corpus_search(root, q, feed=row.feed_id or None, top_k=internal_k)
    if not outcome.get("error"):
        append_query_event(root, str(outcome.get("query_type") or ""))
    scoped = filter_outcome_to_episode(outcome, row.episode_id, top_k)
    return build_search_response(q, scoped)
