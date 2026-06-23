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
from podcast_scraper.server.app_gi_view import insights_from_gi
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_search_view import build_search_response, filter_outcome_to_episode
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.corpus_catalog import _load_metadata_doc, CatalogEpisodeRow
from podcast_scraper.server.schemas import (
    AppEntitiesResponse,
    AppEpisodeDetail,
    AppInsightsResponse,
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


def _corpus_root(request: Request) -> Path:
    """Resolve the single shared corpus root, or 503 if the platform has no corpus."""
    anchor = getattr(request.app.state, "output_dir", None)
    if anchor is None:
        raise HTTPException(status_code=503, detail="No corpus configured for the platform API.")
    return Path(anchor)


def _resolve(request: Request, slug: str) -> tuple[Path, CatalogEpisodeRow]:
    """Resolve ``(corpus_root, row)`` for a slug, or 404 when the slug is unknown."""
    root = _corpus_root(request)
    row = resolve_slug(root, slug)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown episode slug.")
    return root, row


def _content_block(root: Path, metadata_relpath: str) -> dict:
    """Load an episode's ``content`` metadata block (empty dict when absent)."""
    doc = _load_metadata_doc(root / metadata_relpath)
    content = doc.get("content") if isinstance(doc, dict) else None
    return content if isinstance(content, dict) else {}


def _load_artifact(root: Path, relpath: str) -> dict | None:
    """Path-safe JSON load of a corpus artifact (GI/KG); ``None`` when missing/unreadable."""
    if not relpath:
        return None
    safe = safe_relpath_under_corpus_root(root, relpath)
    if not safe:
        return None
    path = root / safe
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable artifact %s: %s", path, exc)
        return None
    return loaded if isinstance(loaded, dict) else None


@router.get("/episodes/{slug}", response_model=AppEpisodeDetail)
async def episode_detail(request: Request, slug: str) -> AppEpisodeDetail:
    """Consumer episode detail (metadata + summary + artifact-availability flags)."""
    root, row = _resolve(request, slug)
    transcript_rel = _content_block(root, row.metadata_relative_path).get("transcript_file")
    has_transcript = isinstance(transcript_rel, str) and bool(transcript_rel.strip())
    has_summary = bool(row.summary_title or row.summary_bullets or row.summary_text)
    return AppEpisodeDetail(
        slug=slug,
        title=row.episode_title,
        feed_id=row.feed_id,
        podcast_title=row.feed_title,
        publish_date=row.publish_date,
        duration_seconds=row.duration_seconds,
        episode_image_url=row.episode_image_url,
        feed_image_url=row.feed_image_url,
        summary_title=row.summary_title,
        summary_bullets=list(row.summary_bullets),
        summary_text=row.summary_text,
        has_transcript=has_transcript,
        has_summary=has_summary,
        has_gi=row.has_gi,
        has_kg=row.has_kg,
        has_bridge=row.has_bridge,
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
    artifact = _load_artifact(root, row.gi_relative_path)
    return AppInsightsResponse(episode_slug=slug, insights=insights_from_gi(artifact))


@router.get("/episodes/{slug}/entities", response_model=AppEntitiesResponse)
async def episode_entities(request: Request, slug: str) -> AppEntitiesResponse:
    """KG persons, organisations, and topics for one episode (empty when no KG)."""
    root, row = _resolve(request, slug)
    if not row.has_kg:
        return AppEntitiesResponse(episode_slug=slug)
    persons, orgs, topics = entities_from_kg(_load_artifact(root, row.kg_relative_path))
    return AppEntitiesResponse(episode_slug=slug, persons=persons, orgs=orgs, topics=topics)


@router.get("/episodes/{slug}/segments", response_model=SegmentsResponse)
async def episode_segments(request: Request, slug: str) -> SegmentsResponse:
    """Serve the transcript ``segments.json`` contract for one episode (by slug)."""
    root, row = _resolve(request, slug)
    transcript_rel = _content_block(root, row.metadata_relative_path).get("transcript_file")
    if not isinstance(transcript_rel, str) or not transcript_rel.strip():
        raise HTTPException(status_code=404, detail="Transcript not available for this episode.")

    for candidate in segments_relpaths_for_transcript(transcript_rel):
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
async def episode_audio_source(request: Request, slug: str) -> AudioSourceResponse:
    """Resolve the origin enclosure URL the client plays directly (bridge, never rehost).

    Reads the already-persisted ``content.media_url`` (RFC-100; G5 resolved). Freshness
    re-resolution and the optional no-store proxy are deferred follow-ups on #1070.
    """
    root, row = _resolve(request, slug)
    content = _content_block(root, row.metadata_relative_path)
    media_url = content.get("media_url")
    if not isinstance(media_url, str) or not media_url.strip():
        raise HTTPException(status_code=404, detail="No origin audio URL for this episode.")

    mime = content.get("media_type")
    media_id = content.get("media_id")
    return AudioSourceResponse(
        episode_slug=slug,
        url=media_url.strip(),
        mime=mime if isinstance(mime, str) and mime.strip() else None,
        duration_seconds=row.duration_seconds,
        media_id=media_id if isinstance(media_id, str) and media_id.strip() else None,
        strategy="direct",
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
    scoped = filter_outcome_to_episode(outcome, row.episode_id, top_k)
    return build_search_response(q, scoped)
