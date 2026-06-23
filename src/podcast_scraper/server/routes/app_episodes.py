"""Consumer Learning Platform episode routes (``/api/app/episodes/*``).

Read-only, slug-addressed surface for the end-user app (RFC-098 §4–§5; #1067/#1070).
Mounted under the ``/api/app`` namespace, separate from the operator routes; access
becomes auth-gated in later Epic-1 tasks (#1063/#1066).
Serves the single shared corpus at ``app.state.output_dir`` — there is no ``?path``
override (that is an operator concern; see the operator ``/api/corpus/*`` routes).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.corpus_catalog import _load_metadata_doc
from podcast_scraper.server.schemas import AudioSourceResponse, SegmentsResponse
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


def _content_block(root: Path, metadata_relpath: str) -> dict:
    """Load an episode's ``content`` metadata block (empty dict when absent)."""
    doc = _load_metadata_doc(root / metadata_relpath)
    content = doc.get("content") if isinstance(doc, dict) else None
    return content if isinstance(content, dict) else {}


@router.get("/episodes/{slug}/segments", response_model=SegmentsResponse)
async def episode_segments(request: Request, slug: str) -> SegmentsResponse:
    """Serve the transcript ``segments.json`` contract for one episode (by slug)."""
    root = _corpus_root(request)
    row = resolve_slug(root, slug)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown episode slug.")

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
    root = _corpus_root(request)
    row = resolve_slug(root, slug)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown episode slug.")

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
