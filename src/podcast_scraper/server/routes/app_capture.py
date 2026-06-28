"""P2 Capture routes — highlights, notes, Markdown export (#1115, PRD-040 / RFC-098 §7).

All auth-gated by ``get_current_user`` and scoped to the signed-in user's plain files under
``<data_dir>/users/<id>/``. No DB; the personal overlay only. The route mints opaque ids and
timestamps; the store stays pure (RFC-098 §3).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_capture_export import (
    EpisodeHighlights,
    HighlightLine,
    render_highlights_markdown,
)
from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    Highlight,
    HighlightCreate,
    HighlightsResponse,
    HighlightUpdate,
    Note,
    NoteCreate,
    NotesResponse,
    NoteUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    # get_current_user has already guaranteed app_data_dir is configured.
    return Path(request.app.state.app_data_dir)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --- highlights ---------------------------------------------------------------


@router.get("/highlights", response_model=HighlightsResponse)
async def list_highlights(
    request: Request, episode: str | None = None, user: User = Depends(get_current_user)
) -> HighlightsResponse:
    """The user's highlights, optionally scoped to one episode (``?episode=<slug>``)."""
    rows = app_user_state.get_highlights(_data_dir(request), user.user_id, episode)
    return HighlightsResponse(items=[Highlight(**r) for r in rows])


@router.post("/highlights", response_model=Highlight, status_code=201)
async def create_highlight(
    request: Request, body: HighlightCreate, user: User = Depends(get_current_user)
) -> Highlight:
    """Capture a highlight (span / moment / insight); mints id + created_at."""
    record = body.model_dump()
    record["id"] = _new_id("h")
    record["created_at"] = int(time.time())
    app_user_state.add_highlight(_data_dir(request), user.user_id, record)
    return Highlight(**record)


@router.patch("/highlights/{highlight_id}", response_model=Highlight)
async def patch_highlight(
    request: Request,
    highlight_id: str,
    body: HighlightUpdate,
    user: User = Depends(get_current_user),
) -> Highlight:
    """Edit a highlight's colour / captured text (404 if it does not exist)."""
    fields = body.model_dump(exclude_none=True)
    updated = app_user_state.update_highlight(
        _data_dir(request), user.user_id, highlight_id, fields
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="highlight not found")
    return Highlight(**updated)


@router.delete("/highlights/{highlight_id}", response_model=HighlightsResponse)
async def delete_highlight(
    request: Request, highlight_id: str, user: User = Depends(get_current_user)
) -> HighlightsResponse:
    """Remove a highlight by id (no-op if absent); returns the remaining list."""
    rows = app_user_state.remove_highlight(_data_dir(request), user.user_id, highlight_id)
    return HighlightsResponse(items=[Highlight(**r) for r in rows])


# --- notes --------------------------------------------------------------------


@router.get("/notes", response_model=NotesResponse)
async def list_notes(
    request: Request,
    target: str | None = None,
    target_id: str | None = None,
    user: User = Depends(get_current_user),
) -> NotesResponse:
    """The user's notes, optionally scoped to one ``?target=&target_id=``."""
    rows = app_user_state.get_notes(_data_dir(request), user.user_id, target, target_id)
    return NotesResponse(items=[Note(**r) for r in rows])


@router.post("/notes", response_model=Note, status_code=201)
async def create_note(
    request: Request, body: NoteCreate, user: User = Depends(get_current_user)
) -> Note:
    """Attach a free-text note to a highlight / insight / episode; mints id + timestamps."""
    now = int(time.time())
    record = body.model_dump()
    record.update({"id": _new_id("n"), "created_at": now, "updated_at": now})
    app_user_state.add_note(_data_dir(request), user.user_id, record)
    return Note(**record)


@router.patch("/notes/{note_id}", response_model=Note)
async def patch_note(
    request: Request, note_id: str, body: NoteUpdate, user: User = Depends(get_current_user)
) -> Note:
    """Edit a note's text (404 if it does not exist)."""
    updated = app_user_state.update_note(
        _data_dir(request), user.user_id, note_id, body.text, int(time.time())
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="note not found")
    return Note(**updated)


@router.delete("/notes/{note_id}", response_model=NotesResponse)
async def delete_note(
    request: Request, note_id: str, user: User = Depends(get_current_user)
) -> NotesResponse:
    """Remove a note by id (no-op if absent); returns the remaining list."""
    rows = app_user_state.remove_note(_data_dir(request), user.user_id, note_id)
    return NotesResponse(items=[Note(**r) for r in rows])


# --- Markdown export ----------------------------------------------------------


def _episode_titles(request: Request, slugs: set[str]) -> dict[str, tuple[str | None, str | None]]:
    """Best-effort (title, show) per slug; never breaks export when the corpus is unavailable."""
    out: dict[str, tuple[str | None, str | None]] = {}
    try:
        root = corpus_root_or_503(request)
    except Exception:  # noqa: BLE001 — export must still render with bare slugs.
        return out
    for slug in slugs:
        try:
            row = resolve_slug(root, slug)
        except Exception:  # noqa: BLE001
            row = None
        if row is not None:
            out[slug] = (row.episode_title, row.feed_title)
    return out


@router.get("/highlights/export.md", response_class=PlainTextResponse)
async def export_highlights_markdown(
    request: Request, user: User = Depends(get_current_user)
) -> PlainTextResponse:
    """Export all of the user's highlights (with attached notes) as a Markdown document."""
    data_dir = _data_dir(request)
    highlights = app_user_state.get_highlights(data_dir, user.user_id)
    notes = app_user_state.get_notes(data_dir, user.user_id)
    notes_by_target: dict[str, list[str]] = {}
    for n in notes:
        notes_by_target.setdefault(str(n.get("target_id")), []).append(str(n.get("text", "")))

    titles = _episode_titles(request, {str(h.get("episode_slug")) for h in highlights})

    grouped: "OrderedDict[str, EpisodeHighlights]" = OrderedDict()
    for h in highlights:
        slug = str(h.get("episode_slug"))
        if slug not in grouped:
            title, show = titles.get(slug, (None, None))
            grouped[slug] = EpisodeHighlights(slug=slug, title=title, show=show)
        grouped[slug].highlights.append(
            HighlightLine(
                kind=str(h.get("kind", "span")),
                start_ms=h.get("start_ms"),
                end_ms=h.get("end_ms"),
                quote_text=h.get("quote_text"),
                speaker=h.get("speaker"),
                color=h.get("color"),
                anchor_status=h.get("anchor_status"),
                notes=notes_by_target.get(str(h.get("id")), []),
            )
        )

    markdown = render_highlights_markdown(list(grouped.values()))
    return PlainTextResponse(
        markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="my-highlights.md"'},
    )
