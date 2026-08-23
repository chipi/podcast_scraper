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

from podcast_scraper.server import app_graph_refs, app_user_state
from podcast_scraper.server.app_capture_export import (
    EpisodeHighlights,
    HighlightLine,
    render_highlights_markdown,
)
from podcast_scraper.server.app_corpus_access import (
    corpus_root_or_503,
    safe_relpath_under_corpus_root,
)
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
from podcast_scraper.server.segments_view import (
    segments_relpaths_for_transcript,
    to_contract_segments,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    # get_current_user has already guaranteed app_data_dir is configured.
    return Path(request.app.state.app_data_dir)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _corpus_root_opt(request: Request) -> Path | None:
    """The corpus root, or None when unavailable — capture must not fail on a missing corpus."""
    try:
        return corpus_root_or_503(request)
    except HTTPException:
        return None


# --- highlights ---------------------------------------------------------------


def _contract_segments(root: Path, slug: str) -> list[dict] | None:
    """The player transcript contract for one episode, or None when unavailable.

    Mirrors GET /episodes/{slug}/segments, but never raises: re-anchoring is a best-effort read
    over the shared corpus and must not turn a working /highlights into a 500 because one
    episode's transcript is missing or unreadable.
    """
    import json

    from podcast_scraper.server.app_content_source import (
        transcript_corpus_relpath,
        transcript_relpath,
    )
    from podcast_scraper.server.app_corpus_access import load_json_artifact

    row = resolve_slug(root, slug)
    if row is None:
        return None
    try:
        doc = load_json_artifact(root, row.metadata_relative_path) or {}
        content = doc.get("content") if isinstance(doc, dict) else None
        transcript_rel = transcript_relpath(content if isinstance(content, dict) else {})
        if transcript_rel is None:
            return None
        corpus_rel = transcript_corpus_relpath(row.metadata_relative_path, transcript_rel)
        for candidate in segments_relpaths_for_transcript(corpus_rel):
            safe = safe_relpath_under_corpus_root(root, candidate)
            if not safe:
                continue
            path = root / safe
            if path.is_file():
                raw = json.loads(path.read_text(encoding="utf-8"))
                return [seg.model_dump() for seg in to_contract_segments(raw)]
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        logger.debug("re-anchor: segments unavailable for %s: %s", slug, exc)
    return None


def _reanchored(root: Path, rows: list[dict]) -> list[dict]:
    """Re-anchor every highlight against the CURRENT transcript (RFC-098: computed on read).

    Segment ids are positional — to_contract_segments mints ``seg_{index}`` from list position — so
    a re-scrape that inserts or drops a segment renumbers every later id. Serving the stored ids
    unchanged made the client highlight the WRONG paragraph as saved, silently, and the drift badge
    it renders could never appear because nothing ever set ``anchor_status``.

    Read-time, not persisted: the anchor is derived from whatever the transcript says right now, so
    there is no write amplification and a transcript that gets fixed re-anchors on the next read.
    One segments load per DISTINCT episode, not per highlight.
    """
    by_slug: dict[str, list[dict]] = {}
    for row in rows:
        by_slug.setdefault(str(row.get("episode_slug") or ""), []).append(row)
    out: list[dict] = []
    for slug, group in by_slug.items():
        segments = _contract_segments(root, slug) if slug else None
        if segments is None:
            out.extend(group)  # nothing to re-anchor against; serve what we stored
            continue
        out.extend(app_user_state.reanchor_highlight(row, segments) for row in group)
    # Preserve the store's newest-last ordering rather than the grouping order.
    order = {id(r): i for i, r in enumerate(rows)}
    by_id = {str(r.get("id")): r for r in rows}
    return sorted(out, key=lambda r: order.get(id(by_id.get(str(r.get("id")))), 0))


@router.get("/highlights", response_model=HighlightsResponse)
async def list_highlights(
    request: Request, episode: str | None = None, user: User = Depends(get_current_user)
) -> HighlightsResponse:
    """The user's highlights, optionally scoped to one episode (``?episode=<slug>``).

    Re-anchored against the current transcript on the way out (RFC-098 / PRD-040 FR3.1a).
    """
    rows = app_user_state.get_highlights(_data_dir(request), user.user_id, episode)
    root = _corpus_root_opt(request)
    if root is not None and rows:
        rows = _reanchored(root, rows)
    return HighlightsResponse(items=[Highlight(**r) for r in rows])


@router.post("/highlights", response_model=Highlight, status_code=201)
async def create_highlight(
    request: Request, body: HighlightCreate, user: User = Depends(get_current_user)
) -> Highlight:
    """Capture a highlight (span / moment / insight); mints id + created_at + graph refs."""
    record = body.model_dump()
    record["id"] = _new_id("h")
    record["created_at"] = int(time.time())
    # Resolve + persist the highlight's canonical graph refs at capture (#1419) so every outbound
    # surface carries the graph. Best-effort: a missing/KG-less corpus just yields no refs.
    root = _corpus_root_opt(request)
    if root is not None:
        record["graph_refs"] = app_graph_refs.refs_for_slug(root, str(record.get("episode_slug")))
    app_user_state.add_highlight(_data_dir(request), user.user_id, record)
    return Highlight(**record)


@router.patch("/highlights/{highlight_id}", response_model=Highlight)
async def patch_highlight(
    request: Request,
    highlight_id: str,
    body: HighlightUpdate,
    user: User = Depends(get_current_user),
) -> Highlight:
    """Edit a highlight's colour / captured text (404 if it does not exist).

    Uses ``exclude_unset`` (not ``exclude_none``) so an explicit ``"color": null`` *clears* the
    colour, while an omitted field is left unchanged — the correct PATCH semantics.
    """
    fields = body.model_dump(exclude_unset=True)
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
    """Remove a highlight by id (no-op if absent), WITH its notes; returns the remaining list.

    The notes go too. They used to survive server-side while the client pruned them locally, so
    they looked deleted and then resurrected on the next full load — the user is told the note is
    gone and it is not. A note on a highlight is an annotation of that anchor; once the anchor is
    gone there is nothing for it to annotate, and the client's existing local filter is the
    intent this now implements for real.
    """
    data_dir = _data_dir(request)
    rows = app_user_state.remove_highlight(data_dir, user.user_id, highlight_id)
    app_user_state.remove_notes_for_target(data_dir, user.user_id, "highlight", highlight_id)
    # The resurfacing schedule goes too (#39). Nothing reads an orphaned entry — select_due
    # iterates highlights and looks state up — so this is unbounded growth, not a wrong answer:
    # one dead key per deleted capture, for ever. It also left resurfacing.json as the one
    # per-user file where a deleted capture still had a trace.
    app_user_state.remove_resurfacing_state(data_dir, user.user_id, highlight_id)
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


class MarkdownResponse(PlainTextResponse):
    """A text response that DOCUMENTS the media type it actually sends.

    The handler below overrides ``media_type`` to ``text/markdown``, but a plain
    ``response_class=PlainTextResponse`` still advertises ``text/plain`` in the OpenAPI schema —
    so the published contract described something the endpoint never returns. Declaring it on the
    response class keeps the two in step, instead of the schema and the wire drifting apart.
    """

    media_type = "text/markdown; charset=utf-8"


@router.get(
    "/highlights/export.md",
    response_class=MarkdownResponse,
    responses={
        200: {"description": "All of the user's highlights, grouped by episode, as Markdown."}
    },
)
async def export_highlights_markdown(
    request: Request, user: User = Depends(get_current_user)
) -> PlainTextResponse:
    """Export all of the user's highlights AND every note, as a Markdown document.

    "With attached notes" used to mean only notes on a highlight: ``notes_by_target`` was consumed
    solely by highlight id, so a note the user wrote on an EPISODE or on a saved INSIGHT was
    silently absent from their export. An export that quietly drops the user's own writing is worse
    than one that never offered it — episode notes now sit under their episode's heading, and
    anything whose target this renderer cannot place goes to a trailing "Other notes" section.
    """
    data_dir = _data_dir(request)
    highlights = app_user_state.get_highlights(data_dir, user.user_id)
    notes = app_user_state.get_notes(data_dir, user.user_id)
    notes_by_target: dict[str, list[str]] = {}
    for n in notes:
        notes_by_target.setdefault(str(n.get("target_id")), []).append(str(n.get("text", "")))

    highlight_ids = {str(h.get("id")) for h in highlights}
    episode_note_slugs = {
        str(n.get("target_id"))
        for n in notes
        if n.get("target") == "episode" and n.get("target_id")
    }
    # Every episode that needs a heading: one the user highlighted, or one they only made a note on.
    titles = _episode_titles(
        request, {str(h.get("episode_slug")) for h in highlights} | episode_note_slugs
    )

    grouped: "OrderedDict[str, EpisodeHighlights]" = OrderedDict()

    def _episode(slug: str) -> EpisodeHighlights:
        if slug not in grouped:
            title, show = titles.get(slug, (None, None))
            grouped[slug] = EpisodeHighlights(slug=slug, title=title, show=show)
        return grouped[slug]

    for h in highlights:
        _episode(str(h.get("episode_slug"))).highlights.append(
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

    for slug in episode_note_slugs:
        _episode(slug).episode_notes.extend(notes_by_target.get(slug, []))

    # Whatever is left: a note on a saved insight, whose target id is an insight, not an episode.
    # There is no insight -> episode mapping here, so rather than drop it, it gets its own section.
    placed = highlight_ids | episode_note_slugs
    orphans = [str(n.get("text", "")) for n in notes if str(n.get("target_id")) not in placed]

    markdown = render_highlights_markdown(list(grouped.values()), orphans)
    return PlainTextResponse(
        markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="my-highlights.md"'},
    )
