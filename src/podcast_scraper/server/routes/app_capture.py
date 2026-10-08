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
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, PlainTextResponse

from podcast_scraper.server import app_graph_refs, app_pkm_export, app_user_state
from podcast_scraper.server.app_capture_export import (
    EpisodeHighlights,
    format_note,
    HighlightLine,
    render_highlights_html,
    render_highlights_markdown,
)
from podcast_scraper.server.app_corpus_access import (
    corpus_root_or_503,
    safe_relpath_under_corpus_root,
)
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes import app_collections
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


def _matches(needle: str, *texts: object) -> bool:
    """Case-insensitive substring, the client's ``matchesQuery``; an empty needle matches all."""
    return not needle or any(needle in str(t or "").casefold() for t in texts)


def _paged_highlights(
    request: Request,
    rows: list[dict],
    user: User,
    *,
    q: str | None,
    color: str | None,
    muted: bool,
    sort: str,
    offset: int,
    limit: int,
    per_episode: int,
) -> HighlightsResponse:
    """One page of EPISODES with their highlights — Saved's Highlights section, on the server.

    Filtering and grouping read only the stored rows; re-anchoring (a transcript load per episode)
    runs for the page alone, which is what made the full list expensive.
    """
    data_dir = _data_dir(request)
    state = app_user_state.get_resurfacing_state(data_dir, user.user_id)
    needle = (q or "").strip().casefold()
    groups: dict[str, list[dict]] = {}
    matched = 0
    for row in rows:
        st = state.get(str(row.get("id") or ""))
        row["retired"] = bool(st.get("retired")) if isinstance(st, dict) else False
        if color and row.get("color") != color:
            continue
        if muted and not row["retired"]:
            continue
        if not _matches(needle, row.get("quote_text"), row.get("speaker")):
            continue
        matched += 1
        groups.setdefault(str(row.get("episode_slug") or ""), []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: int(r.get("created_at") or 0), reverse=True)
    root = _corpus_root_opt(request)
    order = list(groups)
    if sort == "title":

        def title(slug: str) -> str:
            row = resolve_slug(root, slug) if root is not None and slug else None
            return (row.episode_title if row is not None else slug).casefold()

        order.sort(key=title)
    else:
        order.sort(key=lambda slug: int(groups[slug][0].get("created_at") or 0), reverse=True)
    page = order[offset : offset + limit]
    items = [row for slug in page for row in groups[slug][:per_episode]]
    if root is not None and items:
        items = _reanchored(root, items)
    ids = {str(r.get("id")) for r in items}
    notes = [
        Note(**n)
        for n in app_user_state.get_notes(data_dir, user.user_id, "highlight")
        if str(n.get("target_id")) in ids
    ]
    return HighlightsResponse(
        items=[Highlight(**r) for r in items],
        total=matched,
        episode_total=len(order),
        episode_counts={slug: len(groups[slug]) for slug in page},
        notes=notes,
    )


@router.get("/highlights", response_model=HighlightsResponse)
def list_highlights(
    request: Request,
    episode: str | None = None,
    q: str | None = Query(default=None, max_length=200, description="Paged: quote or speaker."),
    color: str | None = Query(default=None, max_length=32, description="Paged: only this colour."),
    muted: bool = Query(default=False, description="Paged: only highlights stopped resurfacing."),
    sort: Literal["recent", "title"] = Query(default="recent", description="Paged: episode order."),
    offset: int = Query(default=0, ge=0, description="Paged: episodes to skip."),
    limit: int | None = Query(
        default=None,
        ge=1,
        le=100,
        description="EPISODES per page. Absent: every highlight, unfiltered (before 1.0.3).",
    ),
    per_episode: int = Query(default=5, ge=1, le=100, description="Paged: highlights per episode."),
    user: User = Depends(get_current_user),
) -> HighlightsResponse:
    """The user's highlights, optionally scoped to one episode (``?episode=<slug>``).

    Re-anchored against the current transcript on the way out (RFC-098 / PRD-040 FR3.1a).

    ``retired`` is joined from the resurfacing state here rather than stored on the highlight, for
    the same reason ``anchor_status`` is computed on read: it is a fact about the SCHEDULE, and
    duplicating it onto the capture would give two places to disagree. Saved needs it because
    retiring must be reversible somewhere, and Saved is the only surface that lists every capture
    — a retired highlight is by definition absent from Revisit, so it cannot be undone there.
    """
    data_dir = _data_dir(request)
    rows = app_user_state.get_highlights(data_dir, user.user_id, episode)
    if limit is not None:
        return _paged_highlights(
            request,
            rows,
            user,
            q=q,
            color=color,
            muted=muted,
            sort=sort,
            offset=offset,
            limit=limit,
            per_episode=per_episode,
        )
    root = _corpus_root_opt(request)
    if root is not None and rows:
        rows = _reanchored(root, rows)
    state = app_user_state.get_resurfacing_state(data_dir, user.user_id)
    for row in rows:
        st = state.get(str(row.get("id") or ""))
        # Mirrors select_due's tolerance: this file is hand-editable and may hold a non-mapping.
        row["retired"] = bool(st.get("retired")) if isinstance(st, dict) else False
    return HighlightsResponse(items=[Highlight(**r) for r in rows])


@router.post("/highlights", response_model=Highlight, status_code=201)
def create_highlight(
    request: Request,
    body: HighlightCreate,
    response: Response,
    user: User = Depends(get_current_user),
) -> Highlight:
    """Capture a highlight (span / moment / insight); mints id + created_at + graph refs.

    Idempotent when the client supplies ``client_id`` (#1925). Capture is append-only, so a POST
    whose RESPONSE was lost — the common offline case, since the write may well have landed —
    could only be retried by risking a duplicate. That is why highlights and notes stayed out of
    the offline outbox. With a client-minted id the first write wins and the retry returns the
    existing row, so the outbox can replay them like any other write.

    A replay answers 200 rather than 201: nothing was created this time, and the client can tell
    the two apart.
    """
    record = body.model_dump(exclude={"client_id"})
    record["id"] = body.client_id or _new_id("h")
    record["created_at"] = int(time.time())
    # Resolve + persist the highlight's canonical graph refs at capture (#1419) so every outbound
    # surface carries the graph. Best-effort: a missing/KG-less corpus just yields no refs.
    root = _corpus_root_opt(request)
    if root is not None:
        record["graph_refs"] = app_graph_refs.refs_for_slug(root, str(record.get("episode_slug")))
    stored, created = app_user_state.add_highlight_if_absent(
        _data_dir(request), user.user_id, record
    )
    if not created:
        response.status_code = 200
    return Highlight(**stored)


@router.patch("/highlights/{highlight_id}", response_model=Highlight)
def patch_highlight(
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


@router.get(
    "/highlights/{highlight_id}/card.png",
    responses={200: {"content": {"image/png": {}}, "description": "The highlight's share card."}},
)
def highlight_card(
    request: Request, highlight_id: str, user: User = Depends(get_current_user)
) -> Response:
    """The share card for one of the user's highlights — the quote card (operator 2026-10-05).

    Drawn by the same server renderer as every other card (``server/og/card.py``), so a shared
    highlight looks like a shared episode or topic. Signed-in and per-user: a highlight is private,
    so unlike ``/og/{kind}/{id}.png`` this is never public and never unfurled. 404 for a highlight
    that is not yours or whose episode has left the corpus; 503 when the renderer is unavailable.
    """
    from podcast_scraper.server.og.build import build_highlight_card
    from podcast_scraper.server.og.card import render_card_png

    rows = app_user_state.get_highlights(_data_dir(request), user.user_id, None)
    row = next((r for r in rows if str(r.get("id")) == highlight_id), None)
    root = _corpus_root_opt(request)
    if row is None or root is None:
        raise HTTPException(status_code=404, detail="highlight not found")
    model = build_highlight_card(root, row)
    if model is None:
        raise HTTPException(status_code=404, detail="episode not in the corpus")
    try:
        png = render_card_png(model)
    except Exception as exc:  # noqa: BLE001 - Pillow missing / font failure → degrade, not 500
        raise HTTPException(status_code=503, detail="Card renderer unavailable.") from exc
    return Response(
        content=png,
        media_type="image/png",
        # Private: it is this user's highlight. Short-lived, since the highlight can be edited.
        headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"},
    )


@router.delete("/highlights/{highlight_id}", response_model=HighlightsResponse)
def delete_highlight(
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
    # A collection cover derived from this highlight's episode must not linger (#2004 M5).
    app_collections.recompute_covers_for_highlight(request, data_dir, user.user_id, highlight_id)
    return HighlightsResponse(items=[Highlight(**r) for r in rows])


# --- notes --------------------------------------------------------------------


@router.get("/notes", response_model=NotesResponse)
def list_notes(
    request: Request,
    target: str | None = None,
    target_id: str | None = None,
    q: str | None = Query(default=None, max_length=200, description="Paged: text match."),
    kinds: list[str] = Query(
        default_factory=list,
        max_length=10,
        description="Paged: only notes on these target kinds (repeat the param); none = all.",
    ),
    match: Literal["phrase", "words"] = Query(
        default="phrase",
        description="Paged: `q` as one phrase, or every word of it in any order (Search).",
    ),
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(
        default=None,
        ge=1,
        le=100,
        description="Page size. Absent: every note (scoped by target), as before 1.0.3.",
    ),
    user: User = Depends(get_current_user),
) -> NotesResponse:
    """The user's notes, optionally scoped to one ``?target=&target_id=``.

    With ``limit``: newest first, matched on ``q``, one page, plus ``total`` and per-target
    ``counts`` (under the same ``q`` and ``target_id``, ignoring ``target``).
    """
    if limit is None:
        rows = app_user_state.get_notes(_data_dir(request), user.user_id, target, target_id)
        return NotesResponse(items=[Note(**r) for r in rows])
    needle = (q or "").strip().casefold()
    words = needle.split() if match == "words" else [needle]
    every = app_user_state.get_notes(_data_dir(request), user.user_id, None, target_id)
    hits = [
        n
        for n in sorted(every, key=lambda n: int(n.get("created_at") or 0), reverse=True)
        if all(_matches(w, n.get("text")) for w in words)
    ]
    counts: dict[str, int] = {}
    for n in hits:
        counts[str(n.get("target"))] = counts.get(str(n.get("target")), 0) + 1
    selected = [
        n
        for n in hits
        if (target is None or n.get("target") == target) and (not kinds or n.get("target") in kinds)
    ]
    page = selected[offset : offset + limit]
    on = {str(n.get("target_id")) for n in page if n.get("target") == "highlight"}
    highlights = (
        [
            Highlight(**h)
            for h in app_user_state.get_highlights(_data_dir(request), user.user_id, None)
            if str(h.get("id")) in on
        ]
        if on
        else []
    )
    return NotesResponse(
        items=[Note(**n) for n in page],
        total=len(selected),
        counts=counts,
        highlights=highlights,
    )


@router.post("/notes", response_model=Note, status_code=201)
def create_note(
    request: Request,
    body: NoteCreate,
    response: Response,
    user: User = Depends(get_current_user),
) -> Note:
    """Attach a free-text note to a highlight / insight / episode; mints id + timestamps.

    Idempotent under ``client_id``, and 200-on-replay — see ``create_highlight`` (#1925).
    """
    now = int(time.time())
    record = body.model_dump(exclude={"client_id"})
    record.update({"id": body.client_id or _new_id("n"), "created_at": now, "updated_at": now})
    stored, created = app_user_state.add_note_if_absent(_data_dir(request), user.user_id, record)
    if not created:
        response.status_code = 200
    return Note(**stored)


@router.patch("/notes/{note_id}", response_model=Note)
def patch_note(
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
def delete_note(
    request: Request, note_id: str, user: User = Depends(get_current_user)
) -> NotesResponse:
    """Remove a note by id (no-op if absent); returns the remaining list."""
    rows = app_user_state.remove_note(_data_dir(request), user.user_id, note_id)
    return NotesResponse(items=[Note(**r) for r in rows])


# --- Markdown export ----------------------------------------------------------


def _episode_meta(request: Request, slugs: set[str]) -> dict[str, dict]:
    """Best-effort episode metadata per slug; never breaks export when the corpus is unavailable.

    Carries what places an episode — title, show, publish date, duration — and all THREE summary
    fields, because the app treats them as three different things and so must the export: a
    headline, the prose the Summary button renders, and the digest that opens the insights panel.
    """
    out: dict[str, dict] = {}
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
            out[slug] = {
                "title": row.episode_title,
                "show": row.feed_title,
                "publish_date": getattr(row, "publish_date", None),
                "duration_seconds": getattr(row, "duration_seconds", None),
                "summary_title": getattr(row, "summary_title", None),
                "summary_text": getattr(row, "summary_text", None),
                "summary_bullets": list(getattr(row, "summary_bullets", ()) or ()),
            }
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
def export_highlights_markdown(
    request: Request,
    user: User = Depends(get_current_user),
    color: str | None = Query(
        default=None,
        description="When set, export only highlights of this colour token (RFC-121 ph. 3 / "
        "#2042) — so 'filter to amber, then Export' gives just the amber highlights. Episode / "
        "insight notes carry no colour, so a colour-filtered export omits them.",
    ),
    muted_only: bool = Query(
        default=False,
        description="When true, export only captures the user stopped resurfacing — the Saved "
        "tab's muted filter, so 'filter, then Export' agrees with the screen.",
    ),
    q: str | None = Query(
        default=None,
        description="When set, export only captures whose quote or speaker matches — the Saved "
        "tab's search box.",
    ),
) -> PlainTextResponse:
    """Export the user's highlights AND (unfiltered) their notes, as a Markdown document.

    "With attached notes" used to mean only notes on a highlight: ``notes_by_target`` was consumed
    solely by highlight id, so a note the user wrote on an EPISODE or on a saved INSIGHT was
    silently absent from their export. An export that quietly drops the user's own writing is worse
    than one that never offered it — episode notes now sit under their episode's heading, and
    anything whose target this renderer cannot place goes to a trailing "Other notes" section.

    ``color`` narrows the export to match the Saved surface's colour filter: only highlights of
    that colour, and only the notes attached to them (episode / insight notes have no colour and
    would otherwise leak past the filter).
    """
    episodes, orphans = _export_document(request, user, color, muted_only, q)
    markdown = render_highlights_markdown(episodes, orphans)
    return PlainTextResponse(
        markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="my-highlights.md"'},
    )


def _export_document(
    request: Request,
    user: User,
    color: str | None,
    muted_only: bool,
    q: str | None,
) -> tuple[list[EpisodeHighlights], list[str]]:
    """The export document, filtered and hydrated — shared by every output format.

    Markdown and the printable HTML render the SAME structure. Splitting the build out is the only
    thing keeping "what an export contains" from being defined twice and drifting, which is the bug
    this whole arc kept finding in other places.
    """
    data_dir = _data_dir(request)
    root = _corpus_root_opt(request)
    highlights = app_user_state.get_highlights(data_dir, user.user_id)
    if color:
        highlights = [h for h in highlights if h.get("color") == color]
    if muted_only:
        # The Saved tab can narrow to muted captures, so Export had to be able to as well —
        # otherwise "filter, then export" silently returned everything and the file disagreed with
        # the screen that produced it.
        state = app_user_state.get_resurfacing_state(data_dir, user.user_id)
        highlights = [
            h
            for h in highlights
            # One expression for the id, used for BOTH the membership test and the lookup. They
            # differed — `str(h.get("id") or "")` then `str(h.get("id"))` — so a highlight with a
            # missing id probed the key "" and then read the key "None". Harmless today because
            # every real highlight has a uuid, but two spellings of one value is how that stops
            # being true (review 2026-09-18).
            if isinstance(state.get(str(h.get("id") or "")), dict)
            and state[str(h.get("id") or "")].get("retired")
        ]
    if q:
        needle = q.strip().lower()
        # Same fields the Saved search matches (quote / speaker) — episode titles are findable via
        # the Episodes section there, and matching them here would return captures the screen did
        # not show.
        highlights = [
            h
            for h in highlights
            if needle in str(h.get("quote_text") or "").lower()
            or needle in str(h.get("speaker") or "").lower()
        ]

    def _ref_labels(h: dict) -> list[str]:
        """People and topics this capture is about, as plain labels.

        Through ``refs_for_highlight``, which is the same resolution the Obsidian export and the
        revisit surfaces use: stored refs when the capture has them, else the episode's KG. Most
        captures store none, so reading `graph_refs` directly would have produced an empty list
        almost every time and looked like "this episode has no entities".

        Best-effort: no corpus root (or an episode with no KG) means no labels, never an error —
        an export must not fail because the graph is unavailable.
        """
        if root is None:
            return []
        return [
            str(r.get("label") or "").strip()
            for r in app_graph_refs.refs_for_highlight(root, h)
            if str(r.get("label") or "").strip()
        ]

    notes = app_user_state.get_notes(data_dir, user.user_id)
    notes_by_target: dict[str, list[str]] = {}
    for n in notes:
        notes_by_target.setdefault(str(n.get("target_id")), []).append(
            format_note(str(n.get("text", "")), n.get("created_at"), n.get("updated_at"))
        )

    highlight_ids = {str(h.get("id")) for h in highlights}
    # A colour filter is about highlights; episode- and insight-level notes have no colour, so a
    # filtered export drops them rather than leaking un-colourable writing past the filter.
    episode_note_slugs = (
        set()
        if color
        else {
            str(n.get("target_id"))
            for n in notes
            if n.get("target") == "episode" and n.get("target_id")
        }
    )
    # Every episode that needs a heading: one the user highlighted, or one they only made a note on.
    titles = _episode_meta(
        request, {str(h.get("episode_slug") or "") for h in highlights} | episode_note_slugs
    )

    grouped: "OrderedDict[str, EpisodeHighlights]" = OrderedDict()

    def _episode(slug: str) -> EpisodeHighlights:
        if slug not in grouped:
            m = titles.get(slug, {})
            grouped[slug] = EpisodeHighlights(
                slug=slug,
                title=m.get("title"),
                show=m.get("show"),
                url=app_pkm_export.episode_url(slug),
                publish_date=m.get("publish_date"),
                duration_seconds=m.get("duration_seconds"),
                summary_title=m.get("summary_title"),
                summary_text=m.get("summary_text"),
                summary_bullets=list(m.get("summary_bullets") or []),
            )
        return grouped[slug]

    for h in highlights:
        # `or ""` — NOT bare str(). `str(None)` is the string "None", so a capture that lost its
        # episode reference produced an export section headed `## None`, with the user's own quotes
        # filed under an episode that does not exist and jump links resolving nowhere. The export is
        # the canonical record; a fabricated episode title in it is permanent (review 2026-09-18).
        # An empty slug groups under the orphan heading, which is what it is.
        _episode(str(h.get("episode_slug") or "")).highlights.append(
            HighlightLine(
                kind=str(h.get("kind", "span")),
                start_ms=h.get("start_ms"),
                quote_text=h.get("quote_text"),
                speaker=h.get("speaker"),
                color=h.get("color"),
                created_at=h.get("created_at"),
                entities=_ref_labels(h),
                jump_url=app_pkm_export.episode_url(
                    str(h.get("episode_slug") or ""), h.get("start_ms")
                ),
                notes=notes_by_target.get(str(h.get("id")), []),
            )
        )

    for slug in episode_note_slugs:
        _episode(slug).episode_notes.extend(notes_by_target.get(slug, []))

    # Whatever is left: a note on a saved insight, whose target id is an insight, not an episode.
    # There is no insight -> episode mapping here, so rather than drop it, it gets its own section.
    # Suppressed under a colour filter — an insight note has no colour and would slip past it.
    placed = highlight_ids | episode_note_slugs
    orphans = (
        []
        if color
        else [
            format_note(str(n.get("text", "")), n.get("created_at"), n.get("updated_at"))
            for n in notes
            if str(n.get("target_id")) not in placed
        ]
    )

    return list(grouped.values()), orphans


class HtmlResponse(HTMLResponse):
    """An HTML response that documents the media type it sends (see ``MarkdownResponse``)."""

    media_type = "text/html; charset=utf-8"


@router.get(
    "/highlights/export.html",
    response_class=HtmlResponse,
    responses={200: {"description": "The same export, styled for printing to PDF."}},
)
def export_highlights_html(
    request: Request,
    user: User = Depends(get_current_user),
    color: str | None = Query(default=None, description="Same colour filter as export.md."),
    muted_only: bool = Query(default=False, description="Same muted filter as export.md."),
    q: str | None = Query(default=None, description="Same search filter as export.md."),
) -> HtmlResponse:
    """The export as a print-styled page — the PDF path, with no PDF library.

    There is no server-side renderer here on purpose. Every option cost something the others did
    not: WeasyPrint drags Cairo/Pango into the API image, ReportLab means hand-building a layout.
    The browser already has a good one, and "Print -> Save as PDF" is native on every platform we
    ship, including the iOS share sheet. So this route emits the same document with a print
    stylesheet and lets the browser do the conversion.

    Same filters as ``export.md``, because it is literally the same document (``_export_document``).
    """
    episodes, orphans = _export_document(request, user, color, muted_only, q)
    return HtmlResponse(render_highlights_html(episodes, orphans))
