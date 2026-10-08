"""The in-app "Your Week" surface (#1412) — your week in review (operator 2026-10-07).

What you listened to, what you saved, and what is rising in your world. It reads the same digest
the email sends but keeps only its trending section: the email's new-in-follows / new-in-interests
are Home's What's new (see ``_IN_APP_DROPS``).

The payload is a view of the user's OWN data, so it is DECOUPLED from email consent: a user who
has turned the digest email OFF still sees Your Week in-app (the ``comms.types.digest.email`` toggle
governs only the outbound email). Read-only, per-user, no outbox/delivery involvement — this
mirrors ``app_digest_personal.assemble_digest_payload`` (the single source of truth) rather than
re-deriving anything.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from podcast_scraper.server import app_digest_personal, app_user_state, artwork
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.catalog_cache import cached_catalog_last_run
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import YourWeekResponse
from podcast_scraper.server.slugs import episode_slug

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    # get_current_user has already guaranteed app_data_dir is configured.
    return Path(request.app.state.app_data_dir)


def _corpus_root(request: Request) -> Path:
    root = getattr(request.app.state, "output_dir", None)
    if root is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="corpus not configured",
        )
    return Path(root)


def _period_label(now: int) -> str:
    """Human 'Your Week' window — the trailing 7 days ending today (UTC), e.g. 'Aug 1 – 7'.

    Day numbers are formatted by hand (not ``%-d``) because that glibc extension is not portable
    to the BSD strftime the test host may run.
    """
    end = dt.datetime.fromtimestamp(now, dt.timezone.utc).date()
    start = end - dt.timedelta(days=6)
    start_s = f"{start.strftime('%b')} {start.day}"
    end_s = f"{end.day}" if start.month == end.month else f"{end.strftime('%b')} {end.day}"
    return f"{start_s} – {end_s}"


def _image_for(row: CatalogEpisodeRow) -> str | None:
    """Best card artwork for an episode row: the episode's own art (served-local, else remote),
    falling back to the show's. Local paths become the /api/app/artwork thumb URL."""
    return (
        artwork.artwork_url(row.episode_image_local_relpath, "thumb")
        or row.episode_image_url
        or artwork.artwork_url(row.feed_image_local_relpath, "thumb")
        or row.feed_image_url
    )


def _enrich_items(catalog: list[CatalogEpisodeRow], sections: list[dict[str, Any]]) -> None:
    """Enrich each item from its catalog row: the episode/show art (``image_url``) for the card
    backdrop, the show name (``podcast_title``), and ``episode_title`` where the assembler omits
    it (the ``trending_in_your_corpus`` items are topic-centric and carry none, which would render
    a blank card title). In-app ONLY —
    the shared assembler and the email envelope contract stay untouched. ``catalog`` is the SAME
    scan the assembler used this request (threaded through), so enrichment adds no extra scan."""
    slugs = {
        it.get("episode_slug")
        for s in sections
        for it in s.get("items", [])
        if it.get("episode_slug")
    }
    if not slugs:
        return
    by_slug: dict[str, CatalogEpisodeRow] = {}
    for row in catalog:
        slug = episode_slug(row.feed_id, row.episode_id, row.metadata_relative_path)
        if slug in slugs:
            by_slug[slug] = row
    for s in sections:
        for it in s.get("items", []):
            match = by_slug.get(it.get("episode_slug"))
            if match is not None:
                it["image_url"] = _image_for(match)
                it.setdefault("episode_title", match.episode_title)
                # The card names the show too (operator 2026-10-08), as Continue listening does.
                it.setdefault("podcast_title", match.feed_title)


#: What the IN-APP Your Week leaves out (operator 2026-10-07: "cut the overlap"): new episodes from
#: your follows and interests are Home's What's new now. The EMAIL keeps them — in an inbox, "what's
#: new from your shows" is the whole point. `revisit` stays in the payload (other surfaces read it;
#: the app's Your Week already skips it, leaving due highlights to the revisit rail).
_IN_APP_DROPS = frozenset({"new_in_follows", "new_in_interests"})
_WEEK_SECONDS = 7 * 24 * 3600
_REVIEW_LIMIT = 6


def _listened_this_week(data_dir: Path, user_id: str, now: int) -> list[dict[str, Any]]:
    """Episodes played in the last 7 days, most recent first — the week in review's first half."""
    since = now - _WEEK_SECONDS
    rows = [
        r
        for r in app_user_state.list_playback(data_dir, user_id)
        if isinstance(r.get("updated_at"), (int, float)) and r["updated_at"] >= since
    ]
    return [
        {"episode_slug": r["slug"], "deep_link": f"/episode/{r['slug']}"}
        for r in rows[:_REVIEW_LIMIT]
    ]


def _saved_this_week(data_dir: Path, user_id: str, now: int) -> list[dict[str, Any]]:
    """Highlights saved in the last 7 days, newest first, each opening AT its moment."""
    since = now - _WEEK_SECONDS
    recent = [
        h
        for h in app_user_state.get_highlights(data_dir, user_id)
        if not h.get("retired") and int(h.get("created_at") or 0) >= since
    ]
    recent.sort(key=lambda h: int(h.get("created_at") or 0), reverse=True)
    out: list[dict[str, Any]] = []
    for h in recent[:_REVIEW_LIMIT]:
        slug = str(h["episode_slug"])
        start_ms = h.get("start_ms")
        item: dict[str, Any] = {
            "episode_slug": slug,
            "deep_link": f"/episode/{slug}"
            + (f"?t={int(start_ms) // 1000}" if isinstance(start_ms, int) else ""),
        }
        if h.get("quote_text"):
            item["quote"] = str(h["quote_text"])
        if isinstance(start_ms, int):
            item["t_ms"] = start_ms
        out.append(item)
    return out


def week_in_review_sections(
    digest_sections: list[dict[str, Any]], data_dir: Path, user_id: str, now: int
) -> list[dict[str, Any]]:
    """The in-app Your Week: what you listened to, what you saved, what is rising in your world.

    Built from the shared digest (only its ``trending_in_your_corpus`` section survives) plus the
    two review sections read from the listener's own state. Empty sections are left out.
    """
    sections = [
        {"kind": "listened_this_week", "items": _listened_this_week(data_dir, user_id, now)},
        {"kind": "saved_this_week", "items": _saved_this_week(data_dir, user_id, now)},
        *(s for s in digest_sections if s.get("kind") not in _IN_APP_DROPS),
    ]
    return [s for s in sections if s.get("items")]


@router.get("/your-week", response_model=YourWeekResponse)
def get_your_week(request: Request, user: User = Depends(get_current_user)) -> YourWeekResponse:
    """The signed-in user's current Your Week rollup (empty ``sections`` when nothing is due yet).

    Consent-decoupled: always visible in-app regardless of the email digest toggle.
    """
    now = int(time.time())
    root = _corpus_root(request)
    # Cached until the next ingest. Uncached this was a full corpus scan PER REQUEST — 2.6 s of
    # GIL-bound JSON parsing on prod (2,464 rows, 2026-10-07) — and Home fires this together with
    # /notifications (which scanned too), so real users waited 9-25 s all day.
    catalog = cached_catalog_last_run(root)
    payload = app_digest_personal.assemble_digest_payload(
        root, _data_dir(request), user.user_id, now, catalog=catalog
    )
    sections = week_in_review_sections(
        payload["sections"] if payload else [], _data_dir(request), user.user_id, now
    )
    _enrich_items(catalog, sections)
    return YourWeekResponse(
        sections=sections,
        period_label=_period_label(now),
        generated_at=dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
