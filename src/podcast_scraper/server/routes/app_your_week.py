"""The in-app "Your Week" surface (#1412).

Serves the SAME personalized rollup the email digest sends (revisit + new-in-follows +
trending-in-your-corpus) synchronously to the signed-in player user, so "Your Week" lives in the
app as the PRIMARY surface and the email is only the edge for when you don't visit.

The payload is a view of the user's OWN data, so it is DECOUPLED from email consent: a user who
has turned the digest email OFF still sees Your Week in-app (the ``comms.digest.enabled`` toggle
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

from podcast_scraper.server import app_artwork, app_digest_personal
from podcast_scraper.server.app_slugs import episode_slug
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.corpus_catalog import build_catalog_rows, CatalogEpisodeRow
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import YourWeekResponse

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
        app_artwork.artwork_url(row.episode_image_local_relpath, "thumb")
        or row.episode_image_url
        or app_artwork.artwork_url(row.feed_image_local_relpath, "thumb")
        or row.feed_image_url
    )


def _enrich_images(root: Path, sections: list[dict[str, Any]]) -> None:
    """Add ``image_url`` to each item so the in-app card can use the episode/show art as a
    backdrop. In-app only — the shared assembler (and the email contract) stay untouched. Builds
    the catalog once and only indexes the slugs actually present."""
    slugs = {
        it.get("episode_slug")
        for s in sections
        for it in s.get("items", [])
        if it.get("episode_slug")
    }
    if not slugs:
        return
    by_slug: dict[str, CatalogEpisodeRow] = {}
    for row in build_catalog_rows(root):
        slug = episode_slug(row.feed_id, row.episode_id, row.metadata_relative_path)
        if slug in slugs:
            by_slug[slug] = row
    for s in sections:
        for it in s.get("items", []):
            match = by_slug.get(it.get("episode_slug"))
            if match is not None:
                it["image_url"] = _image_for(match)


@router.get("/your-week", response_model=YourWeekResponse)
async def get_your_week(
    request: Request, user: User = Depends(get_current_user)
) -> YourWeekResponse:
    """The signed-in user's current Your Week rollup (empty ``sections`` when nothing is due yet).

    Consent-decoupled: always visible in-app regardless of the email digest toggle.
    """
    now = int(time.time())
    payload = app_digest_personal.assemble_digest_payload(
        _corpus_root(request), _data_dir(request), user.user_id, now
    )
    sections = payload["sections"] if payload else []
    _enrich_images(_corpus_root(request), sections)
    return YourWeekResponse(
        sections=sections,
        period_label=_period_label(now),
        generated_at=dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
