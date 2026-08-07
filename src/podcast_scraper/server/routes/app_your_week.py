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

from fastapi import APIRouter, Depends, HTTPException, Request, status

from podcast_scraper.server import app_digest_personal
from podcast_scraper.server.app_user_store import User
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
    return YourWeekResponse(
        sections=sections,
        period_label=_period_label(now),
        generated_at=dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
