"""Native app-exit reasons — ``POST /api/app/app-exits`` (#2279).

The iOS app "starts from scratch" several times a day after a few minutes away, and nothing said
why. GlitchTip cannot: Sentry reports watchdog terminations only when the app was in the
FOREGROUND, and an eviction in the background is not a crash. The device itself knows — MetricKit
on iOS, ``ApplicationExitInfo`` on Android, plus the WebView's own termination callbacks — so the
native shell records those and the web layer forwards them here on the next launch.

Each entry becomes one ``app_exit`` event on the log sink, which ships to VictoriaLogs. Counts and
reasons only: no account id (``signed_in`` says whether there was one), no device identifier.
Open to signed-out callers, because a signed-out app is evicted just the same; the body is
strictly validated and capped, and the route always answers 204.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from podcast_scraper.obs.events import emit_event
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_optional_user
from podcast_scraper.server.schemas import AppExitsBody

router = APIRouter(tags=["app"])


@router.post("/app-exits", status_code=204)
def app_exits(body: AppExitsBody, user: User | None = Depends(get_optional_user)) -> Response:
    """Record why the app ended, as the device reported it (best-effort; never errors)."""
    for entry in body.entries:
        emit_event(
            "app_exit",
            sink="log",
            platform=body.platform,
            app_version=body.app_version or None,
            source=entry.source,
            reason=entry.reason,
            count=entry.count,
            at=entry.at,
            signed_in=user is not None,
        )
    return Response(status_code=204)
