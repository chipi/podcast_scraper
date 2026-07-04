"""Graph-analytics ingest — ``POST /api/app/graph-events``.

A fire-and-forget batch endpoint the viewer posts graph-usage events to (what users do + how the
graph changes + when it breaks). Always 204; a no-op when there's no data dir. Anonymous sessions
(auth off / signed out) are captured under a shared ``anon`` bucket so open usage still counts.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from podcast_scraper.server import app_graph_telemetry
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_admin_user, get_optional_user
from podcast_scraper.server.schemas import AppGraphEventsBody

router = APIRouter(tags=["app"])

# Cap a single batch so a runaway client can't append unbounded rows in one POST.
_MAX_BATCH = 500


@router.post("/graph-events", status_code=204)
async def graph_events(
    request: Request,
    body: AppGraphEventsBody,
    user: User | None = Depends(get_optional_user),
) -> Response:
    """Append a batch of graph events for later analysis (best-effort; never errors the client)."""
    data_dir = getattr(request.app.state, "app_data_dir", None)
    if data_dir is not None and body.events:
        user_id = user.user_id if user is not None else "anon"
        now = int(time.time())
        stamped = [{**e, "ts": e.get("ts") or now} for e in body.events[:_MAX_BATCH]]
        app_graph_telemetry.record_events(Path(data_dir), user_id, stamped)
    return Response(status_code=204)


@router.get("/graph-events/summary")
async def graph_events_summary(
    request: Request, _admin: User = Depends(get_admin_user)
) -> dict[str, Any]:
    """Aggregate graph analytics across all users (admin only) — usage / size / breakage."""
    data_dir = getattr(request.app.state, "app_data_dir", None)
    if data_dir is None:
        return {**app_graph_telemetry.aggregate([]), "users": 0}
    root = Path(data_dir)
    summary = app_graph_telemetry.aggregate(app_graph_telemetry.read_all_events(root))
    users_dir = root / "users"
    summary["users"] = (
        sum(1 for d in users_dir.iterdir() if d.is_dir() and (d / "graph_events.jsonl").is_file())
        if users_dir.is_dir()
        else 0
    )
    return summary
