"""Internal outbox endpoints — the infra delivery worker's view of the seam (#1415, RFC-110 §2).

Mounted under ``/internal`` (NOT ``/api/app`` — this is service-to-service, not user-facing). The
delivery worker (#1412) polls ``/internal/outbox/pending`` and reports terminal status to
``/internal/outbox/{id}/status``. Both are gated by a shared token (``INTERNAL_OUTBOX_TOKEN``,
tailnet-only) carried in the ``X-Internal-Token`` header — v1.1 amendment 6.

The route layer is thin: consent-filtering, expiry, idempotency, and suppression all live in
``app_outbox_store``. When no token is configured the endpoints are hard-disabled (503), so an
unconfigured deployment never exposes the outbox unauthenticated.
"""

from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

from podcast_scraper.server import app_outbox_store
from podcast_scraper.server.schemas import (
    OutboxPendingResponse,
    OutboxStatusBody,
    OutboxStatusResponse,
)

router = APIRouter(tags=["internal"])


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


def require_internal_token(
    request: Request, x_internal_token: str | None = Header(default=None)
) -> None:
    """Gate on the shared ``INTERNAL_OUTBOX_TOKEN``. 503 when unconfigured, 401 on mismatch."""
    configured = getattr(request.app.state, "internal_outbox_token", "") or ""
    if not configured:
        raise HTTPException(status_code=503, detail="internal outbox not configured")
    presented = x_internal_token or ""
    if not hmac.compare_digest(presented, configured):
        raise HTTPException(status_code=401, detail="invalid internal token")


@router.get(
    "/outbox/pending",
    response_model=OutboxPendingResponse,
    dependencies=[Depends(require_internal_token)],
)
async def pending(
    request: Request,
    channel: str = Query(..., pattern="^(email|push)$"),
    limit: int = Query(default=50, ge=1, le=500),
) -> OutboxPendingResponse:
    """Pending envelopes for a channel — current-consent-filtered, non-expired, oldest-first."""
    envelopes = app_outbox_store.list_pending(_data_dir(request), channel=channel, limit=limit)
    return OutboxPendingResponse(envelopes=envelopes)


@router.post(
    "/outbox/{envelope_id}/status",
    response_model=OutboxStatusResponse,
    dependencies=[Depends(require_internal_token)],
)
async def report_status(
    request: Request, envelope_id: str, body: OutboxStatusBody
) -> OutboxStatusResponse:
    """Record a terminal status (idempotent per id); suppresses the channel on bounce/complaint."""
    effective = app_outbox_store.record_status(
        _data_dir(request), envelope_id, body.status, body.detail
    )
    return OutboxStatusResponse(status=effective)
