"""Public OG-card image route (#2036) — ``GET /og/{kind}/{ident}.png``.

Renders the shareable card for an entity to a PNG so a shared LINK unfurls as the card (this is the
URL the SPA head's ``og:image`` points at). UNAUTHENTICATED by design: unfurl bots (iMessage, Slack,
X, WhatsApp) carry no session. It exposes only transcript-derived text + KG metadata already public
on the card — no audio, no private data. Mounted at ``/og`` (outside ``/api/app``) and the ``.png``
suffix lets the edge's static rule route it to the backend without the coming-soon gate.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request, Response

from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.og.build import build_og_model, OG_KINDS
from podcast_scraper.server.og.card import render_card_png

router = APIRouter(tags=["og"])


@router.get("/og/{kind}/{ident}.png")
async def og_card_image(request: Request, kind: str, ident: str) -> Response:
    """Render ``(kind, ident)`` to a PNG card. 404 when the entity can't be resolved."""
    if kind not in OG_KINDS:
        raise HTTPException(status_code=404, detail="Unknown card kind.")
    root = corpus_root_or_503(request)
    # Off the event loop — the build walks KG projections and the render is CPU-bound.
    model = await asyncio.to_thread(build_og_model, root, kind, ident.strip())
    if model is None:
        raise HTTPException(status_code=404, detail="Nothing to render for this entity.")
    try:
        png = await asyncio.to_thread(render_card_png, model)
    except Exception as exc:  # noqa: BLE001 - Pillow missing / font load failure → degrade, not 500
        raise HTTPException(status_code=503, detail="Card renderer unavailable.") from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={
            # Cards are cheap to regenerate and change only when the corpus does — an hour of
            # edge/CDN caching keeps unfurl bots off the render path without pinning stale cards.
            "Cache-Control": "public, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )
