"""Public OG-card image route (#2036) — ``GET /og/{kind}/{ident}.png``.

Renders the shareable card for an entity to a PNG so a shared LINK unfurls as the card (this is the
URL the SPA head's ``og:image`` points at). UNAUTHENTICATED by design: unfurl bots (iMessage, Slack,
X, WhatsApp) carry no session. It exposes only transcript-derived text + KG metadata already public
on the card — no audio, no private data. Mounted at ``/og`` (outside ``/api/app``) and the ``.png``
suffix lets the edge's static rule route it to the backend without the coming-soon gate.
"""

from __future__ import annotations

import asyncio
import html
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse

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


_SHARE_PAGE = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    "<title>{title}</title></head><body>"
    '<p><a href="{url}">{title}</a></p></body></html>'
)


@router.get("/og/page/{route}/{ident}")
async def og_share_page(request: Request, route: str, ident: str) -> Response:
    """The og/twitter tags for one shareable document, for a link-preview bot.

    The player's documents are ``index.html`` straight from nginx, so the head ``SpaStaticFiles``
    injects never reached a shared ``/topic/…`` link: it unfurled as nothing (prod 2026-10-07). The
    player's nginx (and, pre-launch, the edge gate) sends ONLY link-preview bots on entity paths
    here; people keep getting the app. Same tags ``SpaStaticFiles`` would inject, on a minimal page.
    """
    from podcast_scraper.server.og.build import build_og_meta
    from podcast_scraper.server.spa import SpaStaticFiles

    target = SpaStaticFiles._entity_target(f"/{route}/{ident}")
    if target is None:
        raise HTTPException(status_code=404, detail="Not a shareable page.")
    kind, entity = target
    root = corpus_root_or_503(request)
    meta = await asyncio.to_thread(build_og_meta, root, kind, entity)
    if meta is None:
        raise HTTPException(status_code=404, detail="Nothing to share for this entity.")
    origin = SpaStaticFiles._origin(request.scope)
    page_url = f"{origin}/{route}/{quote(ident, safe=':')}"
    image = f"{origin}/og/{kind}/{quote(entity, safe='')}.png"
    shell = _SHARE_PAGE.format(
        title=html.escape(meta.title, quote=True), url=html.escape(page_url, quote=True)
    )
    return HTMLResponse(
        SpaStaticFiles._inject(shell, meta.title, meta.description, image, page_url),
        headers={"Cache-Control": "public, max-age=3600"},
    )
