"""Consumer artwork route — ``GET /api/app/artwork`` (RFC-099).

Serves the locally-stored podcast art (downloaded at ingest), so the consumer app never
re-fetches graphics from the origin host. ``size=thumb`` (≤320px) and ``size=medium`` (≤1024px)
return stored downscales; ``size=large`` returns the original. Content-addressed → served
``immutable`` so the browser + PWA SW cache it after the first fetch. Open read (consistent with
the other ``/api/app`` reads).
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from podcast_scraper.server.artwork import ensure_medium, ensure_thumbnail, safe_artwork_target

router = APIRouter(tags=["app"])

_IMMUTABLE = "public, max-age=31536000, immutable"


def _corpus_root(request: Request) -> Path:
    anchor = getattr(request.app.state, "output_dir", None)
    if anchor is None:
        raise HTTPException(status_code=503, detail="No corpus configured for the platform API.")
    return Path(anchor)


@router.get("/artwork")
def app_artwork(
    request: Request,
    ref: str = Query(..., description="Corpus-relative artwork path (under the corpus-art store)."),
    size: str = Query(
        "large",
        pattern="^(thumb|medium|large)$",
        description="thumb (lists, ≤320px), medium (player, ≤1024px) or large (original).",
    ),
) -> FileResponse:
    """Serve stored podcast art at the requested size (thumb/medium downscale, large=original)."""
    root = _corpus_root(request)
    target = safe_artwork_target(root, ref)
    if not target:
        raise HTTPException(status_code=400, detail="Path is not under the corpus artwork store.")
    # codeql[py/path-injection] -- target is normpath+prefix-validated by safe_artwork_target.
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="Artwork not found.")

    if size == "thumb":
        serve_path, media_type = ensure_thumbnail(root, target)
    elif size == "medium":
        serve_path, media_type = ensure_medium(root, target)
    else:
        guessed, _ = mimetypes.guess_type(os.path.basename(target))
        serve_path, media_type = target, (guessed or "application/octet-stream")

    # codeql[py/path-injection] -- serve_path is the validated target or its derived thumb.
    return FileResponse(
        path=serve_path,
        media_type=media_type,
        headers={"Cache-Control": _IMMUTABLE},
    )
