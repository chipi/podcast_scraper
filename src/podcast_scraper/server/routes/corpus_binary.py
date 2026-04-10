"""GET /api/corpus/binary — serve downloaded podcast artwork (RFC-067 Phase 4)."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.utils.corpus_artwork import CORPUS_ART_REL_PREFIX
from podcast_scraper.utils.path_validation import safe_relpath_under_corpus_root

router = APIRouter(tags=["corpus"])


def _safe_artwork_target_str(base: Path, relpath: str) -> str:
    norm = relpath.strip().replace("\\", "/").lstrip("/")
    prefix = f"{CORPUS_ART_REL_PREFIX}/"
    if norm == CORPUS_ART_REL_PREFIX or not norm.startswith(prefix):
        raise HTTPException(status_code=400, detail="Path is not under corpus artwork store.")
    safe = safe_relpath_under_corpus_root(base, norm)
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid path.")
    return safe


@router.get("/corpus/binary")
async def corpus_binary(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus root (contains .podcast_scraper/corpus-art/).",
    ),
    relpath: str = Query(
        ...,
        description=(
            "POSIX path relative to corpus root; must start with "
            "``.podcast_scraper/corpus-art/``."
        ),
    ),
) -> FileResponse:
    """Return a binary image file from the allowlisted corpus-art subtree."""
    anchor = getattr(request.app.state, "output_dir", None)
    if path is not None and str(path).strip():
        root = resolve_corpus_path_param(path, anchor)
    elif anchor is not None:
        root = Path(anchor).expanduser().resolve()
    else:
        raise HTTPException(
            status_code=400,
            detail="path query parameter is required when the server has no default output_dir.",
        )

    target = _safe_artwork_target_str(root, relpath)

    # codeql[py/path-injection] -- target from normpath+startswith in safe_relpath.
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="File not found.")

    media_type, _ = mimetypes.guess_type(os.path.basename(target))
    # codeql[py/path-injection] -- target sanitized above.
    return FileResponse(
        path=target,
        media_type=media_type or "application/octet-stream",
    )
