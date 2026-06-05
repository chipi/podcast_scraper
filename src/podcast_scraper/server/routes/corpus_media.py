"""GET /api/corpus/media — serve persisted episode audio under ``media/``."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.utils.corpus_media import CORPUS_MEDIA_DIR
from podcast_scraper.utils.path_validation import safe_relpath_under_corpus_root

router = APIRouter(tags=["corpus"])

_ALLOWED_SUFFIXES = (".mp3", ".m4a", ".wav", ".ogg", ".opus", ".aac", ".flac", ".webm")


def _suffix_allowed(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(s) for s in _ALLOWED_SUFFIXES)


def _safe_media_target_str(base: Path, relpath: str) -> str:
    norm = relpath.strip().replace("\\", "/").lstrip("/")
    prefix = f"{CORPUS_MEDIA_DIR}/"
    if norm == CORPUS_MEDIA_DIR or not norm.startswith(prefix):
        raise HTTPException(status_code=400, detail="Path is not under corpus media store.")
    safe = safe_relpath_under_corpus_root(base, norm)
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid path.")
    basename = os.path.basename(safe)
    if not _suffix_allowed(basename):
        raise HTTPException(status_code=400, detail="Unsupported media file type.")
    return safe


@router.get("/corpus/media")
async def corpus_media(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus root (run directory containing ``media/``).",
    ),
    relpath: str = Query(
        ...,
        description="POSIX path relative to corpus root; must start with ``media/``.",
    ),
) -> FileResponse:
    """Return episode audio from the allowlisted ``media/`` subtree."""
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

    target = _safe_media_target_str(root, relpath)

    # codeql[py/path-injection] -- target from safe_relpath_under_corpus_root.
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="File not found.")

    media_type, _ = mimetypes.guess_type(os.path.basename(target))
    # codeql[py/path-injection] -- target sanitized above.
    return FileResponse(
        path=target,
        media_type=media_type or "application/octet-stream",
        headers={"Accept-Ranges": "bytes"},
    )
