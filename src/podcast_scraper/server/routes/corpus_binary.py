"""GET /api/corpus/binary — serve downloaded podcast artwork (RFC-067 Phase 4)."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.utils.corpus_artwork import CORPUS_ART_REL_PREFIX

router = APIRouter(tags=["corpus"])


def _is_under(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_artwork_target(base: Path, relpath: str) -> Path:
    norm = relpath.strip().replace("\\", "/").lstrip("/")
    prefix = f"{CORPUS_ART_REL_PREFIX}/"
    if norm == CORPUS_ART_REL_PREFIX or not norm.startswith(prefix):
        raise HTTPException(status_code=400, detail="Path is not under corpus artwork store.")
    segments = [p for p in norm.split("/") if p and p != "."]
    if any(p == ".." for p in segments):
        raise HTTPException(status_code=400, detail="Invalid path.")
    target = base.joinpath(*segments).resolve()
    if not _is_under(base, target):
        raise HTTPException(status_code=400, detail="Path escapes corpus root.")
    return target


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

    target = _safe_artwork_target(root, relpath)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    media_type, _ = mimetypes.guess_type(target.name)
    return FileResponse(
        path=str(target),
        media_type=media_type or "application/octet-stream",
    )
