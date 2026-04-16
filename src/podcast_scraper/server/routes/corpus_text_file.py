"""GET /api/corpus/text-file — inline transcript-oriented files under corpus root.

Includes plain text (``.txt``, ``.md``, captions) and ``.json`` transcript artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.utils.path_validation import (
    normpath_if_under_root,
    safe_relpath_under_corpus_root,
    safe_resolve_directory,
)

router = APIRouter(tags=["corpus"])

_ALLOWED_SUFFIXES = (".txt", ".md", ".vtt", ".srt", ".json")


def _suffix_allowed(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(s) for s in _ALLOWED_SUFFIXES)


def _inline_media_type(basename: str) -> str:
    lower = basename.lower()
    if lower.endswith(".json"):
        return "application/json; charset=utf-8"
    return "text/plain; charset=utf-8"


def _cleaned_txt_fallback_relpath(norm: str) -> str | None:
    """Map ``.../foo.txt`` to ``.../foo.cleaned.txt`` (skip if already ``*.cleaned.txt``)."""
    p = Path(norm)
    if p.suffix.lower() != ".txt":
        return None
    if p.name.lower().endswith(".cleaned.txt"):
        return None
    cleaned = p.with_name(f"{p.stem}.cleaned.txt")
    return str(cleaned).replace("\\", "/")


def _resolve_readable_file_under_corpus(root: Path, norm: str) -> tuple[str, str] | None:
    """Return ``(absolute_path, basename)`` for an allowed file under *root*, or ``None``.

    If the requested path is missing but looks like a raw Whisper ``.txt``, tries the
    sibling ``.cleaned.txt`` produced when ``save_cleaned_transcript`` is enabled (metadata
    and GI ``transcript_ref`` often still point at the raw path).

    *root* must already be ``safe_resolve_directory`` output (caller responsibility).
    """
    root_s = os.path.normpath(str(root))

    safe = safe_relpath_under_corpus_root(root, norm)
    verified = normpath_if_under_root(safe, root_s) if safe else None
    if not verified:
        return None
    basename = os.path.basename(verified)
    if not _suffix_allowed(basename):
        return None
    if os.path.isfile(verified):
        return verified, basename

    alt_norm = _cleaned_txt_fallback_relpath(norm)
    if alt_norm is None:
        return None
    safe_alt = safe_relpath_under_corpus_root(root, alt_norm)
    verified_alt = normpath_if_under_root(safe_alt, root_s) if safe_alt else None
    if not verified_alt:
        return None
    alt_base = os.path.basename(verified_alt)
    if not _suffix_allowed(alt_base):
        return None
    if os.path.isfile(verified_alt):
        return verified_alt, alt_base
    return None


@router.get("/corpus/text-file")
async def corpus_text_file(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus root directory (contains transcript paths relative to this root).",
    ),
    relpath: str = Query(
        ...,
        description="File path relative to corpus root (POSIX). Must end with a text suffix.",
    ),
) -> FileResponse:
    """Return a small text-oriented file for browser viewing (transcript, captions, notes)."""
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

    norm = relpath.strip().replace("\\", "/").lstrip("/")
    if not norm:
        raise HTTPException(status_code=400, detail="relpath is required.")

    root_sd = safe_resolve_directory(root)
    if root_sd is None:
        raise HTTPException(status_code=400, detail="Invalid corpus path.")
    root = root_sd

    safe = safe_relpath_under_corpus_root(root, norm)
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid path.")

    req_basename = os.path.basename(safe)
    if not _suffix_allowed(req_basename):
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .md, .vtt, .srt, and .json files are allowed.",
        )

    resolved = _resolve_readable_file_under_corpus(root, norm)
    if resolved is None:
        raise HTTPException(status_code=404, detail="File not found.")

    safe_file, basename = resolved

    root_s = os.path.normpath(str(root))
    verified_path = normpath_if_under_root(safe_file, root_s)
    if not verified_path:
        raise HTTPException(status_code=400, detail="Invalid path.")

    return FileResponse(
        path=verified_path,
        media_type=_inline_media_type(basename),
        filename=basename,
        content_disposition_type="inline",
    )
