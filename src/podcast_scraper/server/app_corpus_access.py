"""Shared corpus-access helpers for the consumer ``/api/app/*`` routes.

The consumer surface serves the single shared corpus at ``app.state.output_dir`` — there
is no ``?path`` override (that is an operator concern). These two helpers are the only
ways the consumer routes reach the filesystem: resolve the corpus root, and path-safely
load a JSON artifact under it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import HTTPException, Request

from podcast_scraper.utils.path_validation import safe_relpath_under_corpus_root

logger = logging.getLogger(__name__)


def corpus_root_or_503(request: Request) -> Path:
    """Resolve the single shared corpus root, or 503 if the platform has no corpus."""
    anchor = getattr(request.app.state, "output_dir", None)
    if anchor is None:
        raise HTTPException(status_code=503, detail="No corpus configured for the platform API.")
    return Path(anchor)


def load_json_artifact(root: Path, relpath: str) -> dict | None:
    """Path-safe JSON load of a corpus artifact (GI/KG); ``None`` when missing/unreadable."""
    if not relpath:
        return None
    safe = safe_relpath_under_corpus_root(root, relpath)
    if not safe:
        return None
    path = root / safe
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable artifact %s: %s", path, exc)
        return None
    return loaded if isinstance(loaded, dict) else None
