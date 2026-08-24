"""Shared corpus-access helpers for the consumer ``/api/app/*`` routes.

The consumer surface serves the single shared corpus at ``app.state.output_dir`` — there
is no ``?path`` override (that is an operator concern). These two helpers are the only
ways the consumer routes reach the filesystem: resolve the corpus root, and path-safely
load a JSON artifact under it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import HTTPException, Request

from podcast_scraper import perf_cache
from podcast_scraper.utils.path_validation import (
    safe_relpath_under_corpus_root,
    safe_resolve_directory,
)

logger = logging.getLogger(__name__)

_ARTIFACT_NS = "app_corpus_artifact"


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


def cached_json_artifact(root: Path, relpath: str) -> dict | None:
    """:func:`load_json_artifact`, cached by corpus mtime (bumps on ingest).

    For hot corpus-scope artifacts (``temporal_velocity``, cluster maps, ``grounding_rate``) that
    are read by several routes and often multiple times within one request — the momentum layer
    alone loaded ``temporal_velocity.json`` two-to-three times per ``/trending`` call. The returned
    dict is **shared**: treat it read-only (callers derive new structures from it; none mutate it),
    the same convention as the shared catalog cache.
    """
    # Sanitize + read INLINE (not via load_json_artifact) with a same-function normpath + prefix
    # guard: CodeQL does not propagate the cross-function safe_relpath_under_corpus_root sanitizer
    # (docs/ci/CODEQL_DISMISSALS.md → py/path-injection), so the target must be guarded next to the
    # sink to close the query — the corpus_binary.py pattern.
    resolved = safe_resolve_directory(root)
    if resolved is None:
        return None
    safe_prefix = os.path.normpath(str(resolved)) + os.sep
    target = os.path.normpath(os.path.join(str(resolved), relpath))
    if not target.startswith(safe_prefix):  # traversal → refuse before any filesystem touch
        return None

    def _load() -> dict | None:
        # codeql[py/path-injection] -- target normpath'd + prefix-checked above.
        if not os.path.isfile(target):
            return None
        try:
            # codeql[py/path-injection] -- same: target sanitized above.
            loaded = json.loads(Path(target).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable artifact %s: %s", target, exc)
            return None
        return loaded if isinstance(loaded, dict) else None

    cached: dict | None = perf_cache.get_or_compute(
        _ARTIFACT_NS,
        f"{safe_prefix}::{relpath}",
        perf_cache.corpus_mtime(root),
        _load,
    )
    return cached
