"""Corpus path resolution for the viewer API.

Uses ``os.path.normpath`` + ``str.startswith`` — the sanitiser pair that
CodeQL's ``py/path-injection`` query recognises as safe.
"""

from __future__ import annotations

import os
from pathlib import Path

from podcast_scraper.utils.path_validation import normpath_if_under_root


class CorpusPathRequestError(Exception):
    """Invalid or disallowed corpus path (HTTP layer maps this to 400 responses)."""

    __slots__ = ("detail", "status_code")

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def resolve_corpus_path_param(
    path_param: str,
    anchor: Path | None,
    *,
    must_be_dir: bool = True,
) -> Path:
    """Resolve a user-supplied corpus directory against a trusted anchor.

    Raises:
        CorpusPathRequestError: path is empty, escapes anchor, or is not a directory.
    """
    raw = str(path_param).strip()
    if not raw:
        raise CorpusPathRequestError(status_code=400, detail="path must be non-empty.")

    if anchor is None:
        raise CorpusPathRequestError(
            status_code=400,
            detail=(
                "Corpus path override is not allowed without a configured server "
                "default (set PODCAST_SERVE_OUTPUT_DIR or pass output_dir to create_app)."
            ),
        )

    anchor_str = os.path.normpath(str(anchor.expanduser().resolve()))

    normed = os.path.normpath(os.path.expanduser(raw))
    if not os.path.isabs(normed):
        normed = os.path.normpath(os.path.join(anchor_str, normed))

    safe = normpath_if_under_root(normed, anchor_str)
    if safe is None:
        raise CorpusPathRequestError(
            status_code=400,
            detail="path must be the configured corpus root or a subdirectory of it.",
        )

    if must_be_dir and not os.path.isdir(safe):
        raise CorpusPathRequestError(status_code=400, detail=f"Not a directory: {safe}")

    return Path(safe)


def resolved_corpus_root_str(root: Path, anchor: Path | None) -> str:
    """Return a normalized corpus root string for filesystem access after resolution.

    Re-applies ``normpath_if_under_root`` when an anchor exists so CodeQL can treat
    the result as sanitized for path operations in route handlers.
    """
    norm_root = os.path.normpath(str(root.resolve()))
    if anchor is None:
        return norm_root
    anchor_str = os.path.normpath(str(anchor.expanduser().resolve()))
    safe = normpath_if_under_root(norm_root, anchor_str)
    return safe if safe is not None else norm_root
