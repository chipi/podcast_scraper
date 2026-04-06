"""Corpus path resolution for the viewer API (CodeQL-friendly root checks)."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def resolve_corpus_path_param(
    path_param: str,
    anchor: Path | None,
    *,
    must_be_dir: bool = True,
) -> Path:
    """Resolve a corpus directory from a query string.

    When ``anchor`` is set (``app.state.output_dir``), the resolved path must be
    exactly that directory or a subdirectory, so remote callers cannot pivot to
    arbitrary filesystem locations. When ``anchor`` is unset, query ``path`` is
    not allowed (callers must configure a default corpus root first).

    Raises:
        HTTPException: 400 if the path is missing, not a directory, or escapes
        the configured anchor.
    """
    raw = str(path_param).strip()
    if not raw:
        raise HTTPException(status_code=400, detail="path must be non-empty.")

    # lgtm[py/path-injection] -- Resolved path is restricted to ``anchor`` (below) before any
    # filesystem reads in route handlers; local viewer binds to loopback by default.
    candidate = Path(raw).expanduser().resolve()

    if anchor is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Corpus path override is not allowed without a configured server "
                "default (set PODCAST_SERVE_OUTPUT_DIR or pass output_dir to create_app)."
            ),
        )

    anchor_resolved = anchor.expanduser().resolve()
    if candidate != anchor_resolved and not candidate.is_relative_to(anchor_resolved):
        raise HTTPException(
            status_code=400,
            detail="path must be the configured corpus root or a subdirectory of it.",
        )

    if must_be_dir and not candidate.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {candidate}")

    return candidate
