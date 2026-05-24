"""Corpus path resolution for the viewer API.

Uses ``os.path.normpath`` + ``str.startswith`` — the sanitiser pair that
CodeQL's ``py/path-injection`` query recognises as safe. Manifest reads for
``/api/health`` use ``read_manifest_produced_by_under_anchor``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from podcast_scraper.corpus_version import CORPUS_MANIFEST_FILE, parse_produced_by_from_manifest_doc
from podcast_scraper.utils.path_validation import normpath_if_under_root, safe_resolve_directory


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

    # Resolve symlinks on the user path so it canonicalises the same way as
    # the anchor (which was ``.resolve()``d above). Without this, paths under
    # a symlinked corpus root (macOS ``/var/folders/...`` → ``/private/var/...``;
    # Docker bind mounts where ``/app/output`` may symlink under ``/var/lib/docker``)
    # would diverge between anchor (resolved) and user path (literal), and the
    # ``startswith`` anchor check below would 400 a legitimately-anchored path.
    # ``Path.resolve(strict=False)`` walks up to the first existing ancestor,
    # resolves symlinks there, then re-attaches missing components — so this
    # works correctly when ``must_be_dir=False`` and the leaf doesn't exist yet
    # (first-run UX, see #693).
    try:
        normed = os.path.normpath(str(Path(normed).resolve(strict=False)))
    except (OSError, RuntimeError):
        # ``resolve()`` can raise on Windows networked paths or symlink loops;
        # fall back to the unresolved normpath rather than crash. The anchor
        # check below is still authoritative.
        pass

    # Inline sanitizer: normpath + startswith (CodeQL py/path-injection recognises this).
    normed = os.path.normpath(normed)
    safe_prefix = anchor_str + os.sep
    if normed != anchor_str and not normed.startswith(safe_prefix):
        raise CorpusPathRequestError(
            status_code=400,
            detail="path must be the configured corpus root or a subdirectory of it.",
        )

    if must_be_dir and not os.path.isdir(normed):
        raise CorpusPathRequestError(status_code=400, detail=f"Not a directory: {normed}")

    return Path(normed)


def resolved_corpus_root_str(root: Path, anchor: Path | None) -> str:
    """Return a normalized corpus root string for filesystem access after resolution.

    Inline ``normpath`` + ``startswith`` so CodeQL treats the result as sanitized.
    """
    norm_root = os.path.normpath(str(root.resolve()))
    if anchor is None:
        return norm_root
    anchor_str = os.path.normpath(str(anchor.expanduser().resolve()))
    # Inline sanitizer: normpath + startswith (CodeQL py/path-injection).
    norm_root = os.path.normpath(norm_root)
    safe_prefix = anchor_str + os.sep
    if norm_root != anchor_str and not norm_root.startswith(safe_prefix):
        return anchor_str
    return norm_root


def read_manifest_produced_by_under_anchor(
    corpus_dir: Path,
    anchor: Path,
) -> dict[str, Any] | None:
    """Load ``produced_by`` from ``corpus_manifest.json`` under a verified anchor."""
    root_s = str(safe_resolve_directory(anchor))
    corpus_s = os.path.normpath(str(corpus_dir.resolve()))
    verified = normpath_if_under_root(corpus_s, root_s)
    if not verified:
        return None
    manifest_s = os.path.normpath(os.path.join(verified, CORPUS_MANIFEST_FILE))
    verified_m = normpath_if_under_root(manifest_s, root_s)
    if not verified_m:
        return None
    # codeql[py/path-injection] -- verified_m from normpath_if_under_root(manifest_s, root_s).
    if not os.path.isfile(verified_m):
        return None
    try:
        # codeql[py/path-injection] -- verified_m (same sanitizer chain as isfile above).
        doc = json.loads(Path(verified_m).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parse_produced_by_from_manifest_doc(doc)
