"""Corpus-wide episode identity helpers (GitHub #505 / RFC-063).

Composite ``(feed_id, episode_id)`` scope keys avoid fingerprint and vector row
collisions when multiple feeds share a corpus parent.
"""

from __future__ import annotations

import glob as _glob
import os
from pathlib import Path
from typing import Any, List, Optional

from podcast_scraper.utils import filesystem
from podcast_scraper.utils.path_validation import safe_resolve_directory


def normalize_feed_id(feed_id: Any) -> Optional[str]:
    """Return stripped feed id string, or None if missing."""
    if isinstance(feed_id, str) and feed_id.strip():
        return feed_id.strip()
    return None


def index_fingerprint_scope_key(feed_id: Optional[str], episode_id: str) -> str:
    """Stable key for ``episode_fingerprints.json`` (one row per scoped episode)."""
    fn = normalize_feed_id(feed_id)
    if fn:
        return f"{fn}\x1f{episode_id}"
    return episode_id


def vector_doc_scope_tag(feed_id: Optional[str], episode_id: str) -> str:
    """Segment embedded in FAISS doc ids for uniqueness across feeds."""
    if not feed_id:
        return episode_id
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in feed_id)[:120]
    return f"{safe}__{episode_id}"


def gi_map_lookup_key_from_vector_meta(meta: dict) -> str:
    """Key into the GI path map built from metadata (matches fingerprint scope)."""
    ep = meta.get("episode_id")
    if not isinstance(ep, str) or not ep:
        return ""
    return index_fingerprint_scope_key(normalize_feed_id(meta.get("feed_id")), ep)


def episode_root_from_metadata_path(metadata_path: Path) -> Path:
    """Episode workspace root: parent of ``metadata/`` (RFC-063 §5)."""
    return metadata_path.parent.parent.resolve()


def discover_metadata_files(output_root: Path) -> List[Path]:
    """List episode metadata files (flat output or corpus parent with ``feeds/``).

    Hybrid layout: if ``feeds/`` exists **and** top-level ``metadata/`` exists, both are
    included (GitHub #505 follow-up).
    """
    corpus_root = safe_resolve_directory(output_root)
    if corpus_root is None:
        return []

    # CodeQL py/path-injection sanitiser: normpath then startswith on every
    # tainted value before it reaches a filesystem sink.  ``os.sep`` is used
    # as the non-tainted anchor (ensures the path is absolute).
    root_normed = os.path.normpath(str(corpus_root))
    if not root_normed.startswith(os.sep):
        return []

    safe_prefix = root_normed + os.sep
    patterns = ("*.metadata.json", "*.metadata.yaml", "*.metadata.yml")
    found: List[Path] = []

    def _collect(meta_dir_str: str) -> None:
        md = os.path.normpath(meta_dir_str)
        if not md.startswith(safe_prefix) and md != root_normed:
            return
        if not os.path.isdir(md):
            return
        for pat in patterns:
            for hit_str in _glob.glob(os.path.join(md, pat)):
                h = os.path.normpath(hit_str)
                if not h.startswith(safe_prefix) and h != root_normed:
                    continue
                if os.path.isfile(h):
                    found.append(Path(h))

    feeds_str = os.path.normpath(os.path.join(root_normed, "feeds"))
    if feeds_str.startswith(safe_prefix) and os.path.isdir(feeds_str):
        for dirpath, _dirnames, _filenames in os.walk(root_normed):
            dp = os.path.normpath(dirpath)
            if not dp.startswith(safe_prefix) and dp != root_normed:
                continue
            if os.path.basename(dp) == filesystem.METADATA_SUBDIR:
                _collect(dp)
        _collect(os.path.join(root_normed, filesystem.METADATA_SUBDIR))
    else:
        _collect(os.path.join(root_normed, filesystem.METADATA_SUBDIR))

    return sorted(set(found))
