"""Newest mtime among files that contribute to vector index fingerprints (GitHub #507)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from podcast_scraper.search import indexer as _indexer
from podcast_scraper.search.corpus_scope import (
    discover_metadata_files,
    episode_root_from_metadata_path,
)
from podcast_scraper.utils.path_validation import normpath_if_under_root

# Short TTL to avoid repeated full-tree scans on Dashboard refresh (#507).
_CACHE_TTL_SEC = 10.0
_cache: Dict[str, Tuple[float, Optional[float]]] = {}


def invalidate_newest_index_source_mtime_cache(corpus_resolved_str: str) -> None:
    """Drop cached mtime for a corpus (e.g. after index rebuild)."""
    key = os.path.normpath(os.path.realpath(corpus_resolved_str))
    _cache.pop(key, None)


def _compute_newest_index_source_mtime_epoch(corpus_root: Path) -> Optional[float]:
    meta_files = discover_metadata_files(corpus_root)
    if not meta_files:
        return None

    # codeql[py/path-injection] -- corpus_root is a trusted server-side Path.
    corpus_s = os.path.normpath(os.path.realpath(str(corpus_root.resolve())))
    newest: Optional[float] = None
    for meta_path in meta_files:
        paths: list[str] = []
        meta_s = normpath_if_under_root(str(meta_path), corpus_s)
        if meta_s:
            paths.append(meta_s)
        doc = _indexer._load_metadata_file(meta_path)
        if doc is not None:
            episode_root = episode_root_from_metadata_path(meta_path)
            gi = _indexer._gi_path(episode_root, meta_path, doc)
            gi_s = normpath_if_under_root(str(gi), corpus_s)
            if gi_s and os.path.isfile(gi_s):
                paths.append(gi_s)
            kg = _indexer._kg_path(episode_root, meta_path, doc)
            kg_s = normpath_if_under_root(str(kg), corpus_s)
            if kg_s and os.path.isfile(kg_s):
                paths.append(kg_s)
            tx = _indexer._transcript_path(episode_root, doc)
            if tx is not None:
                tx_s = normpath_if_under_root(str(tx), corpus_s)
                if tx_s and os.path.isfile(tx_s):
                    paths.append(tx_s)
        for p in paths:
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            newest = m if newest is None else max(newest, m)
    return newest


def newest_index_source_mtime_epoch(
    corpus_root: Path,
    *,
    use_cache: bool = True,
) -> Optional[float]:
    """Return max ``st_mtime`` over metadata + GI/KG/transcript paths used for indexing.

    Mirrors :func:`podcast_scraper.search.indexer.index_corpus` discovery and path
    resolution. Returns ``None`` when no episode metadata files exist or no paths
    could be stat'd.

    When ``use_cache`` is True, results are memoized per resolved corpus root for
    :data:`_CACHE_TTL_SEC` seconds (monotonic clock).
    """
    # codeql[py/path-injection] -- corpus_root is a trusted server-side Path.
    key = os.path.normpath(os.path.realpath(str(corpus_root.resolve())))
    now = time.monotonic()
    if use_cache:
        hit = _cache.get(key)
        if hit is not None:
            expires_at, cached_val = hit
            if now < expires_at:
                return cached_val
    val = _compute_newest_index_source_mtime_epoch(corpus_root)
    if use_cache:
        _cache[key] = (now + _CACHE_TTL_SEC, val)
    return val
