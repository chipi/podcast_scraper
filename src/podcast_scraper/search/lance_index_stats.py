"""LanceDB index statistics (ADR-099 / #995) — replaces the retired FAISS stats reader.

Aggregates row counts, doc-type breakdown, and indexed feeds from the two-tier LanceDB index
(``segment`` / ``insight`` / ``aux`` tables) so the ``cli index --stats`` command and the
``GET /api/index/stats`` route can report on the single LanceDB index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .backends.lancedb_backend import LanceDBBackend

# segment/insight rows have an implicit doc_type (the tier); aux rows carry their own.
_TIER_DEFAULT_DOC_TYPE = {"segment": "transcript", "insight": "insight"}


@dataclass
class LanceIndexStats:
    """Mirror of the retired FAISS ``IndexStats`` shape, read from the LanceDB index."""

    total_vectors: int = 0
    doc_type_counts: Dict[str, int] = field(default_factory=dict)
    feeds_indexed: List[str] = field(default_factory=list)
    embedding_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_dim: Optional[int] = None
    last_updated: Optional[str] = None
    index_size_bytes: int = 0


def _dir_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def read_lance_index_stats(lance_dir: Path | str) -> Optional[LanceIndexStats]:
    """Return aggregate stats for the LanceDB index at *lance_dir*, or ``None`` if absent."""
    p = Path(lance_dir)
    if not p.is_dir():
        return None
    try:
        be = LanceDBBackend(str(p))
    except Exception:
        return None
    meta = be.read_index_meta() or {}
    st = LanceIndexStats(
        embedding_model=meta.get("embedding_model"),
        embedding_dim=meta.get("embed_dim"),
    )
    feeds: set[str] = set()
    for tier in ("segment", "insight", "aux"):
        tbl = be._open_if_exists(tier)
        if tbl is None:
            continue
        n = tbl.count_rows()
        st.total_vectors += n
        cols = [c for c in ("doc_type", "show_id") if c in tbl.schema.names]
        if not cols:
            dt = _TIER_DEFAULT_DOC_TYPE.get(tier, tier)
            st.doc_type_counts[dt] = st.doc_type_counts.get(dt, 0) + n
            continue
        for r in tbl.search().limit(max(n, 1)).select(cols).to_list():
            dt = r.get("doc_type") or _TIER_DEFAULT_DOC_TYPE.get(tier, tier)
            st.doc_type_counts[dt] = st.doc_type_counts.get(dt, 0) + 1
            sid = r.get("show_id")
            if sid:
                feeds.add(str(sid))
    st.feeds_indexed = sorted(feeds)
    try:
        st.last_updated = (
            datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except OSError:
        pass
    st.index_size_bytes = _dir_size(p)
    return st
