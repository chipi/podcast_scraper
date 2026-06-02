"""LanceDB two-tier ``SearchBackend`` implementation (RFC-090 §3.7).

Two tables — ``segments`` (Tier 1) and ``insights`` (Tier 2) — each with a
full-text (BM25) index on ``text`` and a vector index on ``embedding``. BM25 and
vector retrieval are exposed separately so the retrieval layer (#856) fuses them
via RRF. ``lancedb`` is imported lazily so importing this module is cheap and only
instantiating the backend requires the dependency.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...utils.path_validation import (
    normpath_if_under_root,
    safe_relpath_under_corpus_root,
    safe_resolve_directory,
)
from ..backend import InsightDocument, ScoredResult, SearchQuery, SegmentDocument, Tier

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa

# all-MiniLM-L6-v2 dimensionality (matches the existing FAISS index).
DEFAULT_EMBED_DIM = 384

_SEGMENT_TABLE = "segments"
_INSIGHT_TABLE = "insights"


def _segment_schema(dim: int) -> "pa.Schema":
    import pyarrow as pa

    return pa.schema(
        [
            ("id", pa.string()),
            ("text", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("show_id", pa.string()),
            ("episode_id", pa.string()),
            ("speaker_id", pa.string()),
            ("start_time", pa.float64()),
            ("end_time", pa.float64()),
            ("linked_insight_ids", pa.list_(pa.string())),
            ("source_tier", pa.string()),
        ]
    )


def _insight_schema(dim: int) -> "pa.Schema":
    import pyarrow as pa

    return pa.schema(
        [
            ("id", pa.string()),
            ("text", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("show_id", pa.string()),
            ("episode_id", pa.string()),
            ("speaker_id", pa.string()),
            ("entity_type", pa.string()),
            ("confidence", pa.float64()),
            ("derived", pa.bool_()),
            ("source_segment_id", pa.string()),
            ("source_tier", pa.string()),
        ]
    )


class LanceDBBackend:
    """Embedded LanceDB backend (segments + insights), BM25 + vector per tier."""

    TABLES = {"segment": _SEGMENT_TABLE, "insight": _INSIGHT_TABLE}

    INDEX_META_FILE = "index_meta.json"

    def __init__(self, path: str, *, embed_dim: int = DEFAULT_EMBED_DIM) -> None:
        import lancedb

        self.path = path
        self.db = lancedb.connect(path)
        self.embed_dim = embed_dim
        self._tables: Dict[str, Any] = {}  # tier -> open table (list_tables() is cached)

    # --- index metadata sidecar ------------------------------------------------

    # Each FS sink below resolves the sidecar inline via the recognized sanitizer chain
    # (mirrors the proven jobs_log_path idiom): safe_resolve_directory →
    # safe_relpath_under_corpus_root → normpath_if_under_root, then an inline
    # ``# codeql[...]`` pragma at the sink (docs/ci/CODEQL_DISMISSALS.md Type 1; the
    # corpus path is sanitized cross-function at the route, which CodeQL cannot model).

    def write_index_meta(self, embedding_model: str) -> None:
        """Record the embedding model + dim alongside the index (queries must match)."""
        import json
        import os

        root_res = safe_resolve_directory(Path(self.path))
        if root_res is None:
            return
        root_s = os.path.normpath(str(root_res))
        verified = safe_relpath_under_corpus_root(root_res, self.INDEX_META_FILE)
        if not verified:
            return
        meta_path = normpath_if_under_root(os.path.normpath(verified), root_s)
        if not meta_path:
            return
        meta = {"embedding_model": embedding_model, "embed_dim": self.embed_dim}
        # codeql[py/path-injection] -- meta_path via normpath_if_under_root (Type 1).
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)

    def read_index_meta(self) -> Optional[Dict[str, Any]]:
        """Return ``{embedding_model, embed_dim}`` written at build time, or ``None``."""
        import json
        import os

        root_res = safe_resolve_directory(Path(self.path))
        if root_res is None:
            return None
        root_s = os.path.normpath(str(root_res))
        verified = safe_relpath_under_corpus_root(root_res, self.INDEX_META_FILE)
        if not verified:
            return None
        meta_path = normpath_if_under_root(os.path.normpath(verified), root_s)
        if not meta_path:
            return None
        # codeql[py/path-injection] -- meta_path via normpath_if_under_root (Type 1).
        if not os.path.isfile(meta_path):
            return None
        try:
            # codeql[py/path-injection] -- meta_path via normpath_if_under_root (Type 1).
            with open(meta_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    # --- table lifecycle -------------------------------------------------------

    def _open_if_exists(self, tier: str):
        cached = self._tables.get(tier)
        if cached is not None:
            return cached
        try:
            table = self.db.open_table(self.TABLES[tier])
        except Exception:  # noqa: BLE001 - table does not exist yet
            return None
        self._tables[tier] = table
        return table

    def _ensure_table(self, tier: str):
        table = self._open_if_exists(tier)
        if table is not None:
            return table
        schema = (
            _segment_schema(self.embed_dim)
            if tier == "segment"
            else _insight_schema(self.embed_dim)
        )
        table = self.db.create_table(self.TABLES[tier], schema=schema)
        self._tables[tier] = table
        return table

    def create_indices(self) -> None:
        """Create FTS (required for BM25) + vector indices on both tables."""
        for tier in ("segment", "insight"):
            table = self._ensure_table(tier)
            table.create_fts_index("text", replace=True)
            try:
                table.create_index(vector_column_name="embedding", replace=True)
            except Exception:  # noqa: BLE001 - small tables use brute force; index optional
                pass

    def _tables_for_tier(self, tier: Tier) -> List[str]:
        if tier == "all":
            return ["segment", "insight"]
        return [tier]

    # --- retrieval -------------------------------------------------------------

    def _to_sql(self, filters: Dict) -> str | None:
        # OQ-3 (RFC-090): naive interpolation — parameterise before any user-facing
        # free-text filter reaches it. Filters today are internal canonical ids.
        if not filters:
            return None
        return " AND ".join(f"{k} = '{v}'" for k, v in filters.items())

    def _run(self, query: SearchQuery, *, query_type: str, score_key: str) -> List[ScoredResult]:
        where = self._to_sql(query.filters)
        hits: List[tuple[float, Dict[str, Any], str]] = []
        for tier in self._tables_for_tier(query.tier):
            table = self._open_if_exists(tier)  # read must not create empty tables on disk
            if table is None:
                continue
            search_target = query.text if query_type == "fts" else query.embedding
            req = table.search(search_target, query_type=query_type)
            if where:
                req = req.where(where)
            for row in req.limit(query.k).to_list():
                raw = float(row.get(score_key, 0.0) or 0.0)
                # BM25 _score: higher is better; vector _distance: lower is better.
                score = raw if query_type == "fts" else 1.0 / (1.0 + raw)
                row.pop("embedding", None)  # keep payload lean
                hits.append((score, row, tier))
        hits.sort(key=lambda h: h[0], reverse=True)
        signal = "bm25" if query_type == "fts" else "vector"
        return [
            ScoredResult(
                doc_id=str(row.get("id")),
                score=score,
                rank=i + 1,
                payload=row,
                signal=signal,
                source_tier=tier,
            )
            for i, (score, row, tier) in enumerate(hits)
        ]

    def search_bm25(self, query: SearchQuery) -> List[ScoredResult]:
        """BM25 (FTS) results over the query's tier(s) (signal ``bm25``)."""
        return self._run(query, query_type="fts", score_key="_score")

    def search_vector(self, query: SearchQuery) -> List[ScoredResult]:
        """Dense-vector results over the query's tier(s) (signal ``vector``)."""
        return self._run(query, query_type="vector", score_key="_distance")

    # --- writes ----------------------------------------------------------------

    def _upsert(self, tier: str, data: Dict[str, Any]) -> None:
        table = self._ensure_table(tier)
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([data])
        )

    def upsert_segment(self, doc: SegmentDocument) -> None:
        """Insert or update a Tier-1 segment document."""
        self._upsert("segment", dataclasses.asdict(doc))

    def upsert_insight(self, doc: InsightDocument) -> None:
        """Insert or update a Tier-2 insight document."""
        self._upsert("insight", dataclasses.asdict(doc))

    def delete(self, doc_id: str, tier: Tier) -> None:
        """Delete a document by id; ``tier="all"`` removes from both tables."""
        tiers = ("segment", "insight") if tier == "all" else (tier,)
        for t in tiers:
            table = self._open_if_exists(t)
            if table is not None:
                table.delete(f"id = '{doc_id}'")

    def health(self) -> Dict:
        """Return backend status + per-table row counts."""
        try:
            seg = self._open_if_exists("segment")
            ins = self._open_if_exists("insight")
            return {
                "status": "ok",
                "segments": seg.count_rows() if seg is not None else 0,
                "insights": ins.count_rows() if ins is not None else 0,
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc)}
