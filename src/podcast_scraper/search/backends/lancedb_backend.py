"""LanceDB two-tier ``SearchBackend`` implementation (RFC-090 §3.7).

Two tables — ``segments`` (Tier 1) and ``insights`` (Tier 2) — each with a
full-text (BM25) index on ``text`` and a vector index on ``embedding``. BM25 and
vector retrieval are exposed separately so the retrieval layer (#856) fuses them
via RRF. ``lancedb`` is imported lazily so importing this module is cheap and only
instantiating the backend requires the dependency.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from ...utils.path_validation import (
    normpath_if_under_root,
    safe_relpath_under_corpus_root,
    safe_resolve_directory,
)
from ..backend import (
    AuxDocument,
    InsightDocument,
    ScoredResult,
    SearchQuery,
    SegmentDocument,
    Tier,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa

# all-MiniLM-L6-v2 dimensionality.
DEFAULT_EMBED_DIM = 384

# Bump whenever the stored table schema changes so existing indexes self-heal:
# read paths report no_index on a stale index and (re)index moments rebuild it.
#   1 — initial two-tier schema (#855)
#   2 — added fields: ``publish_date`` (date/``since`` filter) on all tiers +
#       ``source_id`` (canonical graph node id → "Show on graph") on insight/aux
LANCE_SCHEMA_VERSION = 2

_SEGMENT_TABLE = "segments"
_INSIGHT_TABLE = "insights"
_AUX_TABLE = "aux"


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
            ("publish_date", pa.string()),
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
            ("publish_date", pa.string()),
            ("source_id", pa.string()),
        ]
    )


def _aux_schema(dim: int) -> "pa.Schema":
    import pyarrow as pa

    return pa.schema(
        [
            ("id", pa.string()),
            ("text", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("show_id", pa.string()),
            ("episode_id", pa.string()),
            ("doc_type", pa.string()),  # kg_entity | kg_topic | quote | summary
            ("source_tier", pa.string()),
            ("publish_date", pa.string()),
            ("source_id", pa.string()),
        ]
    )


class LanceDBBackend:
    """Embedded LanceDB backend (segments + insights + aux), BM25 + vector per tier."""

    TABLES = {"segment": _SEGMENT_TABLE, "insight": _INSIGHT_TABLE, "aux": _AUX_TABLE}

    # Minimum rows before building the IVF vector ANN index. Below this the native
    # index build can SIGSEGV (too few rows to train IVF centroids) and LanceDB
    # falls back to brute-force search anyway — so skip it (review / segfault fix).
    _MIN_VECTOR_INDEX_ROWS = 256

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
        meta = {
            "embedding_model": embedding_model,
            "embed_dim": self.embed_dim,
            "schema_version": LANCE_SCHEMA_VERSION,
        }
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

    _SCHEMAS = {"segment": _segment_schema, "insight": _insight_schema, "aux": _aux_schema}

    def _ensure_table(self, tier: str):
        table = self._open_if_exists(tier)
        if table is not None:
            return table
        schema = self._SCHEMAS[tier](self.embed_dim)
        table = self.db.create_table(self.TABLES[tier], schema=schema)
        self._tables[tier] = table
        return table

    def create_indices(self) -> None:
        """Create FTS (required for BM25) + vector indices on all tables that exist.

        The vector ANN (IVF) index is built only above ``_MIN_VECTOR_INDEX_ROWS``:
        training an IVF index on too few rows can **SIGSEGV** in the native lance
        layer — an uncatchable crash that the ``try/except`` below cannot stop (it
        segfaulted the ``cloud_balanced_single`` acceptance arm, leaving an
        ``empty_vector_index``). Below the threshold, LanceDB uses brute-force
        search anyway, so the ANN index is unnecessary. An empty table is skipped
        entirely.
        """
        for tier in ("segment", "insight", "aux"):
            table = self._open_if_exists(tier)
            if table is None:
                continue  # tier never populated (e.g. a corpus with no kg/quote rows)
            try:
                n_rows = int(table.count_rows())
            except Exception:  # noqa: BLE001 - defensive: treat unknown as empty
                n_rows = 0
            if n_rows <= 0:
                continue  # nothing to index; a native build on 0 rows can SIGSEGV
            table.create_fts_index("text", replace=True)
            if n_rows < self._MIN_VECTOR_INDEX_ROWS:
                continue  # brute-force search below the ANN training floor
            try:
                table.create_index(vector_column_name="embedding", replace=True)
            except Exception:  # noqa: BLE001 - small tables use brute force; index optional
                pass

    def compact(self) -> None:
        """Compact data fragments + prune superseded versions on every table.

        LanceDB is MVCC and the indexer upserts **one document at a time**, so each
        build (and every incremental post-pipeline reindex) appends thousands of tiny
        fragments + versions that are never reclaimed — the index grows unbounded
        (observed: a single ``aux`` table at 8k fragments / 2.7G). ``optimize`` merges
        the fragments and ``cleanup_older_than=0`` drops every version but the current
        one, bounding the on-disk size and keeping reads fast (fewer fragments to scan).
        Best-effort: a compaction failure must never fail the build.
        """
        for tier in ("segment", "insight", "aux"):
            table = self._open_if_exists(tier)
            if table is None:
                continue
            try:
                table.optimize(cleanup_older_than=timedelta(0))
            except Exception as exc:  # noqa: BLE001 - optimize is best-effort
                logger.warning("LanceDB compaction skipped for %s table: %s", tier, exc)

    def _tables_for_tier(self, tier: Tier) -> List[str]:
        if tier == "all":
            return ["segment", "insight", "aux"]
        return [tier]

    # --- retrieval -------------------------------------------------------------

    @staticmethod
    def _sql_str(v: object) -> str:
        # Escape single quotes so a value can't break out of the literal even if a
        # caller ever passes user data (review low/lancedb-sql). OQ-3 stop-gap.
        return str(v).replace("'", "''")

    def _to_sql(self, filters: Dict) -> str | None:
        # OQ-3 (RFC-090): naive interpolation — parameterise before any user-facing
        # free-text filter reaches it. Filters today are internal canonical ids.
        if not filters:
            return None
        return " AND ".join(f"{k} = '{self._sql_str(v)}'" for k, v in filters.items())

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

    def search_hybrid(self, query: SearchQuery) -> List[ScoredResult]:
        """Native LanceDB hybrid: vector + BM25 fused in-engine by an RRF reranker.

        One ``query_type="hybrid"`` query per tier runs the dense + full-text search and
        the reciprocal-rank fusion *inside the engine*, returning ``_relevance_score`` —
        replacing the former Python-side vector+BM25+RRF fan-out (ADR-099 Stage 2). The
        heavy ``embedding`` column is projected away (never read back). Per-table RRF
        scores share a scale, so results across tiers are merged by that score. (#995)
        """
        from lancedb.rerankers import RRFReranker

        reranker = RRFReranker()
        where = self._to_sql(query.filters)
        hits: List[tuple[float, Dict[str, Any], str]] = []
        for tier in self._tables_for_tier(query.tier):
            table = self._open_if_exists(tier)  # read must not create empty tables on disk
            if table is None:
                continue
            payload_cols = [c for c in table.schema.names if c != "embedding"]
            req = (
                table.search(query_type="hybrid")
                .vector(query.embedding)
                .text(query.text)
                .rerank(reranker)
            )
            if payload_cols:
                req = req.select(payload_cols)
            if where:
                req = req.where(where)
            for row in req.limit(query.k).to_list():
                score = float(row.get("_relevance_score", 0.0) or 0.0)
                row.pop("embedding", None)  # never shipped back; keep the payload lean
                hits.append((score, row, tier))
        hits.sort(key=lambda h: h[0], reverse=True)
        return [
            ScoredResult(
                doc_id=str(row.get("id")),
                score=score,
                rank=i + 1,
                payload=row,
                signal="hybrid",
                source_tier=tier,
            )
            for i, (score, row, tier) in enumerate(hits)
        ]

    # --- writes ----------------------------------------------------------------

    def _upsert_many(self, tier: str, rows: List[Dict[str, Any]]) -> None:
        """Upsert a batch of rows in a SINGLE merge_insert transaction.

        One transaction per batch (not per row) is what keeps the index from
        fragmenting: LanceDB writes one data file + one version per ``execute``, so
        batching N rows is N× fewer fragments/versions than per-document upserts.
        """
        if not rows:
            return
        table = self._ensure_table(tier)
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def _upsert(self, tier: str, data: Dict[str, Any]) -> None:
        self._upsert_many(tier, [data])

    def upsert_segment(self, doc: SegmentDocument) -> None:
        """Insert or update a Tier-1 segment document."""
        self._upsert("segment", dataclasses.asdict(doc))

    def upsert_insight(self, doc: InsightDocument) -> None:
        """Insert or update a Tier-2 insight document."""
        self._upsert("insight", dataclasses.asdict(doc))

    def upsert_aux(self, doc: AuxDocument) -> None:
        """Insert or update an aux document (kg_entity / kg_topic / quote / summary)."""
        self._upsert("aux", dataclasses.asdict(doc))

    def upsert_segments(self, docs: List[SegmentDocument]) -> None:
        """Batch-upsert Tier-1 segments in one transaction (see ``_upsert_many``)."""
        self._upsert_many("segment", [dataclasses.asdict(d) for d in docs])

    def upsert_insights(self, docs: List[InsightDocument]) -> None:
        """Batch-upsert Tier-2 insights in one transaction."""
        self._upsert_many("insight", [dataclasses.asdict(d) for d in docs])

    def upsert_auxes(self, docs: List[AuxDocument]) -> None:
        """Batch-upsert aux rows in one transaction."""
        self._upsert_many("aux", [dataclasses.asdict(d) for d in docs])

    def delete(self, doc_id: str, tier: Tier) -> None:
        """Delete a document by id; ``tier="all"`` removes from every table."""
        tiers = ("segment", "insight", "aux") if tier == "all" else (tier,)
        for t in tiers:
            table = self._open_if_exists(t)
            if table is not None:
                table.delete(f"id = '{self._sql_str(doc_id)}'")

    def health(self) -> Dict:
        """Return backend status + per-table row counts."""
        try:
            seg = self._open_if_exists("segment")
            ins = self._open_if_exists("insight")
            aux = self._open_if_exists("aux")
            return {
                "status": "ok",
                "segments": seg.count_rows() if seg is not None else 0,
                "insights": ins.count_rows() if ins is not None else 0,
                "aux": aux.count_rows() if aux is not None else 0,
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc)}


def stored_schema_version(lance_path: Path | str) -> Optional[int]:
    """Schema version recorded in a lance index's ``index_meta.json``.

    ``None`` when no index/meta exists; ``1`` for a pre-versioning index (meta present
    but no ``schema_version`` key — the initial #855 layout, before this field existed).
    """
    import json
    import os

    # py/path-injection sanitiser chain (same as read_index_meta): resolve the corpus
    # dir, confine the CONSTANT "index_meta.json" subpath under it, then normalise. The
    # filename literal mirrors LanceDBBackend.INDEX_META_FILE (kept literal so this stays
    # correct when callers monkeypatch the backend class in tests).
    root_res = safe_resolve_directory(Path(lance_path))
    if root_res is None:
        return None
    root_s = os.path.normpath(str(root_res))
    verified = safe_relpath_under_corpus_root(root_res, "index_meta.json")
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
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    v = data.get("schema_version")
    return int(v) if isinstance(v, int) else 1


def lance_index_is_stale(lance_path: Path | str) -> bool:
    """True when a lance index dir exists but its schema predates the code's.

    A stale index lacks columns the current read path expects (e.g. ``publish_date``),
    so read paths must skip it (reporting no_index, since there is no fallback) and
    (re)index moments must rebuild it rather than upsert into the incompatible schema.

    Requires *positive evidence* of an older schema: an existing ``index_meta.json``
    whose version is below the code's. A missing/unreadable meta is treated as
    not-stale — a real build always writes meta, so an absent one is an ambiguous
    partial/empty dir (the build path's own "already exists" no-op handles that), and
    the read path's try/except still reports no_index if such an index can't serve.
    """
    p = safe_resolve_directory(Path(lance_path))
    if p is None or not Path(p).is_dir():
        return False
    v = stored_schema_version(lance_path)
    return v is not None and v < LANCE_SCHEMA_VERSION
