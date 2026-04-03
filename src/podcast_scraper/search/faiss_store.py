"""FAISS-backed local vector store for semantic corpus search (Phase 1 / #484)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from podcast_scraper.search.protocol import IndexStats, SearchResult

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.faiss"
METADATA_FILE = "metadata.json"
ID_MAP_FILE = "id_map.json"
INDEX_META_FILE = "index_meta.json"
FORMAT_VERSION = 1
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Persisted index_kind (back-compat: flat string unchanged)
INDEX_VARIANT_FLAT = "faiss_flat_ip_idmap"
INDEX_VARIANT_IVF = "faiss_ivf_flat_ip_idmap"
INDEX_VARIANT_PQ = "faiss_ivfpq_idmap"

# RFC-061 / #484: scale index structure by approximate corpus size
FAISS_AUTO_IVF_MIN_VECTORS = 100_000
FAISS_AUTO_IVFPQ_MIN_VECTORS = 1_000_000


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row in place (FAISS IP = cosine for unit vectors)."""
    import faiss

    if vectors.ndim != 2:
        raise ValueError("Expected 2-D embedding matrix")
    faiss.normalize_L2(vectors)
    return vectors


def _metadata_matches(meta: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = meta.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _nlist_for_ntotal(ntotal: int) -> int:
    """IVF list count: sqrt heuristic, capped, with min points per centroid (~39)."""
    if ntotal < 4:
        return 1
    raw = int(np.sqrt(float(ntotal)))
    cap_by_train = max(1, ntotal // 39)
    return int(min(max(raw, 4), cap_by_train, 4096))


def _pq_m_for_dim(d: int) -> int:
    """Subquantizer count m so d % m == 0 (prefer 48 for 384-dim MiniLM)."""
    for m in (48, 64, 32, 24, 16, 12, 8):
        if d % m == 0:
            return m
    return 1


def _extract_idmap_vectors(index: Any) -> Tuple[np.ndarray, np.ndarray]:
    """From IndexIDMap over IndexFlatIP, return (xb, faiss_ids)."""
    import faiss

    ntotal = int(index.ntotal)
    if ntotal == 0:
        return np.zeros((0, int(faiss.downcast_index(index.index).d)), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        )
    inner = faiss.downcast_index(index.index)
    d = int(inner.d)
    xb = np.empty((ntotal, d), dtype=np.float32)
    ids = np.empty((ntotal,), dtype=np.int64)
    for i in range(ntotal):
        xb[i] = inner.reconstruct(i)
        ids[i] = int(index.id_map.at(i))
    return xb, ids


def _is_flat_idmap(index: Any) -> bool:
    import faiss

    try:
        inner = faiss.downcast_index(index.index)
        return type(inner).__name__ == "IndexFlatIP"
    except Exception:
        return False


def _is_ivf_idmap(index: Any) -> bool:
    import faiss

    try:
        inner = faiss.downcast_index(index.index)
        return "IndexIVF" in type(inner).__name__
    except Exception:
        return False


def _set_ivf_nprobe(index: Any) -> None:
    import faiss

    try:
        inner = faiss.downcast_index(index.index)
        if hasattr(inner, "nprobe") and hasattr(inner, "nlist"):
            nlist = int(inner.nlist)
            inner.nprobe = int(min(max(nlist // 8, 1), 64, nlist))
    except Exception:
        pass


class FaissVectorStore:
    """FAISS IndexIDMap (FlatIP, IVFFlat, or IVFPQ inner) with JSON sidecars (#484)."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        index_dir: Optional[Path] = None,
    ) -> None:
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        import faiss

        self._embedding_dim = embedding_dim
        self._embedding_model = embedding_model
        self._index_dir = Path(index_dir) if index_dir is not None else None
        self._doc_to_faiss_id: Dict[str, int] = {}
        self._faiss_to_doc: Dict[int, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1
        self._created_at: Optional[str] = None
        self._last_updated = ""
        self._index_variant = INDEX_VARIANT_FLAT
        inner = faiss.IndexFlatIP(embedding_dim)
        self._index: Any = faiss.IndexIDMap(inner)

    @classmethod
    def load(cls, index_dir: Path | str) -> FaissVectorStore:
        """Load index and sidecars from ``index_dir`` (raises if index is missing)."""
        import faiss

        path = Path(index_dir)
        vec_path = path / VECTORS_FILE
        meta_path = path / INDEX_META_FILE
        if not vec_path.is_file():
            raise FileNotFoundError(f"Missing FAISS index file: {vec_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing index metadata: {meta_path}")

        meta_blob = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = int(meta_blob["embedding_dim"])
        model = str(meta_blob.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
        inst = cls(dim, embedding_model=model, index_dir=path)
        inst._created_at = meta_blob.get("created_at")
        inst._last_updated = str(meta_blob.get("last_updated", ""))
        kind = str(meta_blob.get("index_kind", INDEX_VARIANT_FLAT))
        inst._index_variant = (
            kind
            if kind in (INDEX_VARIANT_FLAT, INDEX_VARIANT_IVF, INDEX_VARIANT_PQ)
            else INDEX_VARIANT_FLAT
        )

        inst._index = faiss.read_index(str(vec_path))
        if int(getattr(inst._index, "d", 0)) != dim:
            raise ValueError("FAISS index dimension does not match index_meta.json")

        md_path = path / METADATA_FILE
        id_path = path / ID_MAP_FILE
        if not md_path.is_file() or not id_path.is_file():
            raise FileNotFoundError(f"Missing {METADATA_FILE} or {ID_MAP_FILE} under {path}")
        inst._metadata = json.loads(md_path.read_text(encoding="utf-8"))
        id_map: Dict[str, int] = {
            str(k): int(v) for k, v in json.loads(id_path.read_text(encoding="utf-8")).items()
        }
        inst._doc_to_faiss_id = id_map
        inst._faiss_to_doc = {fid: doc_id for doc_id, fid in id_map.items()}
        if len(inst._faiss_to_doc) != len(inst._doc_to_faiss_id):
            raise ValueError("id_map contains duplicate faiss ids")
        inst._next_id = max(inst._doc_to_faiss_id.values(), default=0) + 1
        if inst._index.ntotal != len(inst._metadata):
            logger.warning(
                "FAISS ntotal (%s) != metadata entries (%s); search may be inconsistent",
                inst._index.ntotal,
                len(inst._metadata),
            )
        return inst

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def index_dir(self) -> Optional[Path]:
        return self._index_dir

    @property
    def index_variant(self) -> str:
        return self._index_variant

    @property
    def ntotal(self) -> int:
        """Number of vectors in the index."""
        return int(self._index.ntotal)

    def upsert(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Insert or replace one vector."""
        self.batch_upsert([doc_id], [embedding], [metadata])

    def batch_upsert(
        self,
        doc_ids: List[str],
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """Bulk upsert; duplicate ``doc_ids`` in one call keep the last row."""
        if not (len(doc_ids) == len(embeddings) == len(metadata_list)):
            raise ValueError("doc_ids, embeddings, and metadata_list must have same length")
        merged: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}
        for doc_id, emb, meta in zip(doc_ids, embeddings, metadata_list):
            merged[doc_id] = (emb, dict(meta))
        if not merged:
            return

        to_remove: List[int] = []
        fids: List[int] = []
        ordered_ids: List[str] = list(merged.keys())
        for doc_id in ordered_ids:
            emb, _ = merged[doc_id]
            if len(emb) != self._embedding_dim:
                raise ValueError(f"Embedding dim {len(emb)} != store dim {self._embedding_dim}")
            if doc_id in self._doc_to_faiss_id:
                fid = self._doc_to_faiss_id[doc_id]
                to_remove.append(fid)
            else:
                fid = self._next_id
                self._next_id += 1
                self._doc_to_faiss_id[doc_id] = fid
                self._faiss_to_doc[fid] = doc_id
            fids.append(fid)

        unique_remove = list(dict.fromkeys(to_remove))
        if unique_remove:
            self._index.remove_ids(np.array(unique_remove, dtype=np.int64))

        mat = np.array([merged[d][0] for d in ordered_ids], dtype=np.float32)
        _l2_normalize_rows(mat)
        id_arr = np.array(fids, dtype=np.int64)
        self._index.add_with_ids(mat, id_arr)

        for doc_id in ordered_ids:
            self._metadata[doc_id] = merged[doc_id][1]

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        *,
        overfetch_factor: int = 3,
    ) -> List[SearchResult]:
        """Inner-product search with optional metadata post-filtering."""
        if len(query_embedding) != self._embedding_dim:
            raise ValueError(f"Query dim {len(query_embedding)} != store dim {self._embedding_dim}")
        if top_k < 1:
            return []
        if self._index.ntotal == 0:
            return []

        import faiss

        _set_ivf_nprobe(self._index)

        k_fetch = min(
            max(top_k * max(overfetch_factor, 1), top_k),
            int(self._index.ntotal),
        )
        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)
        scores, ids = self._index.search(q, k_fetch)
        out: List[SearchResult] = []
        for score, fid in zip(scores[0].tolist(), ids[0].tolist()):
            if int(fid) < 0:
                continue
            doc_id = self._faiss_to_doc.get(int(fid))
            if doc_id is None:
                continue
            meta = self._metadata.get(doc_id, {})
            if not _metadata_matches(meta, filters):
                continue
            out.append(SearchResult(doc_id=doc_id, score=float(score), metadata=dict(meta)))
            if len(out) >= top_k:
                break
        return out

    def doc_ids_for_episode(self, episode_id: str) -> List[str]:
        """Return all indexed doc ids whose metadata ``episode_id`` matches."""
        out: List[str] = []
        for doc_id, meta in self._metadata.items():
            if meta.get("episode_id") == episode_id:
                out.append(doc_id)
        return out

    def delete(self, doc_ids: List[str]) -> None:
        """Remove documents from the index and metadata tables."""
        if not doc_ids:
            return
        fids: List[int] = []
        for doc_id in doc_ids:
            fid = self._doc_to_faiss_id.pop(doc_id, None)
            self._metadata.pop(doc_id, None)
            if fid is not None:
                self._faiss_to_doc.pop(fid, None)
                fids.append(fid)
        unique = list(dict.fromkeys(fids))
        if unique:
            self._index.remove_ids(np.array(unique, dtype=np.int64))

    def maybe_upgrade_approximate_index(
        self,
        mode: str = "auto",
        *,
        ivf_min_vectors: Optional[int] = None,
        pq_min_vectors: Optional[int] = None,
    ) -> None:
        """Rebuild inner index as IVF or PQ when thresholds/mode match (#484).

        Only upgrades from a flat inner index. Safe to call after incremental upserts.
        On failure, logs and keeps the current index.
        """
        import faiss

        ivf_t = ivf_min_vectors if ivf_min_vectors is not None else FAISS_AUTO_IVF_MIN_VECTORS
        pq_t = pq_min_vectors if pq_min_vectors is not None else FAISS_AUTO_IVFPQ_MIN_VECTORS
        ntotal = int(self._index.ntotal)
        if ntotal < 2:
            return

        want: Optional[str] = None
        if mode == "flat":
            return
        if mode == "ivf_flat":
            want = "ivf"
        elif mode == "ivfpq":
            want = "pq"
        elif mode == "auto":
            if ntotal >= pq_t:
                want = "pq"
            elif ntotal >= ivf_t:
                want = "ivf"
            else:
                return
        else:
            return

        if not _is_flat_idmap(self._index):
            # Second hop: IVFFlat → IVFPQ when corpus crosses PQ threshold under auto/ivfpq.
            if not (
                want == "pq"
                and _is_ivf_idmap(self._index)
                and self._index_variant == INDEX_VARIANT_IVF
            ):
                return

        try:
            xb, faiss_ids = _extract_idmap_vectors(self._index)
        except Exception as exc:
            logger.warning("FAISS approximate upgrade skipped (extract): %s", exc)
            return

        d = self._embedding_dim
        nlist = _nlist_for_ntotal(ntotal)
        quantizer = faiss.IndexFlatIP(d)

        try:
            if want == "ivf":
                inner = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                new_index = faiss.IndexIDMap(inner)
                new_index.train(xb)
                new_index.add_with_ids(xb, faiss_ids)
                self._index = new_index
                self._index_variant = INDEX_VARIANT_IVF
                logger.info("FAISS index upgraded to IVFFlat (nlist=%s, ntotal=%s)", nlist, ntotal)
            elif want == "pq":
                m = _pq_m_for_dim(d)
                inner_pq = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                new_index = faiss.IndexIDMap(inner_pq)
                new_index.train(xb)
                new_index.add_with_ids(xb, faiss_ids)
                self._index = new_index
                self._index_variant = INDEX_VARIANT_PQ
                logger.info(
                    "FAISS index upgraded to IVFPQ (nlist=%s, m=%s, ntotal=%s)",
                    nlist,
                    m,
                    ntotal,
                )
        except Exception as exc:
            logger.warning("FAISS approximate upgrade failed; keeping flat index: %s", exc)

    def persist(self, index_dir: Optional[Path] = None) -> None:
        """Write ``vectors.faiss``, ``metadata.json``, ``id_map.json``, ``index_meta.json``."""
        import faiss

        target = Path(index_dir) if index_dir is not None else self._index_dir
        if target is None:
            raise ValueError("index_dir is required to persist (pass or set on constructor)")
        target.mkdir(parents=True, exist_ok=True)
        now = _utc_iso()
        if self._created_at is None:
            self._created_at = now
        self._last_updated = now

        faiss.write_index(self._index, str(target / VECTORS_FILE))
        (target / METADATA_FILE).write_text(
            json.dumps(self._metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (target / ID_MAP_FILE).write_text(
            json.dumps(self._doc_to_faiss_id, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        meta_blob = {
            "format_version": FORMAT_VERSION,
            "embedding_dim": self._embedding_dim,
            "embedding_model": self._embedding_model,
            "index_kind": self._index_variant,
            "created_at": self._created_at,
            "last_updated": self._last_updated,
        }
        (target / INDEX_META_FILE).write_text(
            json.dumps(meta_blob, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._index_dir = target

    def stats(self) -> IndexStats:
        """Aggregate counts; ``index_size_bytes`` uses on-disk files when present."""
        type_counter: Counter[str] = Counter()
        feeds: Set[str] = set()
        for meta in self._metadata.values():
            dt = meta.get("doc_type")
            if isinstance(dt, str) and dt:
                type_counter[dt] += 1
            feed = meta.get("feed_id")
            if isinstance(feed, str) and feed.strip():
                feeds.add(feed.strip())

        size_b = 0
        if self._index_dir is not None:
            for name in (VECTORS_FILE, METADATA_FILE, ID_MAP_FILE, INDEX_META_FILE):
                p = self._index_dir / name
                if p.is_file():
                    size_b += p.stat().st_size

        last = self._last_updated or _utc_iso()
        return IndexStats(
            total_vectors=int(self._index.ntotal),
            doc_type_counts=dict(type_counter),
            feeds_indexed=sorted(feeds),
            embedding_model=self._embedding_model,
            embedding_dim=self._embedding_dim,
            last_updated=last,
            index_size_bytes=size_b,
        )
