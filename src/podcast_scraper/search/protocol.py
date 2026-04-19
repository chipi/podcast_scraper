"""Vector index contracts for semantic corpus search."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class SearchResult:
    """One hit from vector search with similarity score and sidecar metadata."""

    doc_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexStats:
    """Aggregate statistics for a persisted or in-memory vector index."""

    total_vectors: int
    doc_type_counts: Dict[str, int]
    feeds_indexed: List[str]
    embedding_model: str
    embedding_dim: int
    last_updated: str
    index_size_bytes: int


@runtime_checkable
class VectorStore(Protocol):
    """Backend-agnostic vector index (FAISS shipped; Qdrant optional)."""

    def upsert(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Insert or replace one vector and its metadata."""
        ...

    def batch_upsert(
        self,
        doc_ids: List[str],
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """Bulk insert or replace; last wins when ``doc_ids`` contains duplicates."""
        ...

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        *,
        overfetch_factor: int = 3,
    ) -> List[SearchResult]:
        """Return up to ``top_k`` results, optionally post-filtering on metadata."""
        ...

    def delete(self, doc_ids: List[str]) -> None:
        """Remove vectors (and metadata) for the given document ids."""
        ...

    def persist(self, index_dir: Optional[Path] = None) -> None:
        """Write index and sidecars to disk (``index_dir`` or the store's bound path)."""
        ...

    def stats(self) -> IndexStats:
        """Return counts and bookkeeping fields for the current index."""
        ...
