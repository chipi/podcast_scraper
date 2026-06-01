"""Two-tier hybrid search backend contracts (RFC-090 §3.1–3.2).

Coexists with the single-signal ``VectorStore`` (``protocol.py`` / ``FaissVectorStore``)
until FAISS is deprecated (RFC-090 Stage 4, #858). The two-tier ``SearchBackend``
exposes BM25 and dense-vector retrieval **separately** so the retrieval layer can
fuse them — plus a KG-proximity signal (RFC-091) — via RRF (#856).

Tier 1 = transcript segments (raw evidence); Tier 2 = GIL insight nodes
(synthesized intelligence). Results are mixed by score, not grouped by tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Protocol, runtime_checkable

Tier = Literal["segment", "insight", "all"]


@dataclass
class SegmentDocument:
    """Tier 1 — a transcript chunk (raw evidence)."""

    id: str  # "{episode_id}_chunk_{n}"
    text: str  # 200-300 word chunk, 50-word overlap
    show_id: str
    episode_id: str
    start_time: float  # seconds from episode start
    end_time: float
    embedding: List[float] = field(default_factory=list)  # filled by embedding step
    speaker_id: Optional[str] = None  # if diarized
    linked_insight_ids: List[str] = field(default_factory=list)  # GIL insights by timestamp overlap
    source_tier: str = "segment"


@dataclass
class InsightDocument:
    """Tier 2 — a GIL insight node (synthesized claim / grounded quote)."""

    id: str  # GIL insight node id
    text: str
    show_id: str
    episode_id: str
    entity_type: str
    confidence: float
    derived: bool
    embedding: List[float] = field(default_factory=list)
    speaker_id: Optional[str] = None
    source_segment_id: Optional[str] = None  # back-ref to grounding-quote segment
    source_tier: str = "insight"


@dataclass
class SearchQuery:
    """A retrieval request over one or both tiers."""

    text: str
    embedding: List[float]
    filters: Dict = field(default_factory=dict)
    k: int = 20
    tier: Tier = "all"


@dataclass
class ScoredResult:
    """One ranked hit from a single signal (or the fused RRF list)."""

    doc_id: str
    score: float
    rank: int
    payload: Dict  # includes source_tier, linked_insight_ids / source_segment_id
    signal: str  # "bm25" | "vector" | "kg" | "rrf"
    source_tier: str  # "segment" | "insight" | "compound"


@dataclass
class CompoundResult:
    """A merged result when a segment and an insight refer to the same content."""

    doc_id: str  # segment id (primary key)
    score: float  # max(segment_score, insight_score)
    rank: int
    segment: ScoredResult
    insight: ScoredResult
    signal: str = "rrf"
    source_tier: str = "compound"


@runtime_checkable
class SearchBackend(Protocol):
    """Backend that exposes BM25 + vector retrieval separately, over two tiers.

    Decouples signal generation from fusion (RFC-090 KD-1): the retrieval layer
    runs RRF over the separate ranked lists, so a third (KG) signal plugs in
    without touching the backend. LanceDB is the initial implementation; cloud
    backends require no retrieval-layer change.
    """

    def search_bm25(self, query: SearchQuery) -> List[ScoredResult]:
        """BM25 (full-text) ranked results for *query* over its tier(s)."""
        ...

    def search_vector(self, query: SearchQuery) -> List[ScoredResult]:
        """Dense-vector ranked results for *query* over its tier(s)."""
        ...

    def upsert_segment(self, doc: SegmentDocument) -> None:
        """Insert or update a Tier-1 segment document."""
        ...

    def upsert_insight(self, doc: InsightDocument) -> None:
        """Insert or update a Tier-2 insight document."""
        ...

    def delete(self, doc_id: str, tier: Tier) -> None:
        """Delete a document by id from the given tier."""
        ...

    def create_indices(self) -> None:
        """Create the FTS + vector indices required for retrieval."""
        ...

    def health(self) -> Dict:
        """Return backend status and per-tier document counts."""
        ...
