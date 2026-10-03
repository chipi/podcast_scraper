"""An in-memory `SearchBackend` that unit tests can use instead of a real index.

NOT A `MagicMock`. A mock returns whatever it was told to, so a test built on one asserts the
mock's configuration; this is a small real implementation of the port, so a test built on it
asserts behaviour. The difference matters most for the thing the design rests on: a row with no
vector must be unreachable by a dense signal, and here it is unreachable because **there is no
vector to match**, not because a filter excludes it. A fake that filtered on language would pass
every test while modelling the opposite of the guarantee.

Held honest by `tests/search_backend_contract.py`, which runs the same behaviour contract against
this and against the real `LanceDBBackend`. Seven unit files currently hand-roll their own fakes
with no such check — a fake could drift from LanceDB indefinitely and the unit suite would keep
passing, because it would be agreeing with itself.

SCOPE, deliberately small. Keyword search is substring matching, not BM25, and vector search
ranks by Euclidean distance with no index. Neither is a model of LanceDB's *ranking* — if a test
depends on ranking quality it needs the real backend (integration) or real embeddings (E2E). What
this models is the part our code depends on: which rows exist, which tier they are in, which
signal can see them, and what survives a delete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from podcast_scraper.search.backend import (
    AuxDocument,
    InsightDocument,
    ScoredResult,
    SearchQuery,
    SegmentDocument,
    Tier,
)


@dataclass
class _Row:
    """One stored row. ``embedding is None`` is how the keyword-only tier is modelled."""

    doc_id: str
    text: str
    tier: str
    embedding: Optional[List[float]]
    payload: Dict


@dataclass
class FakeSearchBackend:
    """In-memory stand-in for `SearchBackend`.

    Rows live in one dict keyed by ``(tier, doc_id)``, which is what makes "upsert" genuinely an
    upsert and lets ``delete`` be exact.
    """

    rows: Dict[tuple, _Row] = field(default_factory=dict)

    #: Mirrors `LanceDBBackend`: the tiers a dense signal can read, and the one it cannot.
    DENSE_TIERS = ("segment", "insight", "aux")
    KEYWORD_ONLY_TIERS = ("segment_nonen",)

    # --- routing ------------------------------------------------------------------------

    @staticmethod
    def _segment_tier_for(language: Optional[str]) -> str:
        """`None`/`en` is English; anything else is source-language.

        The same rule the real indexer applies, and the reason it is here rather than in a test:
        a fake that routed differently would make every routing assertion meaningless.
        """
        code = (language or "").strip().lower().split("-")[0]
        return "segment" if code in ("", "en") else "segment_nonen"

    # --- writes -------------------------------------------------------------------------

    def upsert_segment(self, doc: SegmentDocument) -> None:
        tier = self._segment_tier_for(getattr(doc, "language", None))
        # THE STRUCTURAL GUARANTEE: the source-language tier stores NO vector. Not "stores one and
        # hides it" — there is nothing for a dense query to match against.
        embedding = None if tier == "segment_nonen" else list(doc.embedding or [])
        self.rows[(tier, doc.id)] = _Row(
            doc_id=doc.id,
            text=doc.text,
            tier=tier,
            embedding=embedding,
            payload={
                "source_tier": tier,
                "episode_id": doc.episode_id,
                "show_id": doc.show_id,
                "language": getattr(doc, "language", None),
                "linked_insight_ids": list(doc.linked_insight_ids or []),
            },
        )

    def upsert_insight(self, doc: InsightDocument) -> None:
        self.rows[("insight", doc.id)] = _Row(
            doc_id=doc.id,
            text=getattr(doc, "text", "") or "",
            tier="insight",
            embedding=list(getattr(doc, "embedding", []) or []),
            payload={"source_tier": "insight"},
        )

    def upsert_aux(self, doc: AuxDocument) -> None:
        self.rows[("aux", doc.id)] = _Row(
            doc_id=doc.id,
            text=getattr(doc, "text", "") or "",
            tier="aux",
            embedding=list(getattr(doc, "embedding", []) or []),
            payload={"source_tier": "aux"},
        )

    def replace_segments(self, docs: List[SegmentDocument]) -> None:
        """Drop every segment tier, then write. Mirrors the real full-reindex write."""
        for key in [k for k in self.rows if k[0] in ("segment", *self.KEYWORD_ONLY_TIERS)]:
            del self.rows[key]
        for doc in docs:
            self.upsert_segment(doc)

    def delete(self, doc_id: str, tier: Tier) -> None:
        """``tier="all"`` means every tier THIS BACKEND HAS, keyword-only included.

        Spelled out because the real backend got it wrong: it resolved `"all"` to `DENSE_TIERS`
        and left source-language rows behind. The contract suite is what caught the disagreement.
        """
        tiers = (*self.DENSE_TIERS, *self.KEYWORD_ONLY_TIERS) if tier == "all" else (str(tier),)
        for t in tiers:
            self.rows.pop((t, doc_id), None)

    # --- reads --------------------------------------------------------------------------

    def _tiers_for(self, query: SearchQuery, *, keyword: bool) -> tuple:
        if query.tier == "all":
            return (*self.DENSE_TIERS, *self.KEYWORD_ONLY_TIERS) if keyword else self.DENSE_TIERS
        return (str(query.tier),)

    def search_bm25(self, query: SearchQuery) -> List[ScoredResult]:
        """Substring match, lowercased. Not BM25 — see the module docstring on scope."""
        needle = (query.text or "").strip().lower()
        tiers = self._tiers_for(query, keyword=True)
        hits = [
            r
            for (tier, _), r in sorted(self.rows.items())
            if tier in tiers and needle and needle in r.text.lower()
        ]
        return [
            ScoredResult(
                doc_id=r.doc_id,
                score=1.0,
                rank=i + 1,
                payload=dict(r.payload),
                signal="bm25",
                source_tier=r.tier,
            )
            for i, r in enumerate(hits[: query.k])
        ]

    def search_vector(self, query: SearchQuery) -> List[ScoredResult]:
        """Euclidean nearest neighbour over rows THAT HAVE a vector.

        A row with ``embedding is None`` is skipped for the only reason that matters: there is
        nothing to compute a distance against.
        """
        tiers = self._tiers_for(query, keyword=False)
        target = list(query.embedding or [])
        scored = []
        for (tier, _), r in sorted(self.rows.items()):
            if tier not in tiers or not r.embedding or not target:
                continue
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(r.embedding, target)))
            scored.append((dist, r))
        scored.sort(key=lambda pair: (pair[0], pair[1].doc_id))
        return [
            ScoredResult(
                doc_id=r.doc_id,
                score=1.0 / (1.0 + dist),
                rank=i + 1,
                payload=dict(r.payload),
                signal="vector",
                source_tier=r.tier,
            )
            for i, (dist, r) in enumerate(scored[: query.k])
        ]

    # --- introspection used by indexer logic --------------------------------------------

    def existing_tier_tables(self) -> List[str]:
        return sorted({tier for tier, _ in self.rows})

    def health(self) -> Dict:
        counts: Dict[str, int] = {}
        for tier, _ in self.rows:
            counts[tier] = counts.get(tier, 0) + 1
        return {"status": "ok", **counts}
