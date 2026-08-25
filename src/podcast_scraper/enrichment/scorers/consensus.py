"""Concrete ``ConsensusScorer`` implementations (ADR-108).

Two shipped:

* :class:`FixedConsensusScorer` — deterministic, dependency-free. Returns scripted
  :class:`ConsensusSignal` for known ``(text_a, text_b)`` pairs (order-independent);
  defaults to a low-similarity signal for unknown pairs. Used by tests + CI smoke
  ([[feedback_no_llm_in_ci]]).

* :class:`NliEmbeddingConsensusScorer` — the production composite: embedding cosine
  (the *shared-question* gate) + max NLI contradiction over both directions (the
  *direction* gate). Real-corpus eval (docs/adr/ADR-108-nli-disagreement-enrichers-gated-dark.md)
  showed symmetric NLI *entailment* has ~0 recall for genuine agreement, while
  embedding proximity recalls it and contradiction filters the similar-but-opposite
  pairs → precision ~0.91 on prod-v2. Both models are CPU-local (no LLM).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Callable

from podcast_scraper.enrichment.scorers.protocol import ConsensusSignal, NliScorer


@dataclass
class FixedConsensusScorer:
    """Deterministic ``ConsensusScorer`` for tests + CI smoke.

    Pre-populate ``signals`` keyed by ``(text_a, text_b)`` tuples (looked up in
    either order); unknown pairs return ``default`` (low cosine → not consensus).
    """

    signals: dict[tuple[str, str], ConsensusSignal] = field(default_factory=dict)
    default: ConsensusSignal = field(
        default_factory=lambda: ConsensusSignal(cosine=0.0, contradiction=0.0)
    )

    async def score(self, text_a: str, text_b: str) -> ConsensusSignal:
        """ConsensusScorer.score impl — scripted signal, order-independent."""
        await asyncio.sleep(0)
        return self.signals.get((text_a, text_b)) or self.signals.get(
            (text_b, text_a), self.default
        )


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors; 0.0 on a degenerate (zero-norm) input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class NliEmbeddingConsensusScorer:
    """Production composite ``ConsensusScorer`` — embedding cosine + NLI contradiction.

    ``embed_text(text) -> vector`` is a sync callable (e.g.
    ``sentence-transformers.SentenceTransformer.encode``); ``nli`` is any
    :class:`NliScorer`. Text embeddings are cached across a run (the enricher
    scores each insight against many partners). Cosine is the shared-question gate;
    ``max(contradiction_ab, contradiction_ba)`` is the direction gate.
    """

    embed_text: Callable[[str], list[float]]
    nli: NliScorer
    # #1817: optional batch encoder (texts -> vectors in one call). When set AND the
    # nli scorer exposes ``score_batch``, the enricher takes the vectorized path:
    # bulk-embed unique texts, matrix cosine gate, then ONE batched NLI pass over
    # only the gate-passing pairs — instead of two model calls per candidate pair.
    embed_texts: Callable[[list[str]], list[list[float]]] | None = None
    _cache: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False)

    @property
    def supports_batch(self) -> bool:
        """True when both the embedding and NLI sides can batch (#1817)."""
        return self.embed_texts is not None and hasattr(self.nli, "score_batch")

    async def _vec(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = await asyncio.to_thread(self.embed_text, text)
        vec = [float(x) for x in (vector or [])]
        self._cache[text] = vec
        return vec

    async def embed_texts_batch(self, texts: list[str]) -> dict[str, list[float]]:
        """Bulk-embed ``texts`` (deduped, cached); requires ``embed_texts``."""
        out: dict[str, list[float]] = {}
        missing: list[str] = []
        for t in texts:
            cached = self._cache.get(t)
            if cached is not None:
                out[t] = cached
            elif t not in missing:
                missing.append(t)
        if missing and self.embed_texts is not None:
            vectors = await asyncio.to_thread(self.embed_texts, missing)
            for t, v in zip(missing, vectors):
                vec = [float(x) for x in (v or [])]
                self._cache[t] = vec
                out[t] = vec
        return out

    async def contradictions_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batched ``max(contradiction_ab, contradiction_ba)`` per pair (#1817).

        Both directions of every pair go into ONE ``score_batch`` call so the
        cross-encoder sees a single large batch.
        """
        if not pairs:
            return []
        both: list[tuple[str, str]] = []
        for a, b in pairs:
            both.append((a, b))
            both.append((b, a))
        scores = await self.nli.score_batch(both)  # type: ignore[attr-defined]
        out: list[float] = []
        for i in range(0, len(scores), 2):
            out.append(max(scores[i].contradiction, scores[i + 1].contradiction))
        return out

    async def score(self, text_a: str, text_b: str) -> ConsensusSignal:
        """ConsensusScorer.score impl — cosine(a,b) + max NLI contradiction both ways."""
        va = await self._vec(text_a)
        vb = await self._vec(text_b)
        cosine = _cosine(va, vb)
        ab = await self.nli.score(text_a, text_b)
        ba = await self.nli.score(text_b, text_a)
        contradiction = max(ab.contradiction, ba.contradiction)
        cost = ab.cost_usd + ba.cost_usd
        return ConsensusSignal(cosine=cosine, contradiction=contradiction, cost_usd=cost)


__all__ = [
    "FixedConsensusScorer",
    "NliEmbeddingConsensusScorer",
]
