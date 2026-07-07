"""Unit tests for the composite ConsensusScorer (ADR-108).

Covers FixedConsensusScorer (scripted, order-independent), the NliEmbeddingConsensusScorer
composite (cosine + max contradiction over both directions), and provider-type registration.
"""

from __future__ import annotations

import asyncio

from podcast_scraper.enrichment.provider_types import get_global_registry
from podcast_scraper.enrichment.scorers.consensus import (
    _cosine,
    FixedConsensusScorer,
    NliEmbeddingConsensusScorer,
)
from podcast_scraper.enrichment.scorers.nli import FixedNliScorer
from podcast_scraper.enrichment.scorers.protocol import ConsensusSignal, NliScore


def test_cosine_basic() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([], [1.0]) == 0.0  # degenerate → 0


def test_fixed_consensus_scorer_is_order_independent() -> None:
    sig = ConsensusSignal(cosine=0.9, contradiction=0.02)
    scorer = FixedConsensusScorer(signals={("a", "b"): sig})
    assert asyncio.run(scorer.score("a", "b")) is sig
    assert asyncio.run(scorer.score("b", "a")) is sig  # either order
    # unknown → default (cosine 0 → not consensus)
    assert asyncio.run(scorer.score("x", "y")).cosine == 0.0


def test_nli_embedding_composite_cosine_and_max_contradiction() -> None:
    vecs = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
    nli = FixedNliScorer(
        scores={
            ("a", "b"): NliScore(contradiction=0.1, neutral=0.8, entailment=0.1),
            ("b", "a"): NliScore(contradiction=0.4, neutral=0.5, entailment=0.1),
        },
        default=NliScore(contradiction=0.0, neutral=1.0, entailment=0.0),
    )
    scorer = NliEmbeddingConsensusScorer(embed_text=lambda t: vecs[t], nli=nli)
    sig = asyncio.run(scorer.score("a", "b"))
    assert sig.cosine == 1.0  # identical vectors
    assert sig.contradiction == 0.4  # max(0.1, 0.4) over both directions
    sig2 = asyncio.run(scorer.score("a", "c"))
    assert sig2.cosine == 0.0  # orthogonal


def test_consensus_provider_types_registered() -> None:
    reg = get_global_registry()
    names = {t.name for t in reg.list_for_protocol("ConsensusScorer")}
    assert {"consensus_local", "fixed_consensus"} <= names
    # fixed_consensus instantiates without model deps.
    scorer = reg.instantiate("fixed_consensus", {"default_cosine": 0.8})
    sig = asyncio.run(scorer.score("p", "q"))
    assert sig.cosine == 0.8 and sig.contradiction == 0.0
