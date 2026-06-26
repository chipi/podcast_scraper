"""Scorer protocols — async, runtime_checkable, injectable.

Real implementations land with their consuming enricher chunk:

* ``LanceDBEmbeddingProvider`` (chunk 3) implements ``EmbeddingProvider``.
* ``DeBERTaV3SmallNliScorer`` (chunk 4) implements ``NliScorer``.
* ``LLMScorer`` real implementations land with follow-on LLM-tier
  query-enricher RFC (Phase 4).

Scenario-driven mocks under ``tests/fixtures/enrichment/mock_scorers.py``
exercise the resilience pipeline without real models in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class NliScore:
    """Output of an NLI scorer for one ``(premise, hypothesis)`` pair.

    The three probabilities sum to ~1.0 (calibration not guaranteed —
    that's what the ``nli_contradiction`` enricher's Brier-score eval
    measures). ``cost_usd`` is populated by remote/paid scorers; local
    CPU scorers (DeBERTa-v3-small) leave it at ``0.0``.
    """

    contradiction: float
    neutral: float
    entailment: float
    cost_usd: float = 0.0


@runtime_checkable
class NliScorer(Protocol):
    """Async NLI scorer. Local DeBERTa + scenario mock both implement this."""

    async def score(self, premise: str, hypothesis: str) -> NliScore:
        """Return the NLI score distribution for the pair."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Async embedding provider keyed by topic id.

    LanceDB-backed implementation (chunk 3) and scenario mock (chunk 1
    fixtures) both implement this. Returns ``None`` when the ``topic_id``
    has no embedding in the backing store (signals "missing input" to
    the enricher, which produces ``BadInputError`` and ``status:
    failed``).
    """

    async def topic_vector(self, topic_id: str) -> list[float] | None:
        """Return the embedding vector for the topic id, or ``None`` if absent."""
        ...


@runtime_checkable
class LLMScorer(Protocol):
    """Async LLM scorer for future LLM-tier query enrichers.

    Not consumed by any chunk-1 enricher; the protocol ships so future
    LLM query enrichers (follow-on RFC) plug in without framework
    changes. Cost tracking flows through the existing
    ``record_provider_call_cost`` chain so the per-enricher
    ``max_cost_usd_per_run`` and run-wide
    ``max_total_cost_usd_per_run`` caps just work.
    """

    async def score(self, prompt: str) -> str:
        """Return the LLM scoring response for the prompt."""
        ...
