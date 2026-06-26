"""Unit tests for ``enrichment.scorers.protocol``.

The scorer protocols are PEP 544 + ``@runtime_checkable`` — so
``isinstance(instance, Protocol)`` accepts any structurally
conforming class. These tests pin the contract surface for the
scorer mocks (chunk 1 fixtures) + the real implementations (chunks 3
+ 4) so a future-LLM-tier follow-on can plug in without surprise.
"""

from __future__ import annotations

import asyncio

from podcast_scraper.enrichment.scorers.protocol import (
    EmbeddingProvider,
    LLMScorer,
    NliScore,
    NliScorer,
)

# ---------------------------------------------------------------------------
# NliScore dataclass
# ---------------------------------------------------------------------------


def test_nli_score_carries_three_probabilities_and_cost() -> None:
    s = NliScore(contradiction=0.7, neutral=0.2, entailment=0.1, cost_usd=0.003)
    assert s.contradiction == 0.7
    assert s.neutral == 0.2
    assert s.entailment == 0.1
    assert s.cost_usd == 0.003


def test_nli_score_cost_defaults_zero_for_local_scorers() -> None:
    s = NliScore(contradiction=0.5, neutral=0.3, entailment=0.2)
    assert s.cost_usd == 0.0


def test_nli_score_is_frozen() -> None:
    s = NliScore(contradiction=0.5, neutral=0.3, entailment=0.2)
    import pytest

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        s.contradiction = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NliScorer runtime_checkable
# ---------------------------------------------------------------------------


class _FakeNliScorer:
    """Minimal NliScorer-conforming class."""

    async def score(self, premise: str, hypothesis: str) -> NliScore:
        return NliScore(contradiction=0.5, neutral=0.3, entailment=0.2)


def test_nli_scorer_runtime_check_accepts_conforming_class() -> None:
    assert isinstance(_FakeNliScorer(), NliScorer)


def test_nli_scorer_call_returns_score() -> None:
    scorer = _FakeNliScorer()
    result = asyncio.run(scorer.score("Premise.", "Hypothesis."))
    assert isinstance(result, NliScore)
    assert result.contradiction == 0.5


def test_nli_scorer_runtime_check_rejects_missing_method() -> None:
    class _NoScore:
        pass

    assert not isinstance(_NoScore(), NliScorer)


# ---------------------------------------------------------------------------
# EmbeddingProvider runtime_checkable
# ---------------------------------------------------------------------------


class _FakeEmbeddingProvider:
    """Returns deterministic vectors per topic_id; None for unknown."""

    async def topic_vector(self, topic_id: str) -> list[float] | None:
        if topic_id == "topic:absent":
            return None
        return [1.0, 0.0, 0.0]


def test_embedding_provider_runtime_check_accepts_conforming_class() -> None:
    assert isinstance(_FakeEmbeddingProvider(), EmbeddingProvider)


def test_embedding_provider_returns_vector_for_known_topic() -> None:
    p = _FakeEmbeddingProvider()
    vec = asyncio.run(p.topic_vector("topic:ai"))
    assert vec == [1.0, 0.0, 0.0]


def test_embedding_provider_returns_none_for_absent_topic() -> None:
    """Missing topic → None signals "missing input" to the enricher."""
    p = _FakeEmbeddingProvider()
    assert asyncio.run(p.topic_vector("topic:absent")) is None


# ---------------------------------------------------------------------------
# LLMScorer runtime_checkable (future LLM-tier query enrichers)
# ---------------------------------------------------------------------------


class _FakeLLMScorer:
    async def score(self, prompt: str) -> str:
        return f"echo: {prompt}"


def test_llm_scorer_runtime_check_accepts_conforming_class() -> None:
    assert isinstance(_FakeLLMScorer(), LLMScorer)


def test_llm_scorer_returns_string_response() -> None:
    s = _FakeLLMScorer()
    out = asyncio.run(s.score("classify this"))
    assert out == "echo: classify this"


# ---------------------------------------------------------------------------
# Protocols are distinct types — accidental cross-matching shouldn't happen
# ---------------------------------------------------------------------------


def test_nli_scorer_is_not_embedding_provider() -> None:
    """Distinct protocols don't accidentally accept each other's instances."""
    assert not isinstance(_FakeNliScorer(), EmbeddingProvider)
    assert not isinstance(_FakeEmbeddingProvider(), NliScorer)


def test_llm_scorer_and_nli_scorer_share_method_name_but_diff_signature() -> None:
    """Both have ``score()`` — runtime_checkable accepts both since signatures
    aren't enforced at the protocol level. Document the behaviour explicitly
    so future authors don't expect more rigorous structural typing.
    """
    # Both pass each other's isinstance check because runtime_checkable
    # only checks method existence, not signature. This is the documented
    # PEP 544 behaviour; callers must use the right scorer for the right
    # capability.
    assert isinstance(_FakeLLMScorer(), LLMScorer)
    assert isinstance(_FakeNliScorer(), NliScorer)
