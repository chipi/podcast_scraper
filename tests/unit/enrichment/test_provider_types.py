"""Unit tests for the provider-type registry and shipped types."""

from __future__ import annotations

import asyncio

import pytest

from podcast_scraper.enrichment.provider_types import (
    ProviderType,
    ProviderTypeRegistry,
    get_global_registry,
    register_provider_type,
)
from podcast_scraper.enrichment.scorers.embedding import TopicEmbeddingProvider
from podcast_scraper.enrichment.scorers.nli import FixedNliScorer
from podcast_scraper.enrichment.scorers.protocol import NliScore

# ---------------------------------------------------------------------------
# ProviderTypeRegistry — basic operations
# ---------------------------------------------------------------------------


def test_register_then_get_returns_the_same_type() -> None:
    reg = ProviderTypeRegistry()
    pt = ProviderType(
        name="t1",
        protocol="EmbeddingProvider",
        description="x",
        params_schema={"type": "object"},
        factory=lambda _: object(),
    )
    reg.register(pt)
    assert reg.get("t1") is pt


def test_register_duplicate_raises() -> None:
    reg = ProviderTypeRegistry()
    pt = ProviderType("dup", "X", "d", {}, lambda _: None)
    reg.register(pt)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(pt)


def test_list_for_protocol_returns_only_matching() -> None:
    reg = ProviderTypeRegistry()
    reg.register(ProviderType("a", "EmbeddingProvider", "d", {}, lambda _: None))
    reg.register(ProviderType("b", "NliScorer", "d", {}, lambda _: None))
    reg.register(ProviderType("c", "EmbeddingProvider", "d", {}, lambda _: None))
    names = [t.name for t in reg.list_for_protocol("EmbeddingProvider")]
    assert names == ["a", "c"]


def test_instantiate_forwards_config_to_factory() -> None:
    reg = ProviderTypeRegistry()
    captured: dict = {}

    def _factory(cfg: dict) -> dict:
        captured.update(cfg)
        return cfg

    reg.register(ProviderType("echo", "X", "d", {}, _factory))
    out = reg.instantiate("echo", {"key": "value"})
    assert out == {"key": "value"}
    assert captured == {"key": "value"}


def test_instantiate_unknown_raises_key_error() -> None:
    reg = ProviderTypeRegistry()
    with pytest.raises(KeyError):
        reg.instantiate("missing", {})


# ---------------------------------------------------------------------------
# Global registry — shipped types present + functional
# ---------------------------------------------------------------------------


def test_global_registry_has_shipped_types() -> None:
    reg = get_global_registry()
    names = {t.name for t in reg.all_types()}
    assert {"fake_for_test", "sentence_transformer_local"} <= names
    assert {"fixed_scripted", "deberta_local"} <= names


def test_fake_for_test_constructs_a_topic_embedding_provider() -> None:
    reg = get_global_registry()
    provider = reg.instantiate("fake_for_test", {"dim": 16})
    assert isinstance(provider, TopicEmbeddingProvider)
    vec = asyncio.run(provider.topic_vector("topic:hello"))
    assert isinstance(vec, list)
    assert len(vec) == 16
    assert all(isinstance(x, float) for x in vec)


def test_fake_for_test_invalid_dim_falls_back_to_default() -> None:
    reg = get_global_registry()
    provider = reg.instantiate("fake_for_test", {"dim": 5000})  # out of range
    vec = asyncio.run(provider.topic_vector("topic:hello"))
    assert len(vec) == 32  # default


def test_fixed_scripted_constructs_a_fixed_nli_scorer() -> None:
    reg = get_global_registry()
    scorer = reg.instantiate(
        "fixed_scripted",
        {"default_contradiction": 0.7, "default_neutral": 0.2, "default_entailment": 0.1},
    )
    assert isinstance(scorer, FixedNliScorer)
    score = asyncio.run(scorer.score("p", "h"))
    assert isinstance(score, NliScore)
    assert score.contradiction == pytest.approx(0.7)


def test_fixed_scripted_clamps_invalid_probabilities() -> None:
    reg = get_global_registry()
    scorer = reg.instantiate(
        "fixed_scripted",
        {"default_contradiction": 9.0, "default_neutral": "bad", "default_entailment": -1},
    )
    score = asyncio.run(scorer.score("p", "h"))
    # All three fell back to defaults: 0.05 / 0.85 / 0.10
    assert score.contradiction == pytest.approx(0.05)
    assert score.neutral == pytest.approx(0.85)
    assert score.entailment == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# register_provider_type (process-scoped helper)
# ---------------------------------------------------------------------------


def test_register_provider_type_module_helper_lands_in_global_registry() -> None:
    reg = get_global_registry()
    # Avoid colliding with shipped names
    name = "__test_only_provider_type__"
    if name in {t.name for t in reg.all_types()}:
        # Cleanup from a previous failing run.
        reg._types.pop(name, None)  # type: ignore[attr-defined]
    register_provider_type(
        name=name,
        protocol="ProbeProtocol",
        description="test",
        params_schema={},
        factory=lambda _: "ok",
    )
    try:
        assert reg.get(name).description == "test"
        assert reg.instantiate(name, {}) == "ok"
    finally:
        reg._types.pop(name, None)  # type: ignore[attr-defined]
