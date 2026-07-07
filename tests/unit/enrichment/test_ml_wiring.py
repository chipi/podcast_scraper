"""Unit tests for the ``--with-ml`` wiring helper."""

from __future__ import annotations

import logging

import pytest

from podcast_scraper.enrichment.ml_wiring import register_ml_enrichers
from podcast_scraper.enrichment.protocol import EnricherSet
from podcast_scraper.enrichment.registry import EnricherRegistry


def test_register_ml_enrichers_constructs_topic_similarity_from_fake_provider() -> None:
    reg = EnricherRegistry()
    s = EnricherSet(
        enabled_enrichers=["topic_similarity"],
        per_enricher_config={
            "topic_similarity": {
                "top_k": 5,
                "provider": {"type": "fake_for_test", "dim": 16},
            },
        },
    )
    register_ml_enrichers(reg, s)
    assert "topic_similarity" in reg.all_ids()
    enricher = reg.get("topic_similarity")
    assert enricher.manifest.id == "topic_similarity"
    assert getattr(enricher, "_top_k") == 5  # explicit knob honored


def test_topic_similarity_no_knob_uses_enricher_tuned_default_not_shadowed_10() -> None:
    """Regression: with no top_k knob, the enricher's tuned default (7, #1105) must be used.

    The builder used to default top_k to 10, silently shadowing the tuning on the --with-ml
    path (no profile sets the knob), so topic_similarity shipped untuned at 10.
    """
    reg = EnricherRegistry()
    s = EnricherSet(
        enabled_enrichers=["topic_similarity"],
        per_enricher_config={
            "topic_similarity": {"provider": {"type": "fake_for_test", "dim": 16}},
        },
    )
    register_ml_enrichers(reg, s)
    assert getattr(reg.get("topic_similarity"), "_top_k") == 7


def test_register_ml_enrichers_constructs_nli_contradiction_from_fixed_scripted() -> None:
    reg = EnricherRegistry()
    s = EnricherSet(
        enabled_enrichers=["nli_contradiction"],
        per_enricher_config={
            "nli_contradiction": {
                "threshold": 0.6,
                "provider": {"type": "fixed_scripted"},
            },
        },
    )
    register_ml_enrichers(reg, s)
    assert "nli_contradiction" in reg.all_ids()


def test_register_ml_enrichers_skips_with_hint_when_provider_block_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reg = EnricherRegistry()
    s = EnricherSet(
        enabled_enrichers=["topic_similarity"],
        # No provider block — operator forgot.
        per_enricher_config={"topic_similarity": {"top_k": 5}},
    )
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.enrichment.ml_wiring"):
        register_ml_enrichers(reg, s)
    assert "topic_similarity" not in reg.all_ids()
    msgs = [r.message for r in caplog.records]
    assert any("no 'provider' block" in m for m in msgs), msgs


def test_register_ml_enrichers_skips_with_hint_on_unknown_provider_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reg = EnricherRegistry()
    s = EnricherSet(
        enabled_enrichers=["topic_similarity"],
        per_enricher_config={
            "topic_similarity": {
                "provider": {"type": "nope_doesnt_exist"},
            },
        },
    )
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.enrichment.ml_wiring"):
        register_ml_enrichers(reg, s)
    assert "topic_similarity" not in reg.all_ids()
    assert any("not registered" in r.message for r in caplog.records)


def test_register_ml_enrichers_noop_for_unknown_enricher_ids() -> None:
    """An enricher id with no entry in the dispatcher map is just left alone."""
    reg = EnricherRegistry()
    s = EnricherSet(enabled_enrichers=["temporal_velocity", "future_ml_enricher"])
    register_ml_enrichers(reg, s)
    # No crash; nothing gets registered (no provider block, no builder match)
    assert reg.all_ids() == []


def test_register_ml_enrichers_preserves_existing_registrations() -> None:
    """If a test fixture already registered topic_similarity, don't double-register."""
    from podcast_scraper.enrichment.enrichers.topic_similarity import (
        TopicSimilarityEnricher,
    )
    from podcast_scraper.enrichment.scorers.embedding import HashEmbedder, TopicEmbeddingProvider

    reg = EnricherRegistry()
    reg.register(
        TopicSimilarityEnricher(provider=TopicEmbeddingProvider(embed_text=HashEmbedder(dim=8)))
    )
    s = EnricherSet(
        enabled_enrichers=["topic_similarity"],
        per_enricher_config={
            "topic_similarity": {"provider": {"type": "fake_for_test", "dim": 16}},
        },
    )
    # Should NOT raise "already registered".
    register_ml_enrichers(reg, s)
    assert reg.all_ids() == ["topic_similarity"]
