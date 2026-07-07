"""Unit tests for ``enrichment.registry``."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherSet,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    STATUS_OK,
)
from podcast_scraper.enrichment.registry import EnricherRegistry


class _FakeEnricher:
    """A minimal Enricher implementation for registry tests."""

    def __init__(self, manifest: EnricherManifest) -> None:
        self._manifest = manifest

    @property
    def manifest(self) -> EnricherManifest:
        return self._manifest

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict,
        ctx: RunContext,
    ) -> EnricherResult:
        return EnricherResult(status=STATUS_OK, data={"x": 1})


def _manifest(
    enricher_id: str,
    *,
    tier: EnricherTier = EnricherTier.DETERMINISTIC,
    requires_opt_in: bool = False,
) -> EnricherManifest:
    return EnricherManifest(
        id=enricher_id,
        version="1.0.0",
        scope=EnricherScope.EPISODE,
        tier=tier,
        reads=[".kg.json"],
        writes=f"{enricher_id}.json",
        description="test",
        requires_opt_in=requires_opt_in,
    )


# ---------------------------------------------------------------------------
# register / get / all_ids / clear
# ---------------------------------------------------------------------------


def test_register_then_get_returns_the_same_instance() -> None:
    reg = EnricherRegistry()
    e = _FakeEnricher(_manifest("topic_cooccurrence"))
    reg.register(e)
    assert reg.get("topic_cooccurrence") is e


def test_register_duplicate_raises_value_error() -> None:
    reg = EnricherRegistry()
    reg.register(_FakeEnricher(_manifest("topic_cooccurrence")))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_FakeEnricher(_manifest("topic_cooccurrence")))


def test_get_missing_raises_key_error() -> None:
    reg = EnricherRegistry()
    with pytest.raises(KeyError):
        reg.get("topic_cooccurrence")


def test_all_ids_preserves_insertion_order() -> None:
    reg = EnricherRegistry()
    reg.register(_FakeEnricher(_manifest("topic_cooccurrence")))
    reg.register(_FakeEnricher(_manifest("temporal_velocity")))
    reg.register(_FakeEnricher(_manifest("grounding_rate")))
    assert reg.all_ids() == [
        "topic_cooccurrence",
        "temporal_velocity",
        "grounding_rate",
    ]


def test_clear_empties_the_registry() -> None:
    reg = EnricherRegistry()
    reg.register(_FakeEnricher(_manifest("topic_cooccurrence")))
    reg.clear()
    assert reg.all_ids() == []


# ---------------------------------------------------------------------------
# list_enabled
# ---------------------------------------------------------------------------


def test_list_enabled_returns_only_enabled_enrichers() -> None:
    reg = EnricherRegistry()
    reg.register(_FakeEnricher(_manifest("topic_cooccurrence")))
    reg.register(_FakeEnricher(_manifest("temporal_velocity")))
    reg.register(_FakeEnricher(_manifest("grounding_rate")))
    enabled = reg.list_enabled(
        EnricherSet(enabled_enrichers=["topic_cooccurrence", "grounding_rate"])
    )
    ids = [e.manifest.id for e in enabled]
    assert ids == ["topic_cooccurrence", "grounding_rate"]


def test_list_enabled_warns_and_skips_unregistered_ids(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reg = EnricherRegistry()
    reg.register(_FakeEnricher(_manifest("topic_cooccurrence")))
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.enrichment.registry"):
        enabled = reg.list_enabled(
            EnricherSet(enabled_enrichers=["topic_cooccurrence", "unknown_enricher"])
        )
    assert [e.manifest.id for e in enabled] == ["topic_cooccurrence"]
    assert any("unknown_enricher" in r.message for r in caplog.records)


def test_list_enabled_hint_for_known_provider_wired_enrichers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When a profile lists ``topic_similarity`` / ``topic_consensus`` but
    the registry doesn't carry them (the deterministic-CLI case), the
    WARNING explains they need provider / scorer wiring at the call site
    rather than leaving the operator to grep for why the run silently
    skipped them.
    """
    reg = EnricherRegistry()
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.enrichment.registry"):
        reg.list_enabled(EnricherSet(enabled_enrichers=["topic_similarity", "topic_consensus"]))
    messages = [r.message for r in caplog.records]
    # topic_similarity → EmbeddingProvider hint
    assert any("topic_similarity" in m and "EmbeddingProvider" in m for m in messages), messages
    # topic_consensus → ConsensusScorer hint
    assert any("topic_consensus" in m and "ConsensusScorer" in m for m in messages), messages
    # Generic "not registered" warning for unknown ids is unchanged.
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.enrichment.registry"):
        reg.list_enabled(EnricherSet(enabled_enrichers=["unknown_id"]))
    assert any("unknown_id" in r.message for r in caplog.records)


def test_list_enabled_double_opt_in_required_for_requires_opt_in_enrichers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LLM-tier enrichers need both ``requires_opt_in=True`` on the manifest AND
    ``opt_in_flags[id]=True`` in the EnricherSet."""
    reg = EnricherRegistry()
    reg.register(
        _FakeEnricher(_manifest("query_synthesis", tier=EnricherTier.LLM, requires_opt_in=True))
    )
    # enabled but missing opt-in flag → skipped with WARNING.
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.enrichment.registry"):
        enabled = reg.list_enabled(EnricherSet(enabled_enrichers=["query_synthesis"]))
    assert enabled == []
    assert any("requires opt_in flag" in r.message for r in caplog.records)


def test_list_enabled_double_opt_in_passes_when_flag_set() -> None:
    reg = EnricherRegistry()
    reg.register(
        _FakeEnricher(_manifest("query_synthesis", tier=EnricherTier.LLM, requires_opt_in=True))
    )
    enabled = reg.list_enabled(
        EnricherSet(
            enabled_enrichers=["query_synthesis"],
            opt_in_flags={"query_synthesis": True},
        )
    )
    assert [e.manifest.id for e in enabled] == ["query_synthesis"]


def test_list_enabled_does_not_require_opt_in_for_non_llm_tiers() -> None:
    reg = EnricherRegistry()
    reg.register(_FakeEnricher(_manifest("topic_similarity", tier=EnricherTier.EMBEDDING)))
    enabled = reg.list_enabled(EnricherSet(enabled_enrichers=["topic_similarity"]))
    assert [e.manifest.id for e in enabled] == ["topic_similarity"]


def test_list_enabled_with_empty_set_returns_empty() -> None:
    reg = EnricherRegistry()
    reg.register(_FakeEnricher(_manifest("topic_cooccurrence")))
    assert reg.list_enabled(EnricherSet()) == []


# ---------------------------------------------------------------------------
# Integration with test fixture pattern (chunk-1 lock audit §B7)
# ---------------------------------------------------------------------------


@pytest.fixture
def registered_topic_cooccurrence() -> EnricherRegistry:
    """Pattern operators use in test suites: direct registry construction.

    Profile-preset gating (test_default = all-off) means tests CANNOT
    rely on a global registry to surface enrichers. Test fixtures
    construct the registry directly.
    """
    reg = EnricherRegistry()
    reg.register(_FakeEnricher(_manifest("topic_cooccurrence")))
    return reg


def test_fixture_pattern_works(
    registered_topic_cooccurrence: EnricherRegistry,
) -> None:
    """The pytest-fixture registration pattern surfaces enrichers under test."""
    e = registered_topic_cooccurrence.get("topic_cooccurrence")
    # Sanity: the enricher is usable.
    ev = asyncio.Event()
    ctx = RunContext(
        run_id="r",
        parent_run_id=None,
        enricher_id="topic_cooccurrence",
        enricher_version="1.0.0",
        tier="deterministic",
        attempt=1,
        job_id="r",
        cancel_event=ev,
    )
    result = asyncio.run(
        e.enrich(
            bundle=None,
            corpus_root=Path("/tmp"),
            all_bundles=None,
            config={},
            ctx=ctx,
        )
    )
    assert result.status == STATUS_OK
