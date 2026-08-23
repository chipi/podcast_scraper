"""``enrichments_available`` must list every episode-scope enricher (#1650).

The catalog row advertised a hardcoded tuple containing only ``insight_density``, while
``insight_sentiment`` had been writing real per-episode sidecars for the whole corpus —
``GET /api/corpus/episode/enrichments/insight_sentiment`` returned 200 with a full payload for
an enricher the list said was unavailable. Nothing was broken in the enricher; the *reporting*
was wrong, which is harder to notice and exactly the class of defect this epic is about.

The list stays hardcoded on purpose — the request path must not import the enrichment registry
— so this test is the thing that keeps it honest. Add an episode-scope enricher without
updating the tuple and CI fails here rather than the endpoint quietly under-reporting.
"""

from __future__ import annotations

import pytest

from podcast_scraper.enrichment.enrichers import register_deterministic_enrichers
from podcast_scraper.enrichment.protocol import EnricherScope
from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.server.routes.corpus_library import _EPISODE_SCOPE_ENRICHER_IDS

pytestmark = [pytest.mark.unit]


def _registry_episode_scope_ids() -> set[str]:
    registry = EnricherRegistry()
    register_deterministic_enrichers(registry)
    return {
        enricher.manifest.id
        for enricher in (registry.get(eid) for eid in registry.all_ids())
        if enricher.manifest.scope is EnricherScope.EPISODE
    }


def test_advertised_ids_match_the_registry_exactly() -> None:
    advertised = set(_EPISODE_SCOPE_ENRICHER_IDS)
    actual = _registry_episode_scope_ids()

    missing = actual - advertised
    assert not missing, (
        f"episode-scope enrichers exist but are not advertised: {sorted(missing)} — "
        "their sidecars are written and served, but consumers cannot discover them (#1650)"
    )

    phantom = advertised - actual
    assert not phantom, (
        f"advertised enrichers that no longer exist: {sorted(phantom)} — "
        "the endpoint would report availability for something never written"
    )


def test_insight_sentiment_is_advertised() -> None:
    """The specific regression: it shipped for the whole corpus while listed as unavailable."""
    assert "insight_sentiment" in _EPISODE_SCOPE_ENRICHER_IDS


def test_no_corpus_scope_enricher_leaks_into_the_episode_list() -> None:
    """Corpus-scope artifacts are catalogued by /api/corpus/enrichments, not per episode."""
    registry = EnricherRegistry()
    register_deterministic_enrichers(registry)
    corpus_scope = {
        enricher.manifest.id
        for enricher in (registry.get(eid) for eid in registry.all_ids())
        if enricher.manifest.scope is EnricherScope.CORPUS
    }
    assert not (set(_EPISODE_SCOPE_ENRICHER_IDS) & corpus_scope)
