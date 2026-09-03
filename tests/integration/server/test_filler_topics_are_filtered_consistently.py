"""Filler must be filtered on EVERY topic surface, or on none.

The guard has two chokepoints — the enrichment loader and the KG read path — and they must agree.
Filtering one and not the other is the worst of the three states: a listener gets a tappable topic
chip whose entity card is guaranteed empty, because the card's signals come from the artifacts
that DID filter it. That is the "fully built, mounted, fetching, never renders" failure this
branch fixed three separate times, re-created by a partial rollout.

The first version of the guard was exactly that: one call site, in `_loaders.topic_nodes`, with a
docstring claiming extraction was also covered. It was not.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration

_FILLER_ID = "topic:welcome-back-to"
_FILLER_LABEL = "welcome back to"
_REAL_ID = "topic:ai-regulation"
_REAL_LABEL = "ai regulation"


def _kg() -> dict[str, Any]:
    return {
        "nodes": [
            {"type": "Topic", "id": _FILLER_ID, "properties": {"label": _FILLER_LABEL}},
            {"type": "Topic", "id": _REAL_ID, "properties": {"label": _REAL_LABEL}},
            {"type": "Person", "id": "person:jane", "properties": {"label": "Jane Doe"}},
        ],
        "edges": [],
    }


def test_the_enrichment_loader_drops_filler() -> None:
    from podcast_scraper.enrichment.enrichers._loaders import topic_nodes

    assert [n["id"] for n in topic_nodes(_kg())] == [_REAL_ID]


def test_the_kg_read_path_drops_the_same_filler() -> None:
    """Episode topic chips, followable interests, discover ranking, digest sections."""
    from podcast_scraper.server.app_kg_view import entities_from_kg

    _persons, _orgs, topics = entities_from_kg(_kg())
    assert [t.id for t in topics] == [_REAL_ID], (
        "a filler chip is rendered and followable while every enrichment surface hides it — "
        "tapping it opens an entity card that can only ever be empty"
    )


def test_show_signals_drop_the_same_filler() -> None:
    """`top_topics` AND the #1932 connectivity metric read through one accumulator."""
    from podcast_scraper.server.feed_signals import _accumulate_kg_entities

    topic_eps: dict[str, tuple[str, set[str]]] = {}
    person_eps: dict[str, tuple[str, set[str]]] = {}
    _accumulate_kg_entities(_kg(), "ep1", topic_eps, person_eps)
    assert list(topic_eps) == [_REAL_ID], (
        "a host catchphrase in every episode becomes the show's #1 topic chip and pairs with "
        "every real topic, inflating recurring_pairs — the metric that decides ingest budget"
    )
    assert "person:jane" in person_eps, "people must be unaffected"


def test_all_three_chokepoints_agree() -> None:
    """The invariant itself, stated once: same input, same verdict, everywhere."""
    from podcast_scraper.enrichment.enrichers._loaders import topic_nodes
    from podcast_scraper.server.app_kg_view import entities_from_kg
    from podcast_scraper.server.feed_signals import _accumulate_kg_entities

    kg = _kg()
    topic_eps: dict[str, tuple[str, set[str]]] = {}
    _accumulate_kg_entities(kg, "ep1", topic_eps, {})
    _p, _o, view_topics = entities_from_kg(kg)

    from_loader = {n["id"] for n in topic_nodes(kg)}
    from_view = {t.id for t in view_topics}
    from_signals = set(topic_eps)

    assert from_loader == from_view == from_signals, (
        f"the topic surfaces disagree: loader={sorted(from_loader)} "
        f"view={sorted(from_view)} signals={sorted(from_signals)}"
    )
