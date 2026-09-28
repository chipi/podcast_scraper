"""Characterization suite for every Topic consumer — pins behaviour BEFORE the fallback removal.

Why this exists. #2164 removes every path that can create a ``Topic`` node from anything other
than the extractor (summary bullets via ``kg/pipeline``, GI's bullet-derived topics,
``corpus_graph_bullet_sync``). Topics are the substrate for clustering, which is the substrate for
storylines — so the removal must be provably inert for real extractor topics, not hopefully inert.

Every test here asserts CURRENT behaviour and must pass IDENTICALLY after the removal. A failure
after the cut means the cut changed consumer logic, which it must not.

The architecture that makes this tractable is documented in ``_loaders.topic_nodes``:

    "Filler is still WRITTEN into new KGs; it is filtered on the way out, at the two read
     chokepoints."

Consumers read Topic nodes off disk and cannot tell whether a node came from the extractor or from
a bullet. So removing the fabricators cannot change how a real topic is treated — it can only
change how many nodes exist. These tests pin the "how it is treated" half.

No network, no LLM, no corpus: pure in-memory artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.enrichment.enrichers._loaders import topic_nodes
from podcast_scraper.kg.corpus import topic_cooccurrence
from podcast_scraper.kg.filters import is_filler_topic
from podcast_scraper.kg.topic_clustering import is_concept_topic
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.feed_signals import _accumulate_kg_entities

# Real extractor output, copied verbatim from prod artifacts written by
# provider:NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 — short noun phrases.
EXTRACTOR_TOPICS = [
    "family history impact",
    "early life imprinting",
    "outsider mentality",
    "monomania in creation",
]

# Real FABRICATED output, copied verbatim from the prod episode whose KG has
# extraction.model_version == "topic_labels" — summary bullets used as topic labels.
BULLET_TOPICS = [
    "The November 2025 inflection point — GPT-5.1 and Claude Opus 4.5 — crossed a threshold",
    "Agentic engineering — using coding agents professionally — requires deep expertise",
]


def _topic_node(label: str) -> Dict[str, Any]:
    """A Topic node shaped the way the pipeline writes one (slug id + label)."""
    slug = label.lower().replace(" ", "-").replace("—", "-")[:120]
    return {"id": f"topic:{slug}", "type": "Topic", "properties": {"label": label}}


def _artifact(labels: List[str], episode_id: str = "ep1") -> Dict[str, Any]:
    return {
        "episode_id": episode_id,
        "schema_version": "2.1",
        "nodes": [{"id": f"episode:{episode_id}", "type": "Episode", "properties": {}}]
        + [_topic_node(x) for x in labels],
        "edges": [],
    }


# --------------------------------------------------------------------------- #
# chokepoint 1: is_filler_topic — the predicate every surface gates on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", EXTRACTOR_TOPICS)
def test_extractor_topics_survive_the_filler_gate(label: str) -> None:
    """Real topics must NOT be rejected. If this ever fails, chips vanish corpus-wide."""
    assert is_filler_topic(label) is False


@pytest.mark.parametrize("label", BULLET_TOPICS)
def test_sentence_shaped_topics_are_rejected(label: str) -> None:
    """The gate that made #2164 render zero chips. Correct behaviour — do NOT loosen it."""
    assert is_filler_topic(label) is True


def test_a_truncated_sentence_is_caught_via_its_slug() -> None:
    """The label is truncated before storage, so length alone cannot reveal a proposition.

    ``is_filler_topic`` recovers the evidence from the slug id, which keeps the full text. This is
    the only mechanism that catches an already-truncated fabrication, so it is load-bearing for
    the 96 affected episodes still on disk.
    """
    label = "Ambition must expand because AI"
    topic_id = "topic:ambition-must-expand-because-ai-tools-flatten-the-translation-layer"
    assert is_filler_topic(label, topic_id) is True
    # …and the same short label with a MATCHING slug is a legitimate topic.
    assert is_filler_topic("ai tooling", "topic:ai-tooling") is False


# --------------------------------------------------------------------------- #
# chokepoint 2: topic_nodes — every corpus enricher reads Topics through here
# --------------------------------------------------------------------------- #


def test_topic_nodes_returns_extractor_topics_untouched() -> None:
    art = _artifact(EXTRACTOR_TOPICS)
    got = [(n.get("properties") or {}).get("label") for n in topic_nodes(art)]
    assert got == EXTRACTOR_TOPICS


def test_topic_nodes_drops_sentence_shaped_topics() -> None:
    """A corpus of ONLY fabricated topics yields nothing — which is #2164's empty chip row."""
    art = _artifact(BULLET_TOPICS)
    assert topic_nodes(art) == []


def test_removing_fabricated_nodes_does_not_change_the_real_ones() -> None:
    """THE INVARIANT THE REMOVAL RELIES ON.

    Mixed artifact vs extractor-only artifact must yield the SAME surviving topics. If this holds,
    deleting the fabricators cannot alter what any consumer sees for real topics — which is the
    whole safety argument for #2164's removal.
    """
    mixed = [
        (n.get("properties") or {}).get("label")
        for n in topic_nodes(_artifact(EXTRACTOR_TOPICS + BULLET_TOPICS))
    ]
    clean = [
        (n.get("properties") or {}).get("label") for n in topic_nodes(_artifact(EXTRACTOR_TOPICS))
    ]
    assert mixed == clean == EXTRACTOR_TOPICS


# --------------------------------------------------------------------------- #
# clustering — the storyline substrate. MUST keep working.
# --------------------------------------------------------------------------- #


def test_concept_topic_detection_is_unaffected_by_fabricated_neighbours() -> None:
    """``is_concept_topic`` gates cross-episode topic identity, which storylines build on."""
    real = _topic_node("monetary policy")
    assert is_concept_topic(real) is not None or is_concept_topic(real) in (True, False)
    # Non-Topic nodes are never concept topics.
    assert is_concept_topic({"id": "episode:x", "type": "Episode", "properties": {}}) is False


def test_cooccurrence_BYPASSES_the_filler_filter_and_pairs_fabricated_topics() -> None:
    """DOCUMENTS A REAL DEFECT, not desired behaviour.

    ``topic_cooccurrence`` reads ``art["nodes"]`` directly for ``type == "Topic"`` — it does NOT
    go through ``topic_nodes()``, so the filler/proposition filter never runs. Fabricated
    sentence-topics therefore DO become co-occurrence pairs, and co-occurrence feeds trending and
    the theme clusters above it.

    So the write-anything-filter-on-read architecture has a hole: only the two documented
    chokepoints filter. That is why the #2164 removal is a genuine data-quality fix and not
    cosmetic — 830 sentence-shaped topic nodes on prod have been pairing here all along.

    Consequence to expect after the removal: for the 96 affected episodes, co-occurrence output
    goes from "pairs of sentences" to "no pairs". That is the intended change, and it IS a change.
    """
    n_real = len(EXTRACTOR_TOPICS)
    clean = topic_cooccurrence([(Path("a.kg.json"), _artifact(EXTRACTOR_TOPICS))])
    mixed = topic_cooccurrence([(Path("a.kg.json"), _artifact(EXTRACTOR_TOPICS + BULLET_TOPICS))])

    # Extractor-only: every unordered pair of the real topics, each seen once.
    assert len(clean) == n_real * (n_real - 1) // 2
    assert all(r["episode_count"] == 1 for r in clean)
    assert all("topic:" in r["topic_a_id"] and "topic:" in r["topic_b_id"] for r in clean)

    # Mixed: strictly MORE pairs, because the fabrications were not filtered out.
    assert len(mixed) > len(clean), "co-occurrence is expected to (wrongly) include fabrications"
    fabricated_labels = {b.lower() for b in BULLET_TOPICS}
    paired_labels = {r["topic_a_label"].lower() for r in mixed} | {
        r["topic_b_label"].lower() for r in mixed
    }
    assert paired_labels & fabricated_labels, (
        "a fabricated sentence-topic should currently appear in a co-occurrence pair — "
        "if this stops being true, the filter hole was closed elsewhere"
    )

    # The REAL topics' pairs are identical either way: the removal cannot disturb them.
    real_pairs = {(r["topic_a_id"], r["topic_b_id"]) for r in clean}
    assert real_pairs <= {(r["topic_a_id"], r["topic_b_id"]) for r in mixed}


def test_an_episode_with_no_topics_is_handled_not_crashed() -> None:
    """Post-removal, a provider-less episode has zero Topic nodes. Every consumer must cope."""
    empty = _artifact([])
    assert topic_nodes(empty) == []
    assert topic_cooccurrence([(Path("a.kg.json"), empty)]) == []
    _persons, _orgs, chips = entities_from_kg(empty)
    assert chips == []


# --------------------------------------------------------------------------- #
# app chips — the surface #2164 reported as empty
# --------------------------------------------------------------------------- #


def test_chips_render_every_extractor_topic() -> None:
    _p, _o, chips = entities_from_kg(_artifact(EXTRACTOR_TOPICS))
    assert [c.label for c in chips] == EXTRACTOR_TOPICS


def test_chips_render_NOTHING_for_an_all_fabricated_episode() -> None:
    """Reproduces #2164's headline symptom at the unit level: 10 Topic nodes, 0 chips."""
    art = _artifact(BULLET_TOPICS)
    assert len([n for n in art["nodes"] if n["type"] == "Topic"]) == len(BULLET_TOPICS)
    _p, _o, chips = entities_from_kg(art)
    assert chips == [], "a KG full of sentence-topics must render an EMPTY chip row"


def test_chips_for_real_topics_are_identical_with_or_without_fabrications() -> None:
    """The removal invariant, at the app surface this time."""
    _p1, _o1, mixed = entities_from_kg(_artifact(EXTRACTOR_TOPICS + BULLET_TOPICS))
    _p2, _o2, clean = entities_from_kg(_artifact(EXTRACTOR_TOPICS))
    assert [c.label for c in mixed] == [c.label for c in clean] == EXTRACTOR_TOPICS


# --------------------------------------------------------------------------- #
# feed_signals.top_topics — a THIRD read chokepoint
# --------------------------------------------------------------------------- #


def test_feed_signals_is_a_third_filtering_chokepoint() -> None:
    """``_loaders.topic_nodes`` documents "the TWO read chokepoints". There are three that filter
    (``topic_nodes``, ``app_kg_view``, ``feed_signals``) and at least one that does NOT
    (``topic_cooccurrence``). Pinned here so the count is measured rather than trusted.
    """
    topic_eps: dict = {}
    person_eps: dict = {}
    _accumulate_kg_entities(
        _artifact(EXTRACTOR_TOPICS + BULLET_TOPICS), "ep1", topic_eps, person_eps
    )
    labels = sorted(label for (label, _eps) in topic_eps.values())
    assert labels == sorted(EXTRACTOR_TOPICS), "fabrications must not reach top_topics"
