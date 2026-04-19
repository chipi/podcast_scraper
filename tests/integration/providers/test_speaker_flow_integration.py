"""Integration test for end-to-end speaker flow (#598).

Validates: NER → detect_hosts → detect_speakers → KG person injection.
Tests both host+guest and host-only podcast scenarios.
Requires spaCy (en_core_web_sm).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pytest

pytestmark = [pytest.mark.integration]

logger = logging.getLogger(__name__)

_nlp: Optional[Any] = None
_has_spacy = False
try:
    import spacy

    _nlp = spacy.load("en_core_web_sm")
    _has_spacy = True
except (ImportError, OSError):
    pass

needs_spacy = pytest.mark.skipif(not _has_spacy, reason="spaCy en_core_web_sm not available")


@needs_spacy
def test_host_guest_podcast_full_flow() -> None:
    """Host+guest podcast: NER finds host from feed, guest from episode."""
    from podcast_scraper.providers.ml.speaker_detection import (
        detect_hosts_from_feed,
        detect_speaker_names,
    )

    # Feed-level: detect host
    hosts = detect_hosts_from_feed(
        feed_title="Capital Allocators with Ted Seides",
        feed_description="Ted Seides explores the people and process behind capital allocation.",
        feed_authors=["Ted Seides"],
        nlp=_nlp,
    )
    assert "Ted Seides" in hosts

    # Episode-level: detect guest
    speakers, detected_hosts, success, used_defaults = detect_speaker_names(
        episode_title="Kieran Goodwin on Private Credit Markets",
        episode_description=(
            "Ted Seides talks with Kieran Goodwin about the evolution of private "
            "credit markets and how institutional investors are adapting."
        ),
        nlp=_nlp,
        known_hosts=hosts,
    )
    assert success is True
    assert used_defaults is False
    assert "Ted Seides" in detected_hosts
    # Guest should be found (Kieran Goodwin has interview indicator "talks with")
    guest_names = [s for s in speakers if s not in hosts]
    assert len(guest_names) >= 1
    assert any("Kieran" in g or "Goodwin" in g for g in guest_names)


@needs_spacy
def test_host_only_podcast_full_flow() -> None:
    """Host-only podcast: NER finds host, no phantom guest injected."""
    from podcast_scraper.providers.ml.speaker_detection import (
        detect_hosts_from_feed,
        detect_speaker_names,
    )

    hosts = detect_hosts_from_feed(
        feed_title="The Long View",
        feed_description="Alex Morgan explores topics from biohacking to sustainability.",
        feed_authors=["Alex Morgan"],
        nlp=_nlp,
    )
    assert "Alex Morgan" in hosts

    speakers, detected_hosts, success, used_defaults = detect_speaker_names(
        episode_title="Biohacking in 2025: From Fringe to Framework",
        episode_description="Alex Morgan maps what is real, what is uncertain, and what you can do.",
        nlp=_nlp,
        known_hosts=hosts,
    )
    assert success is True
    assert used_defaults is False
    # No guest should be detected for solo show
    guest_names = [s for s in speakers if s not in hosts]
    assert len(guest_names) == 0


@needs_spacy
def test_kg_person_injection_host_guest() -> None:
    """KG correctly creates Person nodes with host/guest roles."""
    from podcast_scraper.kg.pipeline import _append_pipeline_entities

    nodes: list = []
    edges: list = []
    _append_pipeline_entities(
        ep_node_id="ep:test",
        detected_hosts=["Nora"],
        detected_guests=["Daniel"],
        nodes=nodes,
        edges=edges,
        existing_entity_keys=set(),
    )

    persons = {n["id"]: n for n in nodes if n.get("type") == "Entity"}
    assert "person:nora" in persons
    assert "person:daniel" in persons
    assert persons["person:nora"]["properties"]["role"] == "host"
    assert persons["person:daniel"]["properties"]["role"] == "guest"

    # Edges should be MENTIONS type
    mention_edges = [e for e in edges if e["type"] == "MENTIONS"]
    assert len(mention_edges) == 2


@needs_spacy
def test_kg_person_injection_host_only() -> None:
    """KG correctly creates only host Person node for solo show."""
    from podcast_scraper.kg.pipeline import _append_pipeline_entities

    nodes: list = []
    edges: list = []
    _append_pipeline_entities(
        ep_node_id="ep:solo",
        detected_hosts=["Alex Morgan"],
        detected_guests=[],
        nodes=nodes,
        edges=edges,
        existing_entity_keys=set(),
    )

    persons = [n for n in nodes if n.get("type") == "Entity"]
    assert len(persons) == 1
    assert persons[0]["properties"]["name"] == "Alex Morgan"
    assert persons[0]["properties"]["role"] == "host"


@needs_spacy
def test_description_snippet_not_truncated() -> None:
    """Guest names deep in description are found (DESCRIPTION_SNIPPET_LENGTH=500)."""
    from podcast_scraper.providers.ml.speaker_detection import detect_speaker_names

    speakers, _, success, _ = detect_speaker_names(
        episode_title="The Future of Energy",
        episode_description=(
            "In this episode we explore renewable energy trends and the latest "
            "breakthroughs in battery storage technology. Our guest Dr. Sarah Chen, "
            "a professor of materials science at MIT, joins us to discuss solar "
            "panel efficiency improvements and what they mean for residential adoption."
        ),
        nlp=_nlp,
        known_hosts={"Alex Morgan"},
    )
    assert success is True
    guest_names = [s for s in speakers if s != "Alex Morgan"]
    assert any(
        "Sarah" in g or "Chen" in g for g in guest_names
    ), f"Expected Sarah Chen in guests, got: {guest_names}"


@needs_spacy
def test_mentioned_person_not_guest() -> None:
    """Person merely mentioned in description is not detected as guest."""
    from podcast_scraper.providers.ml.speaker_detection import detect_speaker_names

    speakers, _, _, _ = detect_speaker_names(
        episode_title="Analysis of Recent Policy Changes",
        episode_description="We discuss Elon Musk's latest decisions and their market impact.",
        nlp=_nlp,
        known_hosts={"Alex Morgan"},
    )
    guest_names = [s for s in speakers if s != "Alex Morgan"]
    assert "Elon Musk" not in guest_names


@needs_spacy
def test_org_author_not_host() -> None:
    """Organization names in RSS authors are not treated as hosts."""
    from podcast_scraper.providers.ml.speaker_detection import detect_hosts_from_feed

    hosts = detect_hosts_from_feed(
        feed_title="NPR News Now",
        feed_description="Top stories from NPR News.",
        feed_authors=["NPR"],
        nlp=_nlp,
    )
    assert "NPR" not in hosts
