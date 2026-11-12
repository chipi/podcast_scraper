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

    # RFC-097 v2.0: persons emit typed ``Person`` nodes (was legacy ``Entity``).
    persons = {n["id"]: n for n in nodes if n.get("type") == "Person"}
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

    # RFC-097 v2.0: persons emit typed ``Person`` nodes (was legacy ``Entity``).
    persons = [n for n in nodes if n.get("type") == "Person"]
    assert len(persons) == 1
    assert persons[0]["properties"]["name"] == "Alex Morgan"
    assert persons[0]["properties"]["role"] == "host"
