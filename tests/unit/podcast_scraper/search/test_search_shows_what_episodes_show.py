"""Search shows only what the episode view shows, under one spelling per person (RFC-131 S6c,
D2 item 7). Invented names only."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.search import corpus_search as CS

pytestmark = pytest.mark.unit


def _gi(person: str, name: str, *, surfaceable: bool = True) -> Dict[str, Any]:
    insight_props: Dict[str, Any] = {"text": "A point.", "tier": 3, "grounded": True}
    if not surfaceable:
        insight_props.update(surfaceable=False, speaker_voice_type="unknown")
    return {
        "nodes": [
            {"id": person, "type": "Person", "properties": {"name": name}},
            {"id": "insight:a", "type": "Insight", "properties": insight_props},
            {
                "id": "quote:a",
                "type": "Quote",
                "properties": {
                    "text": "Said so.",
                    "speaker_id": person,
                    "char_start": 0,
                    "char_end": 40,
                },
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "insight:a", "to": "quote:a"}],
    }


def test_the_scope_carries_the_speakers_canonical_id() -> None:
    (scope,) = CS._insight_scopes_from_gi(
        "e1",
        _gi("person:korin-vaal", "Korin Vaal"),
        id_map={"person:korin-vaal": "person:korin-vael"},
    )
    assert set(scope.speakers) == {"person:korin-vaal", "person:korin-vael"}
