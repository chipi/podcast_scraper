"""The KG extraction prompt shows example entities; a model can copy one into a text that never
mentions it (seen on news, 2026-11-18, and in 21 of 4,715 prod podcast kg.json files). Such an
entity is dropped; the same name, when the text does name it, is kept."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.kg import llm_extract
from podcast_scraper.kg.pipeline import build_artifact

pytestmark = pytest.mark.unit

TEXT = (
    "Datasette Lite now runs the full server-side Datasette application in the browser, "
    "using Pyodide and service workers. Simon Willison shared the code on GitHub."
)


def _partial() -> Dict[str, Any]:
    return {
        "topics": [{"label": "browser-based Python"}],
        "entities": [
            {"name": "Simon Willison", "entity_kind": "person"},
            {"name": "GitHub", "entity_kind": "organization"},
            {"name": "Cascadia Alliance", "entity_kind": "organization"},
        ],
    }


def _names(art: Dict[str, Any]) -> List[str]:
    return sorted(
        (n.get("properties") or {}).get("name", "")
        for n in art["nodes"]
        if n["type"] in ("Person", "Organization", "Object")
    )


def test_an_example_entity_the_text_never_names_is_dropped() -> None:
    art = build_artifact(
        "ep:leak", TEXT, podcast_id="podcast:p1", episode_title="T", prefilled_partial=_partial()
    )
    assert "Cascadia Alliance" not in _names(art)
    assert {"Simon Willison", "GitHub"} <= set(_names(art))


def test_an_example_name_the_text_does_name_is_kept() -> None:
    text = TEXT + " The Cascadia Alliance maintains the trails near Portland."
    art = build_artifact(
        "ep:real", text, podcast_id="podcast:p1", episode_title="T", prefilled_partial=_partial()
    )
    assert "Cascadia Alliance" in _names(art)


def test_the_guard_covers_every_example_the_prompt_shows() -> None:
    prompt = llm_extract.build_kg_transcript_system_prompt(10, 12)
    for name, kind in llm_extract.PROMPT_EXAMPLE_ENTITIES:
        assert f'{{"name":"{name}","entity_kind":"{kind}"}}' in prompt
    entities = [{"name": n.upper()} for n, _ in llm_extract.PROMPT_EXAMPLE_ENTITIES]
    kept, dropped = llm_extract.drop_prompt_example_entities(entities, "unrelated text")
    assert kept == [] and dropped == len(llm_extract.PROMPT_EXAMPLE_ENTITIES)
