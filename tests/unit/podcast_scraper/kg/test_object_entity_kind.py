"""`Object` — the third entity kind, and the catch-all (#2057, KG schema 2.1).

Before it existed the vocabulary was person|organization, and `_normalize_entity_kind` was two
branches: a handful of organisation synonyms, then `return "person"` for everything else. So
`event`, `podcast`, `show`, `place`, `book`, `film`, `product`, `concept` — and a MISSING kind —
all became people.

Measured on prod `top_people` 2026-09-13: 7 of the corpus's top 40 "voices" were not people, and
the #1 voice, with 2,720 grounded insights, was the Norman Conquest.

Forcing those into `organization` would only move the pollution — a battle is not a company — so
the model gained a third bucket rather than a better guess.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.graph_id_utils import entity_node_id, PERSON_ORG_NODE_TYPES
from podcast_scraper.kg.llm_extract import (
    _normalize_entity_kind,
    build_kg_transcript_system_prompt,
    ENTITY_KIND_OBJECT,
    ENTITY_KIND_ORGANIZATION,
    ENTITY_KIND_PERSON,
    ENTITY_KINDS,
    parse_kg_graph_response,
)

pytestmark = pytest.mark.unit


class TestPersonIsNeverAFallback:
    """The rule the whole type exists to enforce."""

    def test_person_requires_the_extractor_to_say_so(self) -> None:
        assert _normalize_entity_kind("person") == ENTITY_KIND_PERSON

    @pytest.mark.parametrize(
        "kind",
        [
            "event",
            "podcast",
            "show",
            "place",
            "location",
            "book",
            "film",
            "movie",
            "product",
            "concept",
            "work_of_art",
            "album",
            "song",
            "standard",
            "protocol",
            "award",
        ],
    )
    def test_a_named_non_person_thing_is_an_object(self, kind: str) -> None:
        assert _normalize_entity_kind(kind) == ENTITY_KIND_OBJECT

    @pytest.mark.parametrize("kind", [None, "", "   ", "banana", "no-idea"])
    def test_an_unknown_or_absent_kind_becomes_object_not_person(self, kind) -> None:
        # Unknown must not mean dropped either — the entity is a real referent, we just cannot
        # place it more precisely.
        assert _normalize_entity_kind(kind) == ENTITY_KIND_OBJECT

    def test_a_body_of_people_is_still_an_organization(self) -> None:
        # The test is "could it employ someone or hold a position?"
        for kind in ("company", "organization", "university", "agency", "band", "government"):
            assert _normalize_entity_kind(kind) == ENTITY_KIND_ORGANIZATION

    def test_every_result_is_in_the_declared_vocabulary(self) -> None:
        for kind in ("person", "company", "event", "banana", None, "", "PODCAST", "Work-Of-Art"):
            assert _normalize_entity_kind(kind) in ENTITY_KINDS

    @pytest.mark.parametrize("kind", ["PERSON", "Person", " person ", "Work-Of-Art", "work of art"])
    def test_case_and_separator_variants_are_understood(self, kind: str) -> None:
        assert _normalize_entity_kind(kind) in ENTITY_KINDS


class TestObjectIsAddressable:
    """A catch-all that cannot be referenced is just a slower drop."""

    def test_objects_get_their_own_id_namespace(self) -> None:
        assert entity_node_id("object", "The Norman Conquest") == "object:the-norman-conquest"

    def test_an_unknown_kind_never_lands_on_a_person_id(self) -> None:
        # entity_node_id had the same `else "person"` bug as the classifier.
        assert entity_node_id("event", "The Norman Conquest").startswith("object:")
        assert entity_node_id(None, "Mystery").startswith("object:")  # type: ignore[arg-type]

    def test_person_and_org_ids_are_unchanged(self) -> None:
        assert entity_node_id("person", "Aaron Levie") == "person:aaron-levie"
        assert entity_node_id("organization", "Box") == "org:box"

    def test_object_is_a_recognised_entity_node_type(self) -> None:
        assert "Object" in PERSON_ORG_NODE_TYPES


class TestTheProductionPollutionIsGone:
    def test_the_reported_top_voices_no_longer_land_on_people(self) -> None:
        payload = json.dumps(
            {
                "topics": [],
                "entities": [
                    {"name": "Aaron Levie", "entity_kind": "person"},
                    {"name": "Box", "entity_kind": "company"},
                    {"name": "The Norman Conquest", "entity_kind": "event"},
                    {"name": "Conversations with Tyler", "entity_kind": "podcast"},
                    {"name": "Machine Learning Street Talk", "entity_kind": "show"},
                    {"name": "Mystery Thing"},
                ],
            }
        )
        ents = (parse_kg_graph_response(payload) or {}).get("entities", [])
        by_kind = {e["name"]: e["entity_kind"] for e in ents}
        assert by_kind["Aaron Levie"] == ENTITY_KIND_PERSON
        assert by_kind["Box"] == ENTITY_KIND_ORGANIZATION
        assert by_kind["The Norman Conquest"] == ENTITY_KIND_OBJECT
        assert by_kind["Conversations with Tyler"] == ENTITY_KIND_OBJECT
        assert by_kind["Machine Learning Street Talk"] == ENTITY_KIND_OBJECT
        assert by_kind["Mystery Thing"] == ENTITY_KIND_OBJECT

    def test_nothing_is_lost_to_the_catch_all(self) -> None:
        # The catch-all must KEEP entities; dropping would trade one silent data loss for another.
        payload = json.dumps({"topics": [], "entities": [{"name": f"Thing {i}"} for i in range(5)]})
        ents = (parse_kg_graph_response(payload) or {}).get("entities", [])
        assert len(ents) == 5


class TestTheExtractionPromptTeachesTheVocabulary:
    """A classifier that silently repairs the model is worse than a model told the rules."""

    def test_the_prompt_names_all_three_kinds(self) -> None:
        prompt = build_kg_transcript_system_prompt(10, 15)
        for kind in ENTITY_KINDS:
            assert f'"{kind}"' in prompt

    def test_the_prompt_forbids_defaulting_to_person(self) -> None:
        prompt = build_kg_transcript_system_prompt(10, 15)
        assert "NEVER default to person" in prompt

    def test_the_prompt_gives_the_organization_test(self) -> None:
        # "could it employ someone" is the discriminator; without it the model guesses.
        assert "employ someone" in build_kg_transcript_system_prompt(10, 15)
