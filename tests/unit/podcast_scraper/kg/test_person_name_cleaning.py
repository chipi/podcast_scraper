"""Display names must not carry extractor punctuation debris (#2055).

Observed in prod on episode "Every Agent Needs a Box — Aaron Levie, Box", Related people:
`Sophia Dew)`, `Ben Mildenhall)`, `Lukasz Kaiser)`.

The cause is an asymmetry the KG pipeline already documents at `kg/pipeline.py:511`: the node ID
is slugified (parens stripped, so `lukasz kaiser)` -> `person:lukasz-kaiser`) while the display
LABEL is stored raw. A `)` that leaks out of the extractor — typically from a parenthetical alias
`(Lukasz Kaiser)` whose opening paren was consumed elsewhere — survives into the UI.

`_dedupe_nodes_by_id` keeps the FIRST node's label, so whether a user sees the debris depends on
extraction order. That is why it appears intermittently rather than always.
"""

from __future__ import annotations

import pytest

from podcast_scraper.kg.llm_extract import clean_entity_display_name

pytestmark = pytest.mark.unit


class TestTheObservedProductionCases:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Sophia Dew)", "Sophia Dew"),
            ("Ben Mildenhall)", "Ben Mildenhall"),
            ("Lukasz Kaiser)", "Lukasz Kaiser"),
            ("(Lukasz Kaiser", "Lukasz Kaiser"),
            ("(Lukasz Kaiser)", "Lukasz Kaiser"),
        ],
    )
    def test_stray_parens_are_stripped(self, raw: str, expected: str) -> None:
        assert clean_entity_display_name(raw) == expected

    def test_the_cleaned_name_matches_what_the_slug_already_assumed(self) -> None:
        # The bug is the label disagreeing with the id. After cleaning they must agree.
        from podcast_scraper.identity.slugify import slugify

        assert slugify(clean_entity_display_name("Lukasz Kaiser)")) == slugify("Lukasz Kaiser")


class TestItDoesNotDamageLegitimateNames:
    """Over-cleaning would silently rewrite real names, which is worse than the debris."""

    @pytest.mark.parametrize(
        "name",
        [
            "Jean-Luc Picard",
            "O'Brien",
            "Ben Mildenhall",
            "J.R.R. Tolkien",
            "Yann LeCun",
            "Søren Kierkegaard",
            "Ursula K. Le Guin",
            "Andrej Karpathy",
            "3Blue1Brown",
            "will.i.am",
            "Sam Altman, Jr.",
        ],
    )
    def test_a_legitimate_name_is_returned_unchanged(self, name: str) -> None:
        assert clean_entity_display_name(name) == name

    def test_an_internal_parenthetical_is_kept_not_mangled(self) -> None:
        # Balanced parens inside a name are meaningful; only UNBALANCED edges are debris.
        assert clean_entity_display_name("Bell Labs (Murray Hill)") == "Bell Labs (Murray Hill)"

    def test_whitespace_is_still_normalised(self) -> None:
        assert clean_entity_display_name("  Aaron   Levie  ") == "Aaron Levie"


class TestDegenerateInputs:
    @pytest.mark.parametrize("raw", ["", "   ", ")", "()", "(", None])
    def test_nothing_usable_yields_empty_rather_than_punctuation(self, raw) -> None:
        assert clean_entity_display_name(raw) == ""

    def test_it_is_length_bounded_like_the_raw_path_was(self) -> None:
        assert len(clean_entity_display_name("x" * 900)) == 500


class TestTheExtractorDropsNamesThatCleanToNothing:
    """A punctuation-only entity must not survive as an empty label on a person card."""

    def _entities(self, raw_names):
        import json

        from podcast_scraper.kg.llm_extract import parse_kg_graph_response

        payload = json.dumps(
            {"topics": [], "entities": [{"name": n, "entity_kind": "person"} for n in raw_names]}
        )
        parsed = parse_kg_graph_response(payload)
        return [e["name"] for e in (parsed or {}).get("entities", [])]

    def test_punctuation_only_entities_are_dropped_not_emptied(self) -> None:
        assert self._entities([")", "()", "   ", "Aaron Levie"]) == ["Aaron Levie"]

    def test_the_production_names_come_through_clean(self) -> None:
        got = self._entities(["Sophia Dew)", "Ben Mildenhall)", "Lukasz Kaiser)"])
        assert got == ["Sophia Dew", "Ben Mildenhall", "Lukasz Kaiser"]
