"""Unit tests for #652 Part B — KG topic + entity filters."""

from __future__ import annotations

import pytest

from podcast_scraper.kg.filters import (
    KNOWN_ORGS,
    normalize_topic_labels,
    repair_entity_kind,
)

pytestmark = [pytest.mark.unit]


class TestNormalizeTopicLabels:
    def test_lowercases_and_strips_leading_stopwords(self):
        out, changes = normalize_topic_labels(["The Prediction Markets"])
        assert out == ["prediction markets"]
        assert changes == 1

    def test_trims_to_four_tokens(self):
        out, _ = normalize_topic_labels(["one two three four five six"])
        # After capping at 4 tokens: "one two three four". None are stopwords.
        assert out == ["one two three four"]

    def test_drops_medial_stopwords(self):
        out, _ = normalize_topic_labels(["Markets in Flux"])
        # "in" is a stopword, dropped as medial.
        assert out == ["markets flux"]

    def test_dedupes_near_matches(self):
        out, changes = normalize_topic_labels(["AI agents", "ai agents", "AI Agents"])
        assert out == ["ai agents"]
        # 3 changes: first was lowercased (1); second + third de-duped (2).
        assert changes == 3

    def test_dedupe_preserves_first_occurrence_order(self):
        out, _ = normalize_topic_labels(["zebra", "apple", "zebra", "banana"])
        assert out == ["zebra", "apple", "banana"]

    def test_empty_label_is_dropped_counted(self):
        out, changes = normalize_topic_labels([" ", "", "valid topic"])
        assert out == ["valid topic"]
        assert changes == 2

    def test_all_stopwords_is_dropped(self):
        out, changes = normalize_topic_labels(["the of and"])
        assert out == []
        assert changes == 1

    def test_punctuation_stripped_but_hyphens_preserved(self):
        out, _ = normalize_topic_labels(["AI-agents, briefly"])
        # "ai-agents" stays hyphenated after punctuation cleanup; "briefly"
        # is not a stopword and survives.
        assert out == ["ai-agents briefly"]

    def test_noop_on_already_canonical(self):
        out, changes = normalize_topic_labels(["nuclear program"])
        assert out == ["nuclear program"]
        assert changes == 0


class TestRepairEntityKind:
    def test_forces_org_on_known_show(self):
        entities = [{"name": "Planet Money", "kind": "person"}]
        out, repaired = repair_entity_kind(entities)
        assert out[0]["kind"] == "org"
        assert repaired == 1

    def test_forces_org_on_known_sponsor_company(self):
        entities = [{"name": "Ramp", "kind": "person"}]
        out, repaired = repair_entity_kind(entities)
        assert out[0]["kind"] == "org"
        assert repaired == 1

    def test_case_insensitive_match(self):
        entities = [{"name": "NPR", "kind": "person"}]
        out, repaired = repair_entity_kind(entities)
        assert out[0]["kind"] == "org"
        assert repaired == 1

    def test_unknown_name_untouched(self):
        entities = [{"name": "Jack Clark", "kind": "person"}]
        out, repaired = repair_entity_kind(entities)
        assert out[0]["kind"] == "person"  # left alone
        assert repaired == 0

    def test_already_org_no_repair(self):
        entities = [{"name": "NPR", "kind": "org"}]
        out, repaired = repair_entity_kind(entities)
        assert out[0]["kind"] == "org"
        assert repaired == 0

    def test_preserves_other_fields(self):
        entities = [{"name": "NPR", "kind": "person", "role": "host", "extra": "data"}]
        out, _ = repair_entity_kind(entities)
        assert out[0]["role"] == "host"
        assert out[0]["extra"] == "data"

    def test_non_dict_entity_passes_through_unchanged(self):
        """Malformed entries aren't the filter's job — downstream schema
        validation handles them. Just don't crash."""
        entities = [{"name": "NPR", "kind": "person"}, "not a dict", 42]
        out, repaired = repair_entity_kind(entities)
        assert out[0]["kind"] == "org"
        assert out[1] == "not a dict"
        assert out[2] == 42
        assert repaired == 1

    def test_known_orgs_set_non_empty_and_lowercase(self):
        # Guard against whitespace / case bugs in the curated list.
        assert len(KNOWN_ORGS) > 0
        for name in KNOWN_ORGS:
            assert name == name.lower().strip()
