"""Unit tests for mega-bundle parser (#643)."""

from __future__ import annotations

import json

import pytest

from podcast_scraper.providers.common.megabundle_parser import (
    MegaBundleParseError,
    parse_extraction_bundle_response,
    parse_megabundle_response,
)

VALID = {
    "title": "Test Episode",
    "summary": "Paragraph one. Paragraph two.",
    "bullets": ["bullet 1", "bullet 2", "bullet 3"],
    "insights": [
        {"text": "Insight one.", "insight_type": "claim"},
        {"text": "Insight two.", "insight_type": "fact"},
    ],
    "topics": ["topic one", "topic two", "topic three"],
    "entities": [
        {"name": "Alice", "kind": "person", "role": "host"},
        {"name": "Acme Inc.", "kind": "org", "role": "mentioned"},
    ],
}


class TestParseMegaBundle:
    def test_parses_clean_json(self):
        r = parse_megabundle_response(json.dumps(VALID))
        assert r.title == "Test Episode"
        assert r.summary.startswith("Paragraph one")
        assert len(r.bullets) == 3
        assert len(r.insights) == 2
        assert r.insights[0]["text"] == "Insight one."
        assert r.insights[0]["insight_type"] == "claim"
        assert r.topics == ["topic one", "topic two", "topic three"]
        assert len(r.entities) == 2
        assert r.entities[0]["name"] == "Alice"

    def test_strips_code_fences(self):
        text = "```json\n" + json.dumps(VALID) + "\n```"
        r = parse_megabundle_response(text)
        assert r.title == "Test Episode"

    def test_strips_triple_backtick_no_lang(self):
        text = "```\n" + json.dumps(VALID) + "\n```"
        r = parse_megabundle_response(text)
        assert r.summary

    def test_raises_on_empty_text(self):
        with pytest.raises(MegaBundleParseError, match="Empty"):
            parse_megabundle_response("")

    def test_raises_on_malformed_json(self):
        with pytest.raises(MegaBundleParseError, match="not valid JSON"):
            parse_megabundle_response("{ not valid json }")

    def test_raises_on_non_object_top_level(self):
        with pytest.raises(MegaBundleParseError, match="top level"):
            parse_megabundle_response("[]")

    def test_raises_on_missing_summary(self):
        bad = dict(VALID)
        del bad["summary"]
        with pytest.raises(MegaBundleParseError, match="Missing 'summary'"):
            parse_megabundle_response(json.dumps(bad))

    def test_raises_on_missing_insights(self):
        bad = dict(VALID)
        del bad["insights"]
        with pytest.raises(MegaBundleParseError, match="Missing 'insights'"):
            parse_megabundle_response(json.dumps(bad))

    def test_raises_on_missing_topics(self):
        bad = dict(VALID)
        del bad["topics"]
        with pytest.raises(MegaBundleParseError, match="Missing 'topics'"):
            parse_megabundle_response(json.dumps(bad))

    def test_normalizes_bare_string_insights(self):
        data = dict(VALID)
        data["insights"] = ["raw insight one", "raw insight two"]
        r = parse_megabundle_response(json.dumps(data))
        assert r.insights[0]["text"] == "raw insight one"
        assert r.insights[0]["insight_type"] == "claim"

    def test_normalizes_type_alias_to_insight_type(self):
        data = dict(VALID)
        data["insights"] = [{"text": "X", "type": "fact"}]
        r = parse_megabundle_response(json.dumps(data))
        assert r.insights[0]["insight_type"] == "fact"

    def test_normalizes_unknown_insight_type_to_claim(self):
        data = dict(VALID)
        data["insights"] = [{"text": "X", "insight_type": "weird"}]
        r = parse_megabundle_response(json.dumps(data))
        assert r.insights[0]["insight_type"] == "claim"

    def test_normalizes_topic_with_label_key(self):
        data = dict(VALID)
        data["topics"] = [{"label": "one"}, {"label": "two"}, "three"]
        r = parse_megabundle_response(json.dumps(data))
        assert r.topics == ["one", "two", "three"]

    def test_normalizes_bare_string_entities(self):
        data = dict(VALID)
        data["entities"] = ["Alice", "Bob"]
        r = parse_megabundle_response(json.dumps(data))
        assert r.entities[0]["name"] == "Alice"
        assert r.entities[0]["kind"] == "person"
        assert r.entities[0]["role"] == "mentioned"

    def test_normalizes_unknown_entity_kind_to_person(self):
        data = dict(VALID)
        data["entities"] = [{"name": "X", "kind": "robot"}]
        r = parse_megabundle_response(json.dumps(data))
        assert r.entities[0]["kind"] == "person"

    def test_to_summary_artifact_shape(self):
        r = parse_megabundle_response(json.dumps(VALID))
        a = r.to_summary_artifact()
        assert a == {
            "title": "Test Episode",
            "summary": "Paragraph one. Paragraph two.",
            "bullets": ["bullet 1", "bullet 2", "bullet 3"],
        }

    def test_raw_is_preserved(self):
        r = parse_megabundle_response(json.dumps(VALID))
        assert r.raw == VALID


class TestParseExtractionBundle:
    def test_summary_not_required(self):
        data = {k: v for k, v in VALID.items() if k not in ("title", "summary", "bullets")}
        r = parse_extraction_bundle_response(json.dumps(data))
        assert r.title == ""
        assert r.summary == ""
        assert r.bullets == []
        assert len(r.insights) == 2
        assert len(r.topics) == 3

    def test_insights_still_required(self):
        bad = {"topics": ["a", "b"], "entities": []}
        with pytest.raises(MegaBundleParseError, match="Missing 'insights'"):
            parse_extraction_bundle_response(json.dumps(bad))

    def test_topics_still_required(self):
        bad = {"insights": [{"text": "X"}], "entities": []}
        with pytest.raises(MegaBundleParseError, match="Missing 'topics'"):
            parse_extraction_bundle_response(json.dumps(bad))
