"""Unit tests for KG LLM JSON parsing helpers."""

import unittest
from unittest.mock import Mock

from podcast_scraper.kg.llm_extract import (
    normalize_bullet_labels_for_kg,
    parse_kg_graph_response,
    resolve_kg_model_id,
    strip_known_ml_bullet_prefixes,
)


class TestKgLlmExtract(unittest.TestCase):
    """Tests for parse_kg_graph_response."""

    def test_parse_plain_json(self) -> None:
        raw = (
            '{"topics":[{"label":"Inflation"}],"entities":['
            '{"name":"Jane Doe","entity_kind":"person"}]}'
        )
        out = parse_kg_graph_response(raw)
        assert out is not None
        self.assertEqual(len(out["topics"]), 1)
        self.assertEqual(out["topics"][0]["label"], "Inflation")
        self.assertEqual(out["entities"][0]["name"], "Jane Doe")
        self.assertEqual(out["entities"][0]["entity_kind"], "person")

    def test_parse_strips_markdown_fence(self) -> None:
        raw = '```json\n{"topics":[],"entities":[{"name":"Org","entity_kind":"organization"}]}\n```'
        out = parse_kg_graph_response(raw)
        assert out is not None
        self.assertEqual(out["entities"][0]["entity_kind"], "organization")

    def test_parse_empty_returns_none(self) -> None:
        self.assertIsNone(parse_kg_graph_response("{}"))
        self.assertIsNone(parse_kg_graph_response('{"topics":[],"entities":[]}'))

    def test_parse_preserves_optional_descriptions(self) -> None:
        """Topic and entity description strings are kept when present (#487)."""
        raw = (
            '{"topics":[{"label":"T","description":"Ctx one"}],'
            '"entities":[{"name":"N","entity_kind":"person","description":"Ctx two"}]}'
        )
        out = parse_kg_graph_response(raw)
        assert out is not None
        self.assertEqual(out["topics"][0].get("description"), "Ctx one")
        self.assertEqual(out["entities"][0].get("description"), "Ctx two")

    def test_parse_truncates_to_max_topics_and_entities(self) -> None:
        raw = (
            '{"topics":['
            '{"label":"A"},{"label":"B"},{"label":"C"}],'
            '"entities":['
            '{"name":"E1","entity_kind":"person"},'
            '{"name":"E2","entity_kind":"person"}]}'
        )
        out = parse_kg_graph_response(raw, max_topics=2, max_entities=1)
        assert out is not None
        self.assertEqual([t["label"] for t in out["topics"]], ["A", "B"])
        self.assertEqual(len(out["entities"]), 1)
        self.assertEqual(out["entities"][0]["name"], "E1")

    def test_resolve_kg_model_id(self) -> None:
        prov = Mock(summary_model="m1")
        self.assertEqual(resolve_kg_model_id(prov, {"kg_extraction_model": "override"}), "override")
        self.assertEqual(resolve_kg_model_id(prov, None), "m1")

    def test_strip_known_ml_bullet_prefixes(self) -> None:
        self.assertEqual(
            strip_known_ml_bullet_prefixes("Unden- inflation is high"),
            "inflation is high",
        )
        self.assertEqual(strip_known_ml_bullet_prefixes("  Exting- Smoke  "), "Smoke")

    def test_normalize_bullet_labels_strips_ml_prefixes(self) -> None:
        out = normalize_bullet_labels_for_kg(["Unden- topic one", "plain"])
        self.assertEqual(out, ["topic one", "plain"])
