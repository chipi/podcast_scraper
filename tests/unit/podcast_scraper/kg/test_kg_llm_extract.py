"""Unit tests for KG LLM JSON parsing helpers."""

import unittest
from unittest.mock import Mock

from podcast_scraper.kg.llm_extract import parse_kg_graph_response, resolve_kg_model_id


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

    def test_resolve_kg_model_id(self) -> None:
        prov = Mock(summary_model="m1")
        self.assertEqual(resolve_kg_model_id(prov, {"kg_extraction_model": "override"}), "override")
        self.assertEqual(resolve_kg_model_id(prov, None), "m1")
