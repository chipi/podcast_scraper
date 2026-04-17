"""Tests for GI/KG bullet sync without LLM (corpus repair)."""

from __future__ import annotations

import json
import unittest

from podcast_scraper.utils.corpus_graph_bullet_sync import (
    bullet_labels_from_summary_bullets,
    kg_should_replace_topics_from_bullets,
    patch_gi_for_bullet_labels,
    patch_kg_for_bullet_labels,
)


class TestBulletLabels(unittest.TestCase):
    def test_strip_and_cap(self):
        labels = bullet_labels_from_summary_bullets(["Unden- One", "  Two  "])
        self.assertEqual(labels, ["One", "Two"])


class TestPatchGi(unittest.TestCase):
    def test_insight_text_and_topics(self):
        gi = {
            "schema_version": "1.0",
            "episode_id": "e1",
            "model_version": "test",
            "nodes": [
                {"id": "episode:e1", "type": "Episode", "properties": {"title": "T"}},
                {
                    "id": "insight:a",
                    "type": "Insight",
                    "properties": {"text": "OLD1", "episode_id": "e1"},
                },
                {
                    "id": "insight:b",
                    "type": "Insight",
                    "properties": {"text": "OLD2", "episode_id": "e1"},
                },
                {"id": "topic:bad", "type": "Topic", "properties": {"label": "```json {}"}},
            ],
            "edges": [
                {"type": "HAS_INSIGHT", "from": "episode:e1", "to": "insight:a"},
                {"type": "HAS_INSIGHT", "from": "episode:e1", "to": "insight:b"},
                {"type": "ABOUT", "from": "insight:a", "to": "topic:bad"},
                {"type": "ABOUT", "from": "insight:b", "to": "topic:bad"},
            ],
        }
        out, msg = patch_gi_for_bullet_labels(gi, ["First line", "Second line"])
        self.assertEqual(msg, "patched-gi")
        assert out is not None
        topics = [n for n in out["nodes"] if n.get("type") == "Topic"]
        self.assertTrue(
            all("```" not in str(t.get("properties", {}).get("label", "")) for t in topics)
        )
        by_id = {n["id"]: n for n in out["nodes"] if isinstance(n, dict)}
        self.assertEqual(by_id["insight:a"]["properties"]["text"], "First line")
        self.assertEqual(by_id["insight:b"]["properties"]["text"], "Second line")
        about = [e for e in out["edges"] if e.get("type") == "ABOUT"]
        self.assertEqual(len(about), 2 * len(topics))


class TestPatchKg(unittest.TestCase):
    def test_skips_provider_kg(self):
        kg = {
            "schema_version": "1.2",
            "episode_id": "e1",
            "extraction": {"model_version": "provider:gpt-4"},
            "nodes": [
                {"id": "episode:e1", "type": "Episode", "properties": {}},
                {"id": "topic:x", "type": "Topic", "properties": {"label": "Real topic"}},
            ],
            "edges": [],
        }
        out, msg = patch_kg_for_bullet_labels(kg, ["A", "B"])
        self.assertIsNone(out)
        self.assertIn("preserved", msg)

    def test_replaces_summary_bullets_kg(self):
        kg = {
            "schema_version": "1.2",
            "episode_id": "e1",
            "extraction": {"model_version": "summary_bullets"},
            "nodes": [
                {"id": "episode:e1", "type": "Episode", "properties": {"title": "T"}},
                {
                    "id": "topic:junk",
                    "type": "Topic",
                    "properties": {"label": "{junk}", "slug": "junk"},
                },
            ],
            "edges": [
                {"from": "topic:junk", "to": "episode:e1", "type": "MENTIONS", "properties": {}},
            ],
        }
        out, msg = patch_kg_for_bullet_labels(kg, ["Clean headline", "Another"])
        self.assertEqual(msg, "patched-kg")
        assert out is not None
        topics = [n for n in out["nodes"] if n.get("type") == "Topic"]
        self.assertEqual(len(topics), 2)
        labels = {str(t["properties"]["label"]) for t in topics}
        self.assertIn("Clean headline", labels)
        self.assertIn("Another", labels)


class TestKgHeuristic(unittest.TestCase):
    def test_corrupt_label_triggers_replace(self):
        kg = {
            "episode_id": "e1",
            "extraction": {"model_version": "provider:gpt-4"},
            "nodes": [
                {"id": "episode:e1", "type": "Episode", "properties": {}},
                {"id": "topic:x", "type": "Topic", "properties": {"label": "```json {}"}},
            ],
            "edges": [],
        }
        self.assertTrue(kg_should_replace_topics_from_bullets(kg))


class TestJsonStable(unittest.TestCase):
    def test_patch_gi_roundtrip_json(self):
        gi = {
            "schema_version": "1.0",
            "episode_id": "e1",
            "nodes": [
                {"id": "episode:e1", "type": "Episode", "properties": {}},
                {"id": "insight:x", "type": "Insight", "properties": {"text": "a"}},
                {"id": "topic:t", "type": "Topic", "properties": {"label": "old"}},
            ],
            "edges": [
                {"type": "HAS_INSIGHT", "from": "episode:e1", "to": "insight:x"},
                {"type": "ABOUT", "from": "insight:x", "to": "topic:t"},
            ],
        }
        out, _ = patch_gi_for_bullet_labels(gi, ["Only"])
        assert out is not None
        json.dumps(out, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
