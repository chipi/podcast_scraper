"""Unit tests for KG pipeline (stub artifact builder)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from podcast_scraper.kg.pipeline import build_artifact
from podcast_scraper.kg.schema import validate_artifact


class TestKgPipeline(unittest.TestCase):
    """Tests for build_artifact."""

    def test_build_artifact_minimal_passes_strict_schema(self) -> None:
        """Episode-only graph validates against kg.schema.json."""
        art = build_artifact(
            "episode:test-1",
            "Hello transcript.",
            podcast_id="podcast:abc",
            episode_title="My Episode",
            publish_date="2024-01-15T12:00:00Z",
            transcript_ref="transcripts/ep.txt",
        )
        self.assertEqual(art["schema_version"], "1.0")
        self.assertEqual(art["episode_id"], "episode:test-1")
        self.assertEqual(len(art["nodes"]), 1)
        self.assertEqual(art["nodes"][0]["type"], "Episode")
        validate_artifact(art, strict=True)

    def test_build_artifact_topic_and_hosts(self) -> None:
        """Topic + host entities produce MENTIONS edges to Episode."""
        art = build_artifact(
            "ep:x",
            "x",
            podcast_id="p:1",
            episode_title="T",
            topic_label="Inflation outlook",
            detected_hosts=["Alice"],
            detected_guests=["Bob"],
        )
        validate_artifact(art, strict=True)
        types = {n["type"] for n in art["nodes"]}
        self.assertIn("Episode", types)
        self.assertIn("Topic", types)
        self.assertIn("Entity", types)
        self.assertTrue(any(e["type"] == "MENTIONS" for e in art["edges"]))

    def test_stub_source_skips_summary_topics(self) -> None:
        """stub + cfg: topic hints do not create Topic nodes."""
        cfg = SimpleNamespace(kg_extraction_source="stub")
        art = build_artifact(
            "ep:stub",
            "transcript",
            podcast_id="p:1",
            episode_title="T",
            cfg=cfg,
            topic_label="Should not appear",
            detected_hosts=["Pat"],
        )
        validate_artifact(art, strict=True)
        types = {n["type"] for n in art["nodes"]}
        self.assertIn("Episode", types)
        self.assertIn("Entity", types)
        self.assertNotIn("Topic", types)
        self.assertEqual(art["extraction"]["model_version"], "stub")

    def test_summary_bullets_multiple_topics_with_cfg(self) -> None:
        """summary_bullets uses up to kg_max_topics labels."""
        cfg = SimpleNamespace(kg_extraction_source="summary_bullets", kg_max_topics=2)
        art = build_artifact(
            "ep:m",
            "x",
            podcast_id="p:1",
            episode_title="T",
            cfg=cfg,
            topic_labels=["One topic", "Two topic", "Three ignored"],
        )
        validate_artifact(art, strict=True)
        topics = [n for n in art["nodes"] if n["type"] == "Topic"]
        self.assertEqual(len(topics), 2)
        self.assertEqual(art["extraction"]["model_version"], "summary_bullets")

    def test_provider_path_uses_llm_partial(self) -> None:
        """provider source calls extract_kg_graph and merges pipeline entities."""
        prov = MagicMock()
        prov.summary_model = "test-model"
        prov.extract_kg_graph.return_value = {
            "topics": [{"label": "AI policy"}],
            "entities": [{"name": "ACME Corp", "entity_kind": "organization"}],
        }
        metrics = SimpleNamespace(kg_provider_extractions=0)
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=True,
        )
        art = build_artifact(
            "ep:llm",
            "We discuss AI policy at ACME Corp.",
            podcast_id="p:1",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
            pipeline_metrics=metrics,
            detected_hosts=["Pat Host"],
        )
        validate_artifact(art, strict=True)
        prov.extract_kg_graph.assert_called_once()
        self.assertEqual(metrics.kg_provider_extractions, 1)
        types = {n["type"] for n in art["nodes"]}
        self.assertIn("Topic", types)
        self.assertIn("Entity", types)
        self.assertTrue(art["extraction"]["model_version"].startswith("provider:"))
