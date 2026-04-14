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
        self.assertEqual(art["schema_version"], "1.2")
        self.assertEqual(art["episode_id"], "episode:test-1")
        self.assertEqual(len(art["nodes"]), 1)
        self.assertEqual(art["nodes"][0]["type"], "Episode")
        self.assertEqual(art["nodes"][0]["id"], "episode:episode:test-1")
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

    def test_summary_bullets_derives_via_provider_when_available(self) -> None:
        """summary_bullets calls extract_kg_from_summary_bullets when implemented."""
        prov = MagicMock(spec=["extract_kg_from_summary_bullets", "summary_model"])
        prov.summary_model = "test-model"
        prov.extract_kg_from_summary_bullets.return_value = {
            "topics": [{"label": "Derived topic"}],
            "entities": [{"name": "Jane Doe", "entity_kind": "person"}],
        }
        metrics = SimpleNamespace(kg_provider_extractions=0)
        cfg = SimpleNamespace(
            kg_extraction_source="summary_bullets",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=True,
        )
        art = build_artifact(
            "ep:bullets-llm",
            "",
            podcast_id="p:1",
            episode_title="E",
            cfg=cfg,
            topic_labels=["Long bullet one", "Long bullet two"],
            kg_extraction_provider=prov,
            pipeline_metrics=metrics,
        )
        validate_artifact(art, strict=True)
        prov.extract_kg_from_summary_bullets.assert_called_once()
        self.assertEqual(metrics.kg_provider_extractions, 1)
        self.assertTrue(art["extraction"]["model_version"].startswith("provider:summary_bullets:"))
        topic_labels = {n["properties"]["label"] for n in art["nodes"] if n["type"] == "Topic"}
        self.assertIn("Derived topic", topic_labels)

    def test_summary_bullets_verbatim_when_provider_lacks_bullet_method(self) -> None:
        """Without extract_kg_from_summary_bullets, labels become topic nodes."""

        class _ProvNoBullets:
            summary_model = "x"

        cfg = SimpleNamespace(kg_extraction_source="summary_bullets", kg_max_topics=2)
        art = build_artifact(
            "ep:verbatim",
            "",
            podcast_id="p:1",
            episode_title="T",
            cfg=cfg,
            topic_labels=["Alpha", "Beta"],
            kg_extraction_provider=_ProvNoBullets(),
        )
        validate_artifact(art, strict=True)
        topic_labels = {n["properties"]["label"] for n in art["nodes"] if n["type"] == "Topic"}
        self.assertEqual(topic_labels, {"Alpha", "Beta"})
        self.assertEqual(art["extraction"]["model_version"], "summary_bullets")

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

    def test_provider_entities_dedup_by_kind_and_name(self) -> None:
        """Same display name may appear as person and organization; both are kept."""
        prov = MagicMock()
        prov.summary_model = "test-model"
        prov.extract_kg_graph.return_value = {
            "topics": [],
            "entities": [
                {"name": "Mercury", "entity_kind": "person"},
                {"name": "Mercury", "entity_kind": "organization"},
                {"name": "Mercury", "entity_kind": "person"},
            ],
        }
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=False,
        )
        art = build_artifact(
            "ep:dedup-kind",
            "transcript",
            podcast_id="p:1",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
        )
        validate_artifact(art, strict=True)
        entities = [n for n in art["nodes"] if n["type"] == "Entity"]
        kinds = sorted((n["properties"]["kind"], n["properties"]["name"]) for n in entities)
        self.assertEqual(
            kinds,
            [("org", "Mercury"), ("person", "Mercury")],
        )
        self.assertEqual(len(entities), 2)

    def test_pipeline_host_skipped_only_if_same_kind_and_name_as_llm(self) -> None:
        """Host merged only when person+name matches; org with same name does not block."""
        prov = MagicMock()
        prov.summary_model = "test-model"
        prov.extract_kg_graph.return_value = {
            "topics": [],
            "entities": [{"name": "ACME", "entity_kind": "organization"}],
        }
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=True,
        )
        art = build_artifact(
            "ep:host-org",
            "x",
            podcast_id="p:1",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
            detected_hosts=["ACME"],
        )
        validate_artifact(art, strict=True)
        entities = [n for n in art["nodes"] if n["type"] == "Entity"]
        kinds = {(n["properties"]["kind"], n["properties"]["name"]) for n in entities}
        self.assertIn(("org", "ACME"), kinds)
        self.assertIn(("person", "ACME"), kinds)
        self.assertEqual(len(entities), 2)

    def test_provider_path_propagates_llm_descriptions(self) -> None:
        """LLM partial topic/entity descriptions land on nodes (#487)."""
        prov = MagicMock()
        prov.summary_model = "m"
        prov.extract_kg_graph.return_value = {
            "topics": [{"label": "X", "description": "Why X matters."}],
            "entities": [
                {"name": "Pat", "entity_kind": "person", "description": "Guest expert."},
            ],
        }
        cfg = SimpleNamespace(
            kg_extraction_source="provider",
            kg_max_topics=5,
            kg_max_entities=10,
            kg_merge_pipeline_entities=False,
        )
        art = build_artifact(
            "ep:desc",
            "t",
            podcast_id="p",
            episode_title="E",
            cfg=cfg,
            kg_extraction_provider=prov,
        )
        validate_artifact(art, strict=True)
        topics = [n for n in art["nodes"] if n["type"] == "Topic"]
        self.assertEqual(topics[0]["properties"].get("description"), "Why X matters.")
        ents = [n for n in art["nodes"] if n["type"] == "Entity"]
        self.assertEqual(ents[0]["properties"].get("description"), "Guest expert.")
