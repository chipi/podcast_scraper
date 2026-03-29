"""Unit tests for KG inspect contracts (Pydantic)."""

import unittest
from pathlib import Path

from podcast_scraper.kg.contracts import (
    build_kg_corpus_bundle_output,
    build_kg_entity_rollup_output,
    build_kg_inspect_output,
    build_kg_topic_pairs_output,
    KgCorpusBundleOutput,
    KgEntityRollupOutput,
    KgInspectOutput,
    KgTopicPairsOutput,
)
from podcast_scraper.kg.corpus import entity_rollup, export_merged_json, topic_cooccurrence
from podcast_scraper.kg.pipeline import build_artifact


class TestKgContracts(unittest.TestCase):
    """Tests for build_kg_inspect_output."""

    def test_build_kg_inspect_output_roundtrip(self) -> None:
        """Validated model matches artifact summary fields."""
        art = build_artifact(
            "ep:c1",
            "x",
            podcast_id="p:1",
            episode_title="Title",
            topic_label="T1",
        )
        out = build_kg_inspect_output(art, artifact_path=Path("metadata") / "x.kg.json")
        self.assertIsInstance(out, KgInspectOutput)
        self.assertEqual(out.episode_id, "ep:c1")
        self.assertTrue(str(out.artifact_path or "").endswith("x.kg.json"))
        self.assertGreaterEqual(out.node_count, 1)
        dumped = out.model_dump(mode="json")
        self.assertIn("topics", dumped)

    def test_entity_rollup_and_topic_contracts(self) -> None:
        """Roll-up and topic-pair CLI JSON validate via Pydantic."""
        art = build_artifact(
            "ep:c1",
            "x",
            podcast_id="p:1",
            episode_title="Title",
            topic_label="T1",
            detected_hosts=["Host A"],
        )
        loaded = [(Path("metadata/x.kg.json"), art)]
        rows = entity_rollup(loaded, min_episodes=1)
        er = build_kg_entity_rollup_output(rows)
        self.assertIsInstance(er, KgEntityRollupOutput)
        self.assertGreaterEqual(len(er.entities), 1)

        pairs = topic_cooccurrence(loaded, min_support=1)
        tp = build_kg_topic_pairs_output(pairs)
        self.assertIsInstance(tp, KgTopicPairsOutput)

    def test_merged_export_contract(self) -> None:
        """Merged export bundle validates."""
        art = build_artifact(
            "ep:c1",
            "x",
            podcast_id="p:1",
            episode_title="Title",
            topic_label="T1",
            detected_hosts=["Host A"],
        )
        loaded = [(Path("metadata/x.kg.json"), art)]
        bundle = export_merged_json(loaded, output_dir=None)
        out = build_kg_corpus_bundle_output(bundle)
        self.assertIsInstance(out, KgCorpusBundleOutput)
        self.assertEqual(out.export_kind, "kg_corpus_bundle")
        self.assertEqual(out.artifact_count, len(out.artifacts))
