"""Unit tests for KG corpus helpers (roll-up, co-occurrence, inspect summary)."""

import unittest
from pathlib import Path

from podcast_scraper.kg.corpus import (
    entity_rollup,
    inspect_summary,
    topic_cooccurrence,
)
from podcast_scraper.kg.pipeline import build_artifact
from podcast_scraper.kg.schema import validate_artifact


class TestKgCorpus(unittest.TestCase):
    """Tests for corpus aggregation."""

    def test_entity_rollup_counts_episodes(self) -> None:
        """Same entity name across two episodes appears in roll-up."""
        a1 = build_artifact(
            "episode:1",
            "t",
            podcast_id="pod:p",
            episode_title="E1",
            detected_hosts=["Pat"],
        )
        a2 = build_artifact(
            "episode:2",
            "t",
            podcast_id="pod:p",
            episode_title="E2",
            detected_hosts=["Pat"],
        )
        validate_artifact(a1, strict=True)
        validate_artifact(a2, strict=True)
        loaded = [(Path("a.kg.json"), a1), (Path("b.kg.json"), a2)]
        rows = entity_rollup(loaded, min_episodes=1, output_dir=None)
        pat = next((r for r in rows if r["name"] == "Pat"), None)
        self.assertIsNotNone(pat)
        assert pat is not None
        self.assertEqual(pat["episode_count"], 2)

    def test_topic_cooccurrence_single_episode(self) -> None:
        """Two topics in one episode produce one unordered pair."""
        art = {
            "schema_version": "1.0",
            "episode_id": "episode:x",
            "extraction": {
                "model_version": "stub",
                "extracted_at": "2026-01-01T00:00:00Z",
                "transcript_ref": "t.txt",
            },
            "nodes": [
                {
                    "id": "kg:ep",
                    "type": "Episode",
                    "properties": {
                        "podcast_id": "p",
                        "title": "T",
                        "publish_date": "2026-01-01T00:00:00Z",
                    },
                },
                {
                    "id": "kg:t:a",
                    "type": "Topic",
                    "properties": {"label": "Alpha", "slug": "alpha"},
                },
                {
                    "id": "kg:t:b",
                    "type": "Topic",
                    "properties": {"label": "Beta", "slug": "beta"},
                },
            ],
            "edges": [
                {"from": "kg:t:a", "to": "kg:ep", "type": "MENTIONS", "properties": {}},
                {"from": "kg:t:b", "to": "kg:ep", "type": "MENTIONS", "properties": {}},
            ],
        }
        validate_artifact(art, strict=True)
        rows = topic_cooccurrence([(Path("x.kg.json"), art)], min_support=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["episode_count"], 1)
        labels = {rows[0]["topic_a_label"], rows[0]["topic_b_label"]}
        self.assertEqual(labels, {"Alpha", "Beta"})

    def test_inspect_summary_shape(self) -> None:
        art = build_artifact("e:9", "x", podcast_id="p", episode_title="Nine")
        s = inspect_summary(art)
        self.assertEqual(s["episode_id"], "e:9")
        self.assertIn("Episode", s["nodes_by_type"])
