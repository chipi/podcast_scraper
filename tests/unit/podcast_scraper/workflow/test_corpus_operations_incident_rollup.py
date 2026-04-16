"""Unit tests for multi-feed corpus_run_summary incident rollup (GitHub #557 / opportunity #6)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.workflow import corpus_operations
from podcast_scraper.workflow.corpus_operations import MultiFeedFeedResult


@pytest.mark.unit
class TestParseCorpusIncidentJsonlWindow(unittest.TestCase):
    def test_mid_file_offset_skips_partial_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "corpus_incidents.jsonl"
            first = {"scope": "episode", "category": "policy", "feed_url": "https://a/x"}
            p.write_text(json.dumps(first) + "\n", encoding="utf-8")
            mid = p.stat().st_size
            second = {"scope": "feed", "category": "soft", "feed_url": "https://b/y"}
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(second) + "\n")
            records, end_off = corpus_operations.parse_corpus_incident_jsonl_window(str(p), mid)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].get("scope"), "feed")
            self.assertEqual(end_off, p.stat().st_size)


@pytest.mark.unit
class TestRollupCorpusIncidentsForMultiFeedSummary(unittest.TestCase):
    def test_dedupes_episode_rows_by_episode_id(self) -> None:
        fu = "https://feeds.example/a.xml"
        rows: List[Dict[str, Any]] = [
            {"scope": "episode", "category": "policy", "feed_url": fu, "episode_id": "e1"},
            {"scope": "episode", "category": "policy", "feed_url": fu, "episode_id": "e1"},
            {"scope": "episode", "category": "soft", "feed_url": fu, "episode_idx": 2},
        ]
        batch, per_feed = corpus_operations.rollup_corpus_incidents_for_multi_feed_summary(rows)
        self.assertEqual(batch["episode_incidents_unique"]["policy"], 1)
        self.assertEqual(batch["episode_incidents_unique"]["soft"], 1)
        self.assertEqual(batch["episode_incidents_unique"]["hard"], 0)
        self.assertEqual(per_feed[fu]["policy"], 1)
        self.assertEqual(per_feed[fu]["soft"], 1)


@pytest.mark.unit
class TestFinalizeMultiFeedBatchIncidentRollup(unittest.TestCase):
    def test_summary_includes_batch_incidents_and_per_feed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as corpus:
            inc = Path(corpus) / "corpus_incidents.jsonl"
            fu_a = "https://a.example/feed.xml"
            fu_b = "https://b.example/feed.xml"
            inc.write_text("", encoding="utf-8")
            start = inc.stat().st_size
            line_a = json.dumps(
                {
                    "scope": "episode",
                    "category": "policy",
                    "feed_url": fu_a,
                    "episode_id": "ea-1",
                }
            )
            line_b = json.dumps(
                {
                    "scope": "feed",
                    "category": "soft",
                    "feed_url": fu_b,
                }
            )
            with inc.open("a", encoding="utf-8") as fh:
                fh.write(line_a + "\n" + line_b + "\n")

            class _Cfg:
                vector_search = False

            batch = [
                MultiFeedFeedResult(fu_a, True, None, 0, finished_at="2026-01-01T00:00:00Z"),
                MultiFeedFeedResult(fu_b, False, "rss", 0, finished_at="2026-01-01T00:00:01Z"),
            ]
            doc = corpus_operations.finalize_multi_feed_batch(
                corpus,
                _Cfg(),  # type: ignore[arg-type]
                batch,
                incident_log_path=str(inc),
                incident_log_start_offset=start,
            )
            self.assertEqual(
                doc.get("schema_version"),
                corpus_operations.CORPUS_RUN_SUMMARY_SCHEMA_VERSION,
            )
            bi = doc.get("batch_incidents")
            self.assertIsInstance(bi, dict)
            assert isinstance(bi, dict)
            self.assertEqual(bi.get("lines_in_window"), 2)
            self.assertEqual(bi.get("episode_incidents_unique", {}).get("policy"), 1)
            self.assertEqual(bi.get("feed_incidents_unique", {}).get("soft"), 1)
            self.assertEqual(bi.get("episodes_documented_skips_unique"), 1)
            self.assertEqual(bi.get("episodes_other_incidents_unique"), 0)
            feeds = doc.get("feeds") or []
            by_url = {str(r["feed_url"]): r for r in feeds}
            self.assertEqual(by_url[fu_a]["episode_incidents_unique"]["policy"], 1)
            self.assertEqual(by_url[fu_b]["episode_incidents_unique"]["policy"], 0)

            summary_path = os.path.join(corpus, "corpus_run_summary.json")
            self.assertTrue(os.path.isfile(summary_path))
