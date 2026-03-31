"""Unit tests for ``gi.compare_runs`` (paired GIL run comparison)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from podcast_scraper.gi.compare_runs import (
    collect_gil_stats_from_run_root,
    format_text_report,
    paired_episode_rows,
    summarize_agreement,
)


class TestCompareGilRuns(unittest.TestCase):
    """Tests for collect / pair / summarize / report."""

    def setUp(self) -> None:
        self.fixture_gi = (
            Path(__file__).resolve().parents[3]
            / "fixtures"
            / "gil_kg_ci_enforce"
            / "metadata"
            / "ci_sample.gi.json"
        )

    def test_collect_and_summarize(self) -> None:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        root = Path(tmp)
        meta = root / "metadata"
        meta.mkdir(parents=True)
        shutil.copy(self.fixture_gi, meta / "ep_a.gi.json")
        data = json.loads(self.fixture_gi.read_text(encoding="utf-8"))
        data["episode_id"] = "ep-b"
        data["nodes"] = [n for n in data["nodes"] if n.get("type") != "Quote"]
        data["edges"] = [e for e in data["edges"] if e.get("type") != "SUPPORTED_BY"]
        ins = [n for n in data["nodes"] if n.get("type") == "Insight"]
        if ins:
            ins[0].setdefault("properties", {})["grounded"] = False
        (meta / "ep_b.gi.json").write_text(json.dumps(data), encoding="utf-8")

        ref = collect_gil_stats_from_run_root(root)
        self.assertEqual(len(ref), 2)
        self.assertEqual(ref["ci-fixture"].n_quotes, 1)
        self.assertEqual(ref["ep-b"].n_quotes, 0)

        rows = paired_episode_rows(ref, ref)
        s = summarize_agreement(rows)
        self.assertEqual(s["both_have_quotes"], 1)
        self.assertEqual(s["neither_has_quotes"], 1)

        text = format_text_report(root, root, rows, s)
        self.assertIn("ci-fixture", text)
        self.assertIn("GIL run comparison", text)

    def test_collect_missing_metadata_uses_root(self) -> None:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        root = Path(tmp)
        shutil.copy(self.fixture_gi, root / "solo.gi.json")
        stats = collect_gil_stats_from_run_root(root)
        self.assertEqual(len(stats), 1)


if __name__ == "__main__":
    unittest.main()
