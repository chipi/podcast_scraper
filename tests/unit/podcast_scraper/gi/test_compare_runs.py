"""Unit tests for ``gi.compare_runs`` (paired GIL run comparison)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import cast, List, Optional, Tuple

import pytest

from podcast_scraper.gi.compare_runs import (
    collect_gil_stats_from_run_root,
    format_text_report,
    GilArtifactStats,
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
        self.assertEqual(ref["ci-fixture"].n_quotes, 2)
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


@pytest.mark.unit
class TestCompareRunsPairingAndEdges:
    """Direct tests for pairing, agreement buckets, and resilient collection."""

    def test_paired_episode_rows_union_sorted(self) -> None:
        ref = {"z": GilArtifactStats("z", "a", 1, 0, 1)}
        cand = {"a": GilArtifactStats("a", "b", 1, 0, 0)}
        rows = paired_episode_rows(ref, cand)
        assert [r[0] for r in rows] == ["a", "z"]
        assert rows[0] == ("a", None, cand["a"])
        assert rows[1] == ("z", ref["z"], None)

    def test_summarize_agreement_ref_only_and_cand_only_quotes(self) -> None:
        ref = {"e1": GilArtifactStats("e1", "p1", 1, 0, 2)}
        cand = {"e1": GilArtifactStats("e1", "p2", 1, 0, 0)}
        rows = paired_episode_rows(ref, cand)
        s = summarize_agreement(rows)
        assert s["episodes_compared"] == 1
        assert s["both_have_quotes"] == 0
        assert s["reference_only_quotes"] == 1
        assert s["candidate_only_quotes"] == 0
        assert s["neither_has_quotes"] == 0
        assert s["missing_in_reference"] == 0
        assert s["missing_in_candidate"] == 0

        cand2 = {"e1": GilArtifactStats("e1", "p2", 1, 0, 1)}
        s2 = summarize_agreement(paired_episode_rows(ref, cand2))
        assert s2["both_have_quotes"] == 1

    def test_summarize_agreement_missing_episode_counts(self) -> None:
        ref = {"a": GilArtifactStats("a", "p", 0, 0, 0)}
        cand = {"b": GilArtifactStats("b", "p", 0, 0, 0)}
        s = summarize_agreement(paired_episode_rows(ref, cand))
        assert s["episodes_compared"] == 2
        assert s["missing_in_reference"] == 1
        assert s["missing_in_candidate"] == 1
        assert s["neither_has_quotes"] == 2

    def test_collect_skips_bad_json_and_non_dict(self, tmp_path: Path) -> None:
        meta = tmp_path / "metadata"
        meta.mkdir()
        (meta / "good.gi.json").write_text(
            '{"episode_id": "ok", "nodes": [{"type": "Quote"}], "edges": []}',
            encoding="utf-8",
        )
        (meta / "bad.gi.json").write_text("{not json", encoding="utf-8")
        (meta / "list.gi.json").write_text("[]", encoding="utf-8")
        stats = collect_gil_stats_from_run_root(tmp_path)
        assert set(stats) == {"ok"}
        assert stats["ok"].n_quotes == 1

    def test_format_text_report_includes_summary_keys(self, tmp_path: Path) -> None:
        ref_root = tmp_path / "ref"
        cand_root = tmp_path / "cand"
        ref_root.mkdir()
        cand_root.mkdir()
        rows = cast(
            List[Tuple[str, Optional[GilArtifactStats], Optional[GilArtifactStats]]],
            [
                (
                    "ep1",
                    GilArtifactStats("ep1", "r", 1, 1, 1),
                    GilArtifactStats("ep1", "c", 1, 0, 0),
                )
            ],
        )
        summary = summarize_agreement(rows)
        text = format_text_report(ref_root, cand_root, rows, summary)
        assert "GIL run comparison" in text
        assert "Summary (quote = at least one Quote node):" in text
        for key in (
            "episodes_compared",
            "both_have_quotes",
            "reference_only_quotes",
            "missing_in_reference",
        ):
            assert key in text


if __name__ == "__main__":
    unittest.main()
