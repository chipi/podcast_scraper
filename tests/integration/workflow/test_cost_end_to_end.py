"""#650 + #651 end-to-end integration: per-stage cost recording → metrics.json
→ aggregator → cost_rollup → ``corpus-cost`` CLI.

Covers what no single unit test can: the stitch between the Metrics class, its
JSON serialisation (``to_dict()``), the corpus-wide aggregator, and the CLI
rendering. If any link in that chain silently drops a new stage field (as
happened pre-PR #661 with ``llm_bundled_clean_summary_cost_usd``), this test
fails.

Hermetic — no real provider calls. Uses the public ``record_llm_*_call``
API to simulate a realistic mixed-provider mega_bundled + staged run.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from podcast_scraper.cli import _run_corpus_cost_cli
from podcast_scraper.workflow.corpus_cost_aggregation import aggregate_corpus_costs
from podcast_scraper.workflow.metrics import Metrics


def _finish_and_write(metrics: Metrics, run_dir: Path) -> dict:
    """Serialise metrics via the public API and write metrics.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    result = metrics.finish()
    (run_dir / "metrics.json").write_text(json.dumps(result), encoding="utf-8")
    return result


@pytest.mark.integration
@pytest.mark.critical_path
class TestCostFlowEndToEnd(unittest.TestCase):
    """Exercise the full cost chain: recorder → metrics.json → rollup → CLI."""

    def test_all_stages_flow_through_to_rollup(self) -> None:
        """One simulated episode invokes every record_llm_*_call with cost_usd;
        assert every value survives through aggregate_corpus_costs + CLI.
        """
        with TemporaryDirectory() as td:
            corpus = Path(td)
            run_dir = corpus / "feeds" / "example" / "run_001"

            # Simulate a realistic mixed-mode run: staged transcription +
            # speaker + cleaning + summarization + GI + KG + bundle.
            m = Metrics()
            m.record_llm_transcription_call(audio_minutes=30.0, cost_usd=0.18)
            m.record_llm_speaker_detection_call(input_tokens=100, output_tokens=50, cost_usd=0.0004)
            m.record_llm_cleaning_call(input_tokens=2000, output_tokens=1500, cost_usd=0.0060)
            m.record_llm_summarization_call(input_tokens=5000, output_tokens=800, cost_usd=0.0130)
            m.record_llm_gi_call(input_tokens=3000, output_tokens=400, cost_usd=0.0075)
            m.record_llm_kg_call(input_tokens=2000, output_tokens=300, cost_usd=0.0050)
            m.record_llm_bundled_clean_summary_call(
                input_tokens=6000, output_tokens=900, cost_usd=0.0150
            )

            result = _finish_and_write(m, run_dir)

            # --- Link 1: Metrics.to_dict() emits every stage field ------------
            for field in (
                "llm_transcription_cost_usd",
                "llm_speaker_detection_cost_usd",
                "llm_cleaning_cost_usd",
                "llm_summarization_cost_usd",
                "llm_gi_cost_usd",
                "llm_kg_cost_usd",
                "llm_bundled_clean_summary_cost_usd",
            ):
                self.assertIn(field, result, f"{field} missing from metrics.json")
                self.assertGreater(result[field], 0, f"{field} zero in metrics.json")

            # Stage total matches the sum (Metrics.total_stage_cost_usd).
            expected_stage_total = round(
                0.18 + 0.0004 + 0.0060 + 0.0130 + 0.0075 + 0.0050 + 0.0150, 6
            )
            self.assertAlmostEqual(result["total_stage_cost_usd"], expected_stage_total, places=6)

            # --- Link 2: corpus-wide aggregator sees every stage --------------
            rollup = aggregate_corpus_costs(corpus)
            self.assertEqual(rollup["run_count"], 1)
            self.assertAlmostEqual(rollup["by_stage"]["llm_transcription_cost_usd"], 0.18, places=6)
            self.assertAlmostEqual(
                rollup["by_stage"]["llm_bundled_clean_summary_cost_usd"], 0.0150, places=6
            )
            self.assertAlmostEqual(rollup["total_transcription_cost_usd"], 0.18, places=6)
            # Non-transcription LLM sum = everything except transcription.
            expected_llm = round(0.0004 + 0.0060 + 0.0130 + 0.0075 + 0.0050 + 0.0150, 6)
            self.assertAlmostEqual(rollup["total_llm_cost_usd"], expected_llm, places=6)
            self.assertAlmostEqual(rollup["total_cost_usd"], expected_stage_total, places=6)
            self.assertEqual(rollup["metrics_files_missing_cost_fields"], 0)

            # --- Link 3: corpus-cost CLI renders every stage ------------------
            args = argparse.Namespace(corpus_path=str(corpus), json=False, update_manifest=False)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _run_corpus_cost_cli(args, logging.getLogger("test"))
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            for stage in (
                "llm_transcription_cost_usd",
                "llm_summarization_cost_usd",
                "llm_speaker_detection_cost_usd",
                "llm_cleaning_cost_usd",
                "llm_gi_cost_usd",
                "llm_kg_cost_usd",
                "llm_bundled_clean_summary_cost_usd",
            ):
                self.assertIn(stage, out, f"{stage} missing from CLI output")

    def test_multi_run_aggregation(self) -> None:
        """Two runs under the corpus sum correctly across all stage fields."""
        with TemporaryDirectory() as td:
            corpus = Path(td)
            m1 = Metrics()
            m1.record_llm_transcription_call(audio_minutes=10.0, cost_usd=0.06)
            m1.record_llm_summarization_call(input_tokens=1000, output_tokens=200, cost_usd=0.003)
            _finish_and_write(m1, corpus / "feeds" / "show_a" / "run_001")

            m2 = Metrics()
            m2.record_llm_transcription_call(audio_minutes=20.0, cost_usd=0.12)
            m2.record_llm_gi_call(input_tokens=2000, output_tokens=300, cost_usd=0.005)
            m2.record_llm_bundled_clean_summary_call(
                input_tokens=6000, output_tokens=900, cost_usd=0.01
            )
            _finish_and_write(m2, corpus / "feeds" / "show_b" / "run_002")

            rollup = aggregate_corpus_costs(corpus)
            self.assertEqual(rollup["run_count"], 2)
            self.assertAlmostEqual(rollup["by_stage"]["llm_transcription_cost_usd"], 0.18, places=6)
            self.assertAlmostEqual(
                rollup["by_stage"]["llm_summarization_cost_usd"], 0.003, places=6
            )
            self.assertAlmostEqual(rollup["by_stage"]["llm_gi_cost_usd"], 0.005, places=6)
            self.assertAlmostEqual(
                rollup["by_stage"]["llm_bundled_clean_summary_cost_usd"], 0.01, places=6
            )
            self.assertAlmostEqual(rollup["total_cost_usd"], 0.198, places=6)

    def test_pre_650_metrics_file_tolerated(self) -> None:
        """A pre-#650 metrics.json (lacking llm_*_cost_usd fields) must not crash
        the aggregator — it's counted in ``metrics_files_missing_cost_fields``.
        """
        with TemporaryDirectory() as td:
            corpus = Path(td)
            run_dir = corpus / "feeds" / "legacy" / "run_old"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(json.dumps({"legacy": True}), encoding="utf-8")

            # And a modern metrics.json alongside, so we know it coexists.
            m = Metrics()
            m.record_llm_transcription_call(audio_minutes=5.0, cost_usd=0.03)
            _finish_and_write(m, corpus / "feeds" / "modern" / "run_new")

            rollup = aggregate_corpus_costs(corpus)
            self.assertEqual(rollup["run_count"], 2)
            self.assertEqual(rollup["metrics_files_missing_cost_fields"], 1)
            self.assertAlmostEqual(rollup["total_transcription_cost_usd"], 0.03, places=6)
