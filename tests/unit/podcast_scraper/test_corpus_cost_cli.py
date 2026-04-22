"""#650 audit — regression guard for the ``corpus-cost`` CLI subcommand.

Covers:
- human-readable output: contains cost lines + per-stage breakdown including
  the new ``llm_bundled_clean_summary_cost_usd`` field.
- ``--json`` mode: parses as JSON with the expected top-level schema.
- ``--update-manifest`` rewrites ``corpus_manifest.json`` with fresh rollup.
- empty corpus (no metrics.json) returns zeros (not a crash).
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


def _write_metrics(run_dir: Path, **fields: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "llm_transcription_cost_usd": 0.0,
        "llm_summarization_cost_usd": 0.0,
        "llm_speaker_detection_cost_usd": 0.0,
        "llm_cleaning_cost_usd": 0.0,
        "llm_gi_cost_usd": 0.0,
        "llm_kg_cost_usd": 0.0,
        "llm_bundled_clean_summary_cost_usd": 0.0,
    }
    payload.update(fields)
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
class TestCorpusCostCli(unittest.TestCase):
    def test_human_output_contains_expected_lines(self) -> None:
        with TemporaryDirectory() as td:
            corpus = Path(td)
            feed = corpus / "feeds" / "example"
            _write_metrics(
                feed / "run_001",
                llm_transcription_cost_usd=0.0123,
                llm_summarization_cost_usd=0.0456,
                llm_bundled_clean_summary_cost_usd=0.0010,
            )
            args = argparse.Namespace(corpus_path=str(corpus), json=False, update_manifest=False)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _run_corpus_cost_cli(args, logging.getLogger("test"))
            out = buf.getvalue()

            self.assertEqual(rc, 0)
            self.assertIn("Runs aggregated:             1", out)
            self.assertIn("Transcription cost:          $0.0123", out)
            # Total ≈ 0.0456 (summarization) + 0.0010 (bundled) = 0.0466.
            self.assertIn("LLM cost (non-transcription): $0.0466", out)
            self.assertIn("llm_bundled_clean_summary_cost_usd: $0.0010", out)
            # Per-stage breakdown prints every stage field including zero ones.
            self.assertIn("llm_transcription_cost_usd:", out)
            self.assertIn("llm_summarization_cost_usd:", out)

    def test_json_mode_emits_parsable_payload(self) -> None:
        with TemporaryDirectory() as td:
            corpus = Path(td)
            feed = corpus / "feeds" / "example"
            _write_metrics(
                feed / "run_001",
                llm_gi_cost_usd=0.02,
                llm_kg_cost_usd=0.01,
            )
            args = argparse.Namespace(corpus_path=str(corpus), json=True, update_manifest=False)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _run_corpus_cost_cli(args, logging.getLogger("test"))
            self.assertEqual(rc, 0)

            payload = json.loads(buf.getvalue())
            self.assertIn("total_cost_usd", payload)
            self.assertIn("total_llm_cost_usd", payload)
            self.assertIn("total_transcription_cost_usd", payload)
            self.assertIn("by_stage", payload)
            self.assertIn("run_count", payload)
            self.assertIn("metrics_files_missing_cost_fields", payload)
            self.assertIn("llm_bundled_clean_summary_cost_usd", payload["by_stage"])
            self.assertAlmostEqual(payload["total_cost_usd"], 0.03, places=4)

    def test_empty_corpus_returns_zeros(self) -> None:
        with TemporaryDirectory() as td:
            corpus = Path(td)
            args = argparse.Namespace(corpus_path=str(corpus), json=True, update_manifest=False)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _run_corpus_cost_cli(args, logging.getLogger("test"))
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["run_count"], 0)
            self.assertEqual(payload["total_cost_usd"], 0.0)

    def test_missing_corpus_returns_exit_1(self) -> None:
        args = argparse.Namespace(corpus_path="/does/not/exist", json=False, update_manifest=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _run_corpus_cost_cli(args, logging.getLogger("test"))
        self.assertEqual(rc, 1)

    def test_update_manifest_rewrites_cost_rollup(self) -> None:
        with TemporaryDirectory() as td:
            corpus = Path(td)
            feed = corpus / "feeds" / "example"
            _write_metrics(
                feed / "run_001",
                llm_summarization_cost_usd=0.0123,
            )
            manifest_path = corpus / "corpus_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "cost_rollup": {"total_cost_usd": 0.0},  # stale
                    }
                ),
                encoding="utf-8",
            )

            args = argparse.Namespace(corpus_path=str(corpus), json=False, update_manifest=True)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _run_corpus_cost_cli(args, logging.getLogger("test"))
            self.assertEqual(rc, 0)

            written = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], "1.1.0")
            self.assertAlmostEqual(written["cost_rollup"]["total_cost_usd"], 0.0123, places=4)

    def test_update_manifest_missing_file_returns_2(self) -> None:
        with TemporaryDirectory() as td:
            corpus = Path(td)
            _write_metrics(
                corpus / "feeds" / "example" / "run_001",
                llm_summarization_cost_usd=0.01,
            )
            args = argparse.Namespace(corpus_path=str(corpus), json=False, update_manifest=True)
            # Intentionally NOT writing corpus_manifest.json — drives the
            # "manifest not found" error path.
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _run_corpus_cost_cli(args, logging.getLogger("test"))
            self.assertEqual(rc, 2)
