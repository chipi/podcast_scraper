"""Unit tests for podcast_scraper.evaluation.scorer module."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.evaluation import scorer as scorer_mod
from podcast_scraper.evaluation.scorer import (
    compute_bleu_vs_reference,
    compute_embedding_similarity,
    compute_intrinsic_metrics,
    compute_rouge_vs_reference,
    compute_vs_reference_metrics,
    compute_wer_vs_reference,
    estimate_tokens,
    load_predictions,
    score_run,
)


@pytest.mark.unit
class TestLoadPredictions:
    """Tests for load_predictions."""

    def test_load_valid_jsonl(self):
        """Load valid JSONL file returns list of dicts."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"episode_id": "e1", "output": {"summary_final": "Hi"}}\n')
            f.write('{"episode_id": "e2", "output": {"summary_final": "Bye"}}\n')
            path = Path(f.name)
        try:
            preds = load_predictions(path)
            assert len(preds) == 2
            assert preds[0]["episode_id"] == "e1"
            assert preds[1]["episode_id"] == "e2"
        finally:
            path.unlink(missing_ok=True)

    def test_load_empty_file_returns_empty_list(self):
        """Empty file returns empty list."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            path = Path(f.name)
        try:
            preds = load_predictions(path)
            assert preds == []
        finally:
            path.unlink(missing_ok=True)

    def test_load_skips_blank_lines(self):
        """Blank lines are skipped."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"id": "a"}\n')
            f.write("\n")
            f.write('{"id": "b"}\n')
            path = Path(f.name)
        try:
            preds = load_predictions(path)
            assert len(preds) == 2
        finally:
            path.unlink(missing_ok=True)


@pytest.mark.unit
class TestEstimateTokens:
    """Tests for estimate_tokens."""

    def test_estimate_tokens_rough_four_chars(self):
        """Token count is roughly len/4."""
        assert estimate_tokens("") == 0
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 8) == 2
        assert estimate_tokens("hello world") == 2  # 11 // 4


@pytest.mark.unit
class TestComputeRougeBleuWerVsReference:
    """Extrinsic metrics vs reference (optional deps)."""

    def test_compute_rouge_vs_reference_basic(self):
        if scorer_mod.rouge_scorer is None:
            pytest.skip("rouge-score not installed")
        preds = [{"episode_id": "e1", "output": {"summary_final": "hello world"}}]
        refs = [{"episode_id": "e1", "output": {"summary_final": "hello there world"}}]
        out = compute_rouge_vs_reference(preds, refs)
        assert out["rouge1_f1"] is not None
        assert out["rouge2_f1"] is not None
        assert out["rougeL_f1"] is not None

    def test_compute_rouge_no_matching_episode_returns_nulls(self):
        if scorer_mod.rouge_scorer is None:
            pytest.skip("rouge-score not installed")
        preds = [{"episode_id": "a", "output": {"summary_final": "x"}}]
        refs = [{"episode_id": "b", "output": {"summary_final": "y"}}]
        out = compute_rouge_vs_reference(preds, refs)
        assert out["rouge1_f1"] is None
        assert out["rouge2_f1"] is None
        assert out["rougeL_f1"] is None

    def test_compute_bleu_vs_reference_basic(self):
        if scorer_mod.sentence_bleu is None or scorer_mod.word_tokenize is None:
            pytest.skip("nltk not installed")
        preds = [{"episode_id": "e1", "output": {"summary_final": "the cat"}}]
        refs = [{"episode_id": "e1", "output": {"summary_final": "the cat sat"}}]
        bleu = compute_bleu_vs_reference(preds, refs)
        assert bleu is not None
        assert 0.0 <= bleu <= 1.0

    def test_compute_wer_vs_reference_basic(self):
        if scorer_mod.jiwer is None:
            pytest.skip("jiwer not installed")
        preds = [{"episode_id": "e1", "output": {"summary_final": "hello world"}}]
        refs = [{"episode_id": "e1", "output": {"summary_final": "hello world"}}]
        wer = compute_wer_vs_reference(preds, refs)
        assert wer is not None
        assert 0.0 <= wer <= 1.0


@pytest.mark.unit
class TestComputeIntrinsicMetrics:
    """Tests for compute_intrinsic_metrics."""

    def test_empty_predictions_returns_structure(self):
        """Empty predictions returns metrics structure with gates and length."""
        out = compute_intrinsic_metrics([], "ds1", "run1")
        assert "gates" in out
        assert "length" in out
        assert "performance" in out
        assert out["length"]["avg_tokens"] == 0

    def test_minimal_predictions_with_summary(self):
        """Predictions with summary_final produce gates and length stats."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "A short summary."},
            },
        ]
        out = compute_intrinsic_metrics(preds, "ds1", "run1")
        assert "gates" in out
        assert "length" in out
        assert out["length"]["avg_tokens"] >= 0
        assert out["length"]["min_tokens"] >= 0

    def test_boilerplate_gate(self):
        """Boilerplate phrase in summary triggers gate."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "Please subscribe to our newsletter for more."},
            },
        ]
        out = compute_intrinsic_metrics(preds, "ds", "run")
        assert out["gates"]["boilerplate_leak_rate"] > 0
        assert "e1" in out["gates"]["failed_episodes"]

    def test_speaker_label_gate(self):
        """Speaker label pattern triggers FAIL gate."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "Host: welcome to the podcast."},
            },
        ]
        out = compute_intrinsic_metrics(preds, "ds", "run")
        assert out["gates"]["speaker_label_leak_rate"] > 0

    def test_truncation_end_marker(self):
        """Summary ending with ellipsis counts as truncation."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "We discussed many topics..."},
            },
        ]
        out = compute_intrinsic_metrics(preds, "ds", "run")
        assert out["gates"]["truncation_rate"] > 0

    def test_expectations_allow_sponsor_skips_boilerplate_gate(self):
        """allow_sponsor_content disables boilerplate detection."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "subscribe to our newsletter"},
            },
        ]
        meta = {"e1": {"expectations": {"allow_sponsor_content": True}}}
        out = compute_intrinsic_metrics(preds, "ds", "run", metadata_map=meta)
        assert out["gates"]["boilerplate_leak_rate"] == 0.0

    def test_speaker_name_warn_with_metadata(self):
        """Speaker real name in summary when not allowed → warning rate."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "Alice Johnson: hello there."},
            },
        ]
        meta = {
            "e1": {
                "speakers": [{"name": "Alice Johnson"}],
                "expectations": {"allow_speaker_names": False},
            },
        }
        out = compute_intrinsic_metrics(preds, "ds", "run", metadata_map=meta)
        assert out["warnings"]["speaker_name_leak_rate"] > 0

    def test_performance_latency_ms(self):
        """processing_time_seconds in metadata becomes avg_latency_ms."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "Hi"},
                "metadata": {"processing_time_seconds": 2.0},
            },
        ]
        out = compute_intrinsic_metrics(preds, "ds", "run")
        assert out["performance"]["avg_latency_ms"] == 2000.0

    def test_performance_latency_percentiles_and_steady_state(self):
        """Multiple episodes get median, p95, and avg excluding first."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "A"},
                "metadata": {"processing_time_seconds": 10.0},
            },
            {
                "episode_id": "e2",
                "output": {"summary_final": "B"},
                "metadata": {"processing_time_seconds": 2.0},
            },
            {
                "episode_id": "e3",
                "output": {"summary_final": "C"},
                "metadata": {"processing_time_seconds": 4.0},
            },
            {
                "episode_id": "e4",
                "output": {"summary_final": "D"},
                "metadata": {"processing_time_seconds": 6.0},
            },
        ]
        out = compute_intrinsic_metrics(preds, "ds", "run")
        perf = out["performance"]
        assert perf["avg_latency_ms"] == 5500.0
        assert perf["median_latency_ms"] == 5000.0
        assert perf["p95_latency_ms"] == 10000.0
        assert perf["avg_latency_ms_excluding_first"] == pytest.approx(4000.0)

    def test_cost_from_metadata_cost_usd(self):
        """Direct cost_usd in metadata populates cost section."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "Hi"},
                "metadata": {"cost_usd": 0.42},
            },
        ]
        out = compute_intrinsic_metrics(preds, "ds", "run")
        assert "cost" in out
        assert out["cost"]["total_cost_usd"] == 0.42
        assert out["cost"]["avg_cost_usd"] == 0.42

    def test_cost_from_usage_gpt4o_mini(self):
        """usage + model gpt-4o-mini computes approximate cost."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "Hi"},
                "metadata": {
                    "model": "gpt-4o-mini",
                    "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
                },
            },
        ]
        out = compute_intrinsic_metrics(preds, "ds", "run")
        assert "cost" in out
        # 1M in @ $0.15 + 1M out @ $0.60
        assert abs(out["cost"]["total_cost_usd"] - 0.75) < 1e-9

    def test_cost_from_usage_gpt4o_full(self):
        """usage + model gpt-4o (not mini) uses full GPT-4o pricing."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "Hi"},
                "metadata": {
                    "model": "gpt-4o",
                    "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
                },
            },
        ]
        out = compute_intrinsic_metrics(preds, "ds", "run")
        assert "cost" in out
        assert abs(out["cost"]["total_cost_usd"] - 12.50) < 1e-6

    def test_cost_skips_when_usage_tokens_zero(self):
        """No cost section when usage has zero tokens and no cost_usd."""
        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "Hi"},
                "metadata": {
                    "model": "gpt-4o-mini",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
            },
        ]
        out = compute_intrinsic_metrics(preds, "ds", "run")
        assert "cost" not in out


@pytest.mark.unit
class TestComputeVsReferencePartialEpisodes:
    """Subset predictions vs full reference (max_episodes / autoresearch)."""

    def test_allows_prediction_subset_of_reference(self, tmp_path: Path) -> None:
        """When predictions cover a subset of reference episodes, score intersection only."""
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        ref_jsonl = ref_dir / "predictions.jsonl"
        ref_jsonl.write_text(
            '{"episode_id": "e1", "output": {"summary_final": "alpha beta"}}\n'
            '{"episode_id": "e2", "output": {"summary_final": "gamma delta"}}\n',
            encoding="utf-8",
        )
        (ref_dir / "baseline.json").write_text(
            '{"reference_quality": "fixture"}',
            encoding="utf-8",
        )

        preds = [
            {"episode_id": "e1", "output": {"summary_final": "alpha beta"}},
        ]
        out = compute_vs_reference_metrics(preds, "fixture_ref", ref_dir)
        assert "error" not in out
        assert out.get("rougeL_f1") is not None

    def test_rejects_extra_prediction_episodes(self, tmp_path: Path) -> None:
        """Predictions must not contain episode_ids absent from reference."""
        ref_dir = tmp_path / "ref2"
        ref_dir.mkdir()
        (ref_dir / "predictions.jsonl").write_text(
            '{"episode_id": "e1", "output": {"summary_final": "a"}}\n',
            encoding="utf-8",
        )
        (ref_dir / "baseline.json").write_text("{}", encoding="utf-8")

        preds = [
            {"episode_id": "e1", "output": {"summary_final": "a"}},
            {"episode_id": "e99", "output": {"summary_final": "orphan"}},
        ]
        with pytest.raises(ValueError, match="extra"):
            compute_vs_reference_metrics(preds, "fixture_ref2", ref_dir)

    def test_numbers_retained_metric(self, tmp_path: Path) -> None:
        """Reference numbers overlapping prediction appear in numbers_retained."""
        ref_dir = tmp_path / "refn"
        ref_dir.mkdir()
        ref_line = (
            '{"episode_id": "e1", "output": {"summary_final": '
            '"Revenue was 42 million in 2024."}}\n'
        )
        (ref_dir / "predictions.jsonl").write_text(ref_line, encoding="utf-8")
        (ref_dir / "baseline.json").write_text("{}", encoding="utf-8")

        preds = [
            {
                "episode_id": "e1",
                "output": {"summary_final": "We noted 42 and 2024 in the discussion."},
            },
        ]
        out = compute_vs_reference_metrics(preds, "refnums", ref_dir)
        assert out.get("numbers_retained") is not None
        assert 0.0 <= out["numbers_retained"] <= 1.0


@pytest.mark.unit
class TestComputeVsReferenceOptionalMetricsBranches:
    """Cover ImportError branches and per-episode exception logging in scorer."""

    @staticmethod
    def _ref_and_preds(tmp_path: Path) -> tuple[Path, list]:
        ref_dir = tmp_path / "ref_opt"
        ref_dir.mkdir()
        line = '{"episode_id": "e1", "output": {"summary_final": "hello world there"}}\n'
        (ref_dir / "predictions.jsonl").write_text(line, encoding="utf-8")
        (ref_dir / "baseline.json").write_text("{}", encoding="utf-8")
        preds = [{"episode_id": "e1", "output": {"summary_final": "hello world there"}}]
        return ref_dir, preds

    def test_rouge_skipped_on_import_error(self, tmp_path: Path, caplog) -> None:
        caplog.set_level(logging.WARNING)
        ref_dir, preds = self._ref_and_preds(tmp_path)
        with patch.object(
            scorer_mod,
            "compute_rouge_vs_reference",
            side_effect=ImportError("rouge missing"),
        ):
            out = compute_vs_reference_metrics(preds, "r1", ref_dir)
        assert out["rouge1_f1"] is None
        assert "ROUGE computation skipped" in caplog.text

    def test_bleu_skipped_on_import_error(self, tmp_path: Path, caplog) -> None:
        caplog.set_level(logging.WARNING)
        ref_dir, preds = self._ref_and_preds(tmp_path)
        with patch.object(
            scorer_mod,
            "compute_bleu_vs_reference",
            side_effect=ImportError("nltk missing"),
        ):
            out = compute_vs_reference_metrics(preds, "r1", ref_dir)
        assert out["bleu"] is None
        assert "BLEU computation skipped" in caplog.text

    def test_wer_skipped_on_import_error(self, tmp_path: Path, caplog) -> None:
        caplog.set_level(logging.WARNING)
        ref_dir, preds = self._ref_and_preds(tmp_path)
        with patch.object(
            scorer_mod,
            "compute_wer_vs_reference",
            side_effect=ImportError("jiwer missing"),
        ):
            out = compute_vs_reference_metrics(preds, "r1", ref_dir)
        assert out["wer"] is None
        assert "WER computation skipped" in caplog.text

    def test_embedding_skipped_on_import_error(self, tmp_path: Path, caplog) -> None:
        caplog.set_level(logging.WARNING)
        ref_dir, preds = self._ref_and_preds(tmp_path)
        with patch.object(
            scorer_mod,
            "compute_embedding_similarity",
            side_effect=ImportError("st missing"),
        ):
            out = compute_vs_reference_metrics(preds, "r1", ref_dir)
        assert out["embedding_cosine"] is None
        assert "Embedding similarity computation skipped" in caplog.text

    def test_bleu_episode_error_logs_safe_message(self, tmp_path: Path, caplog) -> None:
        if scorer_mod.sentence_bleu is None or scorer_mod.word_tokenize is None:
            pytest.skip("nltk not installed")
        caplog.set_level(logging.WARNING)
        ref_dir, preds = self._ref_and_preds(tmp_path)
        refs = [{"episode_id": "e1", "output": {"summary_final": "hello world there"}}]
        with patch.object(scorer_mod, "word_tokenize", side_effect=RuntimeError("token boom")):
            compute_bleu_vs_reference(preds, refs)
        assert "Error computing BLEU" in caplog.text
        assert "e1" in caplog.text

    def test_wer_episode_error_logs_safe_message(self, tmp_path: Path, caplog) -> None:
        if scorer_mod.jiwer is None:
            pytest.skip("jiwer not installed")
        caplog.set_level(logging.WARNING)
        ref_dir, preds = self._ref_and_preds(tmp_path)
        refs = [{"episode_id": "e1", "output": {"summary_final": "hello world there"}}]
        with patch.object(scorer_mod.jiwer, "wer", side_effect=RuntimeError("wer boom")):
            compute_wer_vs_reference(preds, refs)
        assert "Error computing WER" in caplog.text

    def test_embedding_model_load_failure_returns_none(self, caplog) -> None:
        if scorer_mod._SentenceTransformer is None:
            pytest.skip("sentence-transformers not installed")
        caplog.set_level(logging.ERROR)
        preds = [{"episode_id": "e1", "output": {"summary_final": "a"}}]
        refs = [{"episode_id": "e1", "output": {"summary_final": "b"}}]
        with patch.object(
            scorer_mod,
            "_SentenceTransformer",
            side_effect=RuntimeError("cannot load model"),
        ):
            out = compute_embedding_similarity(preds, refs)
        assert out is None
        assert "Failed to load sentence-transformer" in caplog.text


@pytest.mark.unit
class TestScoreRun:
    """score_run top-level orchestration."""

    def test_vs_reference_error_surfaces_under_ref_key(self, tmp_path: Path) -> None:
        """Broken reference path records error string instead of crashing score_run."""
        pred_path = tmp_path / "predictions.jsonl"
        pred_path.write_text(
            '{"episode_id": "e1", "output": {"summary_final": "hello world there"}}\n',
            encoding="utf-8",
        )
        missing_ref = tmp_path / "no_such_ref_dir"
        out = score_run(
            pred_path,
            "ds_x",
            "run_y",
            reference_paths={"broken": missing_ref},
        )
        assert out["vs_reference"] is not None
        assert "error" in out["vs_reference"]["broken"]
        assert "predictions" in out["vs_reference"]["broken"]["error"].lower()

    def test_vs_reference_compute_exception_logs_and_records_error(
        self, tmp_path: Path, caplog
    ) -> None:
        """score_run logs format_exception_for_log when vs-reference compute raises."""
        caplog.set_level(logging.ERROR)
        ref_dir = tmp_path / "good_ref"
        ref_dir.mkdir()
        (ref_dir / "predictions.jsonl").write_text(
            '{"episode_id": "e1", "output": {"summary_final": "x"}}\n',
            encoding="utf-8",
        )
        pred_path = tmp_path / "predictions.jsonl"
        pred_path.write_text(
            '{"episode_id": "e1", "output": {"summary_final": "x"}}\n',
            encoding="utf-8",
        )
        with patch.object(
            scorer_mod,
            "compute_vs_reference_metrics",
            side_effect=RuntimeError("synthetic ref failure"),
        ):
            out = score_run(
                pred_path,
                "ds_z",
                "run_z",
                reference_paths={"ref_a": ref_dir},
            )
        assert "error" in out["vs_reference"]["ref_a"]
        assert "Failed to compute metrics vs reference" in caplog.text
