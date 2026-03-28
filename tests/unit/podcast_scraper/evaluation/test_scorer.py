"""Unit tests for podcast_scraper.evaluation.scorer module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from podcast_scraper.evaluation import scorer as scorer_mod
from podcast_scraper.evaluation.scorer import (
    compute_bleu_vs_reference,
    compute_intrinsic_metrics,
    compute_rouge_vs_reference,
    compute_wer_vs_reference,
    estimate_tokens,
    load_predictions,
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
