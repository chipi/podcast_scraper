"""Unit tests for podcast_scraper.evaluation.scorer module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from podcast_scraper.evaluation.scorer import (
    compute_intrinsic_metrics,
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
