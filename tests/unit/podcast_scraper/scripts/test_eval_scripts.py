#!/usr/bin/env python3
"""Tests for evaluation scripts (eval_cleaning.py and eval_summaries.py).

These tests validate core functionality of the evaluation scripts without
requiring actual evaluation data or ML models.
"""

# Import eval modules directly using importlib (avoiding 'eval' builtin conflict)
import importlib.util

# Import eval script functions
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Calculate project root: tests/unit/podcast_scraper/scripts -> 5 levels up
project_root = Path(__file__).parent.parent.parent.parent.parent
eval_cleaning_path = project_root / "scripts" / "eval" / "eval_cleaning.py"
eval_summaries_path = project_root / "scripts" / "eval" / "eval_summaries.py"

spec_cleaning = importlib.util.spec_from_file_location("eval_cleaning", eval_cleaning_path)
eval_cleaning = importlib.util.module_from_spec(spec_cleaning)  # type: ignore
spec_cleaning.loader.exec_module(eval_cleaning)  # type: ignore
# Register in sys.modules so @patch can find it
sys.modules["eval_cleaning"] = eval_cleaning

spec_summaries = importlib.util.spec_from_file_location("eval_summaries", eval_summaries_path)
eval_summaries = importlib.util.module_from_spec(spec_summaries)  # type: ignore
spec_summaries.loader.exec_module(eval_summaries)  # type: ignore
# Register in sys.modules so @patch can find it
sys.modules["eval_summaries"] = eval_summaries


@pytest.mark.unit
class TestEvalCleaning(unittest.TestCase):
    """Tests for eval_cleaning.py"""

    def test_load_text_success(self):
        """Test loading text from file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test content")
            temp_path = Path(f.name)

        try:
            result = eval_cleaning.load_text(temp_path)
            self.assertEqual(result, "Test content")
        finally:
            temp_path.unlink()

    def test_load_text_missing_file(self):
        """Test loading from non-existent file."""
        missing_path = Path("/nonexistent/path/file.txt")
        result = eval_cleaning.load_text(missing_path)
        self.assertEqual(result, "")

    def test_count_patterns(self):
        """Test pattern counting."""
        text = "This episode is brought to you by our sponsor. Thanks to our sponsor."
        patterns = [r"brought to you by", r"thanks to"]
        counts = eval_cleaning.count_patterns(text, patterns)
        self.assertEqual(counts[r"brought to you by"], 1)
        self.assertEqual(counts[r"thanks to"], 1)

    def test_count_patterns_case_insensitive(self):
        """Test pattern counting is case-insensitive by default."""
        text = "This Episode Is Brought To You By our sponsor."
        patterns = [r"brought to you by"]
        counts = eval_cleaning.count_patterns(text, patterns, case_sensitive=False)
        self.assertEqual(counts[r"brought to you by"], 1)

    def test_count_brand_mentions(self):
        """Test brand mention counting."""
        text = "Check out Figma and Stripe. Also try Figma again."
        brands = ["figma", "stripe"]
        counts = eval_cleaning.count_brand_mentions(text, brands)
        self.assertEqual(counts["figma"], 2)
        self.assertEqual(counts["stripe"], 1)

    def test_compute_removal_stats(self):
        """Test removal statistics computation."""
        raw = "This is a test transcript with some content."
        cleaned = "This is a test transcript."
        stats = eval_cleaning.compute_removal_stats(raw, cleaned)
        self.assertEqual(stats["raw_chars"], len(raw))
        self.assertEqual(stats["cleaned_chars"], len(cleaned))
        self.assertGreater(stats["removed_chars"], 0)
        self.assertGreater(stats["removal_char_pct"], 0)
        self.assertEqual(stats["raw_words"], 8)
        self.assertEqual(stats["cleaned_words"], 5)
        self.assertEqual(stats["removed_words"], 3)

    def test_compute_removal_stats_empty(self):
        """Test removal stats with empty input."""
        stats = eval_cleaning.compute_removal_stats("", "")
        self.assertEqual(stats["removal_char_pct"], 0.0)
        self.assertEqual(stats["removal_word_pct"], 0.0)

    def test_get_diff_snippet(self):
        """Test diff snippet generation."""
        raw = "Line 1\nLine 2\nLine 3"
        cleaned = "Line 1\nLine 3"
        diff = eval_cleaning.get_diff_snippet(raw, cleaned, max_lines=5)
        self.assertGreater(len(diff), 0)
        self.assertIsInstance(diff, list)

    def test_evaluate_cleaning_missing_files(self):
        """Test evaluation with missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "ep01"
            episode_dir.mkdir()

            result = eval_cleaning.evaluate_cleaning(episode_dir)
            self.assertIn("error", result)
            self.assertEqual(result["episode_id"], "ep01")

    def test_evaluate_cleaning_success(self):
        """Test successful evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "ep01"
            episode_dir.mkdir()

            # Create test files
            raw_path = episode_dir / "transcript.raw.txt"
            cleaned_path = episode_dir / "transcript.cleaned.txt"

            raw_path.write_text("This episode is brought to you by our sponsor. Main content here.")
            cleaned_path.write_text("Main content here.")

            result = eval_cleaning.evaluate_cleaning(episode_dir)
            self.assertNotIn("error", result)
            self.assertEqual(result["episode_id"], "ep01")
            self.assertIn("removal_stats", result)
            self.assertIn("sponsor_patterns", result)
            self.assertIn("flags", result)


@pytest.mark.unit
class TestEvalSummaries(unittest.TestCase):
    """Tests for eval_summaries.py"""

    def test_load_text_success(self):
        """Test loading text from file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test content")
            temp_path = Path(f.name)

        try:
            result = eval_summaries.load_text(temp_path)
            self.assertEqual(result, "Test content")
        finally:
            temp_path.unlink()

    def test_load_text_missing_file(self):
        """Test loading from non-existent file."""
        missing_path = Path("/nonexistent/path/file.txt")
        result = eval_summaries.load_text(missing_path)
        self.assertEqual(result, "")

    def test_compute_compression_ratio(self):
        """Test compression ratio computation."""
        ratio = eval_summaries.compute_compression_ratio(1000, 100)
        self.assertEqual(ratio, 10.0)

    def test_compute_compression_ratio_zero_summary(self):
        """Test compression ratio with zero-length summary."""
        ratio = eval_summaries.compute_compression_ratio(1000, 0)
        self.assertEqual(ratio, float("inf"))

    def test_check_repetition_no_repetition(self):
        """Test repetition check with no repetition."""
        text = "This is a unique sentence. Another unique sentence here."
        is_repetitive, repeated = eval_summaries.check_repetition(text)
        self.assertFalse(is_repetitive)
        self.assertEqual(len(repeated), 0)

    def test_check_repetition_with_repetition(self):
        """Test repetition check with repetition."""
        # Create text with repeated 3-grams
        text = " ".join(["same words here"] * 10)
        is_repetitive, repeated = eval_summaries.check_repetition(text, ngram_size=3, threshold=5)
        # Should detect repetition
        self.assertTrue(is_repetitive or len(repeated) > 0)

    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "Python programming language is great. Python is versatile. Programming is fun."
        keywords = eval_summaries.extract_keywords(text, top_n=5)
        self.assertLessEqual(len(keywords), 5)
        self.assertTrue("python" in keywords or "programming" in keywords)

    def test_compute_keyword_coverage(self):
        """Test keyword coverage computation."""
        transcript = "Machine learning artificial intelligence neural networks"
        summary = "Machine learning and neural networks"
        coverage, covered, missing = eval_summaries.compute_keyword_coverage(transcript, summary)
        self.assertGreaterEqual(coverage, 0.0)
        self.assertLessEqual(coverage, 1.0)
        self.assertGreaterEqual(len(covered), 0)
        self.assertGreaterEqual(len(missing), 0)

    @patch("eval_summaries.rouge_scorer")
    def test_compute_rouge_scores(self, mock_rouge):
        """Test ROUGE score computation."""
        # Mock rouge_scorer
        mock_scorer_instance = Mock()
        mock_score_result = {
            "rouge1": Mock(precision=0.5, recall=0.4, fmeasure=0.45),
            "rouge2": Mock(precision=0.3, recall=0.2, fmeasure=0.25),
            "rougeL": Mock(precision=0.4, recall=0.3, fmeasure=0.35),
        }
        mock_scorer_instance.score.return_value = mock_score_result
        mock_rouge.RougeScorer.return_value = mock_scorer_instance

        prediction = "This is a summary."
        reference = "This is the reference summary."
        scores = eval_summaries.compute_rouge_scores(prediction, reference)

        self.assertIn("rouge1", scores)
        self.assertIn("rouge2", scores)
        self.assertIn("rougeL", scores)
        self.assertEqual(scores["rouge1"]["fmeasure"], 0.45)

    def test_evaluate_episode_missing_transcript(self):
        """Test evaluation with missing transcript."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "ep01"
            episode_dir.mkdir()

            # Mock model and config
            mock_model = Mock()
            mock_cfg = Mock()
            mock_cfg.summary_chunk_size = None
            mock_cfg.summary_max_length = 160
            mock_cfg.summary_min_length = 60
            mock_cfg.summary_prompt = None
            mock_cfg.generate_metadata = True
            mock_cfg.generate_summaries = True

            result = eval_summaries.evaluate_episode(episode_dir, mock_model, None, mock_cfg, False)
            self.assertIn("error", result)
            self.assertEqual(result["episode_id"], "ep01")

    @patch("eval_summaries.summarizer.summarize_long_text")
    def test_evaluate_episode_success(self, mock_summarize):
        """Test successful evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "ep01"
            episode_dir.mkdir()

            # Create test files
            transcript_path = episode_dir / "transcript.cleaned.txt"
            reference_path = episode_dir / "summary.gold.long.txt"

            transcript_path.write_text("This is a test transcript with some content.")
            reference_path.write_text("This is a reference summary.")

            # Mock summarization
            mock_summarize.return_value = "This is a generated summary."

            # Mock model and config
            mock_model = Mock()
            mock_cfg = Mock()
            mock_cfg.summary_chunk_size = None
            mock_cfg.summary_max_length = 160
            mock_cfg.summary_min_length = 60
            mock_cfg.summary_prompt = None
            mock_cfg.generate_metadata = True
            mock_cfg.generate_summaries = True

            result = eval_summaries.evaluate_episode(episode_dir, mock_model, None, mock_cfg, False)

            self.assertNotIn("error", result)
            self.assertEqual(result["episode_id"], "ep01")
            self.assertIn("compression_ratio", result)
            self.assertIn("rouge", result)
            self.assertIn("checks", result)


@pytest.mark.unit
class TestEvalScriptsIntegration(unittest.TestCase):
    """Integration tests for eval scripts (require minimal setup)."""

    def test_eval_cleaning_imports(self):
        """Test that eval_cleaning imports correctly."""
        self.assertTrue(hasattr(eval_cleaning, "load_text"))
        self.assertTrue(hasattr(eval_cleaning, "evaluate_cleaning"))
        self.assertTrue(hasattr(eval_cleaning, "main"))

    def test_eval_summaries_imports(self):
        """Test that eval_summaries imports correctly."""
        self.assertTrue(hasattr(eval_summaries, "load_text"))
        self.assertTrue(hasattr(eval_summaries, "evaluate_episode"))
        self.assertTrue(hasattr(eval_summaries, "main"))

    def test_eval_cleaning_constants(self):
        """Test that eval_cleaning constants are defined."""
        self.assertTrue(hasattr(eval_cleaning, "SPONSOR_PATTERNS"))
        self.assertTrue(hasattr(eval_cleaning, "BRAND_NAMES"))
        self.assertTrue(hasattr(eval_cleaning, "OUTRO_PATTERNS"))
        self.assertIsInstance(eval_cleaning.SPONSOR_PATTERNS, list)
        self.assertGreater(len(eval_cleaning.SPONSOR_PATTERNS), 0)

    def test_eval_summaries_constants(self):
        """Test that eval_summaries constants are defined."""
        self.assertTrue(hasattr(eval_summaries, "MIN_COMPRESSION_RATIO"))
        self.assertTrue(hasattr(eval_summaries, "MAX_COMPRESSION_RATIO"))
        self.assertTrue(hasattr(eval_summaries, "REPETITION_NGRAM_SIZE"))
        self.assertIsInstance(eval_summaries.MIN_COMPRESSION_RATIO, float)
