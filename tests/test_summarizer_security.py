#!/usr/bin/env python3
"""Security tests for summarization module.

Tests critical security functions:
- _validate_model_source() - Model source validation
- Revision pinning - Model revision pinning for security
- prune_cache() - Enhanced cache pruning security
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import summarizer, skip tests if dependencies not available
try:
    from podcast_scraper import summarizer

    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    summarizer = types.ModuleType("summarizer")  # type: ignore[assignment]


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestValidateModelSource(unittest.TestCase):
    """Test _validate_model_source() security function."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger_patcher = patch("podcast_scraper.summarizer.logger")
        self.mock_logger = self.logger_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.logger_patcher.stop()

    def test_trusted_source_facebook(self):
        """Test that facebook models are recognized as trusted."""
        summarizer._validate_model_source("facebook/bart-large-cnn")
        self.mock_logger.debug.assert_called_once_with("Loading model from verified trusted source")
        self.mock_logger.warning.assert_not_called()

    def test_trusted_source_google(self):
        """Test that google models are recognized as trusted."""
        summarizer._validate_model_source("google/pegasus-large")
        self.mock_logger.debug.assert_called_once_with("Loading model from verified trusted source")
        self.mock_logger.warning.assert_not_called()

    def test_trusted_source_sshleifer(self):
        """Test that sshleifer models are recognized as trusted."""
        summarizer._validate_model_source("sshleifer/distilbart-cnn-12-6")
        self.mock_logger.debug.assert_called_once_with("Loading model from verified trusted source")
        self.mock_logger.warning.assert_not_called()

    def test_trusted_source_allenai(self):
        """Test that allenai models are recognized as trusted."""
        summarizer._validate_model_source("allenai/led-large-16384")
        self.mock_logger.debug.assert_called_once_with("Loading model from verified trusted source")
        self.mock_logger.warning.assert_not_called()

    def test_untrusted_source_warns(self):
        """Test that untrusted sources trigger security warning."""
        summarizer._validate_model_source("malicious-user/suspicious-model")
        self.mock_logger.debug.assert_not_called()
        self.mock_logger.warning.assert_called_once()
        warning_message = str(self.mock_logger.warning.call_args[0][0])
        self.assertIn("SECURITY NOTICE", warning_message)
        self.assertIn("custom (untrusted) source", warning_message)

    def test_local_model_warns(self):
        """Test that local/non-standard model identifiers trigger warning."""
        summarizer._validate_model_source("local-model")
        self.mock_logger.debug.assert_not_called()
        self.mock_logger.warning.assert_called_once()
        warning_message = str(self.mock_logger.warning.call_args[0][0])
        self.assertIn("SECURITY NOTICE", warning_message)

    def test_no_sensitive_info_logged(self):
        """Test that model names are not logged in clear text."""
        model_name = "malicious-user/suspicious-model"
        summarizer._validate_model_source(model_name)
        # Check that model name is not in any log calls
        all_logs = []
        for call in self.mock_logger.debug.call_args_list:
            all_logs.extend(call[0])
        for call in self.mock_logger.warning.call_args_list:
            all_logs.extend(call[0])
        log_text = " ".join(str(msg) for msg in all_logs)
        # Model name should not appear in logs (security: avoid logging untrusted input)
        self.assertNotIn("malicious-user", log_text)
        self.assertNotIn("suspicious-model", log_text)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestRevisionPinning(unittest.TestCase):
    """Test revision pinning functionality for security and reproducibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.summarizer.AutoTokenizer")
    @patch("podcast_scraper.summarizer.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.pipeline")
    @patch("podcast_scraper.summarizer.torch")
    @patch("podcast_scraper.summarizer.logger")
    def test_revision_pinning_passed_to_tokenizer(
        self, mock_logger, mock_torch, mock_model_class, mock_tokenizer_class, mock_pipeline
    ):
        """Test that revision parameter is passed to tokenizer."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.to.return_value = mock_model  # Model.to() returns self
        mock_model_class.from_pretrained.return_value = mock_model

        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe

        revision = "abc123def456"
        summarizer.SummaryModel(
            model_name="facebook/bart-base",
            device="cpu",
            cache_dir=self.temp_dir,
            revision=revision,
        )

        # Verify revision was passed to tokenizer
        self.assertTrue(mock_tokenizer_class.from_pretrained.called)
        tokenizer_call = mock_tokenizer_class.from_pretrained.call_args
        self.assertIsNotNone(tokenizer_call)
        self.assertIn("revision", tokenizer_call.kwargs)
        self.assertEqual(tokenizer_call.kwargs["revision"], revision)

        # Verify debug log mentions revision
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        revision_logged = any(
            "pinned revision" in str(call).lower() or revision in str(call) for call in debug_calls
        )
        self.assertTrue(revision_logged, "Revision should be logged")

    @patch("podcast_scraper.summarizer.AutoTokenizer")
    @patch("podcast_scraper.summarizer.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.pipeline")
    @patch("podcast_scraper.summarizer.torch")
    def test_revision_pinning_passed_to_model(
        self, mock_torch, mock_pipeline, mock_model_class, mock_tokenizer_class
    ):
        """Test that revision parameter is passed to model."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.to.return_value = mock_model  # Model.to() returns self
        mock_model_class.from_pretrained.return_value = mock_model

        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe

        revision = "main"
        summarizer.SummaryModel(
            model_name="facebook/bart-base",
            device="cpu",
            cache_dir=self.temp_dir,
            revision=revision,
        )

        # Verify revision was passed to model
        self.assertTrue(mock_model_class.from_pretrained.called)
        model_call = mock_model_class.from_pretrained.call_args
        self.assertIsNotNone(model_call)
        self.assertIn("revision", model_call.kwargs)
        self.assertEqual(model_call.kwargs["revision"], revision)

    @patch("podcast_scraper.summarizer.AutoTokenizer")
    @patch("podcast_scraper.summarizer.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.pipeline")
    @patch("podcast_scraper.summarizer.torch")
    def test_no_revision_when_none(
        self, mock_torch, mock_pipeline, mock_model_class, mock_tokenizer_class
    ):
        """Test that revision is not passed when None."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.to.return_value = mock_model  # Model.to() returns self
        mock_model_class.from_pretrained.return_value = mock_model

        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe

        summarizer.SummaryModel(
            model_name="facebook/bart-base",
            device="cpu",
            cache_dir=self.temp_dir,
            revision=None,
        )

        # Verify revision is not in tokenizer kwargs
        self.assertTrue(mock_tokenizer_class.from_pretrained.called)
        tokenizer_call = mock_tokenizer_class.from_pretrained.call_args
        self.assertIsNotNone(tokenizer_call)
        self.assertNotIn("revision", tokenizer_call.kwargs)

        # Verify revision is not in model kwargs
        self.assertTrue(mock_model_class.from_pretrained.called)
        model_call = mock_model_class.from_pretrained.call_args
        self.assertIsNotNone(model_call)
        self.assertNotIn("revision", model_call.kwargs)

    @patch("podcast_scraper.summarizer.AutoTokenizer")
    @patch("podcast_scraper.summarizer.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.pipeline")
    @patch("podcast_scraper.summarizer.torch")
    def test_revision_stored_in_model(
        self, mock_torch, mock_model_class, mock_tokenizer_class, mock_pipeline
    ):
        """Test that revision is stored in model instance."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe

        revision = "abc123def456"
        model = summarizer.SummaryModel(
            model_name="facebook/bart-base",
            device="cpu",
            cache_dir=self.temp_dir,
            revision=revision,
        )

        self.assertEqual(model.revision, revision)

    @patch("podcast_scraper.summarizer.AutoTokenizer")
    @patch("podcast_scraper.summarizer.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.pipeline")
    @patch("podcast_scraper.summarizer.torch")
    def test_revision_none_stored_in_model(
        self, mock_torch, mock_model_class, mock_tokenizer_class, mock_pipeline
    ):
        """Test that None revision is stored in model instance."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe

        model = summarizer.SummaryModel(
            model_name="facebook/bart-base",
            device="cpu",
            cache_dir=self.temp_dir,
            revision=None,
        )

        self.assertIsNone(model.revision)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestPruneCacheSecurity(unittest.TestCase):
    """Test prune_cache() security checks."""

    def setUp(self):
        """Set up test fixtures."""
        self.home = Path.home()
        # Use a subdirectory of home for testing (required for security checks)
        self.temp_dir = tempfile.mkdtemp(dir=str(self.home))

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_allows_subdirectory_of_home(self):
        """Test that subdirectories of home are allowed."""
        cache_path = self.home / "test_cache"
        cache_path.mkdir(exist_ok=True)
        try:
            # Should not raise ValueError
            result = summarizer.prune_cache(cache_dir=str(cache_path), dry_run=True)
            self.assertIsInstance(result, int)
        finally:
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)

    def test_allows_subdirectory_of_cache(self):
        """Test that subdirectories of ~/.cache are allowed."""
        cache_root = self.home / ".cache"
        cache_path = cache_root / "huggingface" / "hub"
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            # Should not raise ValueError
            result = summarizer.prune_cache(cache_dir=str(cache_path), dry_run=True)
            self.assertIsInstance(result, int)
        finally:
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)

    def test_prevents_deletion_of_home(self):
        """Test that home directory itself cannot be deleted."""
        with self.assertRaises(ValueError) as context:
            summarizer.prune_cache(cache_dir=str(self.home), dry_run=True)
        self.assertIn("protected root directory", str(context.exception).lower())
        self.assertIn("security", str(context.exception).lower())

    def test_prevents_deletion_of_cache_root(self):
        """Test that ~/.cache itself cannot be deleted."""
        cache_root = self.home / ".cache"
        with self.assertRaises(ValueError) as context:
            summarizer.prune_cache(cache_dir=str(cache_root), dry_run=True)
        self.assertIn("protected root directory", str(context.exception).lower())
        self.assertIn("security", str(context.exception).lower())

    def test_prevents_deletion_outside_home(self):
        """Test that paths outside home directory are rejected."""
        # Use a path that's definitely outside home
        outside_path = Path("/tmp") / "test_cache"
        with self.assertRaises(ValueError) as context:
            summarizer.prune_cache(cache_dir=str(outside_path), dry_run=True)
        self.assertIn("outside safe locations", str(context.exception).lower())
        self.assertIn("security", str(context.exception).lower())

    def test_prevents_deletion_of_root(self):
        """Test that root directory cannot be deleted."""
        with self.assertRaises(ValueError) as context:
            summarizer.prune_cache(cache_dir="/", dry_run=True)
        self.assertIn("outside safe locations", str(context.exception).lower())
        self.assertIn("security", str(context.exception).lower())

    def test_prevents_deletion_with_path_traversal(self):
        """Test that path traversal attacks are prevented."""
        # Try to use .. to escape to home
        malicious_path = str(self.home / "test" / ".." / ".cache")
        # This should resolve to ~/.cache which should be rejected
        with self.assertRaises(ValueError) as context:
            summarizer.prune_cache(cache_dir=malicious_path, dry_run=True)
        self.assertIn("protected root directory", str(context.exception).lower())

    def test_invalid_path_raises_error(self):
        """Test that invalid paths raise appropriate error."""
        with self.assertRaises(ValueError) as context:
            summarizer.prune_cache(
                cache_dir="/nonexistent/path/that/does/not/exist/../../../etc", dry_run=True
            )
        # Should raise ValueError for invalid path
        self.assertIsInstance(context.exception, ValueError)

    def test_nonexistent_cache_dir_returns_zero(self):
        """Test that nonexistent cache directory returns 0."""
        nonexistent = self.home / "nonexistent_cache_dir_12345"
        result = summarizer.prune_cache(cache_dir=str(nonexistent), dry_run=True)
        self.assertEqual(result, 0)

    def test_dry_run_does_not_delete(self):
        """Test that dry_run=True does not actually delete files."""
        # Use a subdirectory of home (required for security checks)
        cache_path = self.home / "test_cache_security"
        cache_path.mkdir(parents=True, exist_ok=True)
        test_file = cache_path / "test_file.txt"
        test_file.write_text("test content")
        try:
            # Dry run should not delete
            result = summarizer.prune_cache(cache_dir=str(cache_path), dry_run=True)
            self.assertGreater(result, 0)
            self.assertTrue(test_file.exists(), "File should still exist after dry run")
        finally:
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)

    def test_non_dry_run_deletes_files(self):
        """Test that dry_run=False actually deletes files."""
        # Use a subdirectory of home (required for security checks)
        cache_path = self.home / "test_cache_security_delete"
        cache_path.mkdir(parents=True, exist_ok=True)
        test_file = cache_path / "test_file.txt"
        test_file.write_text("test content")
        try:
            # Non-dry run should delete
            result = summarizer.prune_cache(cache_dir=str(cache_path), dry_run=False)
            self.assertGreater(result, 0)
            self.assertFalse(test_file.exists(), "File should be deleted")
        finally:
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)

    def test_returns_deleted_count(self):
        """Test that function returns correct count of deleted files."""
        # Use a subdirectory of home (required for security checks)
        cache_path = self.home / "test_cache_security_count"
        cache_path.mkdir(parents=True, exist_ok=True)
        # Create multiple files
        for i in range(5):
            (cache_path / f"test_file_{i}.txt").write_text(f"content {i}")
        try:
            result = summarizer.prune_cache(cache_dir=str(cache_path), dry_run=True)
            self.assertEqual(result, 5)
        finally:
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
