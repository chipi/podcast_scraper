#!/usr/bin/env python3
"""Integration tests for summarization security features.

These tests require filesystem I/O and are moved from unit tests.
Tests cover:
- Revision pinning - Model revision pinning for security
- prune_cache() - Enhanced cache pruning security
- Memory cleanup - Model unloading
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Try to import summarizer, skip tests if dependencies not available
try:
    from podcast_scraper import config, summarizer

    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    summarizer = types.ModuleType("summarizer")  # type: ignore[assignment]


@pytest.mark.integration
@pytest.mark.slow
@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestRevisionPinning(unittest.TestCase):
    """Test revision pinning functionality for security and reproducibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.pipeline")
    @patch("podcast_scraper.summarizer.torch", create=True)
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
            model_name=config.TEST_DEFAULT_SUMMARY_MODEL,
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

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.pipeline")
    @patch("podcast_scraper.summarizer.torch", create=True)
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
            model_name=config.TEST_DEFAULT_SUMMARY_MODEL,
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

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.pipeline")
    @patch("podcast_scraper.summarizer.torch", create=True)
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

        # The test verifies kwargs before pipeline creation, so we can catch
        # any exceptions from pipeline and still verify the revision parameter
        try:
            summarizer.SummaryModel(
                model_name=config.TEST_DEFAULT_SUMMARY_MODEL,
                device="cpu",
                cache_dir=self.temp_dir,
                revision=None,
            )
        except Exception:
            # Pipeline creation may fail with mocks, but we can still verify kwargs
            pass

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

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.pipeline")
    @patch("podcast_scraper.summarizer.torch", create=True)
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
            model_name=config.TEST_DEFAULT_SUMMARY_MODEL,
            device="cpu",
            cache_dir=self.temp_dir,
            revision=revision,
        )

        self.assertEqual(model.revision, revision)

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.pipeline")
    @patch("podcast_scraper.summarizer.torch", create=True)
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
            model_name=config.TEST_DEFAULT_SUMMARY_MODEL,
            device="cpu",
            cache_dir=self.temp_dir,
            revision=None,
        )

        self.assertIsNone(model.revision)


@pytest.mark.integration
@pytest.mark.slow
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
        outside_path = Path("/tmp") / "test_cache"  # nosec B108
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


@pytest.mark.integration
@pytest.mark.slow
@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestMemoryCleanup(unittest.TestCase):
    """Test memory cleanup and model unloading."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.pipeline")
    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_unload_model_sets_to_none(
        self, mock_torch, mock_pipeline, mock_model_class, mock_tokenizer_class
    ):
        """Test that unload_model() sets model attributes to None."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe

        model = summarizer.SummaryModel(
            model_name=config.TEST_DEFAULT_SUMMARY_MODEL,
            device="cpu",
            cache_dir=self.temp_dir,
        )

        # Verify model is loaded
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.tokenizer)
        self.assertIsNotNone(model.pipeline)

        # Unload model
        summarizer.unload_model(model)

        # Verify model is unloaded
        self.assertIsNone(model.model)
        self.assertIsNone(model.tokenizer)
        self.assertIsNone(model.pipeline)

    def test_unload_model_with_none(self):
        """Test that unload_model() handles None gracefully."""
        # Should not raise an exception
        summarizer.unload_model(None)

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.pipeline")
    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_unload_model_twice(
        self, mock_torch, mock_pipeline, mock_model_class, mock_tokenizer_class
    ):
        """Test that unload_model() can be called multiple times safely."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe

        model = summarizer.SummaryModel(
            model_name=config.TEST_DEFAULT_SUMMARY_MODEL,
            device="cpu",
            cache_dir=self.temp_dir,
        )

        # Unload twice - should not raise an exception
        summarizer.unload_model(model)
        summarizer.unload_model(model)

        # Verify still None
        self.assertIsNone(model.model)


@pytest.mark.integration
@pytest.mark.slow
@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestModelLoadingFailures(unittest.TestCase):
    """Test error conditions during model loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_tokenizer_loading_failure(self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test that tokenizer loading failure raises appropriate error."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        # Simulate tokenizer loading failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network timeout")

        with self.assertRaises(Exception) as context:
            summarizer.SummaryModel(
                model_name=config.TEST_DEFAULT_SUMMARY_MODEL,
                device=None,
                cache_dir=self.temp_dir,
            )
        self.assertIn("Network timeout", str(context.exception))

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_model_loading_failure(self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test that model loading failure raises appropriate error."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Simulate model loading failure
        mock_model_class.from_pretrained.side_effect = OSError("Model not found")

        with self.assertRaises(OSError) as context:
            summarizer.SummaryModel(
                model_name="invalid/model-name",
                device=None,
                cache_dir=self.temp_dir,
            )
        self.assertIn("Model not found", str(context.exception))

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.pipeline")
    @patch("podcast_scraper.summarizer.torch", create=True)
    def test_pipeline_creation_failure(
        self, mock_torch, mock_pipeline, mock_model_class, mock_tokenizer_class
    ):
        """Test that pipeline creation failure raises appropriate error."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Simulate pipeline creation failure
        mock_pipeline.side_effect = RuntimeError("Pipeline initialization failed")

        with self.assertRaises(RuntimeError) as context:
            summarizer.SummaryModel(
                model_name=config.TEST_DEFAULT_SUMMARY_MODEL,
                device=None,
                cache_dir=self.temp_dir,
            )
        self.assertIn("Pipeline initialization failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
