#!/usr/bin/env python3
"""Unit tests for MLProvider transcription (via factory).

These tests verify the Whisper-based transcription provider implementation
using the unified MLProvider returned by the factory.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts
import importlib.util
from pathlib import Path

parent_tests_dir = Path(__file__).parent.parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

parent_conftest_path = parent_tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

create_test_config = parent_conftest.create_test_config

from podcast_scraper import config  # noqa: E402
from podcast_scraper.ml.ml_provider import (  # noqa: E402
    _import_third_party_whisper,
    _intercept_whisper_progress,
)
from podcast_scraper.transcription.factory import create_transcription_provider  # noqa: E402


class TestWhisperTranscriptionProvider(unittest.TestCase):
    """Tests for MLProvider transcription (via factory)."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL.replace(".en", ""),
            language="en",
            transcription_provider="whisper",
        )
        # Save original whisper module if it exists
        self._original_whisper = sys.modules.get("whisper")
        # Clear whisper from sys.modules to ensure clean state
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

    def tearDown(self):
        """Clean up test fixtures."""
        # Always clear whisper from sys.modules after each test
        # This ensures test isolation when running all tests together
        if "whisper" in sys.modules:
            del sys.modules["whisper"]
        # Don't restore original - let each test start fresh
        # This prevents test isolation issues

    def test_init(self):
        """Test WhisperTranscriptionProvider initialization."""
        provider = create_transcription_provider(self.cfg)
        self.assertEqual(provider.cfg, self.cfg)
        self.assertIsNone(provider._whisper_model)
        self.assertFalse(provider._whisper_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_initialize_success(self, mock_import):
        """Test successful initialization."""
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        self.assertEqual(provider._whisper_model, mock_model)
        self.assertTrue(provider._whisper_initialized)
        self.assertTrue(provider.is_initialized)
        # Should prefer .en variant for English and use download_root for cache
        mock_whisper.load_model.assert_called_once()
        call_args = mock_whisper.load_model.call_args
        self.assertEqual(call_args[0][0], config.TEST_DEFAULT_WHISPER_MODEL)
        self.assertIn("download_root", call_args[1])

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_initialize_transcribe_disabled(self, mock_import):
        """Test initialization when transcribe_missing is False."""
        cfg = create_test_config(transcribe_missing=False, transcription_provider="whisper")
        provider = create_transcription_provider(cfg)
        provider.initialize()

        self.assertIsNone(provider._whisper_model)
        self.assertFalse(provider._whisper_initialized)
        mock_import.assert_not_called()

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_initialize_already_initialized(self, mock_import):
        """Test initialization when already initialized."""
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        provider = create_transcription_provider(self.cfg)
        provider.initialize()
        mock_import.reset_mock()

        # Call again
        provider.initialize()

        # Should not import again
        mock_import.assert_not_called()

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_initialize_import_error(self, mock_import):
        """Test initialization when whisper import fails - logs warning but doesn't raise."""
        mock_import.side_effect = ImportError("whisper not found")

        provider = create_transcription_provider(self.cfg)
        # Initialize should not raise - it logs warning and continues
        provider.initialize()

        # Whisper should not be initialized
        self.assertFalse(provider._whisper_initialized)
        # But transcribe should raise when trying to use it
        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/tmp/audio.mp3")
        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_initialize_fallback_models(self, mock_import):
        """Test initialization with fallback to smaller models."""
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)

        # First attempt fails, fallback succeeds
        mock_whisper.load_model.side_effect = [
            FileNotFoundError("Model not found"),
            mock_model,
        ]
        mock_import.return_value = mock_whisper

        cfg = create_test_config(transcribe_missing=True, whisper_model="large", language="en")
        provider = create_transcription_provider(cfg)
        provider.initialize()

        # Should try large.en first, then fallback
        self.assertEqual(mock_whisper.load_model.call_count, 2)

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_initialize_all_models_fail(self, mock_import):
        """Test initialization when all models fail - logs warning but doesn't raise."""
        mock_whisper = Mock()
        mock_whisper.load_model.side_effect = FileNotFoundError("Model not found")
        mock_import.return_value = mock_whisper

        provider = create_transcription_provider(self.cfg)
        # Initialize should not raise - it logs warning and continues
        provider.initialize()

        # Whisper should not be initialized
        self.assertFalse(provider._whisper_initialized)
        # But transcribe should raise when trying to use it
        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/tmp/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider._intercept_whisper_progress")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    @patch("podcast_scraper.ml.ml_provider.time.time")
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_success(self, mock_import, mock_time, mock_progress, mock_intercept):
        """Test successful transcription."""
        # Remove whisper from sys.modules to ensure clean state
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_model.transcribe.return_value = {"text": "Hello world", "segments": []}
        # Set _is_cpu_device attribute directly on the mock
        setattr(mock_model, "_is_cpu_device", False)
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None
        # Use a call counter to track time.time() calls
        # Make it robust to handle any number of calls
        call_count = [0]

        def time_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.0
            # Return 5.0 for all subsequent calls (elapsed time)
            return 5.0

        mock_time.side_effect = time_side_effect

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        result = provider.transcribe("/tmp/audio.mp3")

        self.assertEqual(result, "Hello world")
        mock_model.transcribe.assert_called_once_with(
            "/tmp/audio.mp3", task="transcribe", language="en", verbose=False
        )

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_not_initialized(self, mock_import):
        """Test transcribe raises error when not initialized."""
        provider = create_transcription_provider(self.cfg)

        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/tmp/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider._intercept_whisper_progress")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    @patch("podcast_scraper.ml.ml_provider.time.time")
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_empty_text(self, mock_import, mock_time, mock_progress, mock_intercept):
        """Test transcribe raises error when text is empty."""
        # Remove whisper from sys.modules to ensure clean state
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_model.transcribe.return_value = {"text": "", "segments": []}
        setattr(mock_model, "_is_cpu_device", False)
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None
        # Use a call counter to track time.time() calls
        # Make it robust to handle any number of calls
        call_count = [0]

        def time_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.0
            # Return 5.0 for all subsequent calls (elapsed time)
            return 5.0

        mock_time.side_effect = time_side_effect

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.transcribe("/tmp/audio.mp3")

        self.assertIn("empty text", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider.MLProvider._transcribe_with_whisper")
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_with_segments(self, mock_import, mock_transcribe):
        """Test transcribe_with_segments method.

        This test mocks _transcribe_with_whisper directly to avoid time.time() mocking
        complexity and test isolation issues. This approach is cleaner and more reliable.
        """
        # Remove whisper from sys.modules to ensure clean state
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        # Mock _transcribe_with_whisper to return expected result with elapsed time
        # This bypasses time.time() mocking entirely and avoids test isolation issues
        expected_result = {
            "text": "Hello world",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello world"}],
        }
        mock_transcribe.return_value = (expected_result, 5.0)

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        result_dict, elapsed = provider.transcribe_with_segments("/tmp/audio.mp3")

        # Verify the result structure
        self.assertEqual(result_dict["text"], "Hello world")
        self.assertEqual(len(result_dict["segments"]), 1)
        # Verify elapsed time is returned correctly
        self.assertEqual(elapsed, 5.0)
        # Verify _transcribe_with_whisper was called with correct arguments
        mock_transcribe.assert_called_once_with("/tmp/audio.mp3", "en")

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_with_segments_not_initialized(self, mock_import):
        """Test transcribe_with_segments raises error when not initialized."""
        provider = create_transcription_provider(self.cfg)

        with self.assertRaises(RuntimeError):
            provider.transcribe_with_segments("/tmp/audio.mp3")

    def test_format_screenplay_from_segments(self):
        """Test format_screenplay_from_segments method."""
        provider = create_transcription_provider(self.cfg)

        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 10.0, "end": 15.0, "text": "World"},
        ]

        result = provider.format_screenplay_from_segments(
            segments, num_speakers=2, speaker_names=["Alice", "Bob"], gap_s=1.0
        )

        self.assertIn("Alice:", result)
        self.assertIn("Bob:", result)
        self.assertIn("Hello", result)
        self.assertIn("World", result)
        self.assertTrue(result.endswith("\n"))

    def test_format_screenplay_from_segments_empty(self):
        """Test format_screenplay_from_segments with empty segments."""
        provider = create_transcription_provider(self.cfg)

        result = provider.format_screenplay_from_segments([], 2, [], 1.0)

        self.assertEqual(result, "")

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_cleanup(self, mock_import):
        """Test cleanup method."""
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        provider.cleanup()

        self.assertIsNone(provider._whisper_model)
        self.assertFalse(provider._whisper_initialized)
        self.assertFalse(provider.is_initialized)

    def test_cleanup_not_initialized(self):
        """Test cleanup when not initialized."""
        provider = create_transcription_provider(self.cfg)
        provider.cleanup()  # Should not raise

    def test_model_property(self):
        """Test model property."""
        provider = create_transcription_provider(self.cfg)
        self.assertIsNone(provider.model)

        mock_model = Mock()
        provider._whisper_model = mock_model
        self.assertEqual(provider.model, mock_model)

    def test_is_initialized_property(self):
        """Test is_initialized property."""
        provider = create_transcription_provider(self.cfg)
        self.assertFalse(provider.is_initialized)

        provider._whisper_initialized = True
        self.assertTrue(provider.is_initialized)


class TestImportThirdPartyWhisper(unittest.TestCase):
    """Tests for _import_third_party_whisper function."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear whisper from sys.modules to ensure clean state
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

    def tearDown(self):
        """Clean up test fixtures."""
        # Clear whisper from sys.modules after each test
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

    @patch("importlib.import_module")
    def test_import_success(self, mock_import):
        """Test successful import."""
        mock_whisper = Mock()
        mock_whisper.load_model = Mock()
        mock_import.return_value = mock_whisper

        result = _import_third_party_whisper()

        self.assertEqual(result, mock_whisper)
        mock_import.assert_called_once_with("whisper")

    @patch("importlib.import_module")
    def test_import_missing_load_model(self, mock_import):
        """Test import fails when whisper lacks load_model."""
        mock_whisper = Mock()
        del mock_whisper.load_model
        mock_import.return_value = mock_whisper

        with self.assertRaises(ImportError):
            _import_third_party_whisper()


class TestInterceptWhisperProgress(unittest.TestCase):
    """Tests for _intercept_whisper_progress context manager."""

    @patch("builtins.__import__")
    def test_intercept_no_tqdm(self, mock_import):
        """Test intercept works when tqdm is not available."""
        mock_reporter = Mock()

        def side_effect(name, *args, **kwargs):
            if name == "tqdm":
                raise ImportError("No module named 'tqdm'")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        with _intercept_whisper_progress(mock_reporter):
            pass  # Should not raise


class TestWhisperProviderEdgeCases(unittest.TestCase):
    """Tests for WhisperTranscriptionProvider edge cases and error paths."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL.replace(".en", ""),
            language="en",
            transcription_provider="whisper",
        )
        # Save original whisper module if it exists
        self._original_whisper = sys.modules.get("whisper")
        # Clear whisper from sys.modules to ensure clean state
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

    def tearDown(self):
        """Clean up test fixtures."""
        # Always clear whisper from sys.modules after each test
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

    @patch("podcast_scraper.ml.ml_provider._intercept_whisper_progress")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_file_not_found(self, mock_import, mock_progress, mock_intercept):
        """Test transcribe raises FileNotFoundError when audio file doesn't exist."""
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_model.transcribe.side_effect = FileNotFoundError("Audio file not found")
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        with self.assertRaises(FileNotFoundError):
            provider.transcribe("/nonexistent/audio.mp3")

    @patch("podcast_scraper.ml.ml_provider._intercept_whisper_progress")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_model_exception(self, mock_import, mock_progress, mock_intercept):
        """Test transcribe handles model.transcribe exception."""
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_model.transcribe.side_effect = RuntimeError("Transcription failed")
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        with self.assertRaises(RuntimeError):
            provider.transcribe("/tmp/audio.mp3")

    @patch("podcast_scraper.ml.ml_provider._intercept_whisper_progress")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_language_override(self, mock_import, mock_progress, mock_intercept):
        """Test transcribe with explicit language parameter."""
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_model.transcribe.return_value = {"text": "Bonjour", "segments": []}
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        call_count = [0]

        def time_side_effect():
            call_count[0] += 1
            return 0.0 if call_count[0] == 1 else 5.0

        with patch(
            "podcast_scraper.transcription.whisper_provider.time.time", side_effect=time_side_effect
        ):
            provider = create_transcription_provider(self.cfg)
            provider.initialize()

            result = provider.transcribe("/tmp/audio.mp3", language="fr")

            self.assertEqual(result, "Bonjour")
            mock_model.transcribe.assert_called_once_with(
                "/tmp/audio.mp3", task="transcribe", language="fr", verbose=False
            )

    @patch("podcast_scraper.ml.ml_provider._intercept_whisper_progress")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_with_segments_file_not_found(
        self, mock_import, mock_progress, mock_intercept
    ):
        """Test transcribe_with_segments raises FileNotFoundError when audio file doesn't exist."""
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_model.transcribe.side_effect = FileNotFoundError("Audio file not found")
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        with self.assertRaises(FileNotFoundError):
            provider.transcribe_with_segments("/nonexistent/audio.mp3")

    @patch("podcast_scraper.ml.ml_provider._intercept_whisper_progress")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_with_segments_model_exception(
        self, mock_import, mock_progress, mock_intercept
    ):
        """Test transcribe_with_segments handles model.transcribe exception."""
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_model.transcribe.side_effect = RuntimeError("Transcription failed")
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        with self.assertRaises(RuntimeError):
            provider.transcribe_with_segments("/tmp/audio.mp3")

    def test_format_screenplay_malformed_segments(self):
        """Test format_screenplay_from_segments with malformed segments."""
        provider = create_transcription_provider(self.cfg)

        # Segments missing start/end
        segments = [{"text": "Hello"}, {"text": "World"}]

        result = provider.format_screenplay_from_segments(
            segments=segments, num_speakers=2, speaker_names=["Speaker1", "Speaker2"], gap_s=1.0
        )

        # Should handle gracefully
        self.assertIn("Hello", result)
        self.assertIn("World", result)

    def test_format_screenplay_with_gaps(self):
        """Test format_screenplay_from_segments with gaps triggering speaker rotation."""
        provider = create_transcription_provider(self.cfg)

        segments = [
            {"start": 0.0, "end": 5.0, "text": "First segment"},
            {"start": 10.0, "end": 15.0, "text": "Second segment"},  # 5s gap
            {"start": 16.0, "end": 20.0, "text": "Third segment"},  # 1s gap (no rotation)
        ]

        result = provider.format_screenplay_from_segments(
            segments=segments, num_speakers=2, speaker_names=["Speaker1", "Speaker2"], gap_s=2.0
        )

        # Should rotate speakers on large gap
        self.assertIn("First segment", result)
        self.assertIn("Second segment", result)
        self.assertIn("Third segment", result)

    def test_format_screenplay_no_speaker_names(self):
        """Test format_screenplay_from_segments without speaker names (uses indices)."""
        provider = create_transcription_provider(self.cfg)

        segments = [
            {"start": 0.0, "end": 5.0, "text": "First segment"},
            {"start": 10.0, "end": 15.0, "text": "Second segment"},
        ]

        result = provider.format_screenplay_from_segments(
            segments=segments, num_speakers=2, speaker_names=[], gap_s=2.0
        )

        # Should use default speaker names
        self.assertIn("First segment", result)
        self.assertIn("Second segment", result)

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_initialize_os_error_handling(self, mock_import):
        """Test initialize handles OSError - logs warning but doesn't raise."""
        mock_whisper = Mock()
        mock_whisper.load_model.side_effect = OSError("Disk full")
        mock_import.return_value = mock_whisper

        provider = create_transcription_provider(self.cfg)
        # Initialize should not raise - it logs warning and continues
        provider.initialize()

        # Whisper should not be initialized
        self.assertFalse(provider._whisper_initialized)
        # But transcribe should raise when trying to use it
        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/tmp/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_initialize_non_english_language(self, mock_import):
        """Test initialize with non-English language (removes .en suffix)."""
        cfg = create_test_config(
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            language="fr",
        )
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        provider = create_transcription_provider(cfg)
        provider.initialize()

        # Should use model without .en suffix for non-English, with download_root
        mock_whisper.load_model.assert_called_once()
        call_args = mock_whisper.load_model.call_args
        self.assertEqual(call_args[0][0], config.TEST_DEFAULT_WHISPER_MODEL.replace(".en", ""))
        self.assertIn("download_root", call_args[1])

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_initialize_model_missing_attributes(self, mock_import):
        """Test initialize handles model with missing optional attributes."""
        mock_whisper = Mock()
        mock_model = Mock()
        # Model without device attribute
        delattr(mock_model, "device") if hasattr(mock_model, "device") else None
        mock_model.dtype = None
        # Model without num_parameters
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Should handle gracefully
        self.assertTrue(provider.is_initialized)

    @patch("podcast_scraper.ml.ml_provider._intercept_whisper_progress")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_with_segments_language_override(
        self, mock_import, mock_progress, mock_intercept
    ):
        """Test transcribe_with_segments with explicit language parameter."""
        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_model.transcribe.return_value = {
            "text": "Bonjour",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Bonjour"}],
        }
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        call_count = [0]

        def time_side_effect():
            call_count[0] += 1
            return 0.0 if call_count[0] == 1 else 5.0

        with patch(
            "podcast_scraper.transcription.whisper_provider.time.time", side_effect=time_side_effect
        ):
            provider = create_transcription_provider(self.cfg)
            provider.initialize()

            result, elapsed = provider.transcribe_with_segments("/tmp/audio.mp3", language="fr")

            self.assertEqual(result["text"], "Bonjour")
            mock_model.transcribe.assert_called_once_with(
                "/tmp/audio.mp3", task="transcribe", language="fr", verbose=False
            )


if __name__ == "__main__":
    unittest.main()
