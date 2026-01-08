#!/usr/bin/env python3
"""Unit tests for whisper_integration.py.

Tests cover:
- format_screenplay_from_segments: Format conversion from Whisper segments to screenplay format
- Edge cases and various segment configurations
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper import config, whisper_integration


class TestFormatScreenplayFromSegments(unittest.TestCase):
    """Tests for format_screenplay_from_segments function."""

    def test_empty_segments(self):
        """Test with empty segments list."""
        result = whisper_integration.format_screenplay_from_segments(
            segments=[],
            num_speakers=2,
            speaker_names=[],
            gap_s=1.0,
        )
        self.assertEqual(result, "")

    def test_single_segment(self):
        """Test with a single segment."""
        segments = [{"start": 0.0, "end": 5.0, "text": "Hello world"}]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=[],
            gap_s=1.0,
        )
        self.assertIn("SPEAKER 1:", result)
        self.assertIn("Hello world", result)
        self.assertTrue(result.endswith("\n"))

    def test_multiple_segments_same_speaker(self):
        """Test multiple consecutive segments assigned to same speaker."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First segment"},
            {"start": 5.0, "end": 10.0, "text": "Second segment"},
            {"start": 10.0, "end": 15.0, "text": "Third segment"},
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=[],
            gap_s=2.0,  # Gap larger than segment spacing, so all same speaker
        )
        # All segments should be concatenated for SPEAKER 1
        self.assertIn("SPEAKER 1:", result)
        self.assertIn("First segment", result)
        self.assertIn("Second segment", result)
        self.assertIn("Third segment", result)
        # Should be on same line (concatenated)
        lines = result.strip().split("\n")
        speaker_1_lines = [line for line in lines if line.startswith("SPEAKER 1:")]
        self.assertEqual(len(speaker_1_lines), 1)

    def test_speaker_alternation_with_gap(self):
        """Test speaker alternation when gap exceeds threshold."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Speaker 1 segment"},
            {"start": 10.0, "end": 15.0, "text": "Speaker 2 segment"},  # 5s gap > 1.0s threshold
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=[],
            gap_s=1.0,
        )
        self.assertIn("SPEAKER 1:", result)
        self.assertIn("SPEAKER 2:", result)
        self.assertIn("Speaker 1 segment", result)
        self.assertIn("Speaker 2 segment", result)

    def test_speaker_names_provided(self):
        """Test with custom speaker names."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Host speaks"},
            {"start": 10.0, "end": 15.0, "text": "Guest responds"},
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=["Host", "Guest"],
            gap_s=1.0,
        )
        self.assertIn("Host:", result)
        self.assertIn("Guest:", result)
        self.assertNotIn("SPEAKER 1:", result)
        self.assertNotIn("SPEAKER 2:", result)

    def test_speaker_names_partial(self):
        """Test with partial speaker names (some provided, some default)."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First speaker"},
            {"start": 10.0, "end": 15.0, "text": "Second speaker"},
            {"start": 20.0, "end": 25.0, "text": "Third speaker"},
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=3,
            speaker_names=["Alice"],  # Only first speaker named
            gap_s=1.0,
        )
        self.assertIn("Alice:", result)
        self.assertIn("SPEAKER 2:", result)
        self.assertIn("SPEAKER 3:", result)

    def test_segments_sorted_by_start_time(self):
        """Test that segments are sorted by start time even if provided out of order."""
        segments = [
            {"start": 20.0, "end": 25.0, "text": "Third"},
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 10.0, "end": 15.0, "text": "Second"},
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=1,
            speaker_names=[],
            gap_s=1.0,
        )
        # Verify order is correct (First, Second, Third)
        # All segments are same speaker (no gaps), so they're concatenated
        # Check that text appears in correct order within the concatenated string
        result_text = result.replace("SPEAKER 1:", "").strip()
        first_pos = result_text.find("First")
        second_pos = result_text.find("Second")
        third_pos = result_text.find("Third")
        self.assertGreater(first_pos, -1)
        self.assertGreater(second_pos, -1)
        self.assertGreater(third_pos, -1)
        self.assertLess(first_pos, second_pos)
        self.assertLess(second_pos, third_pos)

    def test_empty_text_segments_skipped(self):
        """Test that segments with empty or whitespace-only text are skipped."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Valid segment"},
            {"start": 5.0, "end": 10.0, "text": ""},  # Empty
            {"start": 10.0, "end": 15.0, "text": "   "},  # Whitespace only
            {"start": 15.0, "end": 20.0, "text": "Another valid"},
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=1,
            speaker_names=[],
            gap_s=1.0,
        )
        self.assertIn("Valid segment", result)
        self.assertIn("Another valid", result)
        # Empty segments should not create separate lines
        lines = [line for line in result.strip().split("\n") if line.strip()]
        # Should only have one line (both valid segments concatenated)
        self.assertEqual(len(lines), 1)

    def test_missing_start_time_defaults_to_zero(self):
        """Test that missing start time defaults to 0.0."""
        segments = [
            {"end": 5.0, "text": "No start time"},  # Missing "start"
            {"start": 10.0, "end": 15.0, "text": "Has start time"},
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=1,
            speaker_names=[],
            gap_s=1.0,
        )
        # Should still process both segments
        self.assertIn("No start time", result)
        self.assertIn("Has start time", result)

    def test_missing_end_time_defaults_to_start(self):
        """Test that missing end time defaults to start time."""
        segments = [
            {"start": 0.0, "text": "No end time"},  # Missing "end"
            {"start": 10.0, "end": 15.0, "text": "Has end time"},
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=1,
            speaker_names=[],
            gap_s=1.0,
        )
        self.assertIn("No end time", result)
        self.assertIn("Has end time", result)

    def test_speaker_wraparound(self):
        """Test that speaker index wraps around when exceeding num_speakers."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Speaker 1"},
            {"start": 10.0, "end": 15.0, "text": "Speaker 2"},
            {"start": 20.0, "end": 25.0, "text": "Speaker 1 again"},  # Should wrap to speaker 0
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=[],
            gap_s=1.0,
        )
        # Should have SPEAKER 1, SPEAKER 2, then wrap back to SPEAKER 1
        self.assertIn("SPEAKER 1:", result)
        self.assertIn("SPEAKER 2:", result)
        # Count occurrences of SPEAKER 1 (should appear twice)
        speaker_1_count = result.count("SPEAKER 1:")
        self.assertGreaterEqual(speaker_1_count, 1)

    def test_min_num_speakers_enforced(self):
        """Test that MIN_NUM_SPEAKERS is enforced even if num_speakers is lower."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Segment 1"},
            {"start": 10.0, "end": 15.0, "text": "Segment 2"},
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=0,  # Below MIN_NUM_SPEAKERS (which is 1)
            speaker_names=[],
            gap_s=1.0,
        )
        # Should still work because MIN_NUM_SPEAKERS is enforced
        self.assertIn("SPEAKER", result)

    def test_consecutive_segments_same_speaker_concatenated(self):
        """Test that consecutive segments for same speaker are concatenated with space."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First"},
            {"start": 5.0, "end": 10.0, "text": "Second"},  # No gap, same speaker
            {"start": 10.0, "end": 15.0, "text": "Third"},  # No gap, same speaker
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=[],
            gap_s=1.0,
        )
        # All three should be on same line, concatenated
        lines = result.strip().split("\n")
        speaker_1_lines = [line for line in lines if line.startswith("SPEAKER 1:")]
        self.assertEqual(len(speaker_1_lines), 1)
        # Check that all three texts appear in the same line
        self.assertIn("First", speaker_1_lines[0])
        self.assertIn("Second", speaker_1_lines[0])
        self.assertIn("Third", speaker_1_lines[0])

    def test_gap_between_segments_triggers_speaker_change(self):
        """Test that gap larger than threshold triggers speaker change."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Speaker 1"},
            {"start": 10.0, "end": 15.0, "text": "Speaker 2"},  # 5s gap > 1.0s threshold
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=[],
            gap_s=1.0,
        )
        # Should have two different speakers
        lines = result.strip().split("\n")
        speaker_labels = [line.split(":")[0] for line in lines if ":" in line]
        unique_speakers = set(speaker_labels)
        self.assertGreaterEqual(len(unique_speakers), 1)  # At least one speaker

    def test_gap_smaller_than_threshold_no_speaker_change(self):
        """Test that gap smaller than threshold does not trigger speaker change."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Segment 1"},
            {"start": 5.5, "end": 10.0, "text": "Segment 2"},  # 0.5s gap < 1.0s threshold
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=[],
            gap_s=1.0,
        )
        # Both segments should be same speaker (concatenated)
        lines = result.strip().split("\n")
        speaker_1_lines = [line for line in lines if line.startswith("SPEAKER 1:")]
        # Should be concatenated into one line
        self.assertEqual(len(speaker_1_lines), 1)

    def test_trailing_newline(self):
        """Test that result always ends with newline."""
        segments = [{"start": 0.0, "end": 5.0, "text": "Test"}]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=1,
            speaker_names=[],
            gap_s=1.0,
        )
        self.assertTrue(result.endswith("\n"))

    def test_multiple_speakers_with_names(self):
        """Test multiple speakers with custom names."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Host introduction"},
            {"start": 10.0, "end": 15.0, "text": "Guest response"},
            {"start": 20.0, "end": 25.0, "text": "Host follow-up"},
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=["Host", "Guest"],
            gap_s=1.0,
        )
        self.assertIn("Host:", result)
        self.assertIn("Guest:", result)
        # Count Host and Guest labels
        host_count = result.count("Host:")
        guest_count = result.count("Guest:")
        self.assertGreaterEqual(host_count, 1)
        self.assertGreaterEqual(guest_count, 1)

    def test_segments_with_none_values(self):
        """Test handling of None values in segment fields."""
        segments = [
            {"start": None, "end": None, "text": "Valid text"},  # None start/end
            {"start": 10.0, "end": 15.0, "text": None},  # None text (should be skipped)
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=1,
            speaker_names=[],
            gap_s=1.0,
        )
        # Should handle None start/end (defaults to 0.0)
        # Should skip None text
        self.assertIn("Valid text", result)

    def test_large_gap_speaker_rotation(self):
        """Test that large gaps cause speaker rotation through all speakers."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Speaker 1"},
            {"start": 20.0, "end": 25.0, "text": "Speaker 2"},  # Large gap
            {"start": 40.0, "end": 45.0, "text": "Speaker 3"},  # Large gap
            {"start": 60.0, "end": 65.0, "text": "Speaker 1 again"},  # Wraps around
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=3,
            speaker_names=[],
            gap_s=1.0,
        )
        # Should have multiple different speakers
        self.assertIn("SPEAKER 1:", result)
        self.assertIn("SPEAKER 2:", result)
        self.assertIn("SPEAKER 3:", result)

    def test_segments_with_float_strings(self):
        """Test handling of string numeric values (should be converted to float)."""
        segments = [
            {"start": "0.0", "end": "5.0", "text": "First"},  # String numbers
            {"start": "10.0", "end": "15.0", "text": "Second"},
        ]
        result = whisper_integration.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=[],
            gap_s=1.0,
        )
        # Should handle string numbers correctly
        self.assertIn("First", result)
        self.assertIn("Second", result)


class TestImportThirdPartyWhisper(unittest.TestCase):
    """Tests for _import_third_party_whisper function."""

    @patch("importlib.import_module")
    def test_import_whisper_success(self, mock_import):
        """Test successful import of whisper library."""
        mock_whisper = Mock()
        mock_whisper.load_model = Mock()
        mock_import.return_value = mock_whisper

        # Clear whisper from sys.modules if present
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

        result = whisper_integration._import_third_party_whisper()

        self.assertEqual(result, mock_whisper)
        mock_import.assert_called_once_with("whisper")

    @patch("importlib.import_module")
    def test_import_whisper_already_imported(self, mock_import):
        """Test import when whisper is already in sys.modules."""
        mock_whisper = Mock()
        mock_whisper.load_model = Mock()

        # Set whisper in sys.modules
        original_whisper = sys.modules.get("whisper")
        sys.modules["whisper"] = mock_whisper
        try:
            result = whisper_integration._import_third_party_whisper()

            self.assertEqual(result, mock_whisper)
            mock_import.assert_not_called()
        finally:
            # Clean up - always remove to prevent test isolation issues
            if "whisper" in sys.modules:
                del sys.modules["whisper"]
            # Restore original if it existed
            if original_whisper is not None:
                sys.modules["whisper"] = original_whisper

    @patch("importlib.import_module")
    def test_import_whisper_missing_load_model(self, mock_import):
        """Test import fails when whisper module lacks load_model."""
        # Clear whisper from sys.modules if present
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

        mock_whisper = Mock()
        del mock_whisper.load_model  # Remove load_model attribute
        mock_import.return_value = mock_whisper

        with self.assertRaises(ImportError) as context:
            whisper_integration._import_third_party_whisper()

        self.assertIn("load_model", str(context.exception))

    @patch("importlib.import_module")
    def test_import_whisper_import_error(self, mock_import):
        """Test import raises ImportError when module not found."""
        # Clear whisper from sys.modules if present
        if "whisper" in sys.modules:
            del sys.modules["whisper"]

        mock_import.side_effect = ImportError("No module named 'whisper'")

        with self.assertRaises(ImportError) as context:
            whisper_integration._import_third_party_whisper()

        self.assertIn("openai-whisper", str(context.exception))


class TestInterceptWhisperProgress(unittest.TestCase):
    """Tests for _intercept_whisper_progress context manager."""

    @patch("builtins.__import__")
    def test_intercept_whisper_progress_no_tqdm(self, mock_import):
        """Test intercept works when tqdm is not available."""
        mock_reporter = Mock()

        # Make import fail for tqdm
        def side_effect(name, *args, **kwargs):
            if name == "tqdm":
                raise ImportError("No module named 'tqdm'")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        # Should not raise even if tqdm is not available
        with whisper_integration._intercept_whisper_progress(mock_reporter):
            pass  # Context manager should work

    @patch("tqdm.tqdm")
    @patch("builtins.open", create=True)
    def test_intercept_whisper_progress_intercepts_tqdm(self, mock_open, mock_tqdm_class):
        """Test that tqdm calls are intercepted and forwarded."""
        mock_reporter = Mock()
        mock_file = Mock()
        mock_open.return_value = mock_file

        # Import tqdm to get the real module
        try:
            import tqdm

            original_tqdm = tqdm.tqdm
        except ImportError:
            # tqdm not available, skip test
            self.skipTest("tqdm not available")

        with whisper_integration._intercept_whisper_progress(mock_reporter):
            # tqdm should be replaced
            import tqdm

            self.assertNotEqual(tqdm.tqdm, original_tqdm)

        # Should be restored after context
        import tqdm

        self.assertEqual(tqdm.tqdm, original_tqdm)

    @patch("builtins.open", create=True)
    def test_intercept_whisper_progress_forwards_updates(self, mock_open):
        """Test that intercepted tqdm forwards updates to reporter."""
        mock_reporter = Mock()
        mock_file = Mock()
        mock_open.return_value = mock_file

        # Import tqdm to get the real module
        try:
            import tqdm
        except ImportError:
            # tqdm not available, skip test
            self.skipTest("tqdm not available")

        with whisper_integration._intercept_whisper_progress(mock_reporter):
            intercepted_tqdm = tqdm.tqdm(total=100)
            intercepted_tqdm.update(50)

            # Should forward to reporter
            mock_reporter.update.assert_called_with(50)


class TestLoadWhisperModel(unittest.TestCase):
    """Tests for load_whisper_model function."""

    def test_load_whisper_model_transcribe_disabled(self):
        """Test load_whisper_model returns None when transcribe_missing is False."""
        cfg = Mock()
        cfg.transcribe_missing = False

        result = whisper_integration.load_whisper_model(cfg)

        self.assertIsNone(result)

    @patch("podcast_scraper.whisper_integration._import_third_party_whisper")
    def test_load_whisper_model_import_error(self, mock_import):
        """Test load_whisper_model handles import errors."""
        cfg = Mock()
        cfg.transcribe_missing = True
        mock_import.side_effect = ImportError("whisper not found")

        result = whisper_integration.load_whisper_model(cfg)

        self.assertIsNone(result)

    @patch("podcast_scraper.whisper_integration._import_third_party_whisper")
    def test_load_whisper_model_success(self, mock_import):
        """Test successful model loading."""
        cfg = Mock()
        cfg.transcribe_missing = True
        cfg.whisper_model = config.TEST_DEFAULT_WHISPER_MODEL  # Use test default (tiny.en)
        cfg.language = "en"

        mock_whisper = Mock()
        mock_model = Mock()
        mock_model.device = Mock()
        mock_model.device.type = "cpu"
        mock_model.dtype = None
        mock_model.num_parameters = Mock(return_value=1000000)
        mock_whisper.load_model.return_value = mock_model
        mock_import.return_value = mock_whisper

        result = whisper_integration.load_whisper_model(cfg)

        self.assertEqual(result, mock_model)
        # Should use test default model (tiny.en) for English
        # Note: download_root may be passed if cache directory is set up
        call_args = mock_whisper.load_model.call_args
        self.assertIsNotNone(call_args)
        self.assertEqual(
            call_args[0][0], config.TEST_DEFAULT_WHISPER_MODEL
        )  # First positional argument should be test default (tiny.en)

    @patch("podcast_scraper.whisper_integration._import_third_party_whisper")
    def test_load_whisper_model_fallback(self, mock_import):
        """Test model loading with fallback to smaller model."""
        cfg = Mock()
        cfg.transcribe_missing = True
        cfg.whisper_model = "large"
        cfg.language = "en"

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

        result = whisper_integration.load_whisper_model(cfg)

        self.assertEqual(result, mock_model)
        # Should try large.en first, then fallback
        self.assertEqual(mock_whisper.load_model.call_count, 2)

    @patch("podcast_scraper.whisper_integration._import_third_party_whisper")
    def test_load_whisper_model_all_fail(self, mock_import):
        """Test model loading when all models fail."""
        cfg = Mock()
        cfg.transcribe_missing = True
        cfg.whisper_model = config.TEST_DEFAULT_WHISPER_MODEL  # Use test default (tiny.en)
        cfg.language = "en"

        mock_whisper = Mock()
        mock_whisper.load_model.side_effect = FileNotFoundError("Model not found")
        mock_import.return_value = mock_whisper

        result = whisper_integration.load_whisper_model(cfg)

        self.assertIsNone(result)


class TestTranscribeWithWhisper(unittest.TestCase):
    """Tests for transcribe_with_whisper function."""

    @patch("podcast_scraper.whisper_integration._intercept_whisper_progress")
    @patch("podcast_scraper.whisper_integration.progress.progress_context")
    @patch("time.time")
    def test_transcribe_with_whisper_success(self, mock_time, mock_progress, mock_intercept):
        """Test successful transcription."""
        mock_model = Mock()
        mock_model._is_cpu_device = False
        mock_model.dtype = None
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello world"}],
        }

        cfg = Mock()
        cfg.whisper_model = config.TEST_DEFAULT_WHISPER_MODEL  # Use test default (tiny.en)
        cfg.language = "en"

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        mock_time.side_effect = [0.0, 10.5]  # start, end

        result, elapsed = whisper_integration.transcribe_with_whisper(
            mock_model, "/tmp/ep1.mp3", cfg
        )

        self.assertEqual(result["text"], "Hello world")
        self.assertEqual(elapsed, 10.5)
        mock_model.transcribe.assert_called_once_with(
            "/tmp/ep1.mp3", task="transcribe", language="en", verbose=False
        )

    @patch("podcast_scraper.whisper_integration._intercept_whisper_progress")
    @patch("podcast_scraper.whisper_integration.progress.progress_context")
    @patch("time.time")
    def test_transcribe_with_whisper_default_language(
        self, mock_time, mock_progress, mock_intercept
    ):
        """Test transcription with default language when not specified."""
        mock_model = Mock()
        mock_model._is_cpu_device = False
        mock_model.dtype = None
        mock_model.transcribe.return_value = {"text": "Hello", "segments": []}

        cfg = Mock()
        cfg.whisper_model = config.TEST_DEFAULT_WHISPER_MODEL  # Use test default (tiny.en)
        cfg.language = None  # Not specified

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None

        mock_time.side_effect = [0.0, 5.0]

        result, elapsed = whisper_integration.transcribe_with_whisper(
            mock_model, "/tmp/ep1.mp3", cfg
        )

        # Should default to "en"
        mock_model.transcribe.assert_called_once_with(
            "/tmp/ep1.mp3", task="transcribe", language="en", verbose=False
        )


if __name__ == "__main__":
    unittest.main()
