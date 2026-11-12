#!/usr/bin/env python3
"""Tests for summarization failure handling.

These tests verify that summarization failures are handled correctly
with fail-fast behavior when generate_summaries=True.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

from podcast_scraper import config
from podcast_scraper.workflow import metrics

pytestmark = [pytest.mark.unit, pytest.mark.module_summarization]


def create_test_config(**kwargs):
    """Create a test configuration with defaults."""
    defaults = {
        "rss_url": "https://example.com/feed.xml",
        "output_dir": tempfile.mkdtemp(),
        "max_episodes": 1,
        "generate_metadata": True,
        "download_podcast_artwork": False,
        "auto_speakers": False,
        "feed_declared_language": "en",  # #2283: an English feed
    }
    defaults.update(kwargs)
    return config.Config(**defaults)


def _restore_transformers(original):
    """Put ``sys.modules['transformers']`` back the way we found it.

    A mocked module left in ``sys.modules`` is not a local lie — it is a GLOBAL one. Every import
    for the rest of the process gets the Mock, including the real ML tests, which then assert
    against ``MagicMock.max_new_tokens`` instead of 600.
    """
    if original is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = original


class TestEpisodeSummarizationFailure(unittest.TestCase):
    """Test episode-level summarization failure handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp()
        # Create a test transcript file
        self.transcript_path = Path(self.output_dir) / "transcripts" / "ep01_test.txt"
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_path.write_text("This is a test transcript with enough content. " * 20)
        # Create metadata subdirectory
        metadata_dir = Path(self.output_dir) / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.output_dir, ignore_errors=True)

    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_episode_summarization_failure_raises_error(self, mock_clean):
        """Test that episode summarization failure raises RuntimeError."""
        import os

        from podcast_scraper.workflow import metadata_generation as metadata

        # Create transcript file
        transcript_path = os.path.join(self.output_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        mock_clean.return_value = "Cleaned transcript text"
        # Create a mock summary provider that raises exception
        mock_provider = Mock()
        mock_provider.summarize.side_effect = Exception("Provider error")

        # Call _generate_episode_summary which should raise RuntimeError
        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        with self.assertRaises(RuntimeError) as context:
            metadata._generate_episode_summary(
                transcript_file_path="transcript.txt",
                output_dir=self.output_dir,
                cfg=cfg,
                episode_idx=1,
                summary_provider=mock_provider,
            )

        self.assertIn("generate_summaries=True", str(context.exception))
        self.assertIn("Failed to generate summary", str(context.exception))

    def test_episode_summarization_without_provider_raises_error(self):
        """Test that episode summarization without provider raises RuntimeError."""
        from podcast_scraper.workflow import metadata_generation as metadata

        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        # Should raise ValueError when summary_provider is None but generate_summaries=True
        with self.assertRaises(ValueError) as context:
            metadata._generate_episode_summary(
                transcript_file_path=str(self.transcript_path.relative_to(self.output_dir)),
                output_dir=self.output_dir,
                cfg=cfg,
                episode_idx=1,
                summary_provider=None,  # No provider
            )

        self.assertIn("generate_summaries=True", str(context.exception))
        self.assertIn("summary_provider is required", str(context.exception))


class TestParallelSummarizationFailure(unittest.TestCase):
    """Test parallel summarization failure handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp()
        self.episodes = [Mock()]
        self.episodes[0].idx = 1
        self.episodes[0].title_safe = "test_episode"
        self.feed = Mock()
        self.feed_metadata = Mock()
        self.host_detection_result = Mock()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_parallel_summarization_with_none_provider_raises_error(self):
        """Test parallel summarization raises RuntimeError when provider is None and generate_summaries=True."""  # noqa: E501
        from podcast_scraper.utils import filesystem

        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        # Create a transcript file so the function finds episodes to summarize
        from pathlib import Path

        transcript_path_str = filesystem.build_whisper_output_path(
            self.episodes[0].idx, self.episodes[0].title_safe, None, self.output_dir
        )
        transcript_path = Path(transcript_path_str)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text("This is a test transcript. " * 20)

        # Create mock feed metadata and host detection result
        from podcast_scraper.workflow.types import FeedMetadata, HostDetectionResult

        feed_metadata = FeedMetadata(None, None, None)
        host_detection_result = HostDetectionResult(
            cached_hosts=set(), heuristics=None, speaker_detector=None
        )

        # Should raise RuntimeError when summary_provider is None but generate_summaries=True
        from podcast_scraper.workflow.stages import summarization

        with self.assertRaises(RuntimeError) as context:
            summarization.parallel_episode_summarization(
                episodes=self.episodes,
                feed=self.feed,
                cfg=cfg,
                effective_output_dir=self.output_dir,
                run_suffix=None,
                feed_metadata=feed_metadata,
                host_detection_result=host_detection_result,
                summary_provider=None,  # No provider
                download_args=[],
                pipeline_metrics=None,
            )

        self.assertIn("generate_summaries=True", str(context.exception))
        self.assertIn("Summary provider not available", str(context.exception))

    def test_parallel_summarization_no_provider_when_generate_summaries_false(self):
        """Test parallel summarization logs warning when provider is None and generate_summaries=False."""  # noqa: E501
        from podcast_scraper.utils import filesystem
        from podcast_scraper.workflow.stages import summarization

        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=False,  # Summaries disabled
            generate_metadata=True,
        )

        # Create a transcript file so the function finds episodes to summarize
        from pathlib import Path

        transcript_path_str = filesystem.build_whisper_output_path(
            self.episodes[0].idx, self.episodes[0].title_safe, None, self.output_dir
        )
        transcript_path = Path(transcript_path_str)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text("This is a test transcript with enough content. " * 20)

        # Should not raise, just log warning and return
        with patch("podcast_scraper.workflow.stages.summarization.logger") as mock_logger:
            summarization.parallel_episode_summarization(
                episodes=self.episodes,
                feed=self.feed,
                cfg=cfg,
                effective_output_dir=self.output_dir,
                run_suffix=None,
                feed_metadata=self.feed_metadata,
                host_detection_result=self.host_detection_result,
                summary_provider=None,  # No provider
                download_args=[],
                pipeline_metrics=metrics.Metrics(),
            )

        # Should log warning when generate_summaries=False
        mock_logger.warning.assert_called_once()
        self.assertIn("not available", str(mock_logger.warning.call_args[0][0]))

    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_empty_summary_raises_error(self, mock_clean):
        """Test that empty summary raises RuntimeError when generate_summaries=True."""
        import os

        from podcast_scraper.workflow import metadata_generation as metadata

        # Create transcript file
        transcript_path = os.path.join(self.output_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        mock_provider.summarize.return_value = {"summary": "", "metadata": {}}  # Empty summary

        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        # Should raise RuntimeError when summary is empty
        with self.assertRaises(RuntimeError) as context:
            metadata._generate_episode_summary(
                transcript_file_path="transcript.txt",
                output_dir=self.output_dir,
                cfg=cfg,
                episode_idx=1,
                summary_provider=mock_provider,
            )

        self.assertIn("empty result", str(context.exception))
        self.assertIn("generate_summaries=True", str(context.exception))

    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_non_string_summary_raises_error(self, mock_clean):
        """Test that non-string summary raises RuntimeError when generate_summaries=True."""
        import os

        from podcast_scraper.workflow import metadata_generation as metadata

        # Create transcript file
        transcript_path = os.path.join(self.output_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        mock_provider.summarize.return_value = {
            "summary": 123,  # Non-string summary
            "metadata": {},
        }

        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        # Should raise RuntimeError when summary is not a string
        with self.assertRaises(RuntimeError) as context:
            metadata._generate_episode_summary(
                transcript_file_path="transcript.txt",
                output_dir=self.output_dir,
                cfg=cfg,
                episode_idx=1,
                summary_provider=mock_provider,
            )

        self.assertIn("not a string", str(context.exception))
        self.assertIn("generate_summaries=True", str(context.exception))
