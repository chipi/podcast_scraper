"""Unit tests for podcast_scraper.workflow.stages.processing module.

Tests for processing stage helper functions.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts
parent_tests_dir = Path(__file__).parent.parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

import pytest

from podcast_scraper import models
from podcast_scraper.workflow.stages import processing
from podcast_scraper.workflow.types import HostDetectionResult

# Import directly from tests.conftest (works with pytest-xdist)
from tests.conftest import create_test_config, create_test_episode  # noqa: E402


@pytest.mark.unit
class TestHandleDryRunHostDetection(unittest.TestCase):
    """Tests for _handle_dry_run_host_detection helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.feed = models.RssFeed(
            title="Test Feed",
            authors=["Host 1", "Host 2"],
            items=[],
            base_url="https://example.com",
        )

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_handle_dry_run_host_detection_with_authors(self, mock_logger):
        """Test dry-run host detection with RSS authors."""
        result = processing._handle_dry_run_host_detection(self.feed)

        self.assertIsInstance(result, HostDetectionResult)
        self.assertEqual(result.cached_hosts, {"Host 1", "Host 2"})
        self.assertIsNone(result.heuristics)
        self.assertIsNone(result.speaker_detector)
        mock_logger.info.assert_called()

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_handle_dry_run_host_detection_without_authors(self, mock_logger):
        """Test dry-run host detection without RSS authors."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )

        result = processing._handle_dry_run_host_detection(feed)

        self.assertIsInstance(result, HostDetectionResult)
        self.assertEqual(result.cached_hosts, set())
        self.assertIsNone(result.heuristics)
        self.assertIsNone(result.speaker_detector)

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_handle_dry_run_host_detection_with_empty_authors(self, mock_logger):
        """Test dry-run host detection with empty authors list."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=[],
            items=[],
            base_url="https://example.com",
        )

        result = processing._handle_dry_run_host_detection(feed)

        self.assertIsInstance(result, HostDetectionResult)
        self.assertEqual(result.cached_hosts, set())


@pytest.mark.unit
class TestCreateSpeakerDetectorIfNeeded(unittest.TestCase):
    """Tests for _create_speaker_detector_if_needed helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(auto_speakers=True)

    def test_create_speaker_detector_if_needed_with_existing(self):
        """Test that existing speaker detector is returned."""
        mock_detector = Mock()
        result = processing._create_speaker_detector_if_needed(self.cfg, mock_detector)

        self.assertEqual(result, mock_detector)

    @patch("podcast_scraper.workflow.stages.processing.logger")
    @patch("podcast_scraper.speaker_detectors.factory.create_speaker_detector")
    def test_create_speaker_detector_if_needed_creates_new(self, mock_create_detector, mock_logger):
        """Test that new speaker detector is created when None provided."""
        mock_detector = Mock()
        mock_detector.initialize = Mock()
        mock_create_detector.return_value = mock_detector

        # Mock sys.modules to not have create_speaker_detector
        with patch("sys.modules") as mock_modules:
            mock_workflow = Mock()
            del mock_workflow.create_speaker_detector  # Attribute doesn't exist
            mock_modules.get.return_value = mock_workflow

            result = processing._create_speaker_detector_if_needed(self.cfg, None)

            self.assertEqual(result, mock_detector)
            mock_detector.initialize.assert_called_once()
            mock_logger.warning.assert_called()

    @patch("podcast_scraper.workflow.stages.processing.logger")
    @patch("podcast_scraper.speaker_detectors.factory.create_speaker_detector")
    def test_create_speaker_detector_if_needed_handles_exception(
        self, mock_create_detector, mock_logger
    ):
        """Test that exception during creation is handled gracefully."""
        mock_create_detector.side_effect = RuntimeError("Init failed")

        result = processing._create_speaker_detector_if_needed(self.cfg, None)

        self.assertIsNone(result)
        mock_logger.error.assert_called()


@pytest.mark.unit
class TestDetectHostsFromFeed(unittest.TestCase):
    """Tests for _detect_hosts_from_feed helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.feed = models.RssFeed(
            title="Test Feed",
            authors=["Host 1"],
            items=[],
            base_url="https://example.com",
        )
        self.speaker_detector = Mock()
        self.speaker_detector.detect_hosts = Mock(return_value={"Host 1", "Host 2"})

    def test_detect_hosts_from_feed_success(self):
        """Test successful host detection from feed."""
        result = processing._detect_hosts_from_feed(self.feed, self.speaker_detector)

        self.assertEqual(result, {"Host 1", "Host 2"})
        self.speaker_detector.detect_hosts.assert_called_once_with(
            feed_title="Test Feed",
            feed_description=None,
            feed_authors=["Host 1"],
        )

    def test_detect_hosts_from_feed_without_authors(self):
        """Test host detection when feed has no authors."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )

        result = processing._detect_hosts_from_feed(feed, self.speaker_detector)

        self.assertEqual(result, {"Host 1", "Host 2"})
        self.speaker_detector.detect_hosts.assert_called_once_with(
            feed_title="Test Feed",
            feed_description=None,
            feed_authors=None,
        )


@pytest.mark.unit
class TestValidateHostsWithFirstEpisode(unittest.TestCase):
    """Tests for _validate_hosts_with_first_episode helper function."""

    def setUp(self):
        """Set up test fixtures."""
        import xml.etree.ElementTree as ET

        self.item = ET.Element("item")
        ET.SubElement(self.item, "title").text = "Episode 1"
        ET.SubElement(self.item, "description").text = "Episode description"

        self.episode = create_test_episode(idx=1, title="Episode 1")
        self.episode.item = self.item

        self.feed = models.RssFeed(
            title="Test Feed",
            authors=None,  # No authors, so validation will run
            items=[self.item],
            base_url="https://example.com",
        )

        self.speaker_detector = Mock()
        self.speaker_detector.detect_speakers = Mock(return_value=(["Host 1", "Host 2"], [], []))

    def test_validate_hosts_with_first_episode_skips_when_authors_exist(self):
        """Test that validation is skipped when feed has authors."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=["Host 1"],
            items=[self.item],
            base_url="https://example.com",
        )

        feed_hosts = {"Host 1", "Host 2"}
        result = processing._validate_hosts_with_first_episode(
            feed_hosts, feed, [self.episode], self.speaker_detector, None
        )

        self.assertEqual(result, feed_hosts)
        self.speaker_detector.detect_speakers.assert_not_called()

    def test_validate_hosts_with_first_episode_skips_when_no_hosts(self):
        """Test that validation is skipped when no hosts detected."""
        feed_hosts = set()
        result = processing._validate_hosts_with_first_episode(
            feed_hosts, self.feed, [self.episode], self.speaker_detector, None
        )

        self.assertEqual(result, feed_hosts)
        self.speaker_detector.detect_speakers.assert_not_called()

    def test_validate_hosts_with_first_episode_skips_when_no_episodes(self):
        """Test that validation is skipped when no episodes."""
        feed_hosts = {"Host 1", "Host 2"}
        result = processing._validate_hosts_with_first_episode(
            feed_hosts, self.feed, [], self.speaker_detector, None
        )

        self.assertEqual(result, feed_hosts)
        self.speaker_detector.detect_speakers.assert_not_called()

    def test_validate_hosts_with_first_episode_validates_successfully(self):
        """Test successful host validation."""
        feed_hosts = {"Host 1", "Host 2"}
        result = processing._validate_hosts_with_first_episode(
            feed_hosts, self.feed, [self.episode], self.speaker_detector, None
        )

        # Both hosts appear in first episode, so both should be validated
        self.assertEqual(result, {"Host 1", "Host 2"})
        self.speaker_detector.detect_speakers.assert_called_once()

    def test_validate_hosts_with_first_episode_filters_invalid_hosts(self):
        """Test that hosts not in first episode are filtered out."""
        feed_hosts = {"Host 1", "Host 2", "Host 3"}
        # Only Host 1 and Host 2 appear in first episode
        self.speaker_detector.detect_speakers = Mock(return_value=(["Host 1", "Host 2"], [], []))

        result = processing._validate_hosts_with_first_episode(
            feed_hosts, self.feed, [self.episode], self.speaker_detector, None
        )

        # Host 3 should be filtered out
        self.assertEqual(result, {"Host 1", "Host 2"})

    def test_validate_hosts_with_first_episode_with_pipeline_metrics(self):
        """Test validation with pipeline_metrics parameter."""
        from podcast_scraper.workflow import metrics

        pipeline_metrics = metrics.Metrics()
        feed_hosts = {"Host 1", "Host 2"}

        # Mock signature to include pipeline_metrics
        import inspect

        def mock_detect_speakers(
            episode_title, episode_description, known_hosts, pipeline_metrics=None
        ):
            return (["Host 1", "Host 2"], [], [])

        self.speaker_detector.detect_speakers = Mock(side_effect=mock_detect_speakers)
        self.speaker_detector.detect_speakers.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter("episode_title", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("episode_description", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("known_hosts", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("pipeline_metrics", inspect.Parameter.KEYWORD_ONLY),
            ]
        )

        result = processing._validate_hosts_with_first_episode(
            feed_hosts, self.feed, [self.episode], self.speaker_detector, pipeline_metrics
        )

        self.assertEqual(result, {"Host 1", "Host 2"})
        # Verify pipeline_metrics was passed
        call_kwargs = self.speaker_detector.detect_speakers.call_args[1]
        self.assertEqual(call_kwargs.get("pipeline_metrics"), pipeline_metrics)


@pytest.mark.unit
class TestFallbackToEpisodeAuthors(unittest.TestCase):
    """Tests for _fallback_to_episode_authors helper function."""

    def setUp(self):
        """Set up test fixtures."""
        import xml.etree.ElementTree as ET

        self.item1 = ET.Element("item")
        ET.SubElement(self.item1, "title").text = "Episode 1"
        author1 = ET.SubElement(self.item1, "{http://www.itunes.com/dtds/podcast-1.0.dtd}author")
        author1.text = "Author 1"

        self.item2 = ET.Element("item")
        ET.SubElement(self.item2, "title").text = "Episode 2"
        author2 = ET.SubElement(self.item2, "{http://www.itunes.com/dtds/podcast-1.0.dtd}author")
        author2.text = "Author 2"

        self.episode1 = create_test_episode(idx=1, title="Episode 1")
        self.episode1.item = self.item1

        self.episode2 = create_test_episode(idx=2, title="Episode 2")
        self.episode2.item = self.item2

    def test_fallback_to_episode_authors_disabled(self):
        """Test that empty set is returned when auto_speakers is disabled."""
        cfg = create_test_config(auto_speakers=False)

        result = processing._fallback_to_episode_authors(cfg, [self.episode1, self.episode2])

        self.assertEqual(result, set())

    def test_fallback_to_episode_authors_no_episodes(self):
        """Test that empty set is returned when no episodes."""
        cfg = create_test_config(auto_speakers=True)

        result = processing._fallback_to_episode_authors(cfg, [])

        self.assertEqual(result, set())

    @patch("podcast_scraper.rss.parser.extract_episode_authors")
    def test_fallback_to_episode_authors_success(self, mock_extract_authors):
        """Test successful extraction of episode authors."""
        mock_extract_authors.side_effect = [["Author 1"], ["Author 2"]]

        cfg = create_test_config(auto_speakers=True)

        result = processing._fallback_to_episode_authors(cfg, [self.episode1, self.episode2])

        self.assertEqual(result, {"Author 1", "Author 2"})
        self.assertEqual(mock_extract_authors.call_count, 2)

    @patch("podcast_scraper.rss.parser.extract_episode_authors")
    def test_fallback_to_episode_authors_filters_organizations(self, mock_extract_authors):
        """Test that organization names are filtered out."""
        # Organization: all caps, short, no spaces
        mock_extract_authors.side_effect = [["NPR"], ["John Doe"]]

        cfg = create_test_config(auto_speakers=True)

        result = processing._fallback_to_episode_authors(cfg, [self.episode1, self.episode2])

        # NPR should be filtered out (organization)
        self.assertEqual(result, {"John Doe"})

    @patch("podcast_scraper.rss.parser.extract_episode_authors")
    def test_fallback_to_episode_authors_limits_to_first_three(self, mock_extract_authors):
        """Test that only first 3 episodes are checked."""
        episodes = [create_test_episode(idx=i, title=f"Episode {i}") for i in range(1, 6)]
        mock_extract_authors.return_value = ["Author"]

        cfg = create_test_config(auto_speakers=True)

        processing._fallback_to_episode_authors(cfg, episodes)

        # Should only call extract_episode_authors 3 times (first 3 episodes)
        self.assertEqual(mock_extract_authors.call_count, 3)


@pytest.mark.unit
class TestLogDetectedHosts(unittest.TestCase):
    """Tests for _log_detected_hosts helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.feed = models.RssFeed(
            title="Test Feed",
            authors=["Host 1"],
            items=[],
            base_url="https://example.com",
        )

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_log_detected_hosts_from_feed_authors(self, mock_logger):
        """Test logging hosts detected from feed authors."""
        cached_hosts = {"Host 1", "Host 2"}
        episode_authors = set()

        processing._log_detected_hosts(cached_hosts, self.feed, episode_authors, None)

        # Verify log was called with correct source
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        feed_log_found = any("RSS author tags" in str(call) for call in log_calls)
        self.assertTrue(feed_log_found, "Should log hosts from RSS author tags")

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_log_detected_hosts_from_episode_authors(self, mock_logger):
        """Test logging hosts detected from episode authors."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )
        cached_hosts = {"Host 1"}
        episode_authors = {"Host 1"}

        processing._log_detected_hosts(cached_hosts, feed, episode_authors, None)

        # Verify log was called with episode-level authors source
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        episode_log_found = any("episode-level authors" in str(call) for call in log_calls)
        self.assertTrue(episode_log_found, "Should log hosts from episode-level authors")

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_log_detected_hosts_from_config(self, mock_logger):
        """Test logging hosts detected from config known_hosts."""
        cfg = create_test_config(known_hosts=["Host 1", "Host 2"])
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )
        cached_hosts = {"Host 1", "Host 2"}
        episode_authors = set()

        processing._log_detected_hosts(cached_hosts, feed, episode_authors, cfg)

        # Verify log was called with config source
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        config_log_found = any("config known_hosts" in str(call) for call in log_calls)
        self.assertTrue(config_log_found, "Should log hosts from config known_hosts")

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_log_detected_hosts_from_ner(self, mock_logger):
        """Test logging hosts detected from NER (default source)."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )
        cached_hosts = {"Host 1"}
        episode_authors = set()
        cfg = create_test_config(auto_speakers=True)

        processing._log_detected_hosts(cached_hosts, feed, episode_authors, cfg)

        # Verify log was called with feed metadata (NER) source
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        ner_log_found = any("feed metadata (NER)" in str(call) for call in log_calls)
        self.assertTrue(ner_log_found, "Should log hosts from feed metadata (NER)")

    @patch("podcast_scraper.workflow.stages.processing.logger")
    def test_log_detected_hosts_empty(self, mock_logger):
        """Test logging when no hosts detected."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=None,
            items=[],
            base_url="https://example.com",
        )
        cached_hosts = set()
        episode_authors = set()
        cfg = create_test_config(auto_speakers=True)

        processing._log_detected_hosts(cached_hosts, feed, episode_authors, cfg)

        # Should log debug message about no hosts
        log_calls = [str(call) for call in mock_logger.debug.call_args_list]
        no_hosts_log_found = any("No hosts detected" in str(call) for call in log_calls)
        self.assertTrue(no_hosts_log_found, "Should log debug message when no hosts")


@pytest.mark.unit
class TestSetupProcessingResources(unittest.TestCase):
    """Tests for setup_processing_resources function."""

    def test_setup_processing_resources_single_worker(self):
        """Test setup with single worker (no locks needed)."""
        cfg = create_test_config(workers=1, transcription_parallelism=1, processing_parallelism=1)

        result = processing.setup_processing_resources(cfg)

        self.assertIsNotNone(result)
        self.assertIsNone(result.processing_jobs_lock)
        self.assertIsNotNone(result.processing_complete_event)

    def test_setup_processing_resources_multiple_workers(self):
        """Test setup with multiple workers (locks needed)."""
        cfg = create_test_config(workers=4)

        result = processing.setup_processing_resources(cfg)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.processing_jobs_lock)
        self.assertIsNotNone(result.processing_complete_event)

    def test_setup_processing_resources_with_parallelism(self):
        """Test setup with transcription parallelism enabled."""
        cfg = create_test_config(workers=1, transcription_parallelism=2)

        result = processing.setup_processing_resources(cfg)

        # Should have lock even with workers=1 if parallelism > 1
        self.assertIsNotNone(result.processing_jobs_lock)
