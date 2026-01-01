#!/usr/bin/env python3
"""Tests for workflow helper functions.

These tests verify pure helper functions in workflow.py that
can be tested without I/O operations or complex orchestration.
"""

import logging
import os
import sys
import threading
import unittest
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper import config, metrics, models, workflow


class TestUpdateMetricSafely(unittest.TestCase):
    """Tests for _update_metric_safely function."""

    def test_update_metric_without_lock(self):
        """Test updating metric without lock."""
        pipeline_metrics = metrics.Metrics()
        workflow._update_metric_safely(pipeline_metrics, "transcripts_downloaded", 5)
        self.assertEqual(pipeline_metrics.transcripts_downloaded, 5)

    def test_update_metric_with_lock(self):
        """Test updating metric with lock."""
        pipeline_metrics = metrics.Metrics()
        lock = threading.Lock()
        workflow._update_metric_safely(pipeline_metrics, "transcripts_downloaded", 3, lock)
        self.assertEqual(pipeline_metrics.transcripts_downloaded, 3)

    def test_update_metric_adds_value(self):
        """Test that metric value is added, not replaced."""
        pipeline_metrics = metrics.Metrics()
        workflow._update_metric_safely(pipeline_metrics, "transcripts_downloaded", 5)
        workflow._update_metric_safely(pipeline_metrics, "transcripts_downloaded", 3)
        self.assertEqual(pipeline_metrics.transcripts_downloaded, 8)

    def test_update_metric_thread_safety(self):
        """Test that lock ensures thread safety."""
        pipeline_metrics = metrics.Metrics()
        lock = threading.Lock()

        def update_metric():
            for _ in range(10):
                workflow._update_metric_safely(pipeline_metrics, "transcripts_downloaded", 1, lock)

        threads = [threading.Thread(target=update_metric) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 50 total updates (5 threads * 10 updates each)
        self.assertEqual(pipeline_metrics.transcripts_downloaded, 50)


class TestExtractFeedMetadataForGeneration(unittest.TestCase):
    """Tests for _extract_feed_metadata_for_generation function."""

    def test_extract_metadata_when_enabled(self):
        """Test extracting metadata when generate_metadata is enabled."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", generate_metadata=True)
        feed = models.RssFeed(
            title="Test Feed",
            authors=[],
            items=[],
            base_url="https://example.com",
        )
        rss_bytes = b"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <description>Feed description</description>
            </channel>
        </rss>"""

        result = workflow._extract_feed_metadata_for_generation(cfg, feed, rss_bytes)
        self.assertIsNotNone(result.description)
        self.assertIn("Feed description", result.description)

    def test_extract_metadata_when_disabled(self):
        """Test that metadata extraction returns None when disabled."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", generate_metadata=False)
        feed = models.RssFeed(
            title="Test Feed",
            authors=[],
            items=[],
            base_url="https://example.com",
        )
        rss_bytes = b"<rss></rss>"

        result = workflow._extract_feed_metadata_for_generation(cfg, feed, rss_bytes)
        self.assertIsNone(result.description)
        self.assertIsNone(result.image_url)
        self.assertIsNone(result.last_updated)

    def test_extract_metadata_with_empty_bytes(self):
        """Test that empty RSS bytes returns None metadata."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", generate_metadata=True)
        feed = models.RssFeed(
            title="Test Feed",
            authors=[],
            items=[],
            base_url="https://example.com",
        )

        result = workflow._extract_feed_metadata_for_generation(cfg, feed, b"")
        self.assertIsNone(result.description)
        self.assertIsNone(result.image_url)
        self.assertIsNone(result.last_updated)

    def test_extract_metadata_handles_exceptions(self):
        """Test that exceptions during extraction are handled gracefully."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", generate_metadata=True)
        feed = models.RssFeed(
            title="Test Feed",
            authors=[],
            items=[],
            base_url="https://example.com",
        )
        # Invalid XML that will cause parse error
        rss_bytes = b"<invalid>xml"

        result = workflow._extract_feed_metadata_for_generation(cfg, feed, rss_bytes)
        # Should return None metadata on error
        self.assertIsNone(result.description)
        self.assertIsNone(result.image_url)
        self.assertIsNone(result.last_updated)


class TestPrepareEpisodesFromFeed(unittest.TestCase):
    """Tests for _prepare_episodes_from_feed function."""

    def test_prepare_episodes_basic(self):
        """Test preparing episodes from feed."""
        # Bandit: tests construct safe XML elements
        import xml.etree.ElementTree as ET  # nosec B405

        item1 = ET.Element("item")
        title1 = ET.SubElement(item1, "title")
        title1.text = "Episode 1"
        item2 = ET.Element("item")
        title2 = ET.SubElement(item2, "title")
        title2.text = "Episode 2"

        feed = models.RssFeed(
            title="Test Feed",
            authors=[],
            items=[item1, item2],
            base_url="https://example.com",
        )
        cfg = config.Config(rss_url="https://example.com/feed.xml")

        episodes = workflow._prepare_episodes_from_feed(feed, cfg)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].idx, 1)
        self.assertEqual(episodes[0].title, "Episode 1")
        self.assertEqual(episodes[1].idx, 2)
        self.assertEqual(episodes[1].title, "Episode 2")

    def test_prepare_episodes_with_max_episodes(self):
        """Test that max_episodes limits the number of episodes."""
        # Bandit: tests construct safe XML elements
        import xml.etree.ElementTree as ET  # nosec B405

        items = []
        for i in range(10):
            item = ET.Element("item")
            title = ET.SubElement(item, "title")
            title.text = f"Episode {i+1}"
            items.append(item)

        feed = models.RssFeed(
            title="Test Feed",
            authors=[],
            items=items,
            base_url="https://example.com",
        )
        cfg = config.Config(rss_url="https://example.com/feed.xml", max_episodes=5)

        episodes = workflow._prepare_episodes_from_feed(feed, cfg)
        self.assertEqual(len(episodes), 5)
        self.assertEqual(episodes[0].idx, 1)
        self.assertEqual(episodes[4].idx, 5)

    def test_prepare_episodes_with_no_max(self):
        """Test that all episodes are prepared when max_episodes is None."""
        # Bandit: tests construct safe XML elements
        import xml.etree.ElementTree as ET  # nosec B405

        items = []
        for i in range(10):
            item = ET.Element("item")
            title = ET.SubElement(item, "title")
            title.text = f"Episode {i+1}"
            items.append(item)

        feed = models.RssFeed(
            title="Test Feed",
            authors=[],
            items=items,
            base_url="https://example.com",
        )
        cfg = config.Config(rss_url="https://example.com/feed.xml", max_episodes=None)

        episodes = workflow._prepare_episodes_from_feed(feed, cfg)
        self.assertEqual(len(episodes), 10)

    def test_prepare_episodes_empty_feed(self):
        """Test preparing episodes from empty feed."""
        feed = models.RssFeed(
            title="Test Feed",
            authors=[],
            items=[],
            base_url="https://example.com",
        )
        cfg = config.Config(rss_url="https://example.com/feed.xml")

        episodes = workflow._prepare_episodes_from_feed(feed, cfg)
        self.assertEqual(len(episodes), 0)


class TestSetupProcessingResources(unittest.TestCase):
    """Tests for _setup_processing_resources function."""

    def test_setup_resources_single_worker(self):
        """Test setting up resources for single worker (no lock needed)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            workers=1,
            transcription_parallelism=1,
            processing_parallelism=1,
        )

        result = workflow._setup_processing_resources(cfg)
        self.assertIsNotNone(result.processing_jobs)
        self.assertIsNone(result.processing_jobs_lock)  # No lock for single worker
        self.assertIsNotNone(result.processing_complete_event)

    def test_setup_resources_multiple_workers(self):
        """Test setting up resources for multiple workers (lock needed)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            workers=4,
            transcription_parallelism=1,
            processing_parallelism=1,
        )

        result = workflow._setup_processing_resources(cfg)
        self.assertIsNotNone(result.processing_jobs)
        self.assertIsNotNone(result.processing_jobs_lock)  # Lock needed for multiple workers
        self.assertIsNotNone(result.processing_complete_event)

    def test_setup_resources_with_transcription_parallelism(self):
        """Test that transcription_parallelism > 1 requires lock."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            workers=1,
            transcription_parallelism=2,
            processing_parallelism=1,
        )

        result = workflow._setup_processing_resources(cfg)
        self.assertIsNotNone(result.processing_jobs_lock)

    def test_setup_resources_with_processing_parallelism(self):
        """Test that processing_parallelism > 1 requires lock."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            workers=1,
            transcription_parallelism=1,
            processing_parallelism=2,
        )

        result = workflow._setup_processing_resources(cfg)
        self.assertIsNotNone(result.processing_jobs_lock)


class TestGeneratePipelineSummary(unittest.TestCase):
    """Tests for _generate_pipeline_summary function."""

    def test_generate_summary_dry_run(self):
        """Test generating summary in dry-run mode."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            dry_run=True,
            transcribe_missing=True,
        )
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=None,
            temp_dir=None,
            transcription_jobs=[Mock(), Mock()],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        pipeline_metrics = metrics.Metrics()

        count, summary = workflow._generate_pipeline_summary(
            cfg,
            saved=5,
            transcription_resources=transcription_resources,
            effective_output_dir="./output",
            pipeline_metrics=pipeline_metrics,
        )

        self.assertEqual(count, 7)  # 5 downloads + 2 transcriptions
        self.assertIn("Dry run complete", summary)
        self.assertIn("Direct downloads planned: 5", summary)
        self.assertIn("Whisper transcriptions planned: 2", summary)

    def test_generate_summary_normal_mode(self):
        """Test generating summary in normal mode."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            dry_run=False,
            generate_metadata=True,
            generate_summaries=True,
        )
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=None,
            temp_dir=None,
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.transcripts_downloaded = 8
        pipeline_metrics.transcripts_transcribed = 2
        pipeline_metrics.metadata_files_generated = 10
        pipeline_metrics.episodes_summarized = 5

        count, summary = workflow._generate_pipeline_summary(
            cfg,
            saved=10,
            transcription_resources=transcription_resources,
            effective_output_dir="./output",
            pipeline_metrics=pipeline_metrics,
        )

        self.assertEqual(count, 10)
        self.assertIn("Done. transcripts_saved=10", summary)
        self.assertIn("Transcripts downloaded: 8", summary)
        self.assertIn("Episodes transcribed: 2", summary)
        self.assertIn("Metadata files generated: 10", summary)
        self.assertIn("Episodes summarized: 5", summary)

    def test_generate_summary_with_errors(self):
        """Test generating summary with error count."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", dry_run=False)
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=None,
            temp_dir=None,
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.errors_total = 3

        count, summary = workflow._generate_pipeline_summary(
            cfg,
            saved=10,
            transcription_resources=transcription_resources,
            effective_output_dir="./output",
            pipeline_metrics=pipeline_metrics,
        )

        self.assertIn("Errors: 3", summary)

    def test_generate_summary_with_skipped_episodes(self):
        """Test generating summary with skipped episodes."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", dry_run=False)
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=None,
            temp_dir=None,
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.episodes_skipped_total = 2

        count, summary = workflow._generate_pipeline_summary(
            cfg,
            saved=10,
            transcription_resources=transcription_resources,
            effective_output_dir="./output",
            pipeline_metrics=pipeline_metrics,
        )

        self.assertIn("Episodes skipped: 2", summary)

    def test_generate_summary_with_performance_metrics(self):
        """Test generating summary with performance metrics."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", dry_run=False)
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=None,
            temp_dir=None,
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.record_download_media_time(2.5)
        pipeline_metrics.record_transcribe_time(45.0)
        pipeline_metrics.record_extract_names_time(1.2)
        pipeline_metrics.record_summarize_time(30.0)

        count, summary = workflow._generate_pipeline_summary(
            cfg,
            saved=10,
            transcription_resources=transcription_resources,
            effective_output_dir="./output",
            pipeline_metrics=pipeline_metrics,
        )

        self.assertIn("Average download time", summary)
        self.assertIn("Average transcription time", summary)
        self.assertIn("Average name extraction time", summary)
        self.assertIn("Average summary time", summary)


class TestCallGenerateMetadata(unittest.TestCase):
    """Tests for _call_generate_metadata function."""

    @patch("podcast_scraper.workflow._generate_episode_metadata")
    def test_call_generate_metadata_basic(self, mock_generate):
        """Test basic metadata generation call."""
        episode = Mock(spec=models.Episode)
        feed = Mock(spec=models.RssFeed)
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        feed_metadata = workflow._FeedMetadata("Feed desc", "https://example.com/image.jpg", None)
        host_result = workflow._HostDetectionResult(set(), None, None)

        workflow._call_generate_metadata(
            episode=episode,
            feed=feed,
            cfg=cfg,
            effective_output_dir="./output",
            run_suffix="test",
            transcript_path="./transcript.txt",
            transcript_source="direct_download",
            whisper_model=None,
            feed_metadata=feed_metadata,
            host_detection_result=host_result,
            detected_names=["Guest Name"],
            summary_provider=Mock(),
            pipeline_metrics=Mock(),
        )

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        self.assertEqual(call_kwargs["feed"], feed)
        self.assertEqual(call_kwargs["episode"], episode)
        self.assertEqual(call_kwargs["feed_url"], "https://example.com/feed.xml")
        self.assertEqual(call_kwargs["output_dir"], "./output")
        self.assertEqual(call_kwargs["run_suffix"], "test")
        self.assertEqual(call_kwargs["transcript_file_path"], "./transcript.txt")
        self.assertEqual(call_kwargs["transcript_source"], "direct_download")

    @patch("podcast_scraper.workflow._generate_episode_metadata")
    def test_call_generate_metadata_with_none_values(self, mock_generate):
        """Test metadata generation call with None values."""
        episode = Mock(spec=models.Episode)
        feed = Mock(spec=models.RssFeed)
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        feed_metadata = workflow._FeedMetadata(None, None, None)
        host_result = workflow._HostDetectionResult(None, None, None)

        workflow._call_generate_metadata(
            episode=episode,
            feed=feed,
            cfg=cfg,
            effective_output_dir="./output",
            run_suffix=None,
            transcript_path=None,
            transcript_source=None,
            whisper_model=None,
            feed_metadata=feed_metadata,
            host_detection_result=host_result,
            detected_names=None,
            summary_provider=Mock(),
            pipeline_metrics=None,
        )

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        self.assertIsNone(call_kwargs["run_suffix"])
        self.assertIsNone(call_kwargs["transcript_file_path"])
        self.assertIsNone(call_kwargs["transcript_source"])
        self.assertIsNone(call_kwargs["detected_hosts"])
        self.assertIsNone(call_kwargs["detected_guests"])

    @patch("podcast_scraper.workflow._generate_episode_metadata")
    def test_call_generate_metadata_with_cached_hosts(self, mock_generate):
        """Test metadata generation call with cached hosts."""
        episode = Mock(spec=models.Episode)
        feed = Mock(spec=models.RssFeed)
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        feed_metadata = workflow._FeedMetadata(None, None, None)
        host_result = workflow._HostDetectionResult({"Host1", "Host2"}, None, None)

        workflow._call_generate_metadata(
            episode=episode,
            feed=feed,
            cfg=cfg,
            effective_output_dir="./output",
            run_suffix=None,
            transcript_path=None,
            transcript_source=None,
            whisper_model="base",
            feed_metadata=feed_metadata,
            host_detection_result=host_result,
            detected_names=["Guest"],
            summary_provider=Mock(),
            pipeline_metrics=Mock(),
        )

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        # cached_hosts is a set, so convert to sorted list for comparison
        self.assertEqual(sorted(call_kwargs["detected_hosts"]), sorted(["Host1", "Host2"]))
        self.assertEqual(call_kwargs["detected_guests"], ["Guest"])
        self.assertEqual(call_kwargs["whisper_model"], "base")


class TestApplyLogLevel(unittest.TestCase):
    """Tests for apply_log_level function."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clear handlers after each test
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

    def test_apply_log_level_debug(self):
        """Test applying DEBUG log level."""
        workflow.apply_log_level("DEBUG")
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.DEBUG)

    def test_apply_log_level_info(self):
        """Test applying INFO log level."""
        workflow.apply_log_level("INFO")
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.INFO)

    def test_apply_log_level_invalid(self):
        """Test applying invalid log level raises ValueError."""
        with self.assertRaises(ValueError) as context:
            workflow.apply_log_level("INVALID")
        self.assertIn("Invalid log level", str(context.exception))

    def test_apply_log_level_with_file(self):
        """Test applying log level with log file (simplified - just verify it doesn't crash)."""
        # Clear handlers before test
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Use tempfile to avoid filesystem I/O restrictions
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
            log_file = tmp.name

        try:
            # This will create a real file handler, which is allowed in temp directories
            workflow.apply_log_level("INFO", log_file=log_file)

            # Should have at least one handler (file handler)
            self.assertGreaterEqual(len(root_logger.handlers), 1)
            # Check that a file handler was added
            file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
            self.assertGreaterEqual(len(file_handlers), 1)
        finally:
            # Clean up
            import os

            if os.path.exists(log_file):
                os.unlink(log_file)


class TestSetupPipelineEnvironment(unittest.TestCase):
    """Tests for _setup_pipeline_environment function."""

    @patch("podcast_scraper.workflow.filesystem.setup_output_directory")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    @patch("os.makedirs")
    def test_setup_environment_basic(self, mock_makedirs, mock_rmtree, mock_exists, mock_setup):
        """Test basic pipeline environment setup."""
        mock_setup.return_value = ("./output", None)
        mock_exists.return_value = False
        cfg = config.Config(rss_url="https://example.com/feed.xml", output_dir="./output")

        output_dir, run_suffix = workflow._setup_pipeline_environment(cfg)

        self.assertEqual(output_dir, "./output")
        self.assertIsNone(run_suffix)
        mock_setup.assert_called_once_with(cfg)
        mock_makedirs.assert_called_once_with("./output", exist_ok=True)

    @patch("podcast_scraper.workflow.filesystem.setup_output_directory")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    @patch("os.makedirs")
    def test_setup_environment_with_clean_output(
        self, mock_makedirs, mock_rmtree, mock_exists, mock_setup
    ):
        """Test pipeline environment setup with clean_output enabled."""
        mock_setup.return_value = ("./output", None)
        mock_exists.return_value = True
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            output_dir="./output",
            clean_output=True,
            dry_run=False,
        )

        output_dir, run_suffix = workflow._setup_pipeline_environment(cfg)

        mock_rmtree.assert_called_once_with("./output")
        mock_makedirs.assert_called_once_with("./output", exist_ok=True)

    @patch("podcast_scraper.workflow.filesystem.setup_output_directory")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    @patch("os.makedirs")
    def test_setup_environment_dry_run(self, mock_makedirs, mock_rmtree, mock_exists, mock_setup):
        """Test pipeline environment setup in dry-run mode."""
        mock_setup.return_value = ("./output", "test_run")
        mock_exists.return_value = True
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            output_dir="./output",
            clean_output=True,
            dry_run=True,
        )

        output_dir, run_suffix = workflow._setup_pipeline_environment(cfg)

        self.assertEqual(output_dir, "./output")
        self.assertEqual(run_suffix, "test_run")
        # Should not create directory in dry-run
        mock_makedirs.assert_not_called()
        # Should not remove directory in dry-run (just log)
        mock_rmtree.assert_not_called()

    @patch("podcast_scraper.workflow.filesystem.setup_output_directory")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_setup_environment_clean_output_failure(self, mock_rmtree, mock_exists, mock_setup):
        """Test that cleanup failure raises RuntimeError."""
        mock_setup.return_value = ("./output", None)
        mock_exists.return_value = True
        mock_rmtree.side_effect = OSError("Permission denied")
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            output_dir="./output",
            clean_output=True,
            dry_run=False,
        )

        with self.assertRaises(RuntimeError) as context:
            workflow._setup_pipeline_environment(cfg)
        self.assertIn("Failed to clean output directory", str(context.exception))

    @patch("podcast_scraper.workflow.filesystem.setup_output_directory")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_setup_environment_with_run_suffix(self, mock_makedirs, mock_exists, mock_setup):
        """Test pipeline environment setup with run suffix."""
        mock_setup.return_value = ("./output/run123", "run123")
        mock_exists.return_value = False
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", output_dir="./output", run_id="run123"
        )

        output_dir, run_suffix = workflow._setup_pipeline_environment(cfg)

        self.assertEqual(output_dir, "./output/run123")
        self.assertEqual(run_suffix, "run123")
        mock_makedirs.assert_called_once_with("./output/run123", exist_ok=True)


class TestFetchAndParseFeed(unittest.TestCase):
    """Tests for _fetch_and_parse_feed function."""

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.rss_parser.parse_rss_items")
    def test_fetch_and_parse_feed_success(self, mock_parse, mock_fetch):
        """Test successful feed fetch and parse."""
        mock_response = Mock()
        mock_response.content = b"<rss><channel><title>Test Feed</title></channel></rss>"
        mock_response.url = "https://example.com/feed.xml"
        mock_fetch.return_value = mock_response
        mock_parse.return_value = ("Test Feed", ["Author"], [])

        cfg = config.Config(rss_url="https://example.com/feed.xml")
        feed, rss_bytes = workflow._fetch_and_parse_feed(cfg)

        self.assertEqual(feed.title, "Test Feed")
        self.assertEqual(feed.base_url, "https://example.com/feed.xml")
        self.assertEqual(rss_bytes, b"<rss><channel><title>Test Feed</title></channel></rss>")
        mock_fetch.assert_called_once_with(
            "https://example.com/feed.xml", cfg.user_agent, cfg.timeout, stream=False
        )
        mock_parse.assert_called_once_with(rss_bytes)
        mock_response.close.assert_called_once()

    def test_fetch_and_parse_feed_no_url(self):
        """Test that missing RSS URL raises ValueError."""
        cfg = config.Config(rss_url=None)

        with self.assertRaises(ValueError) as context:
            workflow._fetch_and_parse_feed(cfg)
        self.assertIn("RSS URL is required", str(context.exception))

    @patch("podcast_scraper.downloader.fetch_url")
    def test_fetch_and_parse_feed_fetch_failure(self, mock_fetch):
        """Test that fetch failure raises ValueError."""
        mock_fetch.return_value = None
        cfg = config.Config(rss_url="https://example.com/feed.xml")

        with self.assertRaises(ValueError) as context:
            workflow._fetch_and_parse_feed(cfg)
        self.assertIn("Failed to fetch RSS feed", str(context.exception))

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.rss_parser.parse_rss_items")
    def test_fetch_and_parse_feed_parse_failure(self, mock_parse, mock_fetch):
        """Test that parse failure raises ValueError."""
        mock_response = Mock()
        mock_response.content = b"<invalid>xml"
        mock_response.url = "https://example.com/feed.xml"
        mock_fetch.return_value = mock_response
        mock_parse.side_effect = ValueError("Invalid XML")

        cfg = config.Config(rss_url="https://example.com/feed.xml")

        with self.assertRaises(ValueError) as context:
            workflow._fetch_and_parse_feed(cfg)
        self.assertIn("Failed to parse RSS XML", str(context.exception))
        mock_response.close.assert_called_once()

    @patch("podcast_scraper.downloader.fetch_url")
    @patch("podcast_scraper.rss_parser.parse_rss_items")
    def test_fetch_and_parse_feed_uses_response_url(self, mock_parse, mock_fetch):
        """Test that response URL is used as base_url when available."""
        mock_response = Mock()
        mock_response.content = b"<rss></rss>"
        mock_response.url = "https://redirected.com/feed.xml"
        mock_fetch.return_value = mock_response
        mock_parse.return_value = ("Feed", [], [])

        cfg = config.Config(rss_url="https://example.com/feed.xml")
        feed, rss_bytes = workflow._fetch_and_parse_feed(cfg)

        self.assertEqual(feed.base_url, "https://redirected.com/feed.xml")
        mock_response.close.assert_called_once()


class TestDetectFeedHostsAndPatterns(unittest.TestCase):
    """Tests for _detect_feed_hosts_and_patterns function."""

    @patch("podcast_scraper.workflow.create_speaker_detector")
    def test_detect_hosts_auto_speakers_disabled(self, mock_create):
        """Test that host detection is skipped when auto_speakers is disabled."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", auto_speakers=False)
        feed = models.RssFeed(title="Feed", authors=[], items=[], base_url="https://example.com")
        episodes = []

        result = workflow._detect_feed_hosts_and_patterns(cfg, feed, episodes)

        self.assertEqual(len(result.cached_hosts), 0)
        self.assertIsNone(result.heuristics)
        mock_create.assert_not_called()

    @patch("podcast_scraper.workflow.create_speaker_detector")
    def test_detect_hosts_cache_disabled(self, mock_create):
        """Test that host detection is skipped when cache_detected_hosts is disabled."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            auto_speakers=True,
            cache_detected_hosts=False,
        )
        feed = models.RssFeed(title="Feed", authors=[], items=[], base_url="https://example.com")
        episodes = []

        result = workflow._detect_feed_hosts_and_patterns(cfg, feed, episodes)

        self.assertEqual(len(result.cached_hosts), 0)
        mock_create.assert_not_called()

    @patch("podcast_scraper.workflow.create_speaker_detector")
    @patch("podcast_scraper.workflow.extract_episode_description")
    def test_detect_hosts_from_authors(self, mock_extract, mock_create):
        """Test host detection from RSS author tags."""
        mock_detector = Mock()
        mock_detector.detect_hosts.return_value = {"Host1", "Host2"}
        mock_create.return_value = mock_detector

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            auto_speakers=True,
            cache_detected_hosts=True,
        )
        feed = models.RssFeed(
            title="Feed",
            authors=["Host1", "Host2"],
            items=[],
            base_url="https://example.com",
        )
        episodes = []

        result = workflow._detect_feed_hosts_and_patterns(cfg, feed, episodes)

        self.assertEqual(result.cached_hosts, {"Host1", "Host2"})
        mock_detector.initialize.assert_called_once()
        mock_detector.detect_hosts.assert_called_once()

    @patch("podcast_scraper.workflow.create_speaker_detector")
    def test_detect_hosts_initialization_failure(self, mock_create):
        """Test that initialization failure returns empty result."""
        mock_create.side_effect = Exception("Failed to initialize")

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            auto_speakers=True,
            cache_detected_hosts=True,
        )
        feed = models.RssFeed(title="Feed", authors=[], items=[], base_url="https://example.com")
        episodes = []

        result = workflow._detect_feed_hosts_and_patterns(cfg, feed, episodes)

        self.assertEqual(len(result.cached_hosts), 0)
        self.assertIsNone(result.heuristics)
        # Provider is None when initialization fails
        self.assertIsNone(result.speaker_detector)


class TestSetupTranscriptionResources(unittest.TestCase):
    """Tests for _setup_transcription_resources function."""

    @patch("podcast_scraper.workflow.create_transcription_provider")
    @patch("os.path.join")
    @patch("os.makedirs")
    def test_setup_transcription_resources_basic(self, mock_makedirs, mock_join, mock_create):
        """Test basic transcription resources setup."""
        mock_join.return_value = "./output/temp"
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            dry_run=False,
            workers=1,
        )

        result = workflow._setup_transcription_resources(cfg, "./output")

        self.assertIsNotNone(result.transcription_provider)
        self.assertEqual(result.temp_dir, "./output/temp")
        self.assertIsNone(result.transcription_jobs_lock)
        self.assertIsNone(result.saved_counter_lock)
        mock_create.assert_called_once()
        mock_makedirs.assert_called_once_with("./output/temp", exist_ok=True)

    @patch("podcast_scraper.workflow.create_transcription_provider")
    def test_setup_transcription_resources_dry_run(self, mock_create):
        """Test transcription resources setup in dry-run mode."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            dry_run=True,
            workers=1,
        )

        result = workflow._setup_transcription_resources(cfg, "./output")

        # Should not initialize provider in dry-run
        mock_create.assert_not_called()
        self.assertIsNone(result.transcription_provider)
        self.assertIsNotNone(result.temp_dir)

    @patch("podcast_scraper.workflow.create_transcription_provider")
    def test_setup_transcription_resources_transcribe_disabled(self, mock_create):
        """Test transcription resources setup when transcribe_missing is disabled."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=False,
            workers=1,
        )

        result = workflow._setup_transcription_resources(cfg, "./output")

        mock_create.assert_not_called()
        self.assertIsNone(result.transcription_provider)
        self.assertIsNone(result.temp_dir)

    @patch("podcast_scraper.workflow.create_transcription_provider")
    @patch("os.path.join")
    @patch("os.makedirs")
    def test_setup_transcription_resources_multiple_workers(
        self, mock_makedirs, mock_join, mock_create
    ):
        """Test transcription resources setup with multiple workers (locks needed)."""
        mock_join.return_value = "./output/temp"
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            dry_run=False,
            workers=4,
        )

        result = workflow._setup_transcription_resources(cfg, "./output")

        self.assertIsNotNone(result.transcription_jobs_lock)
        self.assertIsNotNone(result.saved_counter_lock)

    @patch("podcast_scraper.workflow.create_transcription_provider")
    @patch("os.path.join")
    @patch("os.makedirs")
    def test_setup_transcription_resources_initialization_failure(
        self, mock_makedirs, mock_join, mock_create
    ):
        """Test that provider initialization failure is handled gracefully."""
        mock_join.return_value = "./output/temp"
        mock_provider = Mock()
        mock_provider.initialize.side_effect = Exception("Failed to initialize")
        mock_create.return_value = mock_provider

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            dry_run=False,
            workers=1,
        )

        result = workflow._setup_transcription_resources(cfg, "./output")

        # Should return None provider on failure
        self.assertIsNone(result.transcription_provider)
