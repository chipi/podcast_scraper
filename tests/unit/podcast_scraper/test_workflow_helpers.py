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

# Import from parent conftest explicitly to avoid conflicts
import importlib.util
from pathlib import Path

from podcast_scraper import config, metrics, models, workflow

parent_tests_dir = Path(__file__).parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

parent_conftest_path = parent_tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

create_test_config = parent_conftest.create_test_config
create_test_episode = parent_conftest.create_test_episode
create_test_feed = parent_conftest.create_test_feed


class TestInitializeMLEnvironment(unittest.TestCase):
    """Tests for _initialize_ml_environment function."""

    def setUp(self):
        """Set up test fixtures."""
        # Save original environment
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_initialize_ml_environment_sets_hf_hub_disable_progress_bars(self):
        """Test _initialize_ml_environment sets HF_HUB_DISABLE_PROGRESS_BARS."""
        # Ensure it's not set
        if "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ:
            del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]

        workflow._initialize_ml_environment()

        self.assertEqual(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")

    def test_initialize_ml_environment_respects_existing_hf_hub_disable_progress_bars(self):
        """Test _initialize_ml_environment respects existing HF_HUB_DISABLE_PROGRESS_BARS."""
        # Set it to a different value
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

        workflow._initialize_ml_environment()

        # Should not override existing value
        self.assertEqual(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "0")

    def test_initialize_ml_environment_does_not_set_thread_vars(self):
        """Test _initialize_ml_environment does not set thread-related environment variables."""
        # Ensure they're not set
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"]:
            if var in os.environ:
                del os.environ[var]

        workflow._initialize_ml_environment()

        # Should not be set (allows full CPU utilization in production)
        self.assertNotIn("OMP_NUM_THREADS", os.environ)
        self.assertNotIn("MKL_NUM_THREADS", os.environ)
        self.assertNotIn("TORCH_NUM_THREADS", os.environ)


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
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
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
        self.assertEqual(call_kwargs["whisper_model"], config.TEST_DEFAULT_WHISPER_MODEL)


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
        # Should create base directory and subdirectories (transcripts/, metadata/)
        self.assertEqual(mock_makedirs.call_count, 3)
        mock_makedirs.assert_any_call("./output", exist_ok=True)
        mock_makedirs.assert_any_call("./output/transcripts", exist_ok=True)
        mock_makedirs.assert_any_call("./output/metadata", exist_ok=True)

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
        # Should create base directory and subdirectories (transcripts/, metadata/)
        self.assertEqual(mock_makedirs.call_count, 3)
        mock_makedirs.assert_any_call("./output", exist_ok=True)
        mock_makedirs.assert_any_call("./output/transcripts", exist_ok=True)
        mock_makedirs.assert_any_call("./output/metadata", exist_ok=True)

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
        # Should create base directory and subdirectories (transcripts/, metadata/)
        self.assertEqual(mock_makedirs.call_count, 3)
        mock_makedirs.assert_any_call("./output/run123", exist_ok=True)
        mock_makedirs.assert_any_call("./output/run123/transcripts", exist_ok=True)
        mock_makedirs.assert_any_call("./output/run123/metadata", exist_ok=True)


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


class TestCleanupPipeline(unittest.TestCase):
    """Tests for _cleanup_pipeline function."""

    @patch("podcast_scraper.workflow.shutil.rmtree")
    @patch("podcast_scraper.workflow.os.path.exists")
    def test_cleanup_pipeline_success(self, mock_exists, mock_rmtree):
        """Test successful pipeline cleanup."""
        mock_exists.return_value = True

        workflow._cleanup_pipeline("/tmp/test_dir")

        mock_exists.assert_called_once_with("/tmp/test_dir")
        mock_rmtree.assert_called_once_with("/tmp/test_dir")

    @patch("podcast_scraper.workflow.shutil.rmtree")
    @patch("podcast_scraper.workflow.os.path.exists")
    def test_cleanup_pipeline_dir_not_exists(self, mock_exists, mock_rmtree):
        """Test cleanup when directory doesn't exist."""
        mock_exists.return_value = False

        workflow._cleanup_pipeline("/tmp/test_dir")

        mock_exists.assert_called_once_with("/tmp/test_dir")
        mock_rmtree.assert_not_called()

    @patch("podcast_scraper.workflow.shutil.rmtree")
    @patch("podcast_scraper.workflow.os.path.exists")
    def test_cleanup_pipeline_none_dir(self, mock_exists, mock_rmtree):
        """Test cleanup when temp_dir is None."""
        workflow._cleanup_pipeline(None)

        mock_exists.assert_not_called()
        mock_rmtree.assert_not_called()

    @patch("podcast_scraper.workflow.shutil.rmtree")
    @patch("podcast_scraper.workflow.os.path.exists")
    def test_cleanup_pipeline_handles_os_error(self, mock_exists, mock_rmtree):
        """Test cleanup handles OSError gracefully."""
        mock_exists.return_value = True
        mock_rmtree.side_effect = OSError("Permission denied")

        # Should not raise
        workflow._cleanup_pipeline("/tmp/test_dir")

        mock_rmtree.assert_called_once()


class TestProcessTranscriptionJobs(unittest.TestCase):
    """Tests for _process_transcription_jobs function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(transcribe_missing=True, dry_run=False)
        self.transcription_resources = workflow._TranscriptionResources(
            transcription_provider=Mock(),
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        self.download_args = []
        self.episodes = []
        self.feed = create_test_feed()
        self.feed_metadata = workflow._FeedMetadata(
            description=None, image_url=None, last_updated=None
        )
        self.host_detection_result = workflow._HostDetectionResult(set(), None, None)
        self.pipeline_metrics = metrics.Metrics()

    @patch("podcast_scraper.workflow.transcribe_media_to_text")
    @patch("podcast_scraper.workflow.progress.progress_context")
    def test_process_transcription_jobs_empty(self, mock_progress, mock_transcribe):
        """Test processing when no jobs."""
        # Create new resources with empty jobs list
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=Mock(),
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )

        result = workflow._process_transcription_jobs(
            transcription_resources,
            self.transcription_resources,
            self.download_args,
            self.episodes,
            self.feed,
            self.cfg,
            "/output",
            None,
            self.feed_metadata,
            self.host_detection_result,
            self.pipeline_metrics,
        )

        self.assertEqual(result, 0)
        mock_transcribe.assert_not_called()

    @patch("podcast_scraper.workflow.transcribe_media_to_text")
    @patch("podcast_scraper.workflow.progress.progress_context")
    def test_process_transcription_jobs_transcribe_disabled(self, mock_progress, mock_transcribe):
        """Test processing when transcribe_missing is False."""
        cfg = create_test_config(transcribe_missing=False)
        job = models.TranscriptionJob(
            idx=1, ep_title="Test", ep_title_safe="Test", temp_media="/tmp/test.mp3"
        )
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=Mock(),
            temp_dir="/tmp",
            transcription_jobs=[job],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )

        result = workflow._process_transcription_jobs(
            transcription_resources,
            self.download_args,
            self.episodes,
            self.feed,
            cfg,
            "/output",
            None,
            self.feed_metadata,
            self.host_detection_result,
            self.pipeline_metrics,
        )

        self.assertEqual(result, 0)
        mock_transcribe.assert_not_called()

    @patch("podcast_scraper.workflow.transcribe_media_to_text")
    @patch("podcast_scraper.workflow.progress.progress_context")
    def test_process_transcription_jobs_dry_run(self, mock_progress, mock_transcribe):
        """Test processing in dry-run mode."""
        cfg = create_test_config(transcribe_missing=True, dry_run=True)
        job = models.TranscriptionJob(
            idx=1, ep_title="Test", ep_title_safe="Test", temp_media="/tmp/test.mp3"
        )
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=Mock(),
            temp_dir="/tmp",
            transcription_jobs=[job],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )

        mock_reporter = Mock()
        mock_progress.return_value.__enter__.return_value = mock_reporter
        mock_progress.return_value.__exit__.return_value = None
        # transcribe_media_to_text returns (success, transcript_path, bytes_downloaded)
        mock_transcribe.return_value = (True, "transcript.txt", 0)

        result = workflow._process_transcription_jobs(
            transcription_resources,
            self.download_args,
            self.episodes,
            self.feed,
            cfg,
            "/output",
            None,
            self.feed_metadata,
            self.host_detection_result,
            self.pipeline_metrics,
        )

        # In dry-run, transcribe_media_to_text is called and returns success
        # The function should process the job and return saved count
        # Since transcribe returns success=True, saved should be 1
        self.assertEqual(result, 1)
        mock_transcribe.assert_called_once()


class TestGenerateEpisodeMetadata(unittest.TestCase):
    """Tests for _generate_episode_metadata function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(generate_metadata=True, metadata_format="json")
        self.feed = create_test_feed()
        self.episode = create_test_episode(idx=1, title="Test Episode")

    @patch("podcast_scraper.workflow.metadata.generate_episode_metadata")
    def test_generate_episode_metadata_success(self, mock_generate):
        """Test successful metadata generation."""
        mock_generate.return_value = "/output/metadata.json"

        workflow._generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url="https://example.com/feed.xml",
            cfg=self.cfg,
            output_dir="/output",
            run_suffix=None,
            transcript_file_path="transcript.txt",
            transcript_source="direct_download",
            whisper_model=None,
            detected_hosts=["Host"],
            detected_guests=["Guest"],
            feed_description="Feed description",
            feed_image_url="https://example.com/image.jpg",
            feed_last_updated=None,
            summary_provider=None,
            summary_model=None,
            reduce_model=None,
            pipeline_metrics=None,
        )

        mock_generate.assert_called_once()

    @patch("podcast_scraper.workflow.metadata.generate_episode_metadata")
    def test_generate_episode_metadata_disabled(self, mock_generate):
        """Test metadata generation when disabled."""
        cfg = create_test_config(generate_metadata=False)

        workflow._generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url="https://example.com/feed.xml",
            cfg=cfg,
            output_dir="/output",
            run_suffix=None,
            transcript_file_path="transcript.txt",
            transcript_source="direct_download",
            whisper_model=None,
            detected_hosts=None,
            detected_guests=None,
            feed_description=None,
            feed_image_url=None,
            feed_last_updated=None,
            summary_provider=None,
            summary_model=None,
            reduce_model=None,
            pipeline_metrics=None,
        )

        # Should not call generate_episode_metadata when disabled
        mock_generate.assert_not_called()


class TestSummarizeSingleEpisode(unittest.TestCase):
    """Tests for _summarize_single_episode function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(generate_summaries=True)
        self.feed = create_test_feed()
        self.episode = create_test_episode(idx=1, title="Test Episode")
        self.feed_metadata = workflow._FeedMetadata(
            description=None, image_url=None, last_updated=None
        )
        self.host_detection_result = workflow._HostDetectionResult(set(), None, None)

    @patch("podcast_scraper.workflow.metadata.generate_episode_metadata")
    @patch("podcast_scraper.rss_parser.extract_episode_metadata")
    @patch("podcast_scraper.rss_parser.extract_episode_published_date")
    @patch("os.path.exists")
    def test_summarize_single_episode_success(
        self, mock_exists, mock_extract_date, mock_extract_meta, mock_generate_metadata
    ):
        """Test successful episode summarization."""
        mock_exists.return_value = True
        # extract_episode_metadata returns 6 values
        mock_extract_meta.return_value = (None, None, None, None, None, None)
        mock_extract_date.return_value = None

        workflow._summarize_single_episode(
            episode=self.episode,
            transcript_path="/output/transcript.txt",
            metadata_path="/output/metadata.json",
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            summary_provider=None,
            summary_model=None,
            reduce_model=None,
            detected_names=None,
            pipeline_metrics=None,
        )

        mock_generate_metadata.assert_called_once()

    @patch("podcast_scraper.workflow.metadata.generate_episode_metadata")
    @patch("podcast_scraper.rss_parser.extract_episode_metadata")
    @patch("podcast_scraper.rss_parser.extract_episode_published_date")
    @patch("os.path.exists")
    def test_summarize_single_episode_summaries_disabled(
        self, mock_exists, mock_extract_date, mock_extract_meta, mock_generate_metadata
    ):
        """Test summarization when disabled."""
        mock_exists.return_value = True
        mock_extract_meta.return_value = (None, None, None, None, None, None)
        mock_extract_date.return_value = None
        cfg = create_test_config(generate_summaries=False)

        workflow._summarize_single_episode(
            episode=self.episode,
            transcript_path="/output/transcript.txt",
            metadata_path="/output/metadata.json",
            feed=self.feed,
            cfg=cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            summary_provider=None,
            summary_model=None,
            reduce_model=None,
            detected_names=None,
            pipeline_metrics=None,
        )

        # Should still generate metadata even when summaries disabled
        mock_generate_metadata.assert_called_once()


class TestProcessEpisodes(unittest.TestCase):
    """Tests for _process_episodes function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(workers=1, generate_metadata=True)
        self.episodes = [create_test_episode(idx=1, title="Episode 1")]
        self.feed = create_test_feed()
        self.feed_metadata = workflow._FeedMetadata(
            description=None, image_url=None, last_updated=None
        )
        self.host_detection_result = workflow._HostDetectionResult(set(), None, None)
        self.transcription_resources = workflow._TranscriptionResources(
            transcription_provider=Mock(),
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        self.processing_resources = workflow._ProcessingResources(
            processing_jobs=[],
            processing_jobs_lock=None,
            processing_complete_event=threading.Event(),
        )
        self.pipeline_metrics = metrics.Metrics()

    @patch("podcast_scraper.workflow.process_episode_download")
    def test_process_episodes_empty(self, mock_download):
        """Test _process_episodes with empty download_args."""
        result = workflow._process_episodes(
            download_args=[],
            episodes=self.episodes,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            transcription_resources=self.transcription_resources,
            processing_resources=self.processing_resources,
            pipeline_metrics=self.pipeline_metrics,
            summary_provider=None,
        )

        self.assertEqual(result, 0)
        mock_download.assert_not_called()

    @patch("podcast_scraper.workflow.process_episode_download")
    def test_process_episodes_sequential_success(self, mock_download):
        """Test _process_episodes sequential processing with success."""
        mock_download.return_value = (True, "/output/transcript.txt", "direct_download", 1000)
        episode = create_test_episode(idx=1, title="Episode 1")
        download_args = [
            (
                episode,
                self.cfg,
                "/tmp",
                "/output",
                None,
                [],
                None,
                None,
            )
        ]

        result = workflow._process_episodes(
            download_args=download_args,
            episodes=self.episodes,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            transcription_resources=self.transcription_resources,
            processing_resources=self.processing_resources,
            pipeline_metrics=self.pipeline_metrics,
            summary_provider=None,
        )

        self.assertEqual(result, 1)
        mock_download.assert_called_once()
        # Verify processing job was queued
        self.assertEqual(len(self.processing_resources.processing_jobs), 1)

    @patch("podcast_scraper.workflow.process_episode_download")
    def test_process_episodes_sequential_skipped(self, mock_download):
        """Test _process_episodes when episode is skipped."""
        mock_download.return_value = (False, None, None, 0)
        episode = create_test_episode(idx=1, title="Episode 1")
        download_args = [
            (
                episode,
                self.cfg,
                "/tmp",
                "/output",
                None,
                [],
                None,
                None,
            )
        ]

        result = workflow._process_episodes(
            download_args=download_args,
            episodes=self.episodes,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            transcription_resources=self.transcription_resources,
            processing_resources=self.processing_resources,
            pipeline_metrics=self.pipeline_metrics,
            summary_provider=None,
        )

        self.assertEqual(result, 0)
        # Should track skipped episode
        # Metrics are stored in a dict-like structure
        skipped = getattr(self.pipeline_metrics, "episodes_skipped_total", 0)
        self.assertEqual(skipped, 1)

    @patch("podcast_scraper.workflow.process_episode_download")
    def test_process_episodes_handles_exception(self, mock_download):
        """Test _process_episodes handles exceptions gracefully."""
        mock_download.side_effect = Exception("Download failed")
        episode = create_test_episode(idx=1, title="Episode 1")
        download_args = [
            (
                episode,
                self.cfg,
                "/tmp",
                "/output",
                None,
                [],
                None,
                None,
            )
        ]

        result = workflow._process_episodes(
            download_args=download_args,
            episodes=self.episodes,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            transcription_resources=self.transcription_resources,
            processing_resources=self.processing_resources,
            pipeline_metrics=self.pipeline_metrics,
            summary_provider=None,
        )

        self.assertEqual(result, 0)
        # Should track error
        errors = getattr(self.pipeline_metrics, "errors_total", 0)
        self.assertEqual(errors, 1)


class TestParallelEpisodeSummarization(unittest.TestCase):
    """Tests for _parallel_episode_summarization function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(generate_summaries=True)
        self.episodes = [create_test_episode(idx=1, title="Episode 1")]
        self.feed = create_test_feed()
        self.feed_metadata = workflow._FeedMetadata(
            description=None, image_url=None, last_updated=None
        )
        self.host_detection_result = workflow._HostDetectionResult(set(), None, None)
        self.summary_provider = Mock()
        self.summary_provider._requires_separate_instances = False

    @patch("podcast_scraper.workflow.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    def test_parallel_episode_summarization_no_episodes(self, mock_exists, mock_build_path):
        """Test _parallel_episode_summarization when no episodes need summarization."""
        mock_exists.return_value = False

        workflow._parallel_episode_summarization(
            episodes=self.episodes,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            summary_provider=self.summary_provider,
            download_args=None,
            pipeline_metrics=None,
        )

        # Should return early
        mock_build_path.assert_called()

    @patch("podcast_scraper.workflow._summarize_single_episode")
    @patch("podcast_scraper.workflow.filesystem.build_whisper_output_path")
    @patch("podcast_scraper.workflow.metadata._determine_metadata_path")
    @patch("os.path.exists")
    def test_parallel_episode_summarization_with_episodes(
        self, mock_exists, mock_metadata_path, mock_build_path, mock_summarize
    ):
        """Test _parallel_episode_summarization with episodes that need summarization."""
        mock_build_path.return_value = "/output/transcript.txt"
        mock_metadata_path.return_value = "/output/metadata.json"
        mock_exists.side_effect = lambda path: path == "/output/transcript.txt"

        workflow._parallel_episode_summarization(
            episodes=self.episodes,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            summary_provider=self.summary_provider,
            download_args=None,
            pipeline_metrics=None,
        )

        # Should call summarize for episode
        mock_summarize.assert_called()

    @patch("podcast_scraper.workflow.filesystem.build_whisper_output_path")
    @patch("os.path.exists")
    def test_parallel_episode_summarization_no_provider(self, mock_exists, mock_build_path):
        """Test _parallel_episode_summarization when no summary provider."""
        mock_exists.return_value = True
        mock_build_path.return_value = "/output/transcript.txt"

        workflow._parallel_episode_summarization(
            episodes=self.episodes,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            summary_provider=None,
            download_args=None,
            pipeline_metrics=None,
        )

        # Should return early when no provider
        mock_build_path.assert_called()


class TestPrepareEpisodeDownloadArgs(unittest.TestCase):
    """Tests for _prepare_episode_download_args function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()
        self.episodes = [create_test_episode(idx=1, title="Episode 1")]
        self.transcription_resources = workflow._TranscriptionResources(
            transcription_provider=None,
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        self.host_detection_result = workflow._HostDetectionResult(
            cached_hosts=set(),
            heuristics=None,
            speaker_detector=None,
        )
        self.pipeline_metrics = metrics.Metrics()

    @patch("podcast_scraper.workflow.extract_episode_description")
    def test_prepare_args_auto_speakers_disabled(self, mock_extract):
        """Test preparing args when auto_speakers is disabled."""
        self.cfg = create_test_config(
            auto_speakers=False, screenplay_speaker_names=["Host", "Guest"]
        )

        result = workflow._prepare_episode_download_args(
            episodes=self.episodes,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            transcription_resources=self.transcription_resources,
            host_detection_result=self.host_detection_result,
            pipeline_metrics=self.pipeline_metrics,
        )

        self.assertEqual(len(result), 1)
        args = result[0]
        self.assertEqual(args[0], self.episodes[0])  # episode
        self.assertEqual(args[1], self.cfg)  # cfg
        self.assertEqual(args[4], None)  # run_suffix
        self.assertEqual(args[7], ["Host", "Guest"])  # detected_speaker_names

    @patch("podcast_scraper.workflow.extract_episode_description")
    @patch("podcast_scraper.workflow.time.time")
    def test_prepare_args_auto_speakers_enabled_with_detector(self, mock_time, mock_extract):
        """Test preparing args when auto_speakers is enabled with detector."""
        self.cfg = create_test_config(auto_speakers=True)
        mock_time.side_effect = lambda: 0.0 if mock_time.call_count == 1 else 0.1
        mock_extract.return_value = "Episode description"

        mock_detector = Mock()
        mock_detector.detect_speakers.return_value = (["Host", "Guest"], {"Host"}, True)
        self.host_detection_result = workflow._HostDetectionResult(
            cached_hosts={"Host"},
            heuristics=None,
            speaker_detector=mock_detector,
        )

        result = workflow._prepare_episode_download_args(
            episodes=self.episodes,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            transcription_resources=self.transcription_resources,
            host_detection_result=self.host_detection_result,
            pipeline_metrics=self.pipeline_metrics,
        )

        self.assertEqual(len(result), 1)
        args = result[0]
        self.assertEqual(args[7], ["Host", "Guest"])  # detected_speaker_names
        mock_detector.detect_speakers.assert_called_once()

    @patch("podcast_scraper.workflow.extract_episode_description")
    @patch("podcast_scraper.workflow.time.time")
    def test_prepare_args_auto_speakers_detection_failed_with_fallback(
        self, mock_time, mock_extract
    ):
        """Test preparing args when detection fails and manual fallback is used."""
        self.cfg = create_test_config(
            auto_speakers=True,
            screenplay_speaker_names=["ManualHost", "ManualGuest"],
            cache_detected_hosts=False,
        )
        mock_time.side_effect = lambda: 0.0 if mock_time.call_count == 1 else 0.1
        mock_extract.return_value = "Episode description"

        mock_detector = Mock()
        mock_detector.detect_speakers.return_value = ([], set(), False)  # Detection failed
        self.host_detection_result = workflow._HostDetectionResult(
            cached_hosts=set(),
            heuristics=None,
            speaker_detector=mock_detector,
        )

        result = workflow._prepare_episode_download_args(
            episodes=self.episodes,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            transcription_resources=self.transcription_resources,
            host_detection_result=self.host_detection_result,
            pipeline_metrics=self.pipeline_metrics,
        )

        self.assertEqual(len(result), 1)
        args = result[0]
        # Should use manual fallback
        self.assertEqual(args[7], ["ManualHost", "ManualGuest"])

    @patch("podcast_scraper.workflow.extract_episode_description")
    @patch("podcast_scraper.workflow.time.time")
    def test_prepare_args_auto_speakers_detection_failed_with_detected_hosts(
        self, mock_time, mock_extract
    ):
        """Test preparing args when detection fails but hosts were detected."""
        self.cfg = create_test_config(
            auto_speakers=True,
            screenplay_speaker_names=["ManualHost", "ManualGuest"],
            cache_detected_hosts=True,
        )
        mock_time.side_effect = lambda: 0.0 if mock_time.call_count == 1 else 0.1
        mock_extract.return_value = "Episode description"

        mock_detector = Mock()
        mock_detector.detect_speakers.return_value = (
            [],
            {"DetectedHost"},
            False,
        )  # Detection failed but hosts detected
        self.host_detection_result = workflow._HostDetectionResult(
            cached_hosts={"DetectedHost"},
            heuristics=None,
            speaker_detector=mock_detector,
        )

        result = workflow._prepare_episode_download_args(
            episodes=self.episodes,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            transcription_resources=self.transcription_resources,
            host_detection_result=self.host_detection_result,
            pipeline_metrics=self.pipeline_metrics,
        )

        self.assertEqual(len(result), 1)
        args = result[0]
        # Should use detected hosts + manual guest
        self.assertEqual(args[7], ["DetectedHost", "ManualGuest"])

    @patch("podcast_scraper.workflow.extract_episode_description")
    def test_prepare_args_no_detector_available(self, mock_extract):
        """Test preparing args when no speaker detector is available."""
        self.cfg = create_test_config(auto_speakers=True)
        mock_extract.return_value = "Episode description"

        self.host_detection_result = workflow._HostDetectionResult(
            cached_hosts=set(),
            heuristics=None,
            speaker_detector=None,  # No detector
        )

        result = workflow._prepare_episode_download_args(
            episodes=self.episodes,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            transcription_resources=self.transcription_resources,
            host_detection_result=self.host_detection_result,
            pipeline_metrics=self.pipeline_metrics,
        )

        self.assertEqual(len(result), 1)
        args = result[0]
        # Should be None when no detector
        self.assertIsNone(args[7])

    def test_prepare_args_multiple_episodes(self):
        """Test preparing args for multiple episodes."""
        self.cfg = create_test_config(auto_speakers=False)
        episodes = [
            create_test_episode(idx=1, title="Episode 1"),
            create_test_episode(idx=2, title="Episode 2"),
        ]

        result = workflow._prepare_episode_download_args(
            episodes=episodes,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            transcription_resources=self.transcription_resources,
            host_detection_result=self.host_detection_result,
            pipeline_metrics=self.pipeline_metrics,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], episodes[0])
        self.assertEqual(result[1][0], episodes[1])


class TestProcessTranscriptionJobsConcurrent(unittest.TestCase):
    """Tests for _process_transcription_jobs_concurrent function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            transcription_provider="openai",
            transcription_parallelism=2,
            openai_api_key="test-key-12345",
        )
        self.transcription_resources = workflow._TranscriptionResources(
            transcription_provider=Mock(),
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=threading.Lock(),
            saved_counter_lock=threading.Lock(),
        )
        self.episodes = [create_test_episode(idx=1, title="Episode 1")]
        self.feed = create_test_feed()
        self.feed_metadata = workflow._FeedMetadata(
            description="Feed description",
            image_url="https://example.com/image.jpg",
            last_updated=None,
        )
        self.host_detection_result = workflow._HostDetectionResult(
            cached_hosts=set(),
            heuristics=None,
            speaker_detector=None,
        )
        self.processing_resources = workflow._ProcessingResources(
            processing_jobs=[],
            processing_jobs_lock=threading.Lock(),
            processing_complete_event=threading.Event(),
        )
        self.pipeline_metrics = metrics.Metrics()
        self.download_args = []

    @patch("podcast_scraper.workflow.transcribe_media_to_text")
    def test_process_transcription_jobs_concurrent_empty_queue(self, mock_transcribe):
        """Test concurrent transcription processing with empty queue."""
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=self.transcription_resources.transcription_provider,
            temp_dir=self.transcription_resources.temp_dir,
            transcription_jobs=[],  # Empty queue
            transcription_jobs_lock=self.transcription_resources.transcription_jobs_lock,
            saved_counter_lock=self.transcription_resources.saved_counter_lock,
        )
        downloads_complete_event = threading.Event()
        downloads_complete_event.set()  # Signal completion immediately

        workflow._process_transcription_jobs_concurrent(
            transcription_resources=transcription_resources,
            download_args=self.download_args,
            episodes=self.episodes,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            processing_resources=self.processing_resources,
            pipeline_metrics=self.pipeline_metrics,
            summary_provider=None,
            downloads_complete_event=downloads_complete_event,
        )

        # Should not call transcribe when queue is empty
        mock_transcribe.assert_not_called()

    @patch("podcast_scraper.workflow.transcribe_media_to_text")
    @patch("podcast_scraper.workflow._generate_episode_metadata")
    def test_process_transcription_jobs_concurrent_with_jobs(
        self, mock_generate_metadata, mock_transcribe
    ):
        """Test concurrent transcription processing with jobs in queue."""
        job = models.TranscriptionJob(
            idx=1,
            ep_title="Episode 1",
            ep_title_safe="Episode_1",
            temp_media="/tmp/media.mp3",
            detected_speaker_names=None,
        )
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=self.transcription_resources.transcription_provider,
            temp_dir=self.transcription_resources.temp_dir,
            transcription_jobs=[job],  # Job in queue
            transcription_jobs_lock=self.transcription_resources.transcription_jobs_lock,
            saved_counter_lock=self.transcription_resources.saved_counter_lock,
        )
        mock_transcribe.return_value = (True, "/output/transcript.txt", 1000)
        downloads_complete_event = threading.Event()
        downloads_complete_event.set()  # Signal completion immediately

        workflow._process_transcription_jobs_concurrent(
            transcription_resources=transcription_resources,
            download_args=self.download_args,
            episodes=self.episodes,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            processing_resources=self.processing_resources,
            pipeline_metrics=self.pipeline_metrics,
            summary_provider=None,
            downloads_complete_event=downloads_complete_event,
        )

        # Should call transcribe for the job
        mock_transcribe.assert_called_once()

    @patch("podcast_scraper.workflow.transcribe_media_to_text")
    def test_process_transcription_jobs_concurrent_whisper_sequential(self, mock_transcribe):
        """Test that Whisper provider uses sequential processing regardless of config."""
        self.cfg = create_test_config(
            transcription_provider="whisper",
            transcription_parallelism=5,
            transcribe_missing=True,
        )
        job = models.TranscriptionJob(
            idx=1,
            ep_title="Episode 1",
            ep_title_safe="Episode_1",
            temp_media="/tmp/media.mp3",
            detected_speaker_names=None,
        )
        transcription_resources = workflow._TranscriptionResources(
            transcription_provider=Mock(),
            temp_dir="/tmp",
            transcription_jobs=[job],
            transcription_jobs_lock=threading.Lock(),
            saved_counter_lock=threading.Lock(),
        )
        mock_transcribe.return_value = (True, "/output/transcript.txt", 1000)
        downloads_complete_event = threading.Event()
        downloads_complete_event.set()

        workflow._process_transcription_jobs_concurrent(
            transcription_resources=transcription_resources,
            download_args=self.download_args,
            episodes=self.episodes,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            processing_resources=self.processing_resources,
            pipeline_metrics=self.pipeline_metrics,
            summary_provider=None,
            downloads_complete_event=downloads_complete_event,
        )

        # Should still process the job (sequential)
        mock_transcribe.assert_called_once()


class TestProcessProcessingJobsConcurrent(unittest.TestCase):
    """Tests for _process_processing_jobs_concurrent function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(processing_parallelism=2)
        self.processing_resources = workflow._ProcessingResources(
            processing_jobs=[],
            processing_jobs_lock=threading.Lock(),
            processing_complete_event=threading.Event(),
        )
        self.feed = create_test_feed()
        self.feed_metadata = workflow._FeedMetadata(
            description="Feed description",
            image_url="https://example.com/image.jpg",
            last_updated=None,
        )
        self.host_detection_result = workflow._HostDetectionResult(
            cached_hosts=set(),
            heuristics=None,
            speaker_detector=None,
        )
        self.pipeline_metrics = metrics.Metrics()
        self.transcription_complete_event = threading.Event()

    @patch("podcast_scraper.workflow._generate_episode_metadata")
    @patch("os.path.exists")
    def test_process_processing_jobs_concurrent_empty_queue(
        self, mock_exists, mock_generate_metadata
    ):
        """Test concurrent processing with empty queue."""
        processing_resources = workflow._ProcessingResources(
            processing_jobs=[],  # Empty queue
            processing_jobs_lock=self.processing_resources.processing_jobs_lock,
            processing_complete_event=self.processing_resources.processing_complete_event,
        )
        self.transcription_complete_event.set()  # Signal completion immediately

        workflow._process_processing_jobs_concurrent(
            processing_resources=processing_resources,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            pipeline_metrics=self.pipeline_metrics,
            summary_provider=None,
            transcription_complete_event=self.transcription_complete_event,
        )

        # Should not call generate_metadata when queue is empty
        mock_generate_metadata.assert_not_called()

    @patch("podcast_scraper.workflow._generate_episode_metadata")
    @patch("os.path.exists")
    @patch("os.path.isabs")
    @patch("os.path.join")
    def test_process_processing_jobs_concurrent_with_jobs(
        self, mock_join, mock_isabs, mock_exists, mock_generate_metadata
    ):
        """Test concurrent processing with jobs in queue."""
        episode = create_test_episode(idx=1, title="Episode 1")
        job = workflow._ProcessingJob(
            episode=episode,
            transcript_path="transcript.txt",
            transcript_source="direct_download",
            detected_names=None,
            whisper_model=None,
        )
        processing_resources = workflow._ProcessingResources(
            processing_jobs=[job],  # Job in queue
            processing_jobs_lock=self.processing_resources.processing_jobs_lock,
            processing_complete_event=self.processing_resources.processing_complete_event,
        )
        mock_isabs.return_value = False
        mock_join.return_value = "/output/transcript.txt"
        mock_exists.return_value = True  # File exists
        self.transcription_complete_event.set()  # Signal completion immediately

        workflow._process_processing_jobs_concurrent(
            processing_resources=processing_resources,
            feed=self.feed,
            cfg=self.cfg,
            effective_output_dir="/output",
            run_suffix=None,
            feed_metadata=self.feed_metadata,
            host_detection_result=self.host_detection_result,
            pipeline_metrics=self.pipeline_metrics,
            summary_provider=None,
            transcription_complete_event=self.transcription_complete_event,
        )

        # Should call generate_metadata for the job
        mock_generate_metadata.assert_called_once()

    @patch("podcast_scraper.workflow._generate_episode_metadata")
    @patch("os.path.exists")
    @patch("os.path.isabs")
    def test_process_processing_jobs_concurrent_wait_for_file(
        self, mock_isabs, mock_exists, mock_generate_metadata
    ):
        """Test that processing waits for transcript file to exist."""
        episode = create_test_episode(idx=1, title="Episode 1")
        job = workflow._ProcessingJob(
            episode=episode,
            transcript_path="transcript.txt",
            transcript_source="direct_download",
            detected_names=None,
            whisper_model=None,
        )
        processing_resources = workflow._ProcessingResources(
            processing_jobs=[job],
            processing_jobs_lock=self.processing_resources.processing_jobs_lock,
            processing_complete_event=self.processing_resources.processing_complete_event,
        )
        mock_isabs.return_value = False
        # File doesn't exist initially, then appears
        mock_exists.side_effect = [False, True]
        self.transcription_complete_event.set()

        with patch("podcast_scraper.workflow.time.sleep"):  # Speed up test
            workflow._process_processing_jobs_concurrent(
                processing_resources=processing_resources,
                feed=self.feed,
                cfg=self.cfg,
                effective_output_dir="/output",
                run_suffix=None,
                feed_metadata=self.feed_metadata,
                host_detection_result=self.host_detection_result,
                pipeline_metrics=self.pipeline_metrics,
                summary_provider=None,
                transcription_complete_event=self.transcription_complete_event,
            )

        # Should eventually call generate_metadata after file appears
        mock_generate_metadata.assert_called_once()


class TestRunPipeline(unittest.TestCase):
    """Tests for run_pipeline() main entry point."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()
        self.feed = create_test_feed()
        self.episodes = [create_test_episode(idx=1, title="Episode 1")]

    @patch("podcast_scraper.workflow._cleanup_pipeline")
    @patch("podcast_scraper.workflow._generate_pipeline_summary")
    @patch("podcast_scraper.workflow._parallel_episode_summarization")
    @patch("podcast_scraper.workflow._process_episodes")
    @patch("podcast_scraper.workflow._prepare_episode_download_args")
    @patch("podcast_scraper.workflow._setup_processing_resources")
    @patch("podcast_scraper.workflow._setup_transcription_resources")
    @patch("podcast_scraper.workflow._detect_feed_hosts_and_patterns")
    @patch("podcast_scraper.workflow._prepare_episodes_from_feed")
    @patch("podcast_scraper.workflow._extract_feed_metadata_for_generation")
    @patch("podcast_scraper.workflow._fetch_and_parse_feed")
    @patch("podcast_scraper.workflow._preload_ml_models_if_needed")
    @patch("podcast_scraper.workflow._setup_pipeline_environment")
    @patch("podcast_scraper.workflow._initialize_ml_environment")
    def test_run_pipeline_basic_success(
        self,
        mock_init_env,
        mock_setup_env,
        mock_preload_models,
        mock_fetch_feed,
        mock_extract_metadata,
        mock_prepare_episodes,
        mock_detect_hosts,
        mock_setup_transcription,
        mock_setup_processing,
        mock_prepare_args,
        mock_process_episodes,
        mock_parallel_summarization,
        mock_generate_summary,
        mock_cleanup,
    ):
        """Test basic successful pipeline execution."""
        mock_setup_env.return_value = ("/output", None)
        mock_fetch_feed.return_value = (self.feed, b"<rss></rss>")
        mock_extract_metadata.return_value = workflow._FeedMetadata(None, None, None)
        mock_prepare_episodes.return_value = self.episodes
        mock_detect_hosts.return_value = workflow._HostDetectionResult(
            cached_hosts=set(), heuristics=None, speaker_detector=None
        )
        mock_transcription_resources = workflow._TranscriptionResources(
            transcription_provider=None,
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        mock_setup_transcription.return_value = mock_transcription_resources
        mock_processing_resources = workflow._ProcessingResources(
            processing_jobs=[],
            processing_jobs_lock=None,
            processing_complete_event=threading.Event(),
        )
        mock_setup_processing.return_value = mock_processing_resources
        mock_prepare_args.return_value = []
        mock_process_episodes.return_value = 1
        mock_generate_summary.return_value = (1, "Processed 1 episode")

        count, summary = workflow.run_pipeline(self.cfg)

        self.assertEqual(count, 1)
        self.assertEqual(summary, "Processed 1 episode")
        mock_setup_env.assert_called_once_with(self.cfg)
        mock_fetch_feed.assert_called_once_with(self.cfg)
        mock_process_episodes.assert_called_once()

    @patch("podcast_scraper.workflow._cleanup_pipeline")
    @patch("podcast_scraper.workflow._generate_pipeline_summary")
    @patch("podcast_scraper.workflow._preload_ml_models_if_needed")
    @patch("podcast_scraper.workflow._setup_pipeline_environment")
    @patch("podcast_scraper.workflow._initialize_ml_environment")
    @patch("podcast_scraper.workflow._fetch_and_parse_feed")
    def test_run_pipeline_fetch_feed_failure(
        self,
        mock_fetch_feed,
        mock_init_env,
        mock_setup_env,
        mock_preload_models,
        mock_generate_summary,
        mock_cleanup,
    ):
        """Test pipeline handles RSS feed fetch failure."""
        mock_setup_env.return_value = ("/output", None)
        mock_fetch_feed.side_effect = ValueError("Failed to fetch RSS feed")

        with self.assertRaises(ValueError):
            workflow.run_pipeline(self.cfg)

    @patch("podcast_scraper.workflow._cleanup_pipeline")
    @patch("podcast_scraper.workflow._generate_pipeline_summary")
    @patch("podcast_scraper.workflow._preload_ml_models_if_needed")
    @patch("podcast_scraper.workflow._setup_pipeline_environment")
    @patch("podcast_scraper.workflow._initialize_ml_environment")
    @patch("podcast_scraper.workflow._fetch_and_parse_feed")
    def test_run_pipeline_parse_feed_failure(
        self,
        mock_fetch_feed,
        mock_init_env,
        mock_setup_env,
        mock_preload_models,
        mock_generate_summary,
        mock_cleanup,
    ):
        """Test pipeline handles RSS feed parse failure."""
        mock_setup_env.return_value = ("/output", None)
        mock_fetch_feed.side_effect = ValueError("Failed to parse RSS XML")

        with self.assertRaises(ValueError):
            workflow.run_pipeline(self.cfg)

    @patch("podcast_scraper.workflow._cleanup_pipeline")
    @patch("podcast_scraper.workflow._generate_pipeline_summary")
    @patch("podcast_scraper.workflow._parallel_episode_summarization")
    @patch("podcast_scraper.workflow._process_episodes")
    @patch("podcast_scraper.workflow._prepare_episode_download_args")
    @patch("podcast_scraper.workflow._setup_processing_resources")
    @patch("podcast_scraper.workflow._setup_transcription_resources")
    @patch("podcast_scraper.workflow._detect_feed_hosts_and_patterns")
    @patch("podcast_scraper.workflow._prepare_episodes_from_feed")
    @patch("podcast_scraper.workflow._extract_feed_metadata_for_generation")
    @patch("podcast_scraper.workflow._fetch_and_parse_feed")
    @patch("podcast_scraper.workflow._preload_ml_models_if_needed")
    @patch("podcast_scraper.workflow._setup_pipeline_environment")
    @patch("podcast_scraper.workflow._initialize_ml_environment")
    def test_run_pipeline_dry_run(
        self,
        mock_init_env,
        mock_setup_env,
        mock_preload_models,
        mock_fetch_feed,
        mock_extract_metadata,
        mock_prepare_episodes,
        mock_detect_hosts,
        mock_setup_transcription,
        mock_setup_processing,
        mock_prepare_args,
        mock_process_episodes,
        mock_parallel_summarization,
        mock_generate_summary,
        mock_cleanup,
    ):
        """Test pipeline execution in dry-run mode."""
        cfg = create_test_config(dry_run=True)
        mock_setup_env.return_value = ("/output", None)
        mock_fetch_feed.return_value = (self.feed, b"<rss></rss>")
        mock_extract_metadata.return_value = workflow._FeedMetadata(None, None, None)
        mock_prepare_episodes.return_value = self.episodes
        mock_detect_hosts.return_value = workflow._HostDetectionResult(
            cached_hosts=set(), heuristics=None, speaker_detector=None
        )
        mock_transcription_resources = workflow._TranscriptionResources(
            transcription_provider=None,
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        mock_setup_transcription.return_value = mock_transcription_resources
        mock_processing_resources = workflow._ProcessingResources(
            processing_jobs=[],
            processing_jobs_lock=None,
            processing_complete_event=threading.Event(),
        )
        mock_setup_processing.return_value = mock_processing_resources
        mock_prepare_args.return_value = []
        mock_process_episodes.return_value = 0
        mock_generate_summary.return_value = (0, "Processed 0 episodes")

        count, summary = workflow.run_pipeline(cfg)

        self.assertEqual(count, 0)
        # Should not call parallel summarization in dry-run
        mock_parallel_summarization.assert_not_called()

    @patch("podcast_scraper.workflow._cleanup_pipeline")
    @patch("podcast_scraper.workflow._generate_pipeline_summary")
    @patch("podcast_scraper.workflow._process_transcription_jobs")
    @patch("podcast_scraper.workflow._process_episodes")
    @patch("podcast_scraper.workflow._prepare_episode_download_args")
    @patch("podcast_scraper.workflow._setup_processing_resources")
    @patch("podcast_scraper.workflow._setup_transcription_resources")
    @patch("podcast_scraper.workflow._detect_feed_hosts_and_patterns")
    @patch("podcast_scraper.workflow._prepare_episodes_from_feed")
    @patch("podcast_scraper.workflow._extract_feed_metadata_for_generation")
    @patch("podcast_scraper.workflow._fetch_and_parse_feed")
    @patch("podcast_scraper.workflow._preload_ml_models_if_needed")
    @patch("podcast_scraper.workflow._setup_pipeline_environment")
    @patch("podcast_scraper.workflow._initialize_ml_environment")
    def test_run_pipeline_dry_run_with_transcription(
        self,
        mock_init_env,
        mock_setup_env,
        mock_preload_models,
        mock_fetch_feed,
        mock_extract_metadata,
        mock_prepare_episodes,
        mock_detect_hosts,
        mock_setup_transcription,
        mock_setup_processing,
        mock_prepare_args,
        mock_process_episodes,
        mock_process_transcription,
        mock_generate_summary,
        mock_cleanup,
    ):
        """Test pipeline in dry-run mode with transcription enabled."""
        cfg = create_test_config(dry_run=True, transcribe_missing=True)
        mock_setup_env.return_value = ("/output", None)
        mock_fetch_feed.return_value = (self.feed, b"<rss></rss>")
        mock_extract_metadata.return_value = workflow._FeedMetadata(None, None, None)
        mock_prepare_episodes.return_value = self.episodes
        mock_detect_hosts.return_value = workflow._HostDetectionResult(
            cached_hosts=set(), heuristics=None, speaker_detector=None
        )
        mock_transcription_resources = workflow._TranscriptionResources(
            transcription_provider=None,
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        mock_setup_transcription.return_value = mock_transcription_resources
        mock_processing_resources = workflow._ProcessingResources(
            processing_jobs=[],
            processing_jobs_lock=None,
            processing_complete_event=threading.Event(),
        )
        mock_setup_processing.return_value = mock_processing_resources
        mock_prepare_args.return_value = []
        mock_process_episodes.return_value = 0
        mock_process_transcription.return_value = 0
        mock_generate_summary.return_value = (0, "Processed 0 episodes")

        count, summary = workflow.run_pipeline(cfg)

        # Should process transcription jobs sequentially in dry-run
        mock_process_transcription.assert_called_once()

    @patch("podcast_scraper.workflow._cleanup_pipeline")
    @patch("podcast_scraper.workflow._generate_pipeline_summary")
    @patch("podcast_scraper.workflow.create_summarization_provider")
    @patch("podcast_scraper.workflow._parallel_episode_summarization")
    @patch("podcast_scraper.workflow._process_episodes")
    @patch("podcast_scraper.workflow._prepare_episode_download_args")
    @patch("podcast_scraper.workflow._setup_processing_resources")
    @patch("podcast_scraper.workflow._setup_transcription_resources")
    @patch("podcast_scraper.workflow._detect_feed_hosts_and_patterns")
    @patch("podcast_scraper.workflow._prepare_episodes_from_feed")
    @patch("podcast_scraper.workflow._extract_feed_metadata_for_generation")
    @patch("podcast_scraper.workflow._fetch_and_parse_feed")
    @patch("podcast_scraper.workflow._preload_ml_models_if_needed")
    @patch("podcast_scraper.workflow._setup_pipeline_environment")
    @patch("podcast_scraper.workflow._initialize_ml_environment")
    def test_run_pipeline_with_summarization(
        self,
        mock_init_env,
        mock_setup_env,
        mock_preload_models,
        mock_fetch_feed,
        mock_extract_metadata,
        mock_prepare_episodes,
        mock_detect_hosts,
        mock_setup_transcription,
        mock_setup_processing,
        mock_prepare_args,
        mock_process_episodes,
        mock_parallel_summarization,
        mock_create_summary_provider,
        mock_generate_summary,
        mock_cleanup,
    ):
        """Test pipeline with summarization enabled."""
        cfg = create_test_config(generate_summaries=True, generate_metadata=True)
        mock_setup_env.return_value = ("/output", None)
        mock_fetch_feed.return_value = (self.feed, b"<rss></rss>")
        mock_extract_metadata.return_value = workflow._FeedMetadata(None, None, None)
        mock_prepare_episodes.return_value = self.episodes
        mock_detect_hosts.return_value = workflow._HostDetectionResult(
            cached_hosts=set(), heuristics=None, speaker_detector=None
        )
        mock_transcription_resources = workflow._TranscriptionResources(
            transcription_provider=None,
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        mock_setup_transcription.return_value = mock_transcription_resources
        mock_processing_resources = workflow._ProcessingResources(
            processing_jobs=[],
            processing_jobs_lock=None,
            processing_complete_event=threading.Event(),
        )
        mock_setup_processing.return_value = mock_processing_resources
        mock_prepare_args.return_value = []
        mock_process_episodes.return_value = 1
        mock_summary_provider = Mock()
        mock_summary_provider.cleanup = Mock()
        mock_create_summary_provider.return_value = mock_summary_provider
        mock_generate_summary.return_value = (1, "Processed 1 episode")

        count, summary = workflow.run_pipeline(cfg)

        self.assertEqual(count, 1)
        mock_create_summary_provider.assert_called_once()
        mock_summary_provider.cleanup.assert_called_once()

    @patch("podcast_scraper.workflow._cleanup_pipeline")
    @patch("podcast_scraper.workflow._generate_pipeline_summary")
    @patch("podcast_scraper.workflow._process_episodes")
    @patch("podcast_scraper.workflow._prepare_episode_download_args")
    @patch("podcast_scraper.workflow._setup_processing_resources")
    @patch("podcast_scraper.workflow._setup_transcription_resources")
    @patch("podcast_scraper.workflow._detect_feed_hosts_and_patterns")
    @patch("podcast_scraper.workflow._prepare_episodes_from_feed")
    @patch("podcast_scraper.workflow._extract_feed_metadata_for_generation")
    @patch("podcast_scraper.workflow._fetch_and_parse_feed")
    @patch("podcast_scraper.workflow._preload_ml_models_if_needed")
    @patch("podcast_scraper.workflow._setup_pipeline_environment")
    @patch("podcast_scraper.workflow._initialize_ml_environment")
    def test_run_pipeline_cleanup_on_exception(
        self,
        mock_init_env,
        mock_setup_env,
        mock_preload_models,
        mock_fetch_feed,
        mock_extract_metadata,
        mock_prepare_episodes,
        mock_detect_hosts,
        mock_setup_transcription,
        mock_setup_processing,
        mock_prepare_args,
        mock_process_episodes,
        mock_generate_summary,
        mock_cleanup,
    ):
        """Test that provider cleanup happens even when exception occurs."""
        mock_setup_env.return_value = ("/output", None)
        mock_fetch_feed.return_value = (self.feed, b"<rss></rss>")
        mock_extract_metadata.return_value = workflow._FeedMetadata(None, None, None)
        mock_prepare_episodes.return_value = self.episodes
        mock_detect_hosts.return_value = workflow._HostDetectionResult(
            cached_hosts=set(), heuristics=None, speaker_detector=None
        )
        mock_transcription_provider = Mock()
        mock_transcription_provider.cleanup = Mock()
        mock_transcription_resources = workflow._TranscriptionResources(
            transcription_provider=mock_transcription_provider,
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        mock_setup_transcription.return_value = mock_transcription_resources
        mock_processing_resources = workflow._ProcessingResources(
            processing_jobs=[],
            processing_jobs_lock=None,
            processing_complete_event=threading.Event(),
        )
        mock_setup_processing.return_value = mock_processing_resources
        mock_prepare_args.return_value = []
        mock_process_episodes.side_effect = RuntimeError("Processing failed")

        with self.assertRaises(RuntimeError):
            workflow.run_pipeline(self.cfg)

        # Provider cleanup should still be called (in finally block)
        mock_transcription_provider.cleanup.assert_called_once()
        # _cleanup_pipeline is called after the try-finally, so it won't be called
        # if exception occurs. This is expected behavior - the function raises
        # before reaching that line
