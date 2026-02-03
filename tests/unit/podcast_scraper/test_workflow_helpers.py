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

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts
import importlib.util
from pathlib import Path

from podcast_scraper import config, models
from podcast_scraper.workflow import metrics, orchestration as workflow
from podcast_scraper.workflow.stages import setup

parent_tests_dir = Path(__file__).parent.parent.parent
conftest_path = parent_tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("conftest", conftest_path)
if spec and spec.loader:
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    create_test_config = conftest.create_test_config

logger = logging.getLogger(__name__)


@pytest.mark.unit
class TestInitializeMLEnvironment(unittest.TestCase):
    """Tests for initialize_ml_environment function."""

    def test_initialize_ml_environment_sets_hf_hub_disable_progress_bars(self):
        """Test that initialize_ml_environment sets HF_HUB_DISABLE_PROGRESS_BARS."""
        # Clear the environment variable if it exists
        original_value = os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        try:
            # Call the function
            setup.initialize_ml_environment()

            # Verify the environment variable was set
            self.assertEqual(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = original_value
            elif "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ:
                del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]

    def test_initialize_ml_environment_sets_tokenizers_parallelism(self):
        """Test that initialize_ml_environment sets TOKENIZERS_PARALLELISM."""
        # Clear the environment variable if it exists
        original_value = os.environ.pop("TOKENIZERS_PARALLELISM", None)
        try:
            # Call the function
            setup.initialize_ml_environment()

            # Verify the environment variable was set
            self.assertEqual(os.environ.get("TOKENIZERS_PARALLELISM"), "false")
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["TOKENIZERS_PARALLELISM"] = original_value
            elif "TOKENIZERS_PARALLELISM" in os.environ:
                del os.environ["TOKENIZERS_PARALLELISM"]

    def test_initialize_ml_environment_respects_existing_values(self):
        """Test that initialize_ml_environment respects existing environment variables."""
        # Set custom values
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        try:
            # Call the function
            setup.initialize_ml_environment()

            # Verify the environment variables were not changed
            self.assertEqual(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "0")
            self.assertEqual(os.environ.get("TOKENIZERS_PARALLELISM"), "true")
        finally:
            # Clean up
            del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]
            del os.environ["TOKENIZERS_PARALLELISM"]


@pytest.mark.unit
class TestUpdateMetricSafely(unittest.TestCase):
    """Tests for update_metric_safely helper function."""

    def test_update_metric_safely_without_lock(self):
        """Test that update_metric_safely works without a lock."""
        from podcast_scraper.workflow import helpers

        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.transcripts_downloaded = 5

        helpers.update_metric_safely(pipeline_metrics, "transcripts_downloaded", 3)

        self.assertEqual(pipeline_metrics.transcripts_downloaded, 8)

    def test_update_metric_safely_with_lock(self):
        """Test that update_metric_safely works with a lock."""
        from podcast_scraper.workflow import helpers

        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.transcripts_downloaded = 5
        lock = threading.Lock()

        helpers.update_metric_safely(pipeline_metrics, "transcripts_downloaded", 3, lock)

        self.assertEqual(pipeline_metrics.transcripts_downloaded, 8)

    def test_update_metric_safely_handles_missing_attribute(self):
        """Test that update_metric_safely handles missing metric attributes."""
        from podcast_scraper.workflow import helpers

        pipeline_metrics = metrics.Metrics()

        # Should not raise - getattr returns 0 for missing attributes
        helpers.update_metric_safely(pipeline_metrics, "nonexistent_metric", 5)

        self.assertEqual(getattr(pipeline_metrics, "nonexistent_metric", 0), 5)


@pytest.mark.unit
class TestExtractFeedMetadataForGeneration(unittest.TestCase):
    """Tests for extract_feed_metadata_for_generation function."""

    def test_extract_feed_metadata_for_generation_disabled(self):
        """Test that extract_feed_metadata_for_generation returns empty when disabled."""
        from podcast_scraper.workflow.stages.scraping import extract_feed_metadata_for_generation
        from podcast_scraper.workflow.types import FeedMetadata

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            generate_metadata=False,
        )
        feed = models.RssFeed(
            title="Test Feed", authors=["Test Author"], items=[], base_url="https://example.com"
        )
        rss_bytes = b"<rss></rss>"

        result = extract_feed_metadata_for_generation(cfg, feed, rss_bytes)

        self.assertEqual(result, FeedMetadata(None, None, None))

    def test_extract_feed_metadata_for_generation_with_empty_bytes(self):
        """Test that extract_feed_metadata_for_generation returns empty when rss_bytes is empty."""
        from podcast_scraper.workflow.stages.scraping import extract_feed_metadata_for_generation
        from podcast_scraper.workflow.types import FeedMetadata

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            generate_metadata=True,
        )
        feed = models.RssFeed(
            title="Test Feed", authors=["Test Author"], items=[], base_url="https://example.com"
        )
        rss_bytes = b""

        result = extract_feed_metadata_for_generation(cfg, feed, rss_bytes)

        self.assertEqual(result, FeedMetadata(None, None, None))


@pytest.mark.unit
class TestPrepareEpisodesFromFeed(unittest.TestCase):
    """Tests for prepare_episodes_from_feed function."""

    def test_prepare_episodes_from_feed_basic(self):
        """Test that prepare_episodes_from_feed creates Episode objects."""
        import xml.etree.ElementTree as ET

        from podcast_scraper.workflow.stages.scraping import prepare_episodes_from_feed

        # Create proper RSS items as XML elements
        item1 = ET.Element("item")
        ET.SubElement(item1, "title").text = "Episode 1"
        ET.SubElement(item1, "link").text = "https://example.com/ep1"
        ET.SubElement(item1, "guid").text = "ep1"

        item2 = ET.Element("item")
        ET.SubElement(item2, "title").text = "Episode 2"
        ET.SubElement(item2, "link").text = "https://example.com/ep2"
        ET.SubElement(item2, "guid").text = "ep2"

        feed = models.RssFeed(
            title="Test Feed",
            authors=["Test Author"],
            items=[item1, item2],
            base_url="https://example.com",
        )
        cfg = create_test_config(rss_url="https://example.com/feed.xml")

        episodes = prepare_episodes_from_feed(feed, cfg)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].idx, 1)
        self.assertEqual(episodes[0].title, "Episode 1")
        self.assertEqual(episodes[1].idx, 2)
        self.assertEqual(episodes[1].title, "Episode 2")

    def test_prepare_episodes_from_feed_with_max_episodes(self):
        """Test that prepare_episodes_from_feed respects max_episodes limit."""
        import xml.etree.ElementTree as ET

        from podcast_scraper.workflow.stages.scraping import prepare_episodes_from_feed

        # Create proper RSS items as XML elements
        items = []
        for i in range(1, 4):
            item = ET.Element("item")
            ET.SubElement(item, "title").text = f"Episode {i}"
            ET.SubElement(item, "link").text = f"https://example.com/ep{i}"
            ET.SubElement(item, "guid").text = f"ep{i}"
            items.append(item)

        feed = models.RssFeed(
            title="Test Feed",
            authors=["Test Author"],
            items=items,
            base_url="https://example.com",
        )
        cfg = create_test_config(rss_url="https://example.com/feed.xml", max_episodes=2)

        episodes = prepare_episodes_from_feed(feed, cfg)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].idx, 1)
        self.assertEqual(episodes[1].idx, 2)


@pytest.mark.unit
class TestSetupProcessingResources(unittest.TestCase):
    """Tests for setup_processing_resources function."""

    def test_setup_processing_resources_basic(self):
        """Test that setup_processing_resources creates ProcessingResources."""
        from podcast_scraper.workflow.stages.processing import setup_processing_resources
        from podcast_scraper.workflow.types import ProcessingResources

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            workers=1,
            transcription_parallelism=1,
            processing_parallelism=1,
        )

        result = setup_processing_resources(cfg)

        self.assertIsInstance(result, ProcessingResources)
        self.assertEqual(len(result.processing_jobs), 0)
        # Lock is None when all parallelism settings are 1
        self.assertIsNone(result.processing_jobs_lock)
        self.assertIsNotNone(result.processing_complete_event)

    def test_setup_processing_resources_with_multiple_workers(self):
        """Test that setup_processing_resources creates lock with multiple workers."""
        from podcast_scraper.workflow.stages.processing import setup_processing_resources

        cfg = create_test_config(rss_url="https://example.com/feed.xml", workers=4)

        result = setup_processing_resources(cfg)

        self.assertIsNotNone(result.processing_jobs_lock)


@pytest.mark.unit
class TestGeneratePipelineSummary(unittest.TestCase):
    """Tests for generate_pipeline_summary helper function."""

    def test_generate_pipeline_summary_dry_run(self):
        """Test that generate_pipeline_summary works in dry-run mode."""
        from podcast_scraper.workflow import helpers
        from podcast_scraper.workflow.types import TranscriptionResources

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            dry_run=True,
            transcribe_missing=True,
        )
        transcription_resources = TranscriptionResources(
            transcription_provider=None,
            temp_dir=None,
            transcription_jobs=[Mock(), Mock()],  # 2 jobs
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        pipeline_metrics = metrics.Metrics()
        effective_output_dir = "/tmp/test"

        count, summary = helpers.generate_pipeline_summary(
            cfg,
            saved=5,
            transcription_resources=transcription_resources,
            effective_output_dir=effective_output_dir,
            pipeline_metrics=pipeline_metrics,
        )

        self.assertEqual(count, 7)  # 5 saved + 2 planned transcriptions
        self.assertIn("Dry run complete", summary)
        self.assertIn("transcripts_planned=7", summary)
        self.assertIn("Direct downloads planned: 5", summary)
        self.assertIn("Whisper transcriptions planned: 2", summary)

    def test_generate_pipeline_summary_normal_mode(self):
        """Test that generate_pipeline_summary works in normal mode."""
        from podcast_scraper.workflow import helpers
        from podcast_scraper.workflow.types import TranscriptionResources

        cfg = create_test_config(rss_url="https://example.com/feed.xml")
        transcription_resources = TranscriptionResources(
            transcription_provider=None,
            temp_dir=None,
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.transcripts_downloaded = 3
        pipeline_metrics.transcripts_transcribed = 2
        effective_output_dir = "/tmp/test"

        count, summary = helpers.generate_pipeline_summary(
            cfg,
            saved=5,
            transcription_resources=transcription_resources,
            effective_output_dir=effective_output_dir,
            pipeline_metrics=pipeline_metrics,
        )

        self.assertEqual(count, 5)
        self.assertIn("Done. transcripts_saved=5", summary)
        self.assertIn("Transcripts downloaded: 3", summary)
        self.assertIn("Episodes transcribed: 2", summary)

    def test_generate_pipeline_summary_includes_llm_usage(self):
        """Test that generate_pipeline_summary includes LLM usage when OpenAI providers are used."""
        from podcast_scraper.workflow import helpers
        from podcast_scraper.workflow.types import TranscriptionResources

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            summary_provider="openai",
            openai_transcription_model="whisper-1",
            openai_summary_model="gpt-4o-mini",
            openai_api_key="sk-test123",
        )
        transcription_resources = TranscriptionResources(
            transcription_provider=None,
            temp_dir=None,
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.transcripts_downloaded = 2
        pipeline_metrics.record_llm_transcription_call(10.0)  # 10 minutes
        pipeline_metrics.record_llm_summarization_call(input_tokens=50000, output_tokens=10000)
        effective_output_dir = "/tmp/test"

        count, summary = helpers.generate_pipeline_summary(
            cfg,
            saved=2,
            transcription_resources=transcription_resources,
            effective_output_dir=effective_output_dir,
            pipeline_metrics=pipeline_metrics,
        )

        # Should include LLM usage section
        self.assertIn("LLM API Usage:", summary)
        self.assertIn("Transcription:", summary)
        self.assertIn("Summarization:", summary)
        self.assertIn("Total estimated cost:", summary)


@pytest.mark.unit
class TestCallGenerateMetadata(unittest.TestCase):
    """Tests for call_generate_metadata function."""

    def test_call_generate_metadata_basic(self):
        """Test that call_generate_metadata calls generate_episode_metadata."""
        from podcast_scraper.workflow.stages.metadata import call_generate_metadata
        from podcast_scraper.workflow.types import FeedMetadata, HostDetectionResult

        episode = models.Episode(
            idx=1,
            title="Test Episode",
            title_safe="test-episode",
            item={"title": "Test Episode"},
            transcript_urls=[],
        )
        feed = models.RssFeed(
            title="Test Feed", authors=["Test Author"], items=[], base_url="https://example.com"
        )
        cfg = create_test_config(rss_url="https://example.com/feed.xml")
        feed_metadata = FeedMetadata(None, None, None)
        host_detection_result = HostDetectionResult(
            cached_hosts=set(), heuristics=None, speaker_detector=None
        )

        with patch(
            "podcast_scraper.workflow.stages.metadata.generate_episode_metadata"
        ) as mock_generate:
            call_generate_metadata(
                episode=episode,
                feed=feed,
                cfg=cfg,
                effective_output_dir="/tmp/test",
                run_suffix=None,
                transcript_path="/tmp/test/ep1.txt",
                transcript_source="direct_download",
                whisper_model=None,
                feed_metadata=feed_metadata,
                host_detection_result=host_detection_result,
                detected_names=None,
                summary_provider=None,
                pipeline_metrics=None,
            )

            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            self.assertEqual(call_args.kwargs["episode"], episode)
            self.assertEqual(call_args.kwargs["feed"], feed)


@pytest.mark.unit
class TestApplyLogLevel(unittest.TestCase):
    """Tests for apply_log_level function."""

    def test_apply_log_level_valid(self):
        """Test that apply_log_level works with valid log level."""
        workflow.apply_log_level("DEBUG")

        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.DEBUG)

    def test_apply_log_level_invalid(self):
        """Test that apply_log_level raises ValueError for invalid log level."""
        with self.assertRaises(ValueError):
            workflow.apply_log_level("INVALID_LEVEL")


@pytest.mark.unit
class TestSetupPipelineEnvironment(unittest.TestCase):
    """Tests for setup_pipeline_environment function."""

    def test_setup_pipeline_environment_basic(self):
        """Test that setup_pipeline_environment creates output directory."""
        import tempfile

        from podcast_scraper.workflow.stages.setup import setup_pipeline_environment

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            cfg = create_test_config(
                rss_url="https://example.com/feed.xml",
                output_dir=output_dir,
                transcription_provider="whisper",  # Disable to avoid provider suffix
                speaker_detector_provider="spacy",  # This will create a suffix
                summary_provider="transformers",  # This will create a suffix
            )

            effective_output_dir, run_suffix, full_config_string = setup_pipeline_environment(cfg)

            # run_suffix will include provider info (e.g., "sp_spacy_sm")
            self.assertIsNotNone(run_suffix)
            self.assertIn("run_", effective_output_dir)
            self.assertTrue(os.path.exists(effective_output_dir))
            self.assertTrue(os.path.exists(os.path.join(effective_output_dir, "transcripts")))
            self.assertTrue(os.path.exists(os.path.join(effective_output_dir, "metadata")))

    def test_setup_pipeline_environment_with_run_id(self):
        """Test that setup_pipeline_environment creates run ID subdirectory."""
        import tempfile

        from podcast_scraper.workflow.stages.setup import setup_pipeline_environment

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            cfg = create_test_config(
                rss_url="https://example.com/feed.xml",
                output_dir=output_dir,
                run_id="test_run",
                transcription_provider="whisper",
                speaker_detector_provider="spacy",
                summary_provider="transformers",
            )

            effective_output_dir, run_suffix, full_config_string = setup_pipeline_environment(cfg)

            self.assertIn("test_run", effective_output_dir)
            # run_suffix will be "test_run" + provider suffix (e.g., "test_run_sp_spacy_sm")
            self.assertIn("test_run", run_suffix)
            self.assertTrue(os.path.exists(effective_output_dir))


@pytest.mark.unit
class TestEnsureMLModelsCached(unittest.TestCase):
    """Tests for ensure_ml_models_cached function in workflow.stages.setup."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
        )

    @patch("podcast_scraper.config._is_test_environment")
    def test_ensure_ml_models_cached_skips_in_test(self, mock_is_test):
        """Test that ensure_ml_models_cached skips in test environment."""
        mock_is_test.return_value = True

        # Function should return early in test environment
        setup.ensure_ml_models_cached(self.cfg)
        # The important thing is it doesn't crash

    @patch("podcast_scraper.config._is_test_environment")
    def test_ensure_ml_models_cached_skips_when_disabled(self, mock_is_test):
        """Test that ensure_ml_models_cached skips when preload_models=False."""
        mock_is_test.return_value = False
        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=False,
        )

        with patch("podcast_scraper.cache.directories.get_whisper_cache_dir") as mock_whisper_cache:
            setup.ensure_ml_models_cached(cfg)
            # Should return early without checking cache
            mock_whisper_cache.assert_not_called()

    @patch("podcast_scraper.config._is_test_environment")
    def test_ensure_ml_models_cached_skips_when_dry_run(self, mock_is_test):
        """Test that ensure_ml_models_cached skips when dry_run=True."""
        mock_is_test.return_value = False
        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
            dry_run=True,
        )

        with patch("podcast_scraper.cache.directories.get_whisper_cache_dir") as mock_whisper_cache:
            setup.ensure_ml_models_cached(cfg)
            # Should return early without checking cache
            mock_whisper_cache.assert_not_called()

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.cache.get_whisper_cache_dir")
    def test_ensure_ml_models_cached_whisper_model_cached(self, mock_get_cache, mock_is_test):
        """Test that ensure_ml_models_cached skips download when model is cached."""
        mock_is_test.return_value = False
        import tempfile
        from pathlib import Path

        temp_dir = tempfile.mkdtemp()
        whisper_cache = Path(temp_dir) / "whisper"
        whisper_cache.mkdir(parents=True, exist_ok=True)
        model_file = whisper_cache / f"{config.TEST_DEFAULT_WHISPER_MODEL}.pt"
        model_file.touch()  # Create fake model file

        mock_get_cache.return_value = whisper_cache

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
        )

        with patch(
            "podcast_scraper.providers.ml.model_loader.preload_whisper_models"
        ) as mock_preload:
            setup.ensure_ml_models_cached(cfg)
            # Should not call preload since model is cached
            mock_preload.assert_not_called()

        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.cache.get_transformers_cache_dir")
    @patch("podcast_scraper.cache.get_whisper_cache_dir")
    def test_ensure_ml_models_cached_whisper_model_missing(
        self, mock_get_whisper_cache, mock_get_transformers_cache, mock_is_test
    ):
        """Test that ensure_ml_models_cached downloads when model is missing."""
        mock_is_test.return_value = False
        import tempfile
        from pathlib import Path

        temp_dir = tempfile.mkdtemp()
        whisper_cache = Path(temp_dir) / "whisper"
        whisper_cache.mkdir(parents=True, exist_ok=True)
        # Don't create model file - it's missing

        mock_get_whisper_cache.return_value = whisper_cache
        mock_get_transformers_cache.return_value = Path(temp_dir) / "transformers"

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
        )

        with patch(
            "podcast_scraper.providers.ml.model_loader.preload_whisper_models"
        ) as mock_preload:
            setup.ensure_ml_models_cached(cfg)
            # Should call preload since model is missing
            mock_preload.assert_called_once_with([config.TEST_DEFAULT_WHISPER_MODEL])

        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.cache.get_transformers_cache_dir")
    def test_ensure_ml_models_cached_transformers_model_missing(self, mock_get_cache, mock_is_test):
        """Test that ensure_ml_models_cached downloads when Transformers model is missing."""
        mock_is_test.return_value = False
        import tempfile
        from pathlib import Path

        temp_dir = tempfile.mkdtemp()
        transformers_cache = Path(temp_dir) / "huggingface" / "hub"
        transformers_cache.mkdir(parents=True, exist_ok=True)
        # Don't create model cache - it's missing

        mock_get_cache.return_value = transformers_cache

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
            generate_summaries=True,
            summary_provider="transformers",
            summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
        )

        with patch(
            "podcast_scraper.providers.ml.summarizer.select_summary_model"
        ) as mock_select_map:
            with patch(
                "podcast_scraper.providers.ml.summarizer.select_reduce_model"
            ) as mock_select_reduce:
                mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
                mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_MODEL

                with patch(
                    "podcast_scraper.providers.ml.model_loader.preload_transformers_models"
                ) as mock_preload:
                    setup.ensure_ml_models_cached(cfg)
                    # Should call preload since model is missing
                    mock_preload.assert_called_once_with([config.TEST_DEFAULT_SUMMARY_MODEL])

        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.providers.ml.model_loader.preload_whisper_models")
    def test_ensure_ml_models_cached_handles_import_error(self, mock_preload, mock_is_test):
        """Test that ensure_ml_models_cached handles ImportError gracefully."""
        mock_is_test.return_value = False
        mock_preload.side_effect = ImportError("Model loader not available")

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
        )

        # Should not raise - just logs warning
        with patch("podcast_scraper.cache.get_whisper_cache_dir") as mock_get_cache:
            import tempfile
            from pathlib import Path

            temp_dir = tempfile.mkdtemp()
            whisper_cache = Path(temp_dir) / "whisper"
            whisper_cache.mkdir(parents=True, exist_ok=True)
            mock_get_cache.return_value = whisper_cache

            # Should not raise
            setup.ensure_ml_models_cached(cfg)

            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.cache.get_transformers_cache_dir")
    @patch("podcast_scraper.providers.ml.summarizer.select_summary_model")
    @patch("podcast_scraper.providers.ml.summarizer.select_reduce_model")
    def test_ensure_ml_models_cached_transformers_reduce_model_different(
        self, mock_select_reduce, mock_select_map, mock_get_cache, mock_is_test
    ):
        """Test that ensure_ml_models_cached handles different reduce model."""
        mock_is_test.return_value = False
        import tempfile
        from pathlib import Path

        temp_dir = tempfile.mkdtemp()
        transformers_cache = Path(temp_dir) / "huggingface" / "hub"
        transformers_cache.mkdir(parents=True, exist_ok=True)
        mock_get_cache.return_value = transformers_cache

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
            generate_summaries=True,
            summary_provider="transformers",
            summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
        )

        # Set up mocks to return different models
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL

        with patch(
            "podcast_scraper.providers.ml.model_loader.preload_transformers_models"
        ) as mock_preload:
            setup.ensure_ml_models_cached(cfg)
            # Should call preload with both models
            mock_preload.assert_called_once()
            called_models = mock_preload.call_args[0][0]
            self.assertIn(config.TEST_DEFAULT_SUMMARY_MODEL, called_models)
            self.assertIn(config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL, called_models)

        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.providers.ml.model_loader.preload_whisper_models")
    def test_ensure_ml_models_cached_handles_general_exception(
        self, mock_preload_whisper, mock_is_test
    ):
        """Test that ensure_ml_models_cached handles general exceptions gracefully."""
        mock_is_test.return_value = False
        mock_preload_whisper.side_effect = Exception("Download failed")

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
        )

        with patch("podcast_scraper.cache.get_whisper_cache_dir") as mock_get_cache:
            import logging
            import tempfile
            from pathlib import Path

            temp_dir = tempfile.mkdtemp()
            whisper_cache = Path(temp_dir) / "whisper"
            whisper_cache.mkdir(parents=True, exist_ok=True)
            mock_get_cache.return_value = whisper_cache

            # Capture log output
            with self.assertLogs(
                "podcast_scraper.workflow.stages.setup", level=logging.WARNING
            ) as log:
                # Should not raise - just logs warning
                setup.ensure_ml_models_cached(cfg)

                # Verify warning was logged
                self.assertTrue(
                    any("Could not automatically download models" in msg for msg in log.output)
                )

            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("podcast_scraper.config._is_test_environment")
    def test_ensure_ml_models_cached_handles_outer_import_error(self, mock_is_test):
        """Test that ensure_ml_models_cached handles ImportError in outer try block."""
        mock_is_test.return_value = False

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
        )

        # Mock ImportError when importing cache module (patch the import at module level)
        # Since cache is imported inside the function, we need to patch sys.modules
        import sys

        original_cache = sys.modules.get("podcast_scraper.cache")
        try:
            # Remove cache from sys.modules to simulate ImportError
            if "podcast_scraper.cache" in sys.modules:
                del sys.modules["podcast_scraper.cache"]
            # Should not raise - just passes silently (caught by outer except ImportError)
            setup.ensure_ml_models_cached(cfg)
        finally:
            # Restore original module
            if original_cache is not None:
                sys.modules["podcast_scraper.cache"] = original_cache

    @patch("podcast_scraper.config._is_test_environment")
    def test_ensure_ml_models_cached_handles_outer_exception(self, mock_is_test):
        """Test that ensure_ml_models_cached handles general exceptions in outer try block."""
        mock_is_test.return_value = False

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
        )

        # Mock exception when getting cache dir (this is in the outer try block)
        # Patch at the source module since it's imported inside the function
        with patch(
            "podcast_scraper.cache.get_whisper_cache_dir",
            side_effect=Exception("Cache error"),
        ):
            import logging

            # Capture log output
            with self.assertLogs(
                "podcast_scraper.workflow.stages.setup", level=logging.DEBUG
            ) as log:
                # Should not raise - just logs debug (caught by outer except Exception)
                setup.ensure_ml_models_cached(cfg)

                # Verify debug was logged
                self.assertTrue(any("Error checking model cache" in msg for msg in log.output))


@pytest.mark.unit
class TestNoDuplicateAliases(unittest.TestCase):
    """Tests to verify no duplicate aliases exist in workflow modules."""

    def test_no_duplicate_stage_function_aliases(self):
        """Test that stage functions are not duplicated between workflow.py and stage modules."""
        from podcast_scraper import workflow

        # Functions that should exist in stage modules, not in workflow.py
        stage_functions = [
            "setup_pipeline_environment",
            "fetch_and_parse_feed",
            "extract_feed_metadata_for_generation",
            "prepare_episodes_from_feed",
            "detect_feed_hosts_and_patterns",
            "setup_transcription_resources",
            "setup_processing_resources",
            "prepare_episode_download_args",
            "process_episodes",
            "process_transcription_jobs",
            "process_transcription_jobs_concurrent",
            "process_processing_jobs_concurrent",
            "generate_episode_metadata",
            "parallel_episode_summarization",
            "summarize_single_episode",
        ]

        # Helper functions that should exist in helpers module, not in workflow.py
        helper_functions = [
            "update_metric_safely",
            "cleanup_pipeline",
            "generate_pipeline_summary",
        ]

        # Check that these functions don't exist in workflow.py (without _ prefix)
        workflow_attrs = set(dir(workflow))
        for func_name in stage_functions + helper_functions:
            # Function should not exist in workflow module (it should be in stage/helper modules)
            self.assertNotIn(
                func_name,
                workflow_attrs,
                f"Function {func_name} should not exist in workflow module, "
                f"it should be in workflow.stages or workflow.helpers",
            )

    def test_stage_functions_exist_in_stage_modules(self):
        """Test that stage functions exist in their respective stage modules."""
        from podcast_scraper.workflow import stages

        # Verify functions exist in stage modules
        self.assertTrue(hasattr(stages.setup, "setup_pipeline_environment"))
        self.assertTrue(hasattr(stages.scraping, "fetch_and_parse_feed"))
        self.assertTrue(hasattr(stages.scraping, "extract_feed_metadata_for_generation"))
        self.assertTrue(hasattr(stages.scraping, "prepare_episodes_from_feed"))
        self.assertTrue(hasattr(stages.processing, "detect_feed_hosts_and_patterns"))
        self.assertTrue(hasattr(stages.processing, "setup_processing_resources"))
        self.assertTrue(hasattr(stages.processing, "prepare_episode_download_args"))
        self.assertTrue(hasattr(stages.processing, "process_episodes"))
        self.assertTrue(hasattr(stages.transcription, "setup_transcription_resources"))
        self.assertTrue(hasattr(stages.transcription, "process_transcription_jobs"))
        self.assertTrue(hasattr(stages.transcription, "process_transcription_jobs_concurrent"))
        self.assertTrue(hasattr(stages.processing, "process_processing_jobs_concurrent"))
        self.assertTrue(hasattr(stages.metadata, "generate_episode_metadata"))
        self.assertTrue(hasattr(stages.summarization_stage, "parallel_episode_summarization"))
        self.assertTrue(hasattr(stages.summarization_stage, "summarize_single_episode"))

    def test_helper_functions_exist_in_helpers_module(self):
        """Test that helper functions exist in helpers module."""
        from podcast_scraper.workflow import helpers

        # Verify functions exist in helpers module
        self.assertTrue(hasattr(helpers, "update_metric_safely"))
        self.assertTrue(hasattr(helpers, "cleanup_pipeline"))
        self.assertTrue(hasattr(helpers, "generate_pipeline_summary"))


@pytest.mark.unit
class TestPrepareEpisodeDownloadArgs(unittest.TestCase):
    """Tests for prepare_episode_download_args function."""

    @patch("podcast_scraper.workflow.stages.processing.http_head")
    @patch("podcast_scraper.workflow.stages.processing.create_speaker_detector")
    def test_skip_speaker_detection_when_file_too_large_for_openai(
        self, mock_create_detector, mock_http_head
    ):
        """Test that speaker detection is skipped when file exceeds OpenAI 25MB limit.

        This test verifies the fix for issue #327: Speaker detection should not run
        for episodes that will be skipped due to file size limits when using OpenAI
        transcription provider.
        """
        from podcast_scraper.workflow.stages import processing
        from podcast_scraper.workflow.types import HostDetectionResult, TranscriptionResources

        # Create test config with OpenAI transcription
        cfg = create_test_config(
            transcribe_missing=True,
            transcription_provider="openai",
            auto_speakers=True,
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
        )

        # Create test episode with media URL
        episode = models.Episode(
            idx=1,
            title="Test Episode",
            title_safe="Test_Episode",
            media_url="https://example.com/episode.mp3",
            transcript_urls=[],
            item=Mock(),
        )

        # Mock HTTP HEAD response with file size > 25MB
        mock_response = Mock()
        mock_response.headers = {"Content-Length": str(30 * 1024 * 1024)}  # 30 MB
        mock_http_head.return_value = mock_response

        # Mock speaker detector (should NOT be called)
        mock_detector = Mock()
        mock_create_detector.return_value = mock_detector

        # Create minimal resources
        transcription_resources = TranscriptionResources(
            transcription_provider=None,
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=threading.Lock(),
            saved_counter_lock=threading.Lock(),
        )
        host_detection_result = HostDetectionResult(
            cached_hosts=set(),
            heuristics=None,
            speaker_detector=None,
        )
        pipeline_metrics = metrics.Metrics()

        # Call prepare_episode_download_args
        download_args = processing.prepare_episode_download_args(
            episodes=[episode],
            cfg=cfg,
            effective_output_dir="/output",
            run_suffix=None,
            transcription_resources=transcription_resources,
            host_detection_result=host_detection_result,
            pipeline_metrics=pipeline_metrics,
        )

        # Verify HTTP HEAD was called to check file size
        mock_http_head.assert_called_once_with(episode.media_url, cfg.user_agent, cfg.timeout)

        # Verify speaker detector was NOT created or called
        mock_create_detector.assert_not_called()
        mock_detector.detect_speakers.assert_not_called()

        # Verify download args were still created (episode processing continues)
        self.assertEqual(len(download_args), 1)

        # Verify detected_speaker_names is None (speaker detection was skipped)
        _ = download_args[0]  # Verify args were created

    @patch("podcast_scraper.workflow.stages.processing.http_head")
    @patch("podcast_scraper.workflow.stages.processing.create_speaker_detector")
    def test_prepare_episode_download_args_no_content_length_header(
        self, mock_create_detector, mock_http_head
    ):
        """Test that speaker detection proceeds when Content-Length header is missing."""
        from podcast_scraper.workflow.stages import processing
        from podcast_scraper.workflow.types import HostDetectionResult, TranscriptionResources

        cfg = create_test_config(
            transcribe_missing=True,
            transcription_provider="openai",
            auto_speakers=True,
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
        )

        # Create proper RSS item as XML element
        import xml.etree.ElementTree as ET

        item = ET.Element("item")
        episode = models.Episode(
            idx=1,
            title="Test Episode",
            title_safe="Test_Episode",
            media_url="https://example.com/episode.mp3",
            transcript_urls=[],
            item=item,
        )

        # Mock HTTP HEAD response without Content-Length header
        mock_response = Mock()
        mock_response.headers = {}  # No Content-Length
        mock_http_head.return_value = mock_response

        mock_detector = Mock()
        mock_detector.detect_speakers.return_value = (["Speaker1"], set(), True)
        mock_create_detector.return_value = mock_detector

        transcription_resources = TranscriptionResources(
            transcription_provider=None,
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=threading.Lock(),
            saved_counter_lock=threading.Lock(),
        )
        host_detection_result = HostDetectionResult(
            cached_hosts=set(),
            heuristics=None,
            speaker_detector=mock_detector,  # Set detector so speaker detection proceeds
        )
        pipeline_metrics = metrics.Metrics()

        _ = processing.prepare_episode_download_args(
            episodes=[episode],
            cfg=cfg,
            effective_output_dir="/output",
            run_suffix=None,
            transcription_resources=transcription_resources,
            host_detection_result=host_detection_result,
            pipeline_metrics=pipeline_metrics,
        )

        # Verify speaker detection was called (proceeded despite missing header)
        mock_detector.detect_speakers.assert_called_once()

    @patch("podcast_scraper.workflow.stages.processing.http_head")
    @patch("podcast_scraper.workflow.stages.processing.create_speaker_detector")
    def test_prepare_episode_download_args_head_request_failed(
        self, mock_create_detector, mock_http_head
    ):
        """Test that speaker detection proceeds when HEAD request fails."""
        from podcast_scraper.workflow.stages import processing
        from podcast_scraper.workflow.types import HostDetectionResult, TranscriptionResources

        cfg = create_test_config(
            transcribe_missing=True,
            transcription_provider="openai",
            auto_speakers=True,
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
        )

        # Create proper RSS item as XML element
        import xml.etree.ElementTree as ET

        item = ET.Element("item")
        episode = models.Episode(
            idx=1,
            title="Test Episode",
            title_safe="Test_Episode",
            media_url="https://example.com/episode.mp3",
            transcript_urls=[],
            item=item,
        )

        # Mock HTTP HEAD request failure (returns None)
        mock_http_head.return_value = None

        mock_detector = Mock()
        mock_detector.detect_speakers.return_value = (["Speaker1"], set(), True)
        mock_create_detector.return_value = mock_detector

        transcription_resources = TranscriptionResources(
            transcription_provider=None,
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=threading.Lock(),
            saved_counter_lock=threading.Lock(),
        )
        host_detection_result = HostDetectionResult(
            cached_hosts=set(),
            heuristics=None,
            speaker_detector=mock_detector,  # Set detector so speaker detection proceeds
        )
        pipeline_metrics = metrics.Metrics()

        _ = processing.prepare_episode_download_args(
            episodes=[episode],
            cfg=cfg,
            effective_output_dir="/output",
            run_suffix=None,
            transcription_resources=transcription_resources,
            host_detection_result=host_detection_result,
            pipeline_metrics=pipeline_metrics,
        )

        # Verify speaker detection was called (proceeded despite HEAD failure)
        mock_detector.detect_speakers.assert_called_once()

    @patch("podcast_scraper.workflow.stages.processing.http_head")
    @patch("podcast_scraper.workflow.stages.processing.create_speaker_detector")
    def test_prepare_episode_download_args_invalid_content_length(
        self, mock_create_detector, mock_http_head
    ):
        """Test that speaker detection proceeds when Content-Length header is invalid."""
        from podcast_scraper.workflow.stages import processing
        from podcast_scraper.workflow.types import HostDetectionResult, TranscriptionResources

        cfg = create_test_config(
            transcribe_missing=True,
            transcription_provider="openai",
            auto_speakers=True,
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
        )

        # Create proper RSS item as XML element
        import xml.etree.ElementTree as ET

        item = ET.Element("item")
        episode = models.Episode(
            idx=1,
            title="Test Episode",
            title_safe="Test_Episode",
            media_url="https://example.com/episode.mp3",
            transcript_urls=[],
            item=item,
        )

        # Mock HTTP HEAD response with invalid Content-Length header
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "invalid"}  # Not a number
        mock_http_head.return_value = mock_response

        mock_detector = Mock()
        mock_detector.detect_speakers.return_value = (["Speaker1"], set(), True)
        mock_create_detector.return_value = mock_detector

        transcription_resources = TranscriptionResources(
            transcription_provider=None,
            temp_dir="/tmp",
            transcription_jobs=[],
            transcription_jobs_lock=threading.Lock(),
            saved_counter_lock=threading.Lock(),
        )
        host_detection_result = HostDetectionResult(
            cached_hosts=set(),
            heuristics=None,
            speaker_detector=mock_detector,  # Set detector so speaker detection proceeds
        )
        pipeline_metrics = metrics.Metrics()

        download_args = processing.prepare_episode_download_args(
            episodes=[episode],
            cfg=cfg,
            effective_output_dir="/output",
            run_suffix=None,
            transcription_resources=transcription_resources,
            host_detection_result=host_detection_result,
            pipeline_metrics=pipeline_metrics,
        )

        # Verify speaker detection was called (proceeded despite invalid header)
        mock_detector.detect_speakers.assert_called_once()

        # Verify detected_speaker_names contains the detected speakers
        args_tuple = download_args[0]
        detected_speaker_names = args_tuple[7]  # 8th element (0-indexed)
        self.assertEqual(detected_speaker_names, ["Speaker1"])


@pytest.mark.unit
class TestGetProviderPricing(unittest.TestCase):
    """Tests for _get_provider_pricing helper function."""

    def test_get_provider_pricing_openai_transcription(self):
        """Test that _get_provider_pricing routes to OpenAI provider for transcription."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            transcription_provider="openai",
            openai_transcription_model="whisper-1",
            openai_api_key="sk-test123",
        )

        pricing = helpers._get_provider_pricing(cfg, "openai", "transcription", "whisper-1")
        self.assertEqual(pricing, {"cost_per_minute": 0.006})

    def test_get_provider_pricing_openai_speaker_detection(self):
        """Test that _get_provider_pricing routes to OpenAI provider for speaker detection."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            speaker_detector_provider="openai",
            openai_speaker_model="gpt-4o-mini",
            openai_api_key="sk-test123",
        )

        pricing = helpers._get_provider_pricing(cfg, "openai", "speaker_detection", "gpt-4o-mini")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.15)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.60)

    def test_get_provider_pricing_openai_summarization(self):
        """Test that _get_provider_pricing routes to OpenAI provider for summarization."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            summary_provider="openai",
            openai_summary_model="gpt-4o",
            openai_api_key="sk-test123",
        )

        pricing = helpers._get_provider_pricing(cfg, "openai", "summarization", "gpt-4o")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 2.50)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 10.00)

    def test_get_provider_pricing_unsupported_provider(self):
        """Test that _get_provider_pricing returns empty dict for unsupported provider."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(rss_url="https://example.com/feed.xml")
        pricing = helpers._get_provider_pricing(cfg, "unsupported", "transcription", "model")
        self.assertEqual(pricing, {})


@pytest.mark.unit
class TestGenerateLLMCallSummary(unittest.TestCase):
    """Tests for _generate_llm_call_summary helper function."""

    def test_generate_llm_call_summary_no_llm_providers(self):
        """Test that summary is empty when no LLM providers are used."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            transcription_provider="whisper",  # Local, not LLM
            speaker_detector_provider="spacy",  # Local, not LLM
            summary_provider="transformers",  # Local, not LLM
        )
        pipeline_metrics = metrics.Metrics()

        summary = helpers._generate_llm_call_summary(cfg, pipeline_metrics)
        self.assertEqual(summary, [])

    def test_generate_llm_call_summary_transcription_only(self):
        """Test LLM call summary with transcription only."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            transcription_provider="openai",
            openai_transcription_model="whisper-1",
            openai_api_key="sk-test123",
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.record_llm_transcription_call(10.5)  # 10.5 minutes
        pipeline_metrics.record_llm_transcription_call(5.0)  # 5.0 minutes

        summary = helpers._generate_llm_call_summary(cfg, pipeline_metrics)
        self.assertEqual(len(summary), 2)  # One line for transcription, one for total
        self.assertIn("Transcription: 2 calls", summary[0])
        self.assertIn("15.5 minutes", summary[0])
        self.assertIn("$0.0930", summary[0])  # 15.5 * 0.006 = 0.093
        self.assertIn("whisper-1", summary[0])
        self.assertIn("Total estimated cost: $0.0930", summary[1])

    def test_generate_llm_call_summary_speaker_detection_only(self):
        """Test LLM call summary with speaker detection only."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            speaker_detector_provider="openai",
            openai_speaker_model="gpt-4o-mini",
            openai_api_key="sk-test123",
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.record_llm_speaker_detection_call(input_tokens=1000, output_tokens=500)
        pipeline_metrics.record_llm_speaker_detection_call(input_tokens=2000, output_tokens=750)

        summary = helpers._generate_llm_call_summary(cfg, pipeline_metrics)
        self.assertEqual(len(summary), 2)  # One line for speaker detection, one for total
        self.assertIn("Speaker Detection: 2 calls", summary[0])
        self.assertIn("3,000 input + 1,250 output tokens", summary[0])
        self.assertIn("gpt-4o-mini", summary[0])
        # Cost: (3000/1M * 0.15) + (1250/1M * 0.60) = 0.00045 + 0.00075 = 0.0012
        self.assertIn("$0.0012", summary[0])
        self.assertIn("Total estimated cost: $0.0012", summary[1])

    def test_generate_llm_call_summary_summarization_only(self):
        """Test LLM call summary with summarization only."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            summary_provider="openai",
            openai_summary_model="gpt-4o",
            openai_api_key="sk-test123",
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.record_llm_summarization_call(input_tokens=50000, output_tokens=10000)

        summary = helpers._generate_llm_call_summary(cfg, pipeline_metrics)
        self.assertEqual(len(summary), 2)  # One line for summarization, one for total
        self.assertIn("Summarization: 1 calls", summary[0])
        self.assertIn("50,000 input + 10,000 output tokens", summary[0])
        self.assertIn("gpt-4o", summary[0])
        # Cost: (50000/1M * 2.50) + (10000/1M * 10.00) = 0.125 + 0.10 = 0.225
        self.assertIn("$0.2250", summary[0])
        self.assertIn("Total estimated cost: $0.2250", summary[1])

    def test_generate_llm_call_summary_all_capabilities(self):
        """Test LLM call summary with all capabilities."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
            openai_transcription_model="whisper-1",
            openai_speaker_model="gpt-4o-mini",
            openai_summary_model="gpt-4o",
            openai_api_key="sk-test123",
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.record_llm_transcription_call(10.0)  # 10 minutes
        pipeline_metrics.record_llm_speaker_detection_call(input_tokens=1000, output_tokens=500)
        pipeline_metrics.record_llm_summarization_call(input_tokens=50000, output_tokens=10000)

        summary = helpers._generate_llm_call_summary(cfg, pipeline_metrics)
        self.assertEqual(len(summary), 4)  # 3 capability lines + 1 total line
        # Check all capabilities are included
        transcription_line = [s for s in summary if "Transcription" in s][0]
        speaker_line = [s for s in summary if "Speaker Detection" in s][0]
        summary_line = [s for s in summary if "Summarization" in s][0]
        total_line = [s for s in summary if "Total estimated cost" in s][0]

        self.assertIn("whisper-1", transcription_line)
        self.assertIn("gpt-4o-mini", speaker_line)
        self.assertIn("gpt-4o", summary_line)
        self.assertIn("Total estimated cost", total_line)

    def test_generate_llm_call_summary_no_calls(self):
        """Test LLM call summary when no calls were made."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            transcription_provider="openai",
            openai_api_key="sk-test123",
        )
        pipeline_metrics = metrics.Metrics()
        # No calls recorded

        summary = helpers._generate_llm_call_summary(cfg, pipeline_metrics)
        self.assertEqual(summary, [])

    def test_generate_llm_call_summary_gpt4o_mini_summarization(self):
        """Test LLM call summary with GPT-4o-mini for summarization (test model)."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            summary_provider="openai",
            openai_summary_model="gpt-4o-mini",
            openai_api_key="sk-test123",
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.record_llm_summarization_call(input_tokens=100000, output_tokens=20000)

        summary = helpers._generate_llm_call_summary(cfg, pipeline_metrics)
        self.assertIn("gpt-4o-mini", summary[0])
        # Cost: (100000/1M * 0.15) + (20000/1M * 0.60) = 0.015 + 0.012 = 0.027
        self.assertIn("$0.0270", summary[0])

    def test_generate_llm_call_summary_no_pricing_info(self):
        """Test LLM call summary when pricing info is not available (partial coverage)."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            speaker_detector_provider="openai",
            openai_speaker_model="gpt-4o-mini",
            openai_api_key="sk-test123",
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.record_llm_speaker_detection_call(100, 50)

        # Mock _get_provider_pricing to return pricing without input_cost_per_1m_tokens
        with patch("podcast_scraper.workflow.helpers._get_provider_pricing") as mock_pricing:
            mock_pricing.return_value = {}  # Empty pricing dict
            summary = helpers._generate_llm_call_summary(cfg, pipeline_metrics)

        # Should not include speaker detection in summary when pricing is missing
        self.assertEqual(summary, [])

    def test_generate_llm_call_summary_transcription_no_pricing(self):
        """Test LLM call summary when transcription pricing is missing."""
        from podcast_scraper.workflow import helpers

        cfg = create_test_config(
            transcription_provider="openai",
            openai_transcription_model="whisper-1",
            openai_api_key="sk-test123",
        )
        pipeline_metrics = metrics.Metrics()
        pipeline_metrics.record_llm_transcription_call(5.0)

        # Mock _get_provider_pricing to return pricing without cost_per_minute
        with patch("podcast_scraper.workflow.helpers._get_provider_pricing") as mock_pricing:
            mock_pricing.return_value = {}  # Empty pricing dict
            summary = helpers._generate_llm_call_summary(cfg, pipeline_metrics)

        # Should not include transcription in summary when pricing is missing
        self.assertEqual(summary, [])
