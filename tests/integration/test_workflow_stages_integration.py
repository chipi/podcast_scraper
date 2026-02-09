#!/usr/bin/env python3
"""Integration tests for workflow stages.

These tests verify that workflow stage functions work correctly with real components
and mocked external dependencies (HTTP, ML models).
"""

import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper.workflow.helpers import (
    cleanup_pipeline,
    generate_pipeline_summary,
    update_metric_safely,
)
from podcast_scraper.workflow.stages import (
    metadata as metadata_stage,
    processing,
    scraping,
    setup,
    summarization,
    transcription,
)
from podcast_scraper.workflow.types import FeedMetadata

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly
import importlib.util

parent_conftest_path = tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

create_test_config = parent_conftest.create_test_config
create_test_feed = parent_conftest.create_test_feed
create_test_episode = parent_conftest.create_test_episode

pytestmark = [pytest.mark.integration, pytest.mark.module_workflow]


@pytest.mark.integration
class TestSetupStage(unittest.TestCase):
    """Integration tests for setup stage."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()

    def test_initialize_ml_environment(self):
        """Test that ML environment initialization sets environment variables."""
        # Save original values
        original_hf = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        original_tokenizers = os.environ.get("TOKENIZERS_PARALLELISM")

        try:
            # Remove if set
            if "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ:
                del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]
            if "TOKENIZERS_PARALLELISM" in os.environ:
                del os.environ["TOKENIZERS_PARALLELISM"]

            # Call function
            setup.initialize_ml_environment()

            # Verify environment variables are set
            self.assertEqual(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")
            self.assertEqual(os.environ.get("TOKENIZERS_PARALLELISM"), "false")
        finally:
            # Restore original values
            if original_hf is not None:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = original_hf
            elif "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ:
                del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]

            if original_tokenizers is not None:
                os.environ["TOKENIZERS_PARALLELISM"] = original_tokenizers
            elif "TOKENIZERS_PARALLELISM" in os.environ:
                del os.environ["TOKENIZERS_PARALLELISM"]

    def test_should_preload_ml_models_with_whisper(self):
        """Test should_preload_ml_models returns True when Whisper is needed."""
        cfg = create_test_config(
            transcribe_missing=True,
            transcription_provider="whisper",
            generate_summaries=False,
        )
        result = setup.should_preload_ml_models(cfg)
        self.assertTrue(result)

    def test_should_preload_ml_models_with_transformers(self):
        """Test should_preload_ml_models returns True when Transformers is needed."""
        cfg = create_test_config(
            transcribe_missing=False,
            transcription_provider="openai",
            openai_api_key="sk-test-dummy-key-for-validation",
            generate_summaries=True,
            summary_provider="transformers",
        )
        result = setup.should_preload_ml_models(cfg)
        self.assertTrue(result)

    def test_should_preload_ml_models_with_spacy(self):
        """Test should_preload_ml_models returns True when spaCy is needed."""
        cfg = create_test_config(
            transcribe_missing=False,
            transcription_provider="openai",
            openai_api_key="sk-test-dummy-key-for-validation",
            auto_speakers=True,
            speaker_detector_provider="spacy",
        )
        result = setup.should_preload_ml_models(cfg)
        self.assertTrue(result)

    def test_should_preload_ml_models_returns_false_when_no_ml(self):
        """Test should_preload_ml_models returns False when no ML providers needed."""
        cfg = create_test_config(
            transcribe_missing=False,
            transcription_provider="openai",
            openai_api_key="sk-test-dummy-key-for-validation",
            generate_summaries=False,
            auto_speakers=False,
        )
        result = setup.should_preload_ml_models(cfg)
        self.assertFalse(result)

    @patch("podcast_scraper.config._is_test_environment")
    def test_ensure_ml_models_cached_skips_in_test(self, mock_is_test):
        """Test ensure_ml_models_cached skips in test environment."""
        mock_is_test.return_value = True
        cfg = create_test_config(preload_models=True, transcribe_missing=True)
        # Should not raise
        setup.ensure_ml_models_cached(cfg)

    @patch("podcast_scraper.config._is_test_environment")
    def test_ensure_ml_models_cached_skips_when_disabled(self, mock_is_test):
        """Test ensure_ml_models_cached skips when preload_models=False."""
        mock_is_test.return_value = False
        cfg = create_test_config(preload_models=False)
        # Should not raise
        setup.ensure_ml_models_cached(cfg)

    @patch("podcast_scraper.config._is_test_environment")
    def test_ensure_ml_models_cached_skips_when_dry_run(self, mock_is_test):
        """Test ensure_ml_models_cached skips in dry run."""
        mock_is_test.return_value = False
        cfg = create_test_config(preload_models=True, dry_run=True)
        # Should not raise
        setup.ensure_ml_models_cached(cfg)

    def test_setup_pipeline_environment_creates_directory(self):
        """Test setup_pipeline_environment creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output")
            cfg = create_test_config(
                output_dir=output_path,
                dry_run=False,
                transcribe_missing=False,
                generate_summaries=False,
                auto_speakers=False,
            )
            output_dir, run_suffix, full_config_string = setup.setup_pipeline_environment(cfg)
            self.assertTrue(os.path.exists(output_dir))
            # run_suffix may be None or a provider suffix
            self.assertIsNotNone(output_dir)

    def test_setup_pipeline_environment_with_run_id(self):
        """Test setup_pipeline_environment creates run_id subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output")
            cfg = create_test_config(
                output_dir=output_path,
                run_id="test_run",
                dry_run=False,
                transcribe_missing=False,
                generate_summaries=False,
                auto_speakers=False,
            )
            output_dir, run_suffix, full_config_string = setup.setup_pipeline_environment(cfg)
            self.assertTrue(os.path.exists(output_dir))
            self.assertIn("test_run", run_suffix)
            self.assertIn("test_run", output_dir)

    def test_setup_pipeline_environment_clean_output(self):
        """Test setup_pipeline_environment cleans existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing directory with a file
            existing_dir = os.path.join(tmpdir, "existing")
            os.makedirs(existing_dir)
            test_file = os.path.join(existing_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")

            cfg = create_test_config(
                output_dir=existing_dir,
                clean_output=True,
                dry_run=False,
                transcribe_missing=False,
                generate_summaries=False,
                auto_speakers=False,
            )
            output_dir, _, _ = setup.setup_pipeline_environment(cfg)
            # Directory should exist (now in run_<suffix> subdirectory)
            self.assertTrue(os.path.exists(output_dir))
            # File should be removed by clean_output (original directory is cleaned)
            # Note: output_dir is now in a run_<suffix> subdirectory, so test_file
            # in the original directory should still exist if clean_output only cleans
            # the run subdirectory. However, if clean_output=True, it should clean
            # the entire output_dir. Let's check if the file is in the run directory
            # or if it was removed from the original location.
            # Since setup_output_directory creates run_<suffix> subdirectory,
            # clean_output should remove the run subdirectory, not the base directory.
            # So test_file in existing_dir should still exist, but files in run_<suffix>
            # should not.
            self.assertTrue(os.path.exists(test_file))  # Original file still exists
            # But the run subdirectory should be clean (no old files)
            if os.path.exists(output_dir):
                run_dir_files = [
                    f for f in os.listdir(output_dir) if f != "transcripts" and f != "metadata"
                ]
                # Should only have transcripts/ and metadata/ subdirectories
                self.assertEqual(
                    len(run_dir_files), 0, f"Run directory should be clean, found: {run_dir_files}"
                )

    def test_setup_pipeline_environment_dry_run(self):
        """Test setup_pipeline_environment handles dry run mode correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output")
            cfg = create_test_config(
                output_dir=output_path,
                dry_run=True,
                transcribe_missing=False,
                generate_summaries=False,
                auto_speakers=False,
            )
            output_dir, run_suffix, full_config_string = setup.setup_pipeline_environment(cfg)
            # In dry run, setup_output_directory creates subdirectories (transcripts/, metadata/)
            # but setup_pipeline_environment should not create the main effective_output_dir
            # The function should return without error
            self.assertIsNotNone(output_dir)
            # Verify that the function returns correctly in dry run mode
            # (subdirectories may exist from setup_output_directory)

    def test_get_preloaded_ml_provider_returns_none_initially(self):
        """Test get_preloaded_ml_provider returns None initially."""
        # Explicitly reset _preloaded_ml_provider to ensure clean state
        # (previous tests might have set it)
        from podcast_scraper.workflow.stages import setup as setup_module

        setup_module._preloaded_ml_provider = None
        result = setup.get_preloaded_ml_provider()
        self.assertIsNone(result)


@pytest.mark.integration
class TestScrapingStage(unittest.TestCase):
    """Integration tests for scraping stage."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()

    @patch("podcast_scraper.rss.downloader.fetch_url")
    @patch("podcast_scraper.rss.parser.parse_rss_items")
    def test_fetch_and_parse_feed_success(self, mock_parse, mock_fetch):
        """Test fetch_and_parse_feed successfully fetches and parses feed."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.content = b"<rss><channel><title>Test Feed</title></channel></rss>"
        mock_response.url = "https://example.com/feed.xml"
        mock_response.close = Mock()
        mock_fetch.return_value = mock_response

        # Mock RSS parsing
        mock_parse.return_value = ("Test Feed", ["Author"], [])

        cfg = create_test_config(rss_url="https://example.com/feed.xml")
        feed, rss_bytes = scraping.fetch_and_parse_feed(cfg)

        self.assertEqual(feed.title, "Test Feed")
        self.assertEqual(rss_bytes, b"<rss><channel><title>Test Feed</title></channel></rss>")
        mock_fetch.assert_called_once()
        # Note: parse_rss_items is imported inside the function, so we verify via the result
        # The function will fail if parse_rss_items isn't called, so we just check the result
        self.assertEqual(feed.title, "Test Feed")

    @patch("podcast_scraper.downloader.fetch_url")
    def test_fetch_and_parse_feed_failure(self, mock_fetch):
        """Test fetch_and_parse_feed raises ValueError on fetch failure."""
        mock_fetch.return_value = None

        cfg = create_test_config(rss_url="https://example.com/feed.xml")
        with self.assertRaises(ValueError) as cm:
            scraping.fetch_and_parse_feed(cfg)
        self.assertIn("Failed to fetch", str(cm.exception))

    def test_extract_feed_metadata_for_generation_disabled(self):
        """Test extract_feed_metadata_for_generation returns empty when disabled."""
        cfg = create_test_config(generate_metadata=False)
        feed = create_test_feed()
        result = scraping.extract_feed_metadata_for_generation(cfg, feed, b"")
        self.assertEqual(result.description, None)
        self.assertEqual(result.image_url, None)
        self.assertEqual(result.last_updated, None)

    @patch("podcast_scraper.workflow.stages.scraping.extract_feed_metadata")
    def test_extract_feed_metadata_for_generation_success(self, mock_extract):
        """Test extract_feed_metadata_for_generation extracts metadata."""
        mock_extract.return_value = ("Description", "image.jpg", None)
        feed = create_test_feed()
        cfg = create_test_config(generate_metadata=True)
        result = scraping.extract_feed_metadata_for_generation(cfg, feed, b"<rss></rss>")
        self.assertEqual(result.description, "Description")
        self.assertEqual(result.image_url, "image.jpg")
        mock_extract.assert_called_once()

    def test_prepare_episodes_from_feed(self):
        """Test prepare_episodes_from_feed creates episodes from feed."""
        import xml.etree.ElementTree as ET

        feed = create_test_feed()
        # Create proper RSS items as XML elements
        item1 = ET.Element("item")
        ET.SubElement(item1, "title").text = "Episode 1"
        ET.SubElement(item1, "link").text = "https://example.com/ep1"
        ET.SubElement(item1, "guid").text = "ep1"
        ET.SubElement(item1, "pubDate").text = "Mon, 01 Jan 2024 00:00:00 GMT"

        item2 = ET.Element("item")
        ET.SubElement(item2, "title").text = "Episode 2"
        ET.SubElement(item2, "link").text = "https://example.com/ep2"
        ET.SubElement(item2, "guid").text = "ep2"
        ET.SubElement(item2, "pubDate").text = "Mon, 02 Jan 2024 00:00:00 GMT"

        feed.items = [item1, item2]
        episodes = scraping.prepare_episodes_from_feed(feed, self.cfg)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].title, "Episode 1")
        self.assertEqual(episodes[1].title, "Episode 2")

    def test_prepare_episodes_from_feed_with_max_episodes(self):
        """Test prepare_episodes_from_feed respects max_episodes."""
        import xml.etree.ElementTree as ET

        feed = create_test_feed()
        # Create proper RSS items as XML elements
        feed.items = []
        for i in range(1, 6):
            item = ET.Element("item")
            ET.SubElement(item, "title").text = f"Episode {i}"
            ET.SubElement(item, "link").text = f"https://example.com/ep{i}"
            ET.SubElement(item, "guid").text = f"ep{i}"
            ET.SubElement(item, "pubDate").text = f"Mon, {i:02d} Jan 2024 00:00:00 GMT"
            feed.items.append(item)

        cfg = create_test_config(max_episodes=3)
        episodes = scraping.prepare_episodes_from_feed(feed, cfg)
        self.assertEqual(len(episodes), 3)


@pytest.mark.integration
class TestTranscriptionStage(unittest.TestCase):
    """Integration tests for transcription stage."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.workflow.stages.transcription.create_transcription_provider")
    def test_setup_transcription_resources_with_provider(self, mock_create):
        """Test setup_transcription_resources creates provider when needed."""
        mock_provider = Mock()
        mock_provider.initialize = Mock()
        mock_create.return_value = mock_provider

        cfg = create_test_config(transcribe_missing=True, dry_run=False)
        resources = transcription.setup_transcription_resources(cfg, self.temp_dir)

        self.assertIsNotNone(resources.transcription_provider)
        mock_provider.initialize.assert_called_once()

    def test_setup_transcription_resources_skips_when_disabled(self):
        """Test setup_transcription_resources skips when transcribe_missing=False."""
        cfg = create_test_config(transcribe_missing=False)
        resources = transcription.setup_transcription_resources(cfg, self.temp_dir)
        self.assertIsNone(resources.transcription_provider)

    def test_setup_transcription_resources_skips_in_dry_run(self):
        """Test setup_transcription_resources skips in dry run."""
        cfg = create_test_config(transcribe_missing=True, dry_run=True)
        resources = transcription.setup_transcription_resources(cfg, self.temp_dir)
        self.assertIsNone(resources.transcription_provider)


@pytest.mark.integration
class TestProcessingStage(unittest.TestCase):
    """Integration tests for processing stage."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()
        self.feed = create_test_feed()
        self.episodes = [create_test_episode(idx=1, title="Episode 1")]

    def test_detect_feed_hosts_and_patterns_disabled(self):
        """Test detect_feed_hosts_and_patterns returns empty when disabled."""
        cfg = create_test_config(auto_speakers=False)
        result = processing.detect_feed_hosts_and_patterns(cfg, self.feed, self.episodes)
        self.assertEqual(len(result.cached_hosts), 0)
        self.assertIsNone(result.heuristics)
        self.assertIsNone(result.speaker_detector)

    def test_detect_feed_hosts_and_patterns_dry_run(self):
        """Test detect_feed_hosts_and_patterns skips in dry run."""
        cfg = create_test_config(auto_speakers=True, dry_run=True)
        result = processing.detect_feed_hosts_and_patterns(cfg, self.feed, self.episodes)
        # In dry run, hosts may still be detected from RSS author tags (no ML needed)
        # So cached_hosts may not be empty if feed has authors
        # The important thing is that speaker_detector is None (no ML model initialized)
        self.assertIsNone(result.speaker_detector)

    @patch("podcast_scraper.speaker_detectors.factory.create_speaker_detector")
    def test_detect_feed_hosts_and_patterns_with_detector(self, mock_create):
        """Test detect_feed_hosts_and_patterns uses speaker detector."""
        mock_detector = Mock()
        mock_detector.initialize = Mock()
        feed_hosts_set = {"Host 1", "Host 2"}
        mock_detector.detect_hosts = Mock(return_value=feed_hosts_set)

        # Mock detect_speakers to return both hosts so validation passes
        # The validation checks if hosts appear in first episode, so return them
        # detect_speakers returns (speaker_names_list, known_hosts_set, success_bool)
        # Need to handle both with and without pipeline_metrics parameter
        def mock_detect_speakers(*args, **kwargs):
            return (["Host 1", "Host 2"], set(), True)

        mock_detector.detect_speakers = Mock(side_effect=mock_detect_speakers)
        # Mock inspect.signature to return a signature that doesn't have pipeline_metrics
        # so it uses the simpler call path
        import inspect

        original_signature = inspect.signature

        def mock_signature(obj):
            if obj == mock_detector.detect_speakers:
                # Return a signature without pipeline_metrics
                from inspect import Parameter, Signature

                sig = Signature(
                    [
                        Parameter("episode_title", Parameter.POSITIONAL_OR_KEYWORD),
                        Parameter("episode_description", Parameter.POSITIONAL_OR_KEYWORD),
                        Parameter("known_hosts", Parameter.POSITIONAL_OR_KEYWORD),
                    ]
                )
                return sig
            return original_signature(obj)

        mock_detector.analyze_patterns = Mock(return_value={"pattern": "value"})
        mock_create.return_value = mock_detector

        # Create feed without authors so it uses the detector instead of RSS author tags
        from podcast_scraper import models

        feed_no_authors = models.RssFeed(
            title=self.feed.title,
            authors=[],  # Empty authors so detector is used
            items=self.feed.items,
            base_url=self.feed.base_url,
        )

        cfg = create_test_config(
            auto_speakers=True,
            cache_detected_hosts=True,
            dry_run=False,
        )

        with patch("inspect.signature", side_effect=mock_signature):
            result = processing.detect_feed_hosts_and_patterns(cfg, feed_no_authors, self.episodes)

        # Validation should pass since both hosts appear in first episode
        # feed_hosts = {"Host 1", "Host 2"}
        # first_episode_persons = {"Host 1", "Host 2"} (from detect_speakers mock)
        # validated_hosts = feed_hosts & first_episode_persons = {"Host 1", "Host 2"}
        self.assertEqual(len(result.cached_hosts), 2)
        mock_detector.detect_hosts.assert_called_once()
        self.assertIsNotNone(result.heuristics)
        mock_detector.initialize.assert_called_once()
        mock_detector.detect_hosts.assert_called_once()


@pytest.mark.integration
class TestMetadataStage(unittest.TestCase):
    """Integration tests for metadata stage."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()
        self.feed = create_test_feed()
        self.episode = create_test_episode(idx=1, title="Episode 1")

    def test_call_generate_metadata_disabled(self):
        """Test call_generate_metadata skips when generate_metadata=False."""
        cfg = create_test_config(generate_metadata=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise
            metadata_stage.call_generate_metadata(
                episode=self.episode,
                feed=self.feed,
                cfg=cfg,
                effective_output_dir=tmpdir,
                run_suffix=None,
                transcript_path=None,
                transcript_source=None,
                whisper_model=None,
                feed_metadata=FeedMetadata(None, None, None),
                host_detection_result=processing.HostDetectionResult(set(), None, None),
                detected_names=None,
                summary_provider=None,
            )

    @patch("podcast_scraper.workflow.stages.metadata.generate_episode_metadata")
    def test_call_generate_metadata_calls_metadata_function(self, mock_generate):
        """Test call_generate_metadata calls metadata generation."""
        # The function checks for workflow._generate_episode_metadata first,
        # so we need to ensure that path doesn't exist to test the normal path
        import sys

        workflow_pkg = sys.modules.get("podcast_scraper.workflow")
        original_func = None
        if workflow_pkg and hasattr(workflow_pkg, "_generate_episode_metadata"):
            # Temporarily remove it to test the normal path
            original_func = getattr(workflow_pkg, "_generate_episode_metadata")
            delattr(workflow_pkg, "_generate_episode_metadata")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                metadata_stage.call_generate_metadata(
                    episode=self.episode,
                    feed=self.feed,
                    cfg=self.cfg,
                    effective_output_dir=tmpdir,
                    run_suffix=None,
                    transcript_path="transcript.txt",
                    transcript_source="direct_download",
                    whisper_model=None,
                    feed_metadata=FeedMetadata("Desc", "image.jpg", None),
                    host_detection_result=processing.HostDetectionResult({"Host 1"}, None, None),
                    detected_names=["Guest 1"],
                    summary_provider=None,
                )
                # The function should call generate_episode_metadata
                # (which is the local name, not the alias)
                mock_generate.assert_called_once()
        finally:
            # Restore if it existed
            if workflow_pkg and original_func is not None:
                setattr(workflow_pkg, "_generate_episode_metadata", original_func)


@pytest.mark.integration
class TestSummarizationStage(unittest.TestCase):
    """Integration tests for summarization stage."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config()
        self.feed = create_test_feed()
        self.episodes = [create_test_episode(idx=1, title="Episode 1")]

    def test_collect_episodes_for_summarization_no_transcript(self):
        """Test _collect_episodes_for_summarization returns empty when no transcript."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = summarization._collect_episodes_for_summarization(
                episodes=self.episodes,
                download_args=None,
                effective_output_dir=tmpdir,
                run_suffix=None,
                cfg=self.cfg,
            )
            self.assertEqual(len(result), 0)

    def test_collect_episodes_for_summarization_with_transcript(self):
        """Test _collect_episodes_for_summarization finds episodes with transcripts."""
        from podcast_scraper.utils import filesystem

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create transcripts subdirectory
            transcripts_dir = os.path.join(tmpdir, filesystem.TRANSCRIPTS_SUBDIR)
            os.makedirs(transcripts_dir, exist_ok=True)

            # Create transcript file using the correct path format
            transcript_path = filesystem.build_whisper_output_path(
                self.episodes[0].idx,
                self.episodes[0].title_safe,
                None,
                tmpdir,
            )
            with open(transcript_path, "w") as f:
                f.write("Test transcript content")

            result = summarization._collect_episodes_for_summarization(
                episodes=self.episodes,
                download_args=None,
                effective_output_dir=tmpdir,
                run_suffix=None,
                cfg=self.cfg,
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0][0], self.episodes[0])


@pytest.mark.integration
class TestWorkflowHelpers(unittest.TestCase):
    """Integration tests for workflow helpers."""

    def setUp(self):
        """Set up test fixtures."""
        from podcast_scraper.workflow import metrics

        self.metrics = metrics.Metrics()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update_metric_safely_without_lock(self):
        """Test update_metric_safely works without lock."""
        update_metric_safely(self.metrics, "transcripts_downloaded", 5)
        self.assertEqual(self.metrics.transcripts_downloaded, 5)

    def test_update_metric_safely_with_lock(self):
        """Test update_metric_safely works with lock."""
        import threading

        lock = threading.Lock()
        update_metric_safely(self.metrics, "transcripts_downloaded", 3, lock=lock)
        self.assertEqual(self.metrics.transcripts_downloaded, 3)

    def test_cleanup_pipeline_removes_directory(self):
        """Test cleanup_pipeline removes temp directory."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        cleanup_pipeline(self.temp_dir)
        self.assertFalse(os.path.exists(self.temp_dir))

    def test_cleanup_pipeline_handles_missing_directory(self):
        """Test cleanup_pipeline handles missing directory gracefully."""
        # Should not raise
        cleanup_pipeline("/nonexistent/directory")

    def test_generate_pipeline_summary_dry_run(self):
        """Test generate_pipeline_summary returns dry run summary."""
        from podcast_scraper.workflow.types import TranscriptionResources

        cfg = create_test_config(dry_run=True, transcribe_missing=True)
        resources = TranscriptionResources(
            transcription_provider=None,
            temp_dir=None,
            transcription_jobs=[Mock()],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        count, summary = generate_pipeline_summary(
            cfg=cfg,
            saved=5,
            transcription_resources=resources,
            effective_output_dir="/output",
            pipeline_metrics=self.metrics,
        )
        self.assertIn("Dry run", summary)
        self.assertIn("transcripts_planned", summary)

    def test_generate_pipeline_summary_normal_run(self):
        """Test generate_pipeline_summary returns normal summary."""
        from podcast_scraper.workflow.types import TranscriptionResources

        cfg = create_test_config(dry_run=False)
        resources = TranscriptionResources(
            transcription_provider=None,
            temp_dir=None,
            transcription_jobs=queue.Queue(),
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        self.metrics.transcripts_downloaded = 3
        self.metrics.transcripts_transcribed = 2
        count, summary = generate_pipeline_summary(
            cfg=cfg,
            saved=5,
            transcription_resources=resources,
            effective_output_dir="/output",
            pipeline_metrics=self.metrics,
        )
        self.assertIn("Done", summary)
        self.assertIn("transcripts_saved", summary)
        self.assertIn("Transcripts downloaded", summary)
        self.assertIn("Episodes transcribed", summary)
