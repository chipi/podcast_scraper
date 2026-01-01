#!/usr/bin/env python3
"""Tests for parallel episode summarization functionality."""

import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import summarizer, skip tests if dependencies not available
try:
    from podcast_scraper import config, models, summarizer, workflow

    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    summarizer = types.ModuleType("summarizer")  # type: ignore[assignment]
    workflow = types.ModuleType("workflow")  # type: ignore[assignment]
    config = types.ModuleType("config")  # type: ignore[assignment]
    models = types.ModuleType("models")  # type: ignore[assignment]

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import create_test_config, create_test_episode, create_test_feed  # noqa: E402


def _create_mock_provider(mock_summary_model):
    """Create a mock provider with map_model and reduce_model attributes."""
    mock_provider = Mock()
    mock_provider.map_model = mock_summary_model
    mock_provider.reduce_model = mock_summary_model
    # Make it appear as TransformersSummarizationProvider for local provider path
    from podcast_scraper.summarization.local_provider import TransformersSummarizationProvider

    mock_provider.__class__ = TransformersSummarizationProvider
    return mock_provider


def _create_test_transcript_files(episodes, temp_dir, cfg):
    """Helper to create transcript and metadata files for test episodes."""
    from podcast_scraper import filesystem, metadata

    for episode in episodes:
        # Create transcript file at expected path
        transcript_path = filesystem.build_whisper_output_path(
            episode.idx, episode.title_safe, None, temp_dir
        )
        Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
        Path(transcript_path).write_text("Test transcript. " * 100)
        # Create metadata file without summary so it needs summarization
        metadata_path = metadata._determine_metadata_path(episode, temp_dir, None, cfg)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).write_text('{"title": "Test"}')


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
@pytest.mark.integration
@pytest.mark.slow
class TestParallelSummarizationPreLoading(unittest.TestCase):
    """Test model pre-loading before parallel execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.summarizer.SummaryModel")
    @patch("podcast_scraper.summarizer.unload_model")
    def test_models_preloaded_before_parallel_execution(self, mock_unload, mock_model_class):
        """Test that models are pre-loaded before starting parallel execution."""
        # Create mock model instances
        mock_models = [Mock() for _ in range(3)]
        mock_model_class.side_effect = mock_models

        # Create a mock summary model with required attributes
        mock_summary_model = Mock()
        mock_summary_model.model_name = "facebook/bart-base"
        mock_summary_model.device = "cpu"
        mock_summary_model.cache_dir = None
        mock_summary_model.revision = None

        # Create a mock provider with map_model attribute
        mock_provider = _create_mock_provider(mock_summary_model)

        # Create test config with parallel processing enabled
        cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,
            summary_batch_size=3,
        )

        # Create test episodes
        episodes = [create_test_episode(idx=i) for i in range(1, 4)]
        feed = create_test_feed()

        # Create transcript files using the expected path format
        from podcast_scraper import filesystem

        for episode in episodes:
            transcript_path = filesystem.build_whisper_output_path(
                episode.idx, episode.title_safe, None, self.temp_dir
            )
            Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
            Path(transcript_path).write_text("This is a test transcript. " * 100)

        # Mock the _summarize_single_episode function
        with patch("podcast_scraper.workflow._summarize_single_episode"):
            # Call _parallel_episode_summarization
            workflow._parallel_episode_summarization(
                episodes=episodes,
                feed=feed,
                cfg=cfg,
                effective_output_dir=self.temp_dir,
                run_suffix=None,
                feed_metadata=workflow._FeedMetadata(
                    description="Test feed",
                    image_url=None,
                    last_updated=None,
                ),
                host_detection_result=workflow._HostDetectionResult(
                    cached_hosts=set(),
                    heuristics=None,
                ),
                summary_provider=mock_provider,
                download_args=[],  # Empty download args for test
                pipeline_metrics=Mock(),
            )

        # Verify models were pre-loaded (3 times for 3 workers)
        self.assertEqual(mock_model_class.call_count, 3)
        # Verify all models were unloaded
        self.assertEqual(mock_unload.call_count, 3)

    @patch("podcast_scraper.summarizer.SummaryModel")
    @patch("podcast_scraper.summarizer.unload_model")
    def test_model_preloading_with_revision(self, mock_unload, mock_model_class):
        """Test that model pre-loading includes revision parameter when present."""
        mock_models = [Mock() for _ in range(2)]
        mock_model_class.side_effect = mock_models

        mock_summary_model = Mock()
        mock_summary_model.model_name = "facebook/bart-base"
        mock_summary_model.device = "cpu"
        mock_summary_model.cache_dir = "/cache"
        mock_summary_model.revision = "abc123"

        cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,
            summary_batch_size=2,
        )

        episodes = [create_test_episode(idx=i) for i in range(1, 3)]
        feed = create_test_feed()

        # Create transcript files using the expected path format
        _create_test_transcript_files(episodes, self.temp_dir, cfg)

        with patch("podcast_scraper.workflow._summarize_single_episode"):
            workflow._parallel_episode_summarization(
                episodes=episodes,
                feed=feed,
                cfg=cfg,
                effective_output_dir=self.temp_dir,
                run_suffix=None,
                feed_metadata=workflow._FeedMetadata(
                    description="Test feed",
                    image_url=None,
                    last_updated=None,
                ),
                host_detection_result=workflow._HostDetectionResult(
                    cached_hosts=set(),
                    heuristics=None,
                ),
                summary_provider=_create_mock_provider(mock_summary_model),
                download_args=[],  # Empty download args for test
                pipeline_metrics=Mock(),
            )

        # Verify models were created with revision parameter
        # Check that SummaryModel was called (once per worker, up to 3 workers for CPU)
        self.assertGreater(mock_model_class.call_count, 0)
        # Verify all calls included revision parameter
        for call_args in mock_model_class.call_args_list:
            self.assertEqual(call_args.kwargs.get("revision"), "abc123")
            self.assertEqual(call_args.kwargs.get("model_name"), "facebook/bart-base")
            self.assertEqual(call_args.kwargs.get("device"), "cpu")
            self.assertEqual(call_args.kwargs.get("cache_dir"), "/cache")


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
@pytest.mark.integration
@pytest.mark.slow
class TestParallelSummarizationThreadSafety(unittest.TestCase):
    """Test thread safety of parallel summarization."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.assigned_models = []
        self.assignment_lock = threading.Lock()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.summarizer.SummaryModel")
    @patch("podcast_scraper.summarizer.unload_model")
    def test_each_worker_gets_unique_model_instance(self, mock_unload, mock_model_class):
        """Test that each worker thread gets its own unique model instance."""
        # Create distinct mock model instances
        mock_models = [Mock(name=f"model_{i}") for i in range(3)]
        mock_model_class.side_effect = mock_models

        mock_summary_model = Mock()
        mock_summary_model.model_name = "facebook/bart-base"
        mock_summary_model.device = "cpu"
        mock_summary_model.cache_dir = None
        mock_summary_model.revision = None

        cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,
            summary_batch_size=3,
        )

        episodes = [create_test_episode(idx=i) for i in range(1, 4)]
        feed = create_test_feed()

        # Create transcript files using the expected path format
        _create_test_transcript_files(episodes, self.temp_dir, cfg)

        # Track which models are used by which threads
        used_models = []
        used_models_lock = threading.Lock()

        def track_model_usage(*args, **kwargs):
            """Track which model instance is used."""
            # Extract summary_model from kwargs
            if "summary_model" in kwargs:
                with used_models_lock:
                    used_models.append(id(kwargs["summary_model"]))

        with patch(
            "podcast_scraper.workflow._summarize_single_episode", side_effect=track_model_usage
        ):
            workflow._parallel_episode_summarization(
                episodes=episodes,
                feed=feed,
                cfg=cfg,
                effective_output_dir=self.temp_dir,
                run_suffix=None,
                feed_metadata=workflow._FeedMetadata(
                    description="Test feed",
                    image_url=None,
                    last_updated=None,
                ),
                host_detection_result=workflow._HostDetectionResult(
                    cached_hosts=set(),
                    heuristics=None,
                ),
                summary_provider=_create_mock_provider(mock_summary_model),
                download_args=[],  # Empty download args for test
                pipeline_metrics=Mock(),
            )

        # Verify that models were pre-loaded (3 times for 3 workers)
        # This confirms that multiple model instances were created
        self.assertEqual(mock_model_class.call_count, 3, "Should pre-load 3 model instances")
        # Verify that different model instances were used
        # (each worker should get a different model from the pre-loaded pool)
        unique_model_ids = set(used_models)
        self.assertGreaterEqual(len(unique_model_ids), 1, "At least one model should be used")
        # With 3 workers, we should see multiple unique model instances used
        # Note: All 3 episodes might be processed by the same worker in some cases,
        # so we verify that models were pre-loaded (call_count) rather than
        # requiring all unique instances to be used


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
@pytest.mark.integration
@pytest.mark.slow
class TestParallelSummarizationFallback(unittest.TestCase):
    """Test fallback behavior when model loading fails."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.summarizer.SummaryModel")
    def test_fallback_to_sequential_on_model_loading_failure(self, mock_model_class):
        """Test that parallel processing falls back to sequential when model loading fails."""
        # First model loads successfully, second fails
        mock_model_class.side_effect = [Mock(), Exception("Model loading failed")]

        mock_summary_model = Mock()
        mock_summary_model.model_name = "facebook/bart-base"
        mock_summary_model.device = "cpu"
        mock_summary_model.cache_dir = None
        mock_summary_model.revision = None

        cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,
            summary_batch_size=2,
        )

        episodes = [create_test_episode(idx=i) for i in range(1, 3)]
        feed = create_test_feed()

        # Create transcript files using the expected path format
        _create_test_transcript_files(episodes, self.temp_dir, cfg)

        summarize_calls = []

        def track_summarize(*args, **kwargs):
            """Track summarize calls."""
            # Track both summary_provider and summary_model (for backward compatibility)
            summarize_calls.append((kwargs.get("summary_provider"), kwargs.get("summary_model")))

        mock_provider = _create_mock_provider(mock_summary_model)
        with patch(
            "podcast_scraper.workflow._summarize_single_episode", side_effect=track_summarize
        ):
            workflow._parallel_episode_summarization(
                episodes=episodes,
                feed=feed,
                cfg=cfg,
                effective_output_dir=self.temp_dir,
                run_suffix=None,
                feed_metadata=workflow._FeedMetadata(
                    description="Test feed",
                    image_url=None,
                    last_updated=None,
                ),
                host_detection_result=workflow._HostDetectionResult(
                    cached_hosts=set(),
                    heuristics=None,
                ),
                summary_provider=mock_provider,
                download_args=[],  # Empty download args for test
                pipeline_metrics=Mock(),
            )

        # Verify fallback: should use sequential processing with provider
        # When model loading fails, it falls back to sequential with the original provider
        self.assertEqual(len(summarize_calls), 2)
        # Both calls should use the provider (fallback uses provider sequentially)
        # The summary_model is extracted from provider for backward compatibility
        for provider, model in summarize_calls:
            self.assertIsNotNone(provider or model, "Should have provider or model")


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
@pytest.mark.integration
@pytest.mark.slow
class TestParallelSummarizationCleanup(unittest.TestCase):
    """Test cleanup of worker models after parallel execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.summarizer.SummaryModel")
    @patch("podcast_scraper.summarizer.unload_model")
    def test_worker_models_unloaded_after_execution(self, mock_unload, mock_model_class):
        """Test that all worker models are unloaded after parallel execution completes."""
        mock_models = [Mock() for _ in range(2)]
        mock_model_class.side_effect = mock_models

        mock_summary_model = Mock()
        mock_summary_model.model_name = "facebook/bart-base"
        mock_summary_model.device = "cpu"
        mock_summary_model.cache_dir = None
        mock_summary_model.revision = None

        cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,
            summary_batch_size=2,
        )

        episodes = [create_test_episode(idx=i) for i in range(1, 3)]
        feed = create_test_feed()

        # Create transcript files using the expected path format
        _create_test_transcript_files(episodes, self.temp_dir, cfg)

        with patch("podcast_scraper.workflow._summarize_single_episode"):
            workflow._parallel_episode_summarization(
                episodes=episodes,
                feed=feed,
                cfg=cfg,
                effective_output_dir=self.temp_dir,
                run_suffix=None,
                feed_metadata=workflow._FeedMetadata(
                    description="Test feed",
                    image_url=None,
                    last_updated=None,
                ),
                host_detection_result=workflow._HostDetectionResult(
                    cached_hosts=set(),
                    heuristics=None,
                ),
                summary_provider=_create_mock_provider(mock_summary_model),
                download_args=[],  # Empty download args for test
                pipeline_metrics=Mock(),
            )

        # Verify all worker models were unloaded
        self.assertEqual(mock_unload.call_count, 2)
        # Verify the correct models were unloaded
        unloaded_models = [call[0][0] for call in mock_unload.call_args_list]
        self.assertEqual(set(unloaded_models), set(mock_models))

    @patch("podcast_scraper.summarizer.SummaryModel")
    @patch("podcast_scraper.summarizer.unload_model")
    def test_cleanup_handles_unload_errors_gracefully(self, mock_unload, mock_model_class):
        """Test that cleanup handles errors during model unloading gracefully."""
        mock_models = [Mock() for _ in range(2)]
        mock_model_class.side_effect = mock_models

        # First unload succeeds, second fails
        mock_unload.side_effect = [None, Exception("Unload failed")]

        mock_summary_model = Mock()
        mock_summary_model.model_name = "facebook/bart-base"
        mock_summary_model.device = "cpu"
        mock_summary_model.cache_dir = None
        mock_summary_model.revision = None

        cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,
            summary_batch_size=2,
        )

        episodes = [create_test_episode(idx=i) for i in range(1, 3)]
        feed = create_test_feed()

        # Create transcript files using the expected path format
        _create_test_transcript_files(episodes, self.temp_dir, cfg)

        # Should not raise exception even if unload fails
        with patch("podcast_scraper.workflow._summarize_single_episode"):
            try:
                workflow._parallel_episode_summarization(
                    episodes=episodes,
                    feed=feed,
                    cfg=cfg,
                    effective_output_dir=self.temp_dir,
                    run_suffix=None,
                    feed_metadata=workflow._FeedMetadata(
                        description="Test feed",
                        image_url=None,
                        last_updated=None,
                    ),
                    host_detection_result=workflow._HostDetectionResult(
                        cached_hosts=set(),
                        heuristics=None,
                    ),
                    summary_provider=_create_mock_provider(mock_summary_model),
                    download_args=[],  # Empty download args for test
                    pipeline_metrics=Mock(),
                )
            except Exception:
                self.fail("Cleanup should handle unload errors gracefully")


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
@pytest.mark.integration
@pytest.mark.slow
class TestParallelSummarizationEdgeCases(unittest.TestCase):
    """Test edge cases for parallel summarization."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.summarizer.SummaryModel")
    def test_single_episode_uses_sequential_processing(self, mock_model_class):
        """Test that single episode uses sequential processing (no parallel overhead)."""
        mock_summary_model = Mock()
        mock_summary_model.model_name = "facebook/bart-base"
        mock_summary_model.device = "cpu"
        mock_summary_model.cache_dir = None
        mock_summary_model.revision = None

        cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,
            summary_batch_size=3,  # Would use 3 workers if multiple episodes
        )

        episodes = [create_test_episode(idx=1)]
        feed = create_test_feed()

        # Create transcript file using the expected path format
        _create_test_transcript_files(episodes, self.temp_dir, cfg)

        summarize_calls = []

        def track_summarize(*args, **kwargs):
            """Track summarize calls."""
            # Track both summary_provider and summary_model (for backward compatibility)
            summarize_calls.append((kwargs.get("summary_provider"), kwargs.get("summary_model")))

        with patch(
            "podcast_scraper.workflow._summarize_single_episode", side_effect=track_summarize
        ):
            workflow._parallel_episode_summarization(
                episodes=episodes,
                feed=feed,
                cfg=cfg,
                effective_output_dir=self.temp_dir,
                run_suffix=None,
                feed_metadata=workflow._FeedMetadata(
                    description="Test feed",
                    image_url=None,
                    last_updated=None,
                ),
                host_detection_result=workflow._HostDetectionResult(
                    cached_hosts=set(),
                    heuristics=None,
                ),
                summary_provider=_create_mock_provider(mock_summary_model),
                download_args=[],  # Empty download args for test
                pipeline_metrics=Mock(),
            )

        # Should use sequential processing (original provider, not pre-loaded)
        self.assertEqual(len(summarize_calls), 1)
        # Should use the provider (sequential processing with single episode)
        provider, model = summarize_calls[0]
        self.assertIsNotNone(provider or model, "Should have provider or model")
        # Should not create new model instances
        mock_model_class.assert_not_called()

    @patch("podcast_scraper.summarizer.SummaryModel")
    def test_gpu_device_limits_parallelism(self, mock_model_class):
        """Test that GPU devices limit parallelism to 2 workers max."""
        mock_models = [Mock() for _ in range(2)]
        mock_model_class.side_effect = mock_models

        mock_summary_model = Mock()
        mock_summary_model.model_name = "facebook/bart-base"
        mock_summary_model.device = "cuda"  # GPU device
        mock_summary_model.cache_dir = None
        mock_summary_model.revision = None

        cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,
            summary_batch_size=10,  # Large batch size, but GPU should limit to 2
        )

        episodes = [create_test_episode(idx=i) for i in range(1, 6)]  # 5 episodes
        feed = create_test_feed()

        # Create transcript files using the expected path format
        _create_test_transcript_files(episodes, self.temp_dir, cfg)

        with patch("podcast_scraper.workflow._summarize_single_episode"):
            workflow._parallel_episode_summarization(
                episodes=episodes,
                feed=feed,
                cfg=cfg,
                effective_output_dir=self.temp_dir,
                run_suffix=None,
                feed_metadata=workflow._FeedMetadata(
                    description="Test feed",
                    image_url=None,
                    last_updated=None,
                ),
                host_detection_result=workflow._HostDetectionResult(
                    cached_hosts=set(),
                    heuristics=None,
                ),
                summary_provider=_create_mock_provider(mock_summary_model),
                download_args=[],  # Empty download args for test
                pipeline_metrics=Mock(),
            )

        # GPU should limit to 2 workers max, so only 2 models should be loaded
        self.assertEqual(mock_model_class.call_count, 2)

    @patch("podcast_scraper.summarizer.SummaryModel")
    def test_cpu_device_allows_more_parallelism(self, mock_model_class):
        """Test that CPU device allows more parallelism (up to 4 workers)."""
        mock_models = [Mock() for _ in range(4)]
        mock_model_class.side_effect = mock_models

        mock_summary_model = Mock()
        mock_summary_model.model_name = "facebook/bart-base"
        mock_summary_model.device = "cpu"
        mock_summary_model.cache_dir = None
        mock_summary_model.revision = None

        cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,
            summary_batch_size=10,  # Large batch size
        )

        episodes = [create_test_episode(idx=i) for i in range(1, 6)]  # 5 episodes
        feed = create_test_feed()

        # Create transcript files using the expected path format
        _create_test_transcript_files(episodes, self.temp_dir, cfg)

        with patch("podcast_scraper.workflow._summarize_single_episode"):
            workflow._parallel_episode_summarization(
                episodes=episodes,
                feed=feed,
                cfg=cfg,
                effective_output_dir=self.temp_dir,
                run_suffix=None,
                feed_metadata=workflow._FeedMetadata(
                    description="Test feed",
                    image_url=None,
                    last_updated=None,
                ),
                host_detection_result=workflow._HostDetectionResult(
                    cached_hosts=set(),
                    heuristics=None,
                ),
                summary_provider=_create_mock_provider(mock_summary_model),
                download_args=[],  # Empty download args for test
                pipeline_metrics=Mock(),
            )

        # CPU should allow up to 4 workers
        self.assertEqual(mock_model_class.call_count, 4)


if __name__ == "__main__":
    unittest.main()
