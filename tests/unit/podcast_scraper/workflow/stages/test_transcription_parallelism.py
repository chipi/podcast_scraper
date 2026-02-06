#!/usr/bin/env python3
"""Tests for transcription parallelism configuration and Whisper parallel processing.

This module tests the transcription_parallelism configuration and the new
Whisper parallel processing functionality.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from podcast_scraper import config, models
from podcast_scraper.workflow import metrics
from podcast_scraper.workflow.stages import transcription
from podcast_scraper.workflow.types import ProcessingResources, TranscriptionResources


class TestTranscriptionParallelismConfiguration(unittest.TestCase):
    """Test transcription parallelism configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
        )
        self.transcription_resources = TranscriptionResources(
            transcription_provider=Mock(),
            temp_dir=None,
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        self.download_args = []
        self.episodes = []
        self.feed = models.RssFeed(
            title="Test Feed",
            items=[],
            base_url="https://example.com",
            authors=[],
        )
        self.effective_output_dir = "/tmp/test"
        self.run_suffix = None
        self.feed_metadata = Mock()
        self.host_detection_result = Mock()
        # Use a mutable list for processing_jobs (NamedTuple contains mutable list)
        processing_jobs_list = []
        self.processing_resources = ProcessingResources(
            processing_jobs=processing_jobs_list,
            processing_jobs_lock=None,
            processing_complete_event=None,
        )
        self.pipeline_metrics = metrics.Metrics()
        self.summary_provider = None
        self.downloads_complete_event = Mock()
        self.downloads_complete_event.is_set.return_value = True
        self.saved_counter = [0]

    def test_whisper_sequential_processing_default(self):
        """Test Whisper uses sequential processing by default (parallelism=1)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
            transcription_parallelism=1,
        )

        with patch("podcast_scraper.workflow.stages.transcription.logger") as mock_logger:
            # Mock the transcription function to avoid actual processing
            with patch("podcast_scraper.workflow.stages.transcription.transcribe_media_to_text"):
                transcription.process_transcription_jobs_concurrent(
                    transcription_resources=self.transcription_resources,
                    download_args=self.download_args,
                    episodes=self.episodes,
                    feed=self.feed,
                    cfg=cfg,
                    effective_output_dir=self.effective_output_dir,
                    run_suffix=self.run_suffix,
                    feed_metadata=self.feed_metadata,
                    host_detection_result=self.host_detection_result,
                    processing_resources=self.processing_resources,
                    pipeline_metrics=self.pipeline_metrics,
                    summary_provider=self.summary_provider,
                    downloads_complete_event=self.downloads_complete_event,
                    saved_counter=self.saved_counter,
                )

        # Should log info (not warning) for sequential processing
        info_calls = [
            log_call
            for log_call in mock_logger.info.call_args_list
            if "configured" in str(log_call) or "effective" in str(log_call)
        ]
        self.assertGreater(len(info_calls), 0, "Should log info for sequential processing")

    def test_whisper_parallel_processing_warning(self):
        """Test Whisper logs warning when parallelism > 1 (experimental)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
            transcription_parallelism=2,
        )

        with patch("podcast_scraper.workflow.stages.transcription.logger") as mock_logger:
            # Mock the transcription function to avoid actual processing
            with patch("podcast_scraper.workflow.stages.transcription.transcribe_media_to_text"):
                transcription.process_transcription_jobs_concurrent(
                    transcription_resources=self.transcription_resources,
                    download_args=self.download_args,
                    episodes=self.episodes,
                    feed=self.feed,
                    cfg=cfg,
                    effective_output_dir=self.effective_output_dir,
                    run_suffix=self.run_suffix,
                    feed_metadata=self.feed_metadata,
                    host_detection_result=self.host_detection_result,
                    processing_resources=self.processing_resources,
                    pipeline_metrics=self.pipeline_metrics,
                    summary_provider=self.summary_provider,
                    downloads_complete_event=self.downloads_complete_event,
                    saved_counter=self.saved_counter,
                )

        # Should log warning for experimental parallel processing
        warning_calls = [
            log_call
            for log_call in mock_logger.warning.call_args_list
            if "EXPERIMENTAL" in str(log_call) or "parallel processing" in str(log_call).lower()
        ]
        self.assertGreater(
            len(warning_calls), 0, "Should log warning for experimental parallel processing"
        )

    def test_openai_parallel_processing_info(self):
        """Test OpenAI provider logs info when using parallelism."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            transcribe_missing=True,
            transcription_parallelism=3,
            openai_api_key="sk-test123",
        )

        with patch("podcast_scraper.workflow.stages.transcription.logger") as mock_logger:
            # Mock the transcription function to avoid actual processing
            with patch("podcast_scraper.workflow.stages.transcription.transcribe_media_to_text"):
                transcription.process_transcription_jobs_concurrent(
                    transcription_resources=self.transcription_resources,
                    download_args=self.download_args,
                    episodes=self.episodes,
                    feed=self.feed,
                    cfg=cfg,
                    effective_output_dir=self.effective_output_dir,
                    run_suffix=self.run_suffix,
                    feed_metadata=self.feed_metadata,
                    host_detection_result=self.host_detection_result,
                    processing_resources=self.processing_resources,
                    pipeline_metrics=self.pipeline_metrics,
                    summary_provider=self.summary_provider,
                    downloads_complete_event=self.downloads_complete_event,
                    saved_counter=self.saved_counter,
                )

        # Should log info (not warning) for OpenAI parallel processing
        info_calls = [
            log_call
            for log_call in mock_logger.info.call_args_list
            if "configured" in str(log_call) or "effective" in str(log_call)
        ]
        self.assertGreater(len(info_calls), 0, "Should log info for OpenAI parallel processing")

    def test_whisper_parallel_processing_uses_threadpool(self):
        """Test Whisper with parallelism > 1 uses ThreadPoolExecutor."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
            transcription_parallelism=2,
        )

        # Create a mock transcription job
        job = models.TranscriptionJob(
            idx=1,
            ep_title="Test Episode",
            ep_title_safe="test_episode",
            temp_media="/tmp/test.mp3",
        )
        transcription_resources = TranscriptionResources(
            transcription_provider=self.transcription_resources.transcription_provider,
            temp_dir=self.transcription_resources.temp_dir,
            transcription_jobs=[job],
            transcription_jobs_lock=self.transcription_resources.transcription_jobs_lock,
            saved_counter_lock=self.transcription_resources.saved_counter_lock,
        )

        with (
            patch(
                "podcast_scraper.workflow.stages.transcription.ThreadPoolExecutor"
            ) as mock_executor_class,
            patch(
                "podcast_scraper.workflow.stages.transcription.transcribe_media_to_text"
            ) as mock_transcribe,
            patch(
                "podcast_scraper.workflow.stages.transcription.as_completed"
            ) as mock_as_completed,
        ):
            mock_transcribe.return_value = (True, "/tmp/test.txt", 0)
            mock_executor = Mock()
            mock_future = Mock()
            mock_future.result.return_value = (True, "/tmp/test.txt", 0)
            # Make submit return the same future object each time
            mock_executor.submit.return_value = mock_future
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            mock_executor_class.return_value.__exit__.return_value = None

            # Mock as_completed to return the same future that was submitted
            # This ensures the futures dictionary lookup works correctly
            def as_completed_side_effect(futures_list, timeout=None):
                # Return the futures that were passed in (they should match mock_future)
                return iter(futures_list)

            mock_as_completed.side_effect = as_completed_side_effect

            transcription.process_transcription_jobs_concurrent(
                transcription_resources=transcription_resources,
                download_args=self.download_args,
                episodes=self.episodes,
                feed=self.feed,
                cfg=cfg,
                effective_output_dir=self.effective_output_dir,
                run_suffix=self.run_suffix,
                feed_metadata=self.feed_metadata,
                host_detection_result=self.host_detection_result,
                processing_resources=self.processing_resources,
                pipeline_metrics=self.pipeline_metrics,
                summary_provider=self.summary_provider,
                downloads_complete_event=self.downloads_complete_event,
                saved_counter=self.saved_counter,
            )

        # Should create ThreadPoolExecutor with max_workers=2
        mock_executor_class.assert_called_once()
        call_kwargs = mock_executor_class.call_args[1]
        self.assertEqual(call_kwargs.get("max_workers"), 2)

    def test_whisper_sequential_processing_uses_loop(self):
        """Test Whisper with parallelism=1 uses sequential loop (not ThreadPoolExecutor)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
            transcription_parallelism=1,
        )

        # Create a mock transcription job
        job = models.TranscriptionJob(
            idx=1,
            ep_title="Test Episode",
            ep_title_safe="test_episode",
            temp_media="/tmp/test.mp3",
        )
        transcription_resources = TranscriptionResources(
            transcription_provider=self.transcription_resources.transcription_provider,
            temp_dir=self.transcription_resources.temp_dir,
            transcription_jobs=[job],
            transcription_jobs_lock=self.transcription_resources.transcription_jobs_lock,
            saved_counter_lock=self.transcription_resources.saved_counter_lock,
        )

        with (
            patch(
                "podcast_scraper.workflow.stages.transcription.ThreadPoolExecutor"
            ) as mock_executor_class,
            patch(
                "podcast_scraper.workflow.stages.transcription.transcribe_media_to_text"
            ) as mock_transcribe,
        ):
            mock_transcribe.return_value = (True, "/tmp/test.txt", 0)

            transcription.process_transcription_jobs_concurrent(
                transcription_resources=transcription_resources,
                download_args=self.download_args,
                episodes=self.episodes,
                feed=self.feed,
                cfg=cfg,
                effective_output_dir=self.effective_output_dir,
                run_suffix=self.run_suffix,
                feed_metadata=self.feed_metadata,
                host_detection_result=self.host_detection_result,
                processing_resources=self.processing_resources,
                pipeline_metrics=self.pipeline_metrics,
                summary_provider=self.summary_provider,
                downloads_complete_event=self.downloads_complete_event,
                saved_counter=self.saved_counter,
            )

        # Should NOT create ThreadPoolExecutor for sequential processing
        mock_executor_class.assert_not_called()
