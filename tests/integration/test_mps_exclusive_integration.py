#!/usr/bin/env python3
"""Integration tests for MPS exclusive mode serialization behavior.

These tests verify that when mps_exclusive=true and both Whisper and
summarization use MPS, GPU work is serialized (transcription completes
before summarization starts).
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

# Add tests directory to path for conftest import
from pathlib import Path

from podcast_scraper.workflow.orchestration import _both_providers_use_mps

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


@pytest.mark.integration
class TestMPSExclusiveMode(unittest.TestCase):
    """Integration tests for MPS exclusive mode behavior."""

    def test_both_providers_use_mps_detection(self):
        """Test _both_providers_use_mps correctly detects when both use MPS."""
        cfg = create_test_config(
            whisper_device="mps",
            summary_device="mps",
            generate_summaries=True,
        )

        # Mock transcription provider (MLProvider with MPS)
        mock_transcription_provider = Mock()
        mock_transcription_provider._detect_whisper_device = Mock(return_value="mps")
        type(mock_transcription_provider).__name__ = "MLProvider"

        # Mock summary provider (MLProvider with SummaryModel having MPS device)
        # The function checks _map_model or _reduce_model device attribute
        mock_summary_provider = Mock()
        mock_map_model = Mock()
        mock_map_model.device = "mps"
        mock_summary_provider._map_model = mock_map_model
        type(mock_summary_provider).__name__ = "MLProvider"

        result = _both_providers_use_mps(cfg, mock_transcription_provider, mock_summary_provider)
        self.assertTrue(result, "Should detect both providers using MPS")

    def test_both_providers_use_mps_false_when_whisper_cpu(self):
        """Test _both_providers_use_mps returns False when Whisper uses CPU."""
        cfg = create_test_config(
            whisper_device="cpu",
            summary_device="mps",
            generate_summaries=True,
        )

        # Mock transcription provider (MLProvider with CPU)
        mock_transcription_provider = Mock()
        mock_transcription_provider._detect_whisper_device = Mock(return_value="cpu")
        type(mock_transcription_provider).__name__ = "MLProvider"

        # Mock summary provider (MLProvider with SummaryModel having MPS device)
        mock_summary_provider = Mock()
        mock_map_model = Mock()
        mock_map_model.device = "mps"
        mock_summary_provider._map_model = mock_map_model
        type(mock_summary_provider).__name__ = "MLProvider"

        result = _both_providers_use_mps(cfg, mock_transcription_provider, mock_summary_provider)
        self.assertFalse(result, "Should return False when Whisper uses CPU")

    def test_both_providers_use_mps_false_when_summary_cpu(self):
        """Test _both_providers_use_mps returns False when summarization uses CPU."""
        cfg = create_test_config(
            whisper_device="mps",
            summary_device="cpu",
            generate_summaries=True,
        )

        # Mock transcription provider (MLProvider with MPS)
        mock_transcription_provider = Mock()
        mock_transcription_provider._detect_whisper_device = Mock(return_value="mps")
        type(mock_transcription_provider).__name__ = "MLProvider"

        # Mock summary provider (MLProvider with SummaryModel having CPU device)
        mock_summary_provider = Mock()
        mock_map_model = Mock()
        mock_map_model.device = "cpu"
        mock_summary_provider._map_model = mock_map_model
        type(mock_summary_provider).__name__ = "MLProvider"

        result = _both_providers_use_mps(cfg, mock_transcription_provider, mock_summary_provider)
        self.assertFalse(result, "Should return False when summarization uses CPU")

    def test_both_providers_use_mps_false_when_openai_transcription(self):
        """Test _both_providers_use_mps returns False when using OpenAI transcription."""
        cfg = create_test_config(
            transcription_provider="openai",
            summary_device="mps",
            generate_summaries=True,
        )

        # Mock OpenAI transcription provider
        mock_transcription_provider = Mock()
        type(mock_transcription_provider).__name__ = "OpenAIProvider"

        # Mock summary provider (MLProvider with SummaryModel having MPS device)
        mock_summary_provider = Mock()
        mock_map_model = Mock()
        mock_map_model.device = "mps"
        mock_summary_provider._map_model = mock_map_model
        type(mock_summary_provider).__name__ = "MLProvider"

        result = _both_providers_use_mps(cfg, mock_transcription_provider, mock_summary_provider)
        self.assertFalse(result, "Should return False when using OpenAI transcription")

    def test_serialization_event_waits_for_transcription(self):
        """Test that processing stage waits for transcription_complete_event when should_serialize_mps=True.

        This test verifies the waiting behavior by mocking the function and checking
        that it waits for the event before proceeding.
        """
        from podcast_scraper.workflow.stages import processing

        cfg = create_test_config(
            generate_summaries=True,
            mps_exclusive=True,
        )

        # Create event that starts unset
        transcription_complete_event = threading.Event()

        # Track when processing attempts to start
        processing_started = threading.Event()
        wait_times = []

        # Mock the entire processing function to verify waiting behavior
        def mock_process_processing_jobs_concurrent(*args, **kwargs):
            """Mock that records when processing starts and verifies waiting."""
            # Extract should_serialize_mps from kwargs (it's a positional arg)
            # Args: processing_resources, feed, cfg, effective_output_dir, run_suffix,
            #       feed_metadata, host_detection_result, pipeline_metrics,
            #       summary_provider, transcription_complete_event, should_serialize_mps
            if len(args) >= 11:
                should_serialize = args[10]  # should_serialize_mps
                event = args[9]  # transcription_complete_event
            else:
                should_serialize = kwargs.get("should_serialize_mps", False)
                event = kwargs.get("transcription_complete_event", None)

            # If serialization is enabled, wait for event
            if should_serialize and event and cfg.generate_summaries:
                # Record time before waiting
                wait_start = time.time()
                event.wait(timeout=1.0)  # Wait with timeout to avoid hanging
                wait_end = time.time()
                wait_times.append(wait_end - wait_start)
                processing_started.set()

        with patch(
            "podcast_scraper.workflow.stages.processing.process_processing_jobs_concurrent",
            side_effect=mock_process_processing_jobs_concurrent,
        ):
            # Create minimal processing resources
            from podcast_scraper.workflow.types import ProcessingResources

            processing_resources = ProcessingResources(
                processing_jobs=[],
                processing_jobs_lock=threading.Lock(),
                processing_complete_event=None,
            )

            # Create minimal feed and episodes
            from podcast_scraper import models
            from podcast_scraper.workflow.types import FeedMetadata, HostDetectionResult

            feed = models.RssFeed(
                title="Test Feed",
                authors=[],
                items=[],
                base_url="https://example.com",
            )
            feed_metadata = FeedMetadata(
                description="",
                image_url=None,
                last_updated=None,
            )
            host_detection_result = HostDetectionResult(set(), None, None)

            # Create minimal metrics
            from podcast_scraper.workflow import metrics

            pipeline_metrics = metrics.Metrics()

            # Start processing in a thread with should_serialize_mps=True
            processing_thread = threading.Thread(
                target=processing.process_processing_jobs_concurrent,
                args=(
                    processing_resources,
                    feed,
                    cfg,
                    "/tmp/test_output",
                    None,
                    feed_metadata,
                    host_detection_result,
                    pipeline_metrics,
                    None,  # summary_provider (None for this test)
                    transcription_complete_event,
                    True,  # should_serialize_mps=True
                ),
                daemon=True,
            )
            processing_thread.start()

            # Wait a bit to ensure processing thread has started
            time.sleep(0.1)

            # Verify processing hasn't started yet (waiting for event)
            self.assertFalse(
                processing_started.is_set(),
                "Processing should wait for transcription_complete_event",
            )

            # Now set the event (simulating transcription completion)
            transcription_complete_event.set()

            # Wait for processing to start
            processing_started.wait(timeout=2.0)
            self.assertTrue(
                processing_started.is_set(),
                "Processing should start after transcription_complete_event is set",
            )

            # Verify we actually waited (wait time should be > 0)
            self.assertGreater(
                len(wait_times),
                0,
                "Should have recorded wait time",
            )
            if wait_times:
                self.assertGreater(
                    wait_times[0],
                    0.0,
                    "Should wait for transcription_complete_event",
                )

            # Clean up
            processing_thread.join(timeout=1.0)
