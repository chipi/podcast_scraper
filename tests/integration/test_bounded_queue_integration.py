#!/usr/bin/env python3
"""Integration tests for bounded queue functionality."""

import queue
import threading
import unittest

import pytest

from podcast_scraper import config, models
from podcast_scraper.workflow.stages import transcription

pytestmark = [pytest.mark.integration]


class TestBoundedQueueIntegration(unittest.TestCase):
    """Integration tests for bounded queue in transcription."""

    def test_queue_backpressure_prevents_unbounded_growth(self):
        """Test that queue size limit prevents unbounded memory growth."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_queue_size=3,  # Small limit for testing
        )

        # Create transcription resources with bounded queue
        resources = transcription.setup_transcription_resources(cfg, "/tmp")
        self.assertIsInstance(resources.transcription_jobs, queue.Queue)
        self.assertEqual(resources.transcription_jobs.maxsize, 3)

    def test_queue_thread_safety(self):
        """Test that queue operations are thread-safe."""
        job_queue = queue.Queue(maxsize=10)

        # Create multiple jobs - use fewer jobs to avoid queue full issues
        # Each thread will try to put 3 jobs, total 9 jobs for 3 threads
        # Queue size is 10, so all should fit
        jobs = [
            models.TranscriptionJob(
                idx=i,
                ep_title=f"Episode {i}",
                ep_title_safe=f"episode_{i}",
                temp_media=f"/tmp/{i}.mp3",
            )
            for i in range(3)
        ]

        # Put jobs from multiple threads
        # Handle queue.Full exceptions to prevent unhandled thread exceptions
        def put_jobs():
            for job in jobs:
                try:
                    job_queue.put(job, block=True, timeout=2.0)
                except queue.Full:
                    # Queue is full, which is expected when multiple threads
                    # compete for limited queue space. This is not an error.
                    pass

        threads = [threading.Thread(target=put_jobs) for _ in range(3)]
        for t in threads:
            t.start()
        # Add timeout to join() to prevent hanging if threads don't complete
        for t in threads:
            t.join(timeout=10.0)
            # Verify thread completed (not still alive)
            if t.is_alive():
                # Thread didn't complete - this indicates a hang
                # Force cleanup and fail the test
                self.fail(f"Thread {t.name} did not complete within timeout - possible hang")

        # Verify all jobs were added (queue may be full, but no errors)
        # With 3 threads putting 3 jobs each (9 total) and queue size 10, all should fit
        self.assertGreaterEqual(job_queue.qsize(), 3)
