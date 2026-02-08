#!/usr/bin/env python3
"""Unit tests for bounded queue functionality in transcription."""

import queue
import unittest
from unittest.mock import Mock

import pytest

from podcast_scraper import config, models
from podcast_scraper.workflow.types import TranscriptionResources

pytestmark = [pytest.mark.unit]


class TestBoundedQueue(unittest.TestCase):
    """Tests for bounded queue in transcription resources."""

    def test_transcription_resources_uses_queue(self):
        """Test that TranscriptionResources uses Queue instead of list."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_queue_size=50,
        )

        resources = TranscriptionResources(
            transcription_provider=Mock(),
            temp_dir="/tmp",
            transcription_jobs=queue.Queue(maxsize=cfg.transcription_queue_size),
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )

        self.assertIsInstance(resources.transcription_jobs, queue.Queue)

    def test_queue_size_limit(self):
        """Test that queue respects size limit."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_queue_size=5,
        )

        job_queue = queue.Queue(maxsize=cfg.transcription_queue_size)
        self.assertEqual(job_queue.maxsize, 5)

    def test_queue_put_with_backpressure(self):
        """Test that queue.put() blocks when queue is full (backpressure)."""
        job_queue = queue.Queue(maxsize=2)

        # Fill queue
        job1 = models.TranscriptionJob(
            idx=1, ep_title="Test 1", ep_title_safe="test_1", temp_media="/tmp/1.mp3"
        )
        job2 = models.TranscriptionJob(
            idx=2, ep_title="Test 2", ep_title_safe="test_2", temp_media="/tmp/2.mp3"
        )

        job_queue.put(job1, block=False)
        job_queue.put(job2, block=False)

        # Queue should be full
        self.assertEqual(job_queue.qsize(), 2)

        # Next put should raise queue.Full if block=False
        job3 = models.TranscriptionJob(
            idx=3, ep_title="Test 3", ep_title_safe="test_3", temp_media="/tmp/3.mp3"
        )
        with self.assertRaises(queue.Full):
            job_queue.put(job3, block=False)

    def test_queue_get_removes_item(self):
        """Test that queue.get() removes and returns items."""
        job_queue = queue.Queue()

        job = models.TranscriptionJob(
            idx=1, ep_title="Test", ep_title_safe="test", temp_media="/tmp/test.mp3"
        )
        job_queue.put(job)

        self.assertEqual(job_queue.qsize(), 1)
        retrieved_job = job_queue.get()
        self.assertEqual(job_queue.qsize(), 0)
        self.assertEqual(retrieved_job, job)

    def test_queue_empty_check(self):
        """Test queue.empty() method."""
        job_queue = queue.Queue()
        self.assertTrue(job_queue.empty())

        job = models.TranscriptionJob(
            idx=1, ep_title="Test", ep_title_safe="test", temp_media="/tmp/test.mp3"
        )
        job_queue.put(job)
        self.assertFalse(job_queue.empty())
