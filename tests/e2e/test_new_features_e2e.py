#!/usr/bin/env python3
"""E2E tests for new features: bounded queue, transcript cache, JSONL metrics, degradation."""

import os
import tempfile
import unittest

import pytest

from tests.conftest import create_test_config

pytestmark = [pytest.mark.e2e]


class TestBoundedQueueE2E(unittest.TestCase):
    """E2E tests for bounded queue functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bounded_queue_configuration(self):
        """Test that bounded queue size is configurable."""
        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            transcription_queue_size=25,
        )
        self.assertEqual(cfg.transcription_queue_size, 25)


class TestTranscriptCacheE2E(unittest.TestCase):
    """E2E tests for transcript caching."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_transcript_cache_configuration(self):
        """Test that transcript cache can be enabled/disabled."""
        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            transcript_cache_enabled=True,
            transcript_cache_dir=os.path.join(self.temp_dir, "cache"),
        )
        self.assertTrue(cfg.transcript_cache_enabled)
        self.assertIsNotNone(cfg.transcript_cache_dir)


class TestJSONLMetricsE2E(unittest.TestCase):
    """E2E tests for JSONL metrics output."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_jsonl_metrics_configuration(self):
        """Test that JSONL metrics can be configured."""
        jsonl_path = os.path.join(self.temp_dir, "metrics.jsonl")
        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            jsonl_metrics_enabled=True,
            jsonl_metrics_path=jsonl_path,
        )
        self.assertTrue(cfg.jsonl_metrics_enabled)
        self.assertEqual(cfg.jsonl_metrics_path, jsonl_path)


class TestDegradationPolicyE2E(unittest.TestCase):
    """E2E tests for graceful degradation policy."""

    def test_degradation_policy_configuration(self):
        """Test that degradation policy can be configured."""
        policy_dict = {
            "save_transcript_on_summarization_failure": True,
            "save_summary_on_entity_extraction_failure": True,
            "continue_on_stage_failure": True,
        }

        cfg = create_test_config(
            rss_url="https://example.com/feed.xml",
            degradation_policy=policy_dict,
        )
        self.assertIsNotNone(cfg.degradation_policy)
        self.assertTrue(cfg.degradation_policy["save_transcript_on_summarization_failure"])
