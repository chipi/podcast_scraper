#!/usr/bin/env python3
"""Unit tests for environment variable loading in Config.

Moved from tests/e2e/ as part of Phase 3 test pyramid refactoring - these
test Config component behavior, not user workflows.
"""

import os
import sys
import unittest

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper import config


@pytest.mark.unit
class TestEnvironmentVariables(unittest.TestCase):
    """Test environment variable loading for Config."""

    def setUp(self):
        """Clear environment variables before each test."""
        self.env_vars_to_clear = [
            "LOG_LEVEL",
            "OUTPUT_DIR",
            "LOG_FILE",
            "SUMMARY_CACHE_DIR",
            "CACHE_DIR",
            "WORKERS",
            "TRANSCRIPTION_PARALLELISM",
            "PROCESSING_PARALLELISM",
            "SUMMARY_BATCH_SIZE",
            "SUMMARY_CHUNK_PARALLELISM",
            "TIMEOUT",
            "SUMMARY_DEVICE",
        ]
        for var in self.env_vars_to_clear:
            os.environ.pop(var, None)

    def tearDown(self):
        """Clean up environment variables after each test."""
        for var in self.env_vars_to_clear:
            os.environ.pop(var, None)

    def test_log_level_from_env(self):
        """Test LOG_LEVEL environment variable loading."""
        os.environ["LOG_LEVEL"] = "DEBUG"
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.log_level, "DEBUG")

    def test_log_level_env_overrides_config(self):
        """Test LOG_LEVEL env var takes precedence over config file."""
        os.environ["LOG_LEVEL"] = "ERROR"
        cfg = config.Config(rss_url="https://test.com", log_level="WARNING")
        self.assertEqual(cfg.log_level, "ERROR")

    def test_output_dir_from_env(self):
        """Test OUTPUT_DIR environment variable loading."""
        os.environ["OUTPUT_DIR"] = "/tmp/test_output"  # nosec B108
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.output_dir, "/tmp/test_output")  # nosec B108

    def test_output_dir_config_overrides_env(self):
        """Test config file value takes precedence over OUTPUT_DIR env var."""
        os.environ["OUTPUT_DIR"] = "/env/path"
        cfg = config.Config(rss_url="https://test.com", output_dir="/config/path")
        self.assertEqual(cfg.output_dir, "/config/path")

    def test_log_file_from_env(self):
        """Test LOG_FILE environment variable loading."""
        os.environ["LOG_FILE"] = "/tmp/test.log"  # nosec B108
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.log_file, "/tmp/test.log")  # nosec B108

    def test_log_file_config_overrides_env(self):
        """Test config file value takes precedence over LOG_FILE env var."""
        os.environ["LOG_FILE"] = "/env/log.log"
        cfg = config.Config(rss_url="https://test.com", log_file="/config/log.log")
        self.assertEqual(cfg.log_file, "/config/log.log")

    def test_summary_cache_dir_from_env(self):
        """Test SUMMARY_CACHE_DIR environment variable loading."""
        os.environ["SUMMARY_CACHE_DIR"] = "/tmp/cache"  # nosec B108
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.summary_cache_dir, "/tmp/cache")  # nosec B108

    def test_cache_dir_alias(self):
        """Test CACHE_DIR environment variable derives summary_cache_dir."""
        os.environ["CACHE_DIR"] = "/tmp/cache2"  # nosec B108
        cfg = config.Config(rss_url="https://test.com")
        # CACHE_DIR now derives summary_cache_dir as CACHE_DIR/huggingface/hub
        self.assertEqual(cfg.summary_cache_dir, "/tmp/cache2/huggingface/hub")  # nosec B108

    def test_summary_cache_dir_precedence(self):
        """Test SUMMARY_CACHE_DIR takes precedence over CACHE_DIR."""
        os.environ["SUMMARY_CACHE_DIR"] = "/tmp/cache1"  # nosec B108
        os.environ["CACHE_DIR"] = "/tmp/cache2"  # nosec B108
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.summary_cache_dir, "/tmp/cache1")  # nosec B108

    def test_workers_from_env(self):
        """Test WORKERS environment variable loading."""
        os.environ["WORKERS"] = "6"
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.workers, 6)

    def test_workers_config_overrides_env(self):
        """Test config file value takes precedence over WORKERS env var."""
        os.environ["WORKERS"] = "8"
        cfg = config.Config(rss_url="https://test.com", workers=4)
        self.assertEqual(cfg.workers, 4)

    def test_workers_invalid_env_ignored(self):
        """Test invalid WORKERS env var is ignored (uses default)."""
        os.environ["WORKERS"] = "invalid"
        cfg = config.Config(rss_url="https://test.com")
        self.assertIsInstance(cfg.workers, int)
        self.assertGreater(cfg.workers, 0)

    def test_transcription_parallelism_from_env(self):
        """Test TRANSCRIPTION_PARALLELISM environment variable loading."""
        os.environ["TRANSCRIPTION_PARALLELISM"] = "3"
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.transcription_parallelism, 3)

    def test_processing_parallelism_from_env(self):
        """Test PROCESSING_PARALLELISM environment variable loading."""
        os.environ["PROCESSING_PARALLELISM"] = "4"
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.processing_parallelism, 4)

    def test_summary_batch_size_from_env(self):
        """Test SUMMARY_BATCH_SIZE environment variable loading."""
        os.environ["SUMMARY_BATCH_SIZE"] = "5"
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.summary_batch_size, 5)

    def test_summary_chunk_parallelism_from_env(self):
        """Test SUMMARY_CHUNK_PARALLELISM environment variable loading."""
        os.environ["SUMMARY_CHUNK_PARALLELISM"] = "2"
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.summary_chunk_parallelism, 2)

    def test_timeout_from_env(self):
        """Test TIMEOUT environment variable loading."""
        os.environ["TIMEOUT"] = "60"
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.timeout, 60)

    def test_timeout_invalid_env_ignored(self):
        """Test invalid TIMEOUT env var is ignored (uses default)."""
        os.environ["TIMEOUT"] = "invalid"
        cfg = config.Config(rss_url="https://test.com")
        self.assertIsInstance(cfg.timeout, int)
        self.assertGreaterEqual(cfg.timeout, 1)

    def test_summary_device_from_env(self):
        """Test SUMMARY_DEVICE environment variable loading."""
        os.environ["SUMMARY_DEVICE"] = "cpu"
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.summary_device, "cpu")

    def test_summary_device_cuda_from_env(self):
        """Test SUMMARY_DEVICE=cuda environment variable loading."""
        os.environ["SUMMARY_DEVICE"] = "cuda"
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.summary_device, "cuda")

    def test_summary_device_mps_from_env(self):
        """Test SUMMARY_DEVICE=mps environment variable loading."""
        os.environ["SUMMARY_DEVICE"] = "mps"
        cfg = config.Config(rss_url="https://test.com")
        self.assertEqual(cfg.summary_device, "mps")

    def test_summary_device_empty_string(self):
        """Test SUMMARY_DEVICE empty string (should be None/auto-detect)."""
        os.environ["SUMMARY_DEVICE"] = ""
        cfg = config.Config(rss_url="https://test.com")
        self.assertIsNone(cfg.summary_device)

    def test_summary_device_invalid_ignored(self):
        """Test invalid SUMMARY_DEVICE env var is ignored (uses default)."""
        os.environ["SUMMARY_DEVICE"] = "invalid_device"
        cfg = config.Config(rss_url="https://test.com")
        # Should use default (None or valid value)
        self.assertIn(cfg.summary_device, (None, "cpu", "cuda", "mps"))

    def test_multiple_env_vars(self):
        """Test multiple environment variables loaded together."""
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["OUTPUT_DIR"] = "/tmp/output"  # nosec B108
        os.environ["WORKERS"] = "4"
        os.environ["TIMEOUT"] = "30"
        os.environ["SUMMARY_DEVICE"] = "cpu"

        cfg = config.Config(rss_url="https://test.com")

        self.assertEqual(cfg.log_level, "DEBUG")
        self.assertEqual(cfg.output_dir, "/tmp/output")  # nosec B108
        self.assertEqual(cfg.workers, 4)
        self.assertEqual(cfg.timeout, 30)
        self.assertEqual(cfg.summary_device, "cpu")

    def test_env_vars_with_config_file(self):
        """Test environment variables work alongside config file values."""
        os.environ["LOG_LEVEL"] = "ERROR"  # Takes precedence
        os.environ["WORKERS"] = "8"
        os.environ["TIMEOUT"] = "60"

        # Config file values
        cfg = config.Config(
            rss_url="https://test.com",
            log_level="WARNING",  # Should be overridden by env
            workers=4,  # Should override env
            timeout=30,  # Should override env
        )

        self.assertEqual(cfg.log_level, "ERROR")  # Env takes precedence for LOG_LEVEL
        self.assertEqual(cfg.workers, 4)  # Config takes precedence
        self.assertEqual(cfg.timeout, 30)  # Config takes precedence
