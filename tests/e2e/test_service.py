#!/usr/bin/env python3
"""Tests for service API (podcast_scraper.service)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import service module directly (not via __getattr__) to avoid recursion
# Use direct import path to bypass __getattr__
import importlib

import podcast_scraper
from podcast_scraper import config

service_module = importlib.import_module("podcast_scraper.service")
ServiceResult = service_module.ServiceResult

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import (  # noqa: E402
    build_rss_xml_with_transcript,
    create_rss_response,
    create_transcript_response,
)

from podcast_scraper import downloader  # noqa: E402


@pytest.mark.e2e
@pytest.mark.slow
class TestServiceResult(unittest.TestCase):
    """Tests for ServiceResult dataclass."""

    def test_service_result_success(self):
        """ServiceResult with success=True should have correct attributes."""
        result = ServiceResult(
            episodes_processed=5,
            summary="Processed 5 episodes",
            success=True,
            error=None,
        )

        self.assertEqual(result.episodes_processed, 5)
        self.assertEqual(result.summary, "Processed 5 episodes")
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_service_result_failure(self):
        """ServiceResult with success=False should have error message."""
        result = ServiceResult(
            episodes_processed=0,
            summary="",
            success=False,
            error="Configuration file not found",
        )

        self.assertEqual(result.episodes_processed, 0)
        self.assertEqual(result.summary, "")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Configuration file not found")

    def test_service_result_defaults(self):
        """ServiceResult should have correct defaults."""
        result = ServiceResult(episodes_processed=0, summary="")

        self.assertEqual(result.episodes_processed, 0)
        self.assertEqual(result.summary, "")
        self.assertTrue(result.success)  # Default is True
        self.assertIsNone(result.error)  # Default is None


@pytest.mark.e2e
@pytest.mark.slow
class TestServiceRun(unittest.TestCase):
    """Tests for service.run()."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _mock_http_map(self, responses):
        """Create HTTP mock side effect function."""

        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            normalized = downloader.normalize_url(url)
            resp = responses.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect

    def test_service_run_success(self):
        """service.run() should return successful ServiceResult on success."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=self.temp_dir,
                max_episodes=1,
            )

            result = service_module.run(cfg)

            # Verify ServiceResult structure
            self.assertIsInstance(result, ServiceResult)
            self.assertTrue(result.success)
            self.assertIsNone(result.error)
            self.assertIsInstance(result.episodes_processed, int)
            self.assertGreater(result.episodes_processed, 0)
            self.assertIsInstance(result.summary, str)
            self.assertIn("transcripts", result.summary.lower())

    def test_service_run_with_logging_config(self):
        """service.run() should apply logging configuration if specified."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        http_mock = self._mock_http_map(responses)
        log_file = os.path.join(self.temp_dir, "test.log")
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with patch("podcast_scraper.workflow.apply_log_level") as mock_apply_log:
                cfg = config.Config(
                    rss_url=rss_url,
                    output_dir=self.temp_dir,
                    max_episodes=1,
                    log_level="DEBUG",
                    log_file=log_file,
                )

                result = service_module.run(cfg)

                # Verify logging was configured
                mock_apply_log.assert_called_once_with(
                    level="DEBUG",
                    log_file=log_file,
                )
                self.assertTrue(result.success)

    def test_service_run_handles_exceptions(self):
        """service.run() should catch exceptions and return failed ServiceResult."""
        rss_url = "https://invalid-feed.example.com/feed.xml"

        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            raise Exception("Network error")

        with patch("podcast_scraper.downloader.fetch_url", side_effect=_side_effect):
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=self.temp_dir,
            )

            result = service_module.run(cfg)

            # Verify ServiceResult indicates failure
            self.assertIsInstance(result, ServiceResult)
            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)
            self.assertIn("Network error", result.error)
            self.assertEqual(result.episodes_processed, 0)
            self.assertEqual(result.summary, "")

    def test_service_run_with_dry_run(self):
        """service.run() should work with dry-run mode."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Test Feed", transcript_url)

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            cfg = config.Config(
                rss_url=rss_url,
                output_dir=self.temp_dir,
                max_episodes=1,
                dry_run=True,
            )

            result = service_module.run(cfg)

            self.assertTrue(result.success)
            self.assertGreater(result.episodes_processed, 0)
            self.assertIn("dry run", result.summary.lower())


@pytest.mark.e2e
@pytest.mark.slow
class TestServiceRunFromConfigFile(unittest.TestCase):
    """Tests for service.run_from_config_file()."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _mock_http_map(self, responses):
        """Create HTTP mock side effect function."""

        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            normalized = downloader.normalize_url(url)
            resp = responses.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect

    def test_run_from_config_file_json_success(self):
        """run_from_config_file() should load JSON config and run successfully."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        # Create JSON config file
        cfg_path = os.path.join(self.temp_dir, "config.json")
        config_data = {
            "rss": rss_url,
            "output_dir": self.temp_dir,
            "max_episodes": 1,
        }
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh)

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            result = service_module.run_from_config_file(cfg_path)

            self.assertTrue(result.success)
            self.assertIsNone(result.error)
            self.assertGreater(result.episodes_processed, 0)

    def test_run_from_config_file_yaml_success(self):
        """run_from_config_file() should load YAML config and run successfully."""
        import yaml

        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        # Create YAML config file
        cfg_path = os.path.join(self.temp_dir, "config.yaml")
        config_data = {
            "rss": rss_url,
            "output_dir": self.temp_dir,
            "max_episodes": 1,
        }
        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.dump(config_data, fh)

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            result = service_module.run_from_config_file(cfg_path)

            self.assertTrue(result.success)
            self.assertIsNone(result.error)
            self.assertGreater(result.episodes_processed, 0)

    def test_run_from_config_file_not_found(self):
        """run_from_config_file() should return failed ServiceResult for missing file."""
        non_existent_path = os.path.join(self.temp_dir, "nonexistent.json")

        result = service_module.run_from_config_file(non_existent_path)

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("not found", result.error.lower())
        self.assertEqual(result.episodes_processed, 0)

    def test_run_from_config_file_invalid_json(self):
        """run_from_config_file() should return failed ServiceResult for invalid JSON."""
        cfg_path = os.path.join(self.temp_dir, "invalid.json")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            fh.write("{ invalid json }")

        result = service_module.run_from_config_file(cfg_path)

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("failed to load", result.error.lower())
        self.assertEqual(result.episodes_processed, 0)

    def test_run_from_config_file_invalid_config(self):
        """run_from_config_file() should return failed ServiceResult for invalid config."""
        cfg_path = os.path.join(self.temp_dir, "invalid_config.json")
        config_data = {
            "rss": "https://example.com/feed.xml",
            "output_dir": self.temp_dir,
            "invalid_field": "should not be here",  # This will cause validation error
        }
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh)

        result = service_module.run_from_config_file(cfg_path)

        # Should fail during Config validation
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_run_from_config_file_path_object(self):
        """run_from_config_file() should accept Path objects."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        # Create JSON config file
        cfg_path = Path(self.temp_dir) / "config.json"
        config_data = {
            "rss": rss_url,
            "output_dir": self.temp_dir,
            "max_episodes": 1,
        }
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh)

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            result = service_module.run_from_config_file(cfg_path)

            self.assertTrue(result.success)
            self.assertGreater(result.episodes_processed, 0)


@pytest.mark.e2e
@pytest.mark.slow
class TestServiceMain(unittest.TestCase):
    """Tests for service.main() CLI entry point."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _mock_http_map(self, responses):
        """Create HTTP mock side effect function."""

        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            normalized = downloader.normalize_url(url)
            resp = responses.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect

    def test_main_success(self):
        """service.main() should return 0 on success."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Test Feed", transcript_url)

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        }

        # Create config file
        cfg_path = os.path.join(self.temp_dir, "config.json")
        config_data = {
            "rss": rss_url,
            "output_dir": self.temp_dir,
            "max_episodes": 1,
            "dry_run": True,  # Use dry-run to avoid actual downloads
        }
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh)

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with patch("sys.argv", ["service", "--config", cfg_path]):
                exit_code = service_module.main()

                self.assertEqual(exit_code, 0)

    def test_main_failure(self):
        """service.main() should return 1 on failure."""
        non_existent_path = os.path.join(self.temp_dir, "nonexistent.json")

        with patch("sys.argv", ["service", "--config", non_existent_path]):
            exit_code = service_module.main()

            self.assertEqual(exit_code, 1)

    def test_main_version_flag(self):
        """service.main() should handle --version flag."""
        with patch("sys.argv", ["service", "--version"]):
            with self.assertRaises(SystemExit) as cm:
                service_module.main()
            # argparse version action exits with 0
            self.assertEqual(cm.exception.code, 0)

    def test_main_missing_config_argument(self):
        """service.main() should fail when --config is missing."""
        with patch("sys.argv", ["service"]):
            with self.assertRaises(SystemExit):
                service_module.main()


@pytest.mark.e2e
@pytest.mark.slow
class TestServiceAPIIntegration(unittest.TestCase):
    """Integration tests for service API with public API."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_service_api_imports(self):
        """Verify service API can be imported from main package."""
        # Test that service is accessible via __getattr__
        self.assertTrue(hasattr(podcast_scraper, "service"))
        self.assertEqual(podcast_scraper.service, service_module)

        # Test that ServiceResult is accessible
        self.assertTrue(hasattr(service_module, "ServiceResult"))
        self.assertEqual(service_module.ServiceResult, ServiceResult)

        # Test that run functions are accessible
        self.assertTrue(hasattr(service_module, "run"))
        self.assertTrue(hasattr(service_module, "run_from_config_file"))
        self.assertTrue(hasattr(service_module, "main"))

    def test_service_result_equality(self):
        """ServiceResult should support equality comparison."""
        result1 = ServiceResult(episodes_processed=5, summary="Test")
        result2 = ServiceResult(episodes_processed=5, summary="Test")
        result3 = ServiceResult(episodes_processed=3, summary="Test")

        self.assertEqual(result1, result2)
        self.assertNotEqual(result1, result3)

    def test_service_result_repr(self):
        """ServiceResult should have useful string representation."""
        result = ServiceResult(
            episodes_processed=5,
            summary="Processed 5 episodes",
            success=True,
            error=None,
        )

        repr_str = repr(result)
        self.assertIn("ServiceResult", repr_str)
        self.assertIn("5", repr_str)
        self.assertIn("Processed 5 episodes", repr_str)
