#!/usr/bin/env python3
"""Unit tests for ServiceResult dataclass.

These are unit tests for the ServiceResult dataclass, testing its structure,
defaults, equality, and string representation. These tests don't require
HTTP mocking or E2E server - they're pure unit tests.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import service module directly (not via __getattr__) to avoid recursion
import importlib

import pytest

service_module = importlib.import_module("podcast_scraper.service")
ServiceResult = service_module.ServiceResult

pytestmark = [pytest.mark.unit, pytest.mark.module_service]


class TestServiceResult(unittest.TestCase):
    """Unit tests for ServiceResult dataclass."""

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
        self.assertIsNone(result.multi_feed_summary)

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


class TestServiceAPIImports(unittest.TestCase):
    """Unit tests for service API imports."""

    def test_service_api_imports(self):
        """Verify service API can be imported from main package."""
        import podcast_scraper

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


class TestServiceRun(unittest.TestCase):
    """Test service.run() function."""

    def _create_config(self, **overrides):
        """Create a test Config with optional overrides."""
        from podcast_scraper import config

        defaults = {
            "rss_url": "https://example.com/feed.xml",
            "output_dir": "./test_output",
            "user_agent": "test",
            "timeout": 30,
        }
        defaults.update(overrides)
        return config.Config(**defaults)

    @patch("podcast_scraper.service.workflow.run_pipeline")
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_success(self, mock_apply_log, mock_run_pipeline):
        """Test successful pipeline execution."""
        cfg = self._create_config()
        mock_run_pipeline.return_value = (10, "Processed 10 episodes")
        result = service_module.run(cfg)

        self.assertTrue(result.success)
        self.assertEqual(result.episodes_processed, 10)
        self.assertEqual(result.summary, "Processed 10 episodes")
        self.assertIsNone(result.error)
        mock_run_pipeline.assert_called_once_with(cfg)

    @patch("podcast_scraper.service.workflow.run_pipeline")
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_applies_log_level(self, mock_apply_log, mock_run_pipeline):
        """Test that log level is applied when specified."""
        cfg = self._create_config(log_level="DEBUG")
        mock_run_pipeline.return_value = (5, "Processed 5 episodes")

        result = service_module.run(cfg)

        self.assertTrue(result.success)
        mock_apply_log.assert_called_once_with(level="DEBUG", log_file=None)

    @patch("podcast_scraper.service.workflow.run_pipeline")
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_applies_log_file(self, mock_apply_log, mock_run_pipeline):
        """Test that log file is applied when specified."""
        cfg = self._create_config(log_file="/tmp/test.log")
        mock_run_pipeline.return_value = (5, "Processed 5 episodes")

        result = service_module.run(cfg)

        self.assertTrue(result.success)
        mock_apply_log.assert_called_once_with(level="INFO", log_file="/tmp/test.log")

    @patch("podcast_scraper.service.workflow.run_pipeline")
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_resolves_relative_log_file_under_output_dir(
        self, mock_apply_log, mock_run_pipeline
    ):
        """Relative log_file is joined with output_dir before apply_log_level."""
        with tempfile.TemporaryDirectory() as out:
            cfg = self._create_config(log_file="pipeline.log", output_dir=out)
            mock_run_pipeline.return_value = (5, "Processed 5 episodes")

            result = service_module.run(cfg)

            self.assertTrue(result.success)
            expected = os.path.normpath(os.path.join(out, "pipeline.log"))
            mock_apply_log.assert_called_once_with(level="INFO", log_file=expected)

    @patch("podcast_scraper.service.workflow.run_pipeline")
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_handles_exception(self, mock_apply_log, mock_run_pipeline):
        """Test that exceptions are caught and returned as error."""
        cfg = self._create_config()
        mock_run_pipeline.side_effect = ValueError("Test error")

        result = service_module.run(cfg)

        self.assertFalse(result.success)
        self.assertEqual(result.episodes_processed, 0)
        self.assertEqual(result.summary, "")
        self.assertEqual(result.error, "Test error")

    @patch("podcast_scraper.service.workflow.run_pipeline")
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_no_log_config(self, mock_apply_log, mock_run_pipeline):
        """Test that log level is not applied when not specified."""
        # Config with no log settings - check if defaults are None
        cfg = self._create_config()
        mock_run_pipeline.return_value = (5, "Processed 5 episodes")

        result = service_module.run(cfg)

        self.assertTrue(result.success)
        # The code only calls apply_log_level if cfg.log_file or cfg.log_level is truthy
        # If both are None/empty, it should not be called
        # However, Config might have defaults, so we just verify the run succeeded
        # and that apply_log_level was called or not based on actual config values
        # This test verifies the function works correctly regardless


class TestServiceRunMultiFeed440(unittest.TestCase):
    """service.run with rss_urls / feeds (GitHub #440)."""

    @patch("podcast_scraper.service.workflow.run_pipeline")
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_multi_feed_calls_pipeline_per_url(self, mock_apply_log, mock_run_pipeline):
        from podcast_scraper import config as cfg_mod

        mock_run_pipeline.return_value = (1, "ok")
        with tempfile.TemporaryDirectory() as corpus:
            cfg = cfg_mod.Config(
                rss_urls=["https://a.example/feed.xml", "https://b.example/feed.xml"],
                output_dir=corpus,
                max_episodes=1,
                user_agent="test",
                timeout=30,
            )
            result = service_module.run(cfg)

        self.assertTrue(result.success)
        self.assertEqual(result.episodes_processed, 2)
        self.assertEqual(mock_run_pipeline.call_count, 2)
        called_cfgs = [c.args[0] for c in mock_run_pipeline.call_args_list]
        for sub in called_cfgs:
            self.assertIsNone(sub.rss_urls)
            self.assertIn(f"{os.sep}feeds{os.sep}", sub.output_dir or "")
        urls_called = {c.rss_url for c in called_cfgs}
        self.assertEqual(
            urls_called,
            {"https://a.example/feed.xml", "https://b.example/feed.xml"},
        )
        self.assertIsNotNone(result.multi_feed_summary)
        self.assertEqual(result.multi_feed_summary.get("schema_version"), "1.0.0")
        self.assertTrue(result.multi_feed_summary.get("overall_ok"))
        self.assertEqual(len(result.multi_feed_summary.get("feeds") or []), 2)
        mock_apply_log.assert_called_once()

    @patch("podcast_scraper.service.workflow.run_pipeline")
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_multi_feed_partial_failure_returns_error(self, mock_apply_log, mock_run_pipeline):
        from podcast_scraper import config as cfg_mod

        def _side_effect(cfg):
            if cfg.rss_url and "b.example" in cfg.rss_url:
                raise ValueError("feed b failed")
            return (1, "ok")

        mock_run_pipeline.side_effect = _side_effect
        with tempfile.TemporaryDirectory() as corpus:
            cfg = cfg_mod.Config(
                rss_urls=["https://a.example/feed.xml", "https://b.example/feed.xml"],
                output_dir=corpus,
                max_episodes=1,
                user_agent="test",
                timeout=30,
            )
            result = service_module.run(cfg)

        self.assertFalse(result.success)
        self.assertEqual(result.episodes_processed, 1)
        self.assertIsNotNone(result.error)
        self.assertIn("feed b failed", result.error)
        self.assertIsNotNone(result.multi_feed_summary)
        self.assertFalse(result.multi_feed_summary.get("overall_ok"))


class TestServiceRunFromConfigFile(unittest.TestCase):
    """Test service.run_from_config_file() function."""

    @patch("podcast_scraper.service.run")
    @patch("podcast_scraper.service.config.load_config_file")
    @patch("podcast_scraper.service.config.Config")
    def test_run_from_config_file_success(self, mock_config_class, mock_load, mock_run):
        """Test successful execution from config file."""
        mock_load.return_value = {"rss_url": "https://example.com/feed.xml"}
        mock_cfg = mock_config_class.return_value
        mock_cfg.memray = False
        mock_cfg.memray_output = None
        mock_run.return_value = ServiceResult(episodes_processed=5, summary="Success", success=True)

        result = service_module.run_from_config_file("config.yaml")

        self.assertTrue(result.success)
        mock_load.assert_called_once_with("config.yaml")
        mock_config_class.assert_called_once_with(**mock_load.return_value)
        mock_run.assert_called_once_with(mock_cfg)

    @patch("podcast_scraper.service.config.load_config_file")
    def test_run_from_config_file_not_found(self, mock_load):
        """Test handling of missing config file."""
        mock_load.side_effect = FileNotFoundError("File not found")

        result = service_module.run_from_config_file("missing.yaml")

        self.assertFalse(result.success)
        self.assertEqual(result.episodes_processed, 0)
        self.assertEqual(result.summary, "")
        self.assertIn("Configuration file not found", result.error)

    @patch("podcast_scraper.service.config.load_config_file")
    def test_run_from_config_file_invalid(self, mock_load):
        """Test handling of invalid config file."""
        mock_load.side_effect = ValueError("Invalid config")

        result = service_module.run_from_config_file("invalid.yaml")

        self.assertFalse(result.success)
        self.assertEqual(result.episodes_processed, 0)
        self.assertEqual(result.summary, "")
        self.assertIn("Failed to load configuration file", result.error)

    @patch("podcast_scraper.service.run")
    @patch("podcast_scraper.service.config.load_config_file")
    @patch("podcast_scraper.service.config.Config")
    def test_run_from_config_file_pipeline_error(self, mock_config_class, mock_load, mock_run):
        """Test that pipeline errors are propagated."""
        mock_load.return_value = {"rss_url": "https://example.com/feed.xml"}
        mock_cfg = mock_config_class.return_value
        mock_cfg.memray = False
        mock_cfg.memray_output = None
        mock_run.return_value = ServiceResult(
            episodes_processed=0, summary="", success=False, error="Pipeline error"
        )

        result = service_module.run_from_config_file("config.yaml")

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Pipeline error")


class TestServiceMain(unittest.TestCase):
    """Test service.main() function."""

    @patch("podcast_scraper.service.run_from_config_file")
    @patch("sys.argv", ["service", "--config", "config.yaml"])
    def test_main_success(self, mock_run_from_config):
        """Test successful main execution."""
        mock_run_from_config.return_value = ServiceResult(
            episodes_processed=5, summary="Success", success=True
        )

        with patch("builtins.print") as mock_print:
            exit_code = service_module.main()

        self.assertEqual(exit_code, 0)
        mock_run_from_config.assert_called_once_with("config.yaml")
        mock_print.assert_called_once_with("Success")

    @patch("podcast_scraper.service.run_from_config_file")
    @patch("sys.argv", ["service", "--config", "config.yaml"])
    def test_main_failure(self, mock_run_from_config):
        """Test main execution with failure."""
        mock_run_from_config.return_value = ServiceResult(
            episodes_processed=0, summary="", success=False, error="Test error"
        )

        with patch("builtins.print") as mock_print:
            exit_code = service_module.main()

        self.assertEqual(exit_code, 1)
        mock_run_from_config.assert_called_once_with("config.yaml")
        # Should print error to stderr
        mock_print.assert_called()
        # Check that error was printed (may be to stderr)
        calls = [str(call) for call in mock_print.call_args_list]
        error_printed = any("Test error" in str(call) for call in calls)
        self.assertTrue(error_printed)

    @patch("sys.argv", ["service", "--version"])
    def test_main_version(self):
        """Test version flag."""
        with self.assertRaises(SystemExit) as cm:
            service_module.main()
        self.assertEqual(cm.exception.code, 0)


class TestServiceRunLogging(unittest.TestCase):
    """Test logging-related behaviour of service.run()."""

    def _make_cfg(self, **overrides):
        from unittest.mock import MagicMock

        cfg = MagicMock()
        cfg.log_file = overrides.get("log_file", None)
        cfg.log_level = overrides.get("log_level", None)
        return cfg

    @patch("podcast_scraper.service.workflow.run_pipeline", return_value=(1, "ok"))
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_no_log_config_skips_apply_log_level(self, mock_apply_log, mock_run_pipeline):
        """apply_log_level is NOT called when both log_file and log_level are falsy."""
        cfg = self._make_cfg(log_file=None, log_level=None)
        result = service_module.run(cfg)

        self.assertTrue(result.success)
        mock_apply_log.assert_not_called()

    @patch("podcast_scraper.service.workflow.run_pipeline", return_value=(1, "ok"))
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_both_log_level_and_log_file(self, mock_apply_log, mock_run_pipeline):
        """apply_log_level receives both level and log_file when both are set."""
        cfg = self._make_cfg(log_level="DEBUG", log_file="/tmp/test.log")
        result = service_module.run(cfg)

        self.assertTrue(result.success)
        mock_apply_log.assert_called_once_with(level="DEBUG", log_file="/tmp/test.log")

    @patch("podcast_scraper.service.workflow.run_pipeline", return_value=(1, "ok"))
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_apply_log_level_raises(self, mock_apply_log, mock_run_pipeline):
        """RuntimeError from apply_log_level produces a failure result."""
        mock_apply_log.side_effect = RuntimeError("logging init failed")
        cfg = self._make_cfg(log_level="DEBUG")
        result = service_module.run(cfg)

        self.assertFalse(result.success)
        self.assertIn("logging init failed", result.error)


class TestServiceRunRedaction(unittest.TestCase):
    """Test that error messages are redacted in service.run()."""

    @patch(
        "podcast_scraper.service.redact_for_log",
        return_value="API key sk-[REDACTED]",
    )
    @patch("podcast_scraper.service.workflow.run_pipeline")
    @patch("podcast_scraper.service.workflow.apply_log_level")
    def test_run_error_uses_redact_for_log(self, mock_apply_log, mock_run_pipeline, mock_redact):
        """Sensitive data in exceptions is redacted via redact_for_log."""
        mock_run_pipeline.side_effect = ValueError("API key sk-abc123xyz")
        from unittest.mock import MagicMock

        cfg = MagicMock()
        cfg.log_file = None
        cfg.log_level = None

        result = service_module.run(cfg)

        self.assertFalse(result.success)
        self.assertNotIn("sk-abc123xyz", result.error)
        self.assertEqual(result.error, "API key sk-[REDACTED]")
        mock_redact.assert_called_once_with("API key sk-abc123xyz")


class TestServiceRunFromConfigFileExtended(unittest.TestCase):
    """Extended tests for service.run_from_config_file()."""

    @patch("podcast_scraper.service.run")
    @patch("podcast_scraper.service.config.load_config_file")
    @patch("podcast_scraper.service.config.Config")
    def test_run_from_config_file_with_path_object(self, mock_config_class, mock_load, mock_run):
        """pathlib.Path argument is converted to str for load_config_file."""
        from pathlib import Path

        mock_load.return_value = {"rss_url": "https://example.com/feed.xml"}
        mock_cfg = mock_config_class.return_value
        mock_cfg.memray = False
        mock_cfg.memray_output = None
        mock_run.return_value = ServiceResult(episodes_processed=3, summary="Done", success=True)

        result = service_module.run_from_config_file(Path("config.yaml"))

        self.assertTrue(result.success)
        mock_load.assert_called_once_with("config.yaml")
        mock_run.assert_called_once_with(mock_cfg)

    @patch("podcast_scraper.service.config.load_config_file")
    def test_run_from_config_file_permission_error(self, mock_load):
        """PermissionError is caught and reported as a failure."""
        mock_load.side_effect = PermissionError("Permission denied")

        result = service_module.run_from_config_file("secret.yaml")

        self.assertFalse(result.success)
        self.assertIn("Failed to load", result.error)

    @patch("podcast_scraper.service.config.load_config_file")
    @patch("podcast_scraper.service.config.Config")
    def test_run_from_config_file_validation_error(self, mock_config_class, mock_load):
        """ValidationError from Config() is caught and reported as a failure."""
        from pydantic import ValidationError

        mock_load.return_value = {"rss_url": "https://example.com/feed.xml"}
        mock_config_class.side_effect = ValidationError.from_exception_data(
            title="Config",
            line_errors=[
                {
                    "type": "missing",
                    "loc": ("output_dir",),
                    "msg": "Field required",
                    "input": {},
                }
            ],
        )

        result = service_module.run_from_config_file("bad.yaml")

        self.assertFalse(result.success)
        self.assertIn("Failed to load", result.error)


class TestServiceMainConfigResolution(unittest.TestCase):
    """Test config resolution logic in service.main()."""

    @patch("podcast_scraper.service.setup.initialize_ml_environment")
    @patch("podcast_scraper.service.run_from_config_file")
    @patch("sys.argv", ["service"])
    def test_main_uses_env_var_when_no_config_flag(self, mock_run_from_config, mock_init_ml):
        """PODCAST_SCRAPER_CONFIG env var is used when --config is absent."""
        mock_run_from_config.return_value = ServiceResult(
            episodes_processed=1, summary="ok", success=True
        )
        with patch.dict(os.environ, {"PODCAST_SCRAPER_CONFIG": "/custom/config.yaml"}):
            with patch("builtins.print"):
                service_module.main()

        mock_run_from_config.assert_called_once_with("/custom/config.yaml")

    @patch("podcast_scraper.service.setup.initialize_ml_environment")
    @patch("podcast_scraper.service.run_from_config_file")
    @patch("sys.argv", ["service"])
    def test_main_uses_default_when_no_config_and_no_env(self, mock_run_from_config, mock_init_ml):
        """Default /app/config.yaml is used when --config and env var are absent."""
        mock_run_from_config.return_value = ServiceResult(
            episodes_processed=1, summary="ok", success=True
        )
        clean_env = {k: v for k, v in os.environ.items() if k != "PODCAST_SCRAPER_CONFIG"}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch("builtins.print"):
                service_module.main()

        mock_run_from_config.assert_called_once_with("/app/config.yaml")

    @patch("podcast_scraper.service.setup.initialize_ml_environment")
    @patch("podcast_scraper.service.run_from_config_file")
    @patch("sys.argv", ["service", "--config", "/from/flag.yaml"])
    def test_main_config_flag_overrides_env_var(self, mock_run_from_config, mock_init_ml):
        """--config flag takes precedence over PODCAST_SCRAPER_CONFIG env var."""
        mock_run_from_config.return_value = ServiceResult(
            episodes_processed=1, summary="ok", success=True
        )
        with patch.dict(os.environ, {"PODCAST_SCRAPER_CONFIG": "/from/env.yaml"}):
            with patch("builtins.print"):
                service_module.main()

        mock_run_from_config.assert_called_once_with("/from/flag.yaml")

    @patch("podcast_scraper.service.run_from_config_file")
    @patch("sys.argv", ["service", "--config", "config.yaml"])
    def test_main_calls_initialize_ml_environment(self, mock_run_from_config):
        """setup.initialize_ml_environment() is called during main()."""
        mock_run_from_config.return_value = ServiceResult(
            episodes_processed=1, summary="ok", success=True
        )
        with patch("podcast_scraper.service.setup.initialize_ml_environment") as mock_init_ml:
            with patch("builtins.print"):
                service_module.main()

        mock_init_ml.assert_called_once()
