#!/usr/bin/env python3
"""E2E tests for Service API.

These tests verify the Service API works end-to-end using real HTTP client and E2E server:
- service.run(config) - Basic service execution
- service.run_from_config_file(path) - Config file execution
- service.main() - CLI entry point

All tests use real HTTP client (no mocking) and E2E server fixture.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import Config, config as config_module, service

pytestmark = [pytest.mark.e2e, pytest.mark.module_service]

# Removed TestServiceAPIBasic class (4 tests) as part of Phase 3 consolidation:
# - test_service_run_basic: Duplicate of test_basic_e2e.py (critical path already covered)
# - test_service_run_with_metadata: Extended feature, not core workflow
# - test_service_run_with_summaries: Extended feature, not core workflow
# - test_service_run_all_features: Extended feature, not core workflow
#
# Critical path Service API tests are covered in test_basic_e2e.py::TestBasicServiceAPIE2E


@pytest.mark.e2e
@pytest.mark.slow
class TestServiceAPIConfigFile:
    """Service API config file E2E tests."""

    def test_service_run_from_config_file_json(self, e2e_server):
        """Test service.run_from_config_file() with JSON config."""
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "rss_url": rss_url,
                "output_dir": tmpdir,
                "max_episodes": 1,
                "generate_metadata": True,
                "metadata_format": "json",
                "transcribe_missing": True,
                "whisper_model": config_module.TEST_DEFAULT_WHISPER_MODEL,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f)

            # Run service from config file
            result = service.run_from_config_file(config_path)

            # Verify ServiceResult
            assert result.success is True, f"Service should succeed, got: {result.error}"
            assert result.episodes_processed > 0, "At least one episode should be processed"

            # Verify transcript file was created
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be created"

            # Verify metadata file was created
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "At least one metadata file should be created"

    def test_service_run_from_config_file_yaml(self, e2e_server):
        """Test service.run_from_config_file() with YAML config."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_path = os.path.join(tmpdir, "config.yaml")
            config_data = {
                "rss_url": rss_url,
                "output_dir": tmpdir,
                "max_episodes": 1,
                "dry_run": True,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            # Run service from config file
            result = service.run_from_config_file(config_path)

            # Verify ServiceResult
            assert result.success is True, f"Service should succeed, got: {result.error}"
            assert result.episodes_processed >= 0, "Count should be non-negative"

            # Verify no files were created (dry-run)
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) == 0, "Dry-run should not create files"

    def test_service_run_from_config_file_not_found(self):
        """Test service.run_from_config_file() with missing config file."""
        result = service.run_from_config_file("/nonexistent/config.json")

        # Verify ServiceResult indicates failure
        assert result.success is False, "Service should fail with missing config file"
        assert result.error is not None, "Service should have error message"
        assert (
            "not found" in result.error.lower() or "Configuration file" in result.error
        ), "Error should mention missing file"
        assert result.episodes_processed == 0, "No episodes should be processed on error"


@pytest.mark.e2e
@pytest.mark.slow
class TestServiceAPIErrorHandling:
    """Service API error handling E2E tests."""

    def test_service_run_error_handling(self, e2e_server):
        """Test service.run() error handling with invalid config."""
        # Use invalid RSS URL (will cause error)
        invalid_url = e2e_server.urls.base() + "/nonexistent/feed.xml"

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=invalid_url,
                output_dir=tmpdir,
                max_episodes=1,
            )

            # Run service
            result = service.run(cfg)

            # Verify ServiceResult indicates failure
            assert result.success is False, "Service should fail with invalid RSS URL"
            assert result.error is not None, "Service should have error message"
            assert result.episodes_processed == 0, "No episodes should be processed on error"

    def test_service_run_from_config_file_invalid_config(self, e2e_server):
        """Test service.run_from_config_file() with invalid config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid config file (not JSON or YAML)
            config_path = os.path.join(tmpdir, "config.invalid")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("not valid json or yaml")

            # Run service from config file
            result = service.run_from_config_file(config_path)

            # Verify ServiceResult indicates failure
            assert result.success is False, "Service should fail with invalid config file"
            assert result.error is not None, "Service should have error message"
            assert result.episodes_processed == 0, "No episodes should be processed on error"


@pytest.mark.e2e
@pytest.mark.slow
class TestServiceAPIReturnValues:
    """Service API return value E2E tests."""

    def test_service_run_return_structure(self, e2e_server):
        """Test that service.run() returns correct ServiceResult structure."""
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
            )

            result = service.run(cfg)

            # Verify ServiceResult has all required fields
            assert hasattr(
                result, "episodes_processed"
            ), "ServiceResult should have episodes_processed"
            assert hasattr(result, "summary"), "ServiceResult should have summary"
            assert hasattr(result, "success"), "ServiceResult should have success"
            assert hasattr(result, "error"), "ServiceResult should have error"

            # Verify types
            assert isinstance(result.episodes_processed, int), "episodes_processed should be int"
            assert isinstance(result.summary, str), "summary should be str"
            assert isinstance(result.success, bool), "success should be bool"
            assert result.error is None or isinstance(
                result.error, str
            ), "error should be str or None"

    def test_service_run_success_vs_error(self, e2e_server):
        """Test that service.run() correctly sets success flag."""
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test success case
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                whisper_model=config_module.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            )
            result = service.run(cfg)
            assert result.success is True, "Service should succeed with valid config"
            assert result.error is None, "Service should have no error on success"
            assert result.episodes_processed > 0, "Service should process episodes on success"

            # Test error case
            invalid_url = e2e_server.urls.base() + "/nonexistent/feed.xml"
            cfg_error = Config(
                rss_url=invalid_url,
                output_dir=tmpdir,
                max_episodes=1,
            )
            result_error = service.run(cfg_error)
            assert result_error.success is False, "Service should fail with invalid config"
            assert result_error.error is not None, "Service should have error message on failure"
            assert (
                result_error.episodes_processed == 0
            ), "Service should process no episodes on failure"


@pytest.mark.e2e
@pytest.mark.slow
class TestServiceMainCLI:
    """E2E tests for service.main() CLI entry point."""

    def test_service_main_success(self, e2e_server):
        """Test service.main() with valid config file."""
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            cfg_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "rss_url": rss_url,
                "output_dir": tmpdir,
                "max_episodes": 1,
                "dry_run": True,  # Use dry-run to avoid actual downloads
            }
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f)

            # Patch sys.argv to simulate CLI call
            from unittest.mock import patch

            with patch("sys.argv", ["service", "--config", cfg_path]):
                exit_code = service.main()

                assert exit_code == 0, f"service.main() should return 0 on success, got {exit_code}"

    def test_service_main_failure(self):
        """Test service.main() with non-existent config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent_path = os.path.join(tmpdir, "nonexistent.json")

            from unittest.mock import patch

            with patch("sys.argv", ["service", "--config", non_existent_path]):
                exit_code = service.main()

                assert exit_code == 1, f"service.main() should return 1 on failure, got {exit_code}"

    def test_service_main_version_flag(self):
        """Test service.main() with --version flag."""
        from unittest.mock import patch

        with patch("sys.argv", ["service", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                service.main()
            # argparse version action exits with 0
            assert exc_info.value.code == 0, "service.main() should exit with 0 for --version"

    def test_service_main_missing_config_argument(self):
        """Test service.main() fails when --config is missing."""
        from unittest.mock import patch

        with patch("sys.argv", ["service"]):
            with pytest.raises(SystemExit):
                service.main()
