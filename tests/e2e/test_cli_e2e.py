#!/usr/bin/env python3
"""E2E tests for CLI commands.

These tests verify CLI commands work end-to-end using real HTTP client and E2E server:
- Basic transcript download
- --transcribe-missing (Whisper fallback)
- --config (config file workflow)
- --dry-run
- --generate-metadata
- --generate-summaries
- Combined features

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

import pytest

import podcast_scraper.cli as cli
from podcast_scraper import config as config_module

pytestmark = [pytest.mark.e2e, pytest.mark.module_cli]

# Removed TestCLIBasicCommands class (5 tests) as part of Phase 3 consolidation:
# - test_basic_transcript_download: Duplicate of test_basic_e2e.py (critical path already covered)
# - test_dry_run: Utility feature, not core workflow
# - test_generate_metadata: Extended feature, not core workflow
# - test_generate_summaries: Extended feature, not core workflow
# - test_all_features_combined: Extended feature, not core workflow
#
# Critical path CLI tests are covered in test_basic_e2e.py::TestBasicCLIE2E


@pytest.mark.e2e
@pytest.mark.slow
class TestCLIConfigFile:
    """CLI config file E2E tests."""

    def test_config_file_json(self, e2e_server):
        """Test --config with JSON file: podcast-scraper --config <config.json>."""
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "rss": rss_url,
                "output_dir": tmpdir,
                "max_episodes": 1,
                "generate_metadata": True,
                "metadata_format": "json",
                "transcribe_missing": True,
                "whisper_model": config_module.TEST_DEFAULT_WHISPER_MODEL,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f)

            exit_code = cli.main(["--config", config_path])

            assert exit_code == 0, f"CLI should succeed with config file, got exit code {exit_code}"

            # Verify transcript file was downloaded/transcribed
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be downloaded"

            # Verify metadata file was created (from config)
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "At least one metadata file should be created"

    def test_config_file_yaml(self, e2e_server):
        """Test --config with YAML file: podcast-scraper --config <config.yaml>."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_path = os.path.join(tmpdir, "config.yaml")
            config_data = {
                "rss": rss_url,
                "output_dir": tmpdir,
                "max_episodes": 1,
                "dry_run": True,
                "transcribe_missing": True,
                "whisper_model": config_module.TEST_DEFAULT_WHISPER_MODEL,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            exit_code = cli.main(["--config", config_path])

            assert (
                exit_code == 0
            ), f"CLI should succeed with YAML config file, got exit code {exit_code}"

            # Verify no files were created (dry-run from config)
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) == 0, "Dry-run from config should not create files"

    def test_cli_overrides_config(self, e2e_server):
        """Test CLI arguments override config file values."""
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file with max_episodes=2
            config_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "rss": rss_url,
                "output_dir": tmpdir,
                "max_episodes": 2,
                "transcribe_missing": True,
                "whisper_model": config_module.TEST_DEFAULT_WHISPER_MODEL,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f)

            # Override max_episodes to 1 via CLI
            exit_code = cli.main(["--config", config_path, "--max-episodes", "1"])

            assert exit_code == 0, f"CLI should succeed, got exit code {exit_code}"

            # Verify only 1 episode was processed (CLI override)
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) == 1, "CLI should override config max_episodes"


@pytest.mark.e2e
@pytest.mark.slow
class TestCLITranscribeMissing:
    """CLI --transcribe-missing E2E tests (marked as slow for Whisper)."""

    def test_transcribe_missing_with_real_whisper(self, e2e_server):
        """Test --transcribe-missing flag with real Whisper.

        This test uses real Whisper to transcribe audio.
        Requires Whisper model to be cached.
        """
        from podcast_scraper import config
        from tests.integration.ml_model_cache_helpers import require_whisper_model_cached

        # Require Whisper model to be cached (skip if not available)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main(
                [
                    rss_url,
                    "--output-dir",
                    tmpdir,
                    "--max-episodes",
                    "1",
                    "--transcribe-missing",
                    "--whisper-model",
                    config.TEST_DEFAULT_WHISPER_MODEL,
                ]
            )

            # The command should complete successfully
            assert exit_code == 0, f"CLI should succeed, got exit code {exit_code}"

            # Verify transcript file was created
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be created"


@pytest.mark.e2e
@pytest.mark.slow
class TestCLIErrorHandling:
    """CLI error handling E2E tests."""

    def test_invalid_rss_url(self, e2e_server):
        """Test CLI with invalid RSS URL."""
        invalid_url = e2e_server.urls.base() + "/nonexistent/feed.xml"

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main([invalid_url, "--output-dir", tmpdir, "--max-episodes", "1"])

            # Should fail gracefully
            assert exit_code != 0, "CLI should fail with invalid RSS URL"

    def test_missing_config_file(self):
        """Test CLI with missing config file."""
        exit_code = cli.main(["--config", "/nonexistent/config.json"])

        # Should fail gracefully
        assert exit_code != 0, "CLI should fail with missing config file"

    def test_invalid_config_file(self, e2e_server):
        """Test CLI with invalid config file."""
        e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid config file (not JSON or YAML)
            config_path = os.path.join(tmpdir, "config.invalid")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("not valid json or yaml")

            exit_code = cli.main(["--config", config_path])

            # Should fail gracefully
            assert exit_code != 0, "CLI should fail with invalid config file"
