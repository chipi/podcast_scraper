#!/usr/bin/env python3
"""E2E tests for Library API.

These tests verify the public library API works end-to-end using real HTTP client and E2E server:
- run_pipeline(config) - Basic pipeline
- run_pipeline(config) with all features (metadata, summaries)
- load_config_file(path) + run_pipeline() - Config file workflow

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

from podcast_scraper import Config, config as config_module, load_config_file, run_pipeline

# Removed TestLibraryAPIBasic class (4 tests) as part of Phase 3 consolidation:
# - test_run_pipeline_basic: Duplicate of test_basic_e2e.py (critical path already covered)
# - test_run_pipeline_with_metadata: Extended feature, not core workflow
# - test_run_pipeline_with_summaries: Extended feature, not core workflow
# - test_run_pipeline_all_features: Extended feature, not core workflow
#
# Critical path Library API tests are covered in test_basic_e2e.py::TestBasicLibraryAPIE2E


@pytest.mark.e2e
@pytest.mark.slow
class TestLibraryAPIConfigFile:
    """Library API config file E2E tests."""

    def test_load_config_file_and_run_pipeline_json(self, e2e_server):
        """Test load_config_file() + run_pipeline() with JSON config."""
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

            # Load config file
            config_dict = load_config_file(config_path)

            # Create Config object
            cfg = Config(**config_dict)

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify return values
            assert count > 0, "Should process at least one episode"
            assert isinstance(summary, str), "Summary should be a string"

            # Verify transcript file was created
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be created"

            # Verify metadata file was created
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "At least one metadata file should be created"

    def test_load_config_file_and_run_pipeline_yaml(self, e2e_server):
        """Test load_config_file() + run_pipeline() with YAML config."""
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

            # Load config file
            config_dict = load_config_file(config_path)

            # Create Config object
            cfg = Config(**config_dict)

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify return values
            assert count >= 0, "Count should be non-negative"
            assert isinstance(summary, str), "Summary should be a string"

            # Verify no files were created (dry-run)
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) == 0, "Dry-run should not create files"


@pytest.mark.e2e
@pytest.mark.slow
class TestLibraryAPIReturnValues:
    """Library API return value E2E tests."""

    def test_run_pipeline_return_count(self, e2e_server):
        """Test that run_pipeline() returns correct count."""
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=2,  # Process 2 episodes
                transcribe_missing=True,
                whisper_model=config_module.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            )

            count, summary = run_pipeline(cfg)

            # Verify count matches number of episodes processed
            assert count >= 1, "Should process at least one episode"
            assert count <= 2, "Should not process more than max_episodes"

            # Verify actual files match count
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) == count, "Count should match number of files created"

    def test_run_pipeline_return_summary(self, e2e_server):
        """Test that run_pipeline() returns meaningful summary."""
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                whisper_model=config_module.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            )

            count, summary = run_pipeline(cfg)

            # Verify summary contains useful information
            assert isinstance(summary, str), "Summary should be a string"
            assert len(summary) > 0, "Summary should not be empty"
            # Summary should mention processing or episodes
            assert (
                "episode" in summary.lower()
                or "processed" in summary.lower()
                or "done" in summary.lower()
                or str(count) in summary
            ), "Summary should contain processing information"
