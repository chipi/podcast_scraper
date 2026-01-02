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

from podcast_scraper import Config, config, load_config_file, run_pipeline


@pytest.mark.e2e
class TestLibraryAPIBasic:
    """Basic Library API E2E tests."""

    @pytest.mark.slow
    def test_run_pipeline_basic(self, e2e_server):
        """Test basic run_pipeline() with minimal config.

        This test is marked as slow because it duplicates test_library_api_basic_pipeline
        from test_basic_e2e.py. The critical path only needs one Library API end-to-end test.
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify return values
            assert count > 0, "Should process at least one episode"
            assert isinstance(summary, str), "Summary should be a string"
            assert len(summary) > 0, "Summary should not be empty"

            # Verify transcript file was created
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be created"

            # Verify file content
            transcript_file = transcript_files[0]
            content = transcript_file.read_text(encoding="utf-8")
            assert len(content) > 0, "Transcript file should not be empty"

    @pytest.mark.slow
    def test_run_pipeline_with_metadata(self, e2e_server):
        """Test run_pipeline() with metadata generation.

        This test is marked as slow because it tests an extended feature,
        not the core use case.
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                generate_metadata=True,
                metadata_format="json",
                transcribe_missing=True,
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify return values
            assert count > 0, "Should process at least one episode"
            assert (
                "processed" in summary.lower() or "done" in summary.lower()
            ), "Summary should indicate processing completed"

            # Verify transcript file was created
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be created"

            # Verify metadata file was created
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "At least one metadata file should be created"

            # Verify metadata content
            if metadata_files:
                metadata_content = json.loads(metadata_files[0].read_text(encoding="utf-8"))
                assert (
                    "episode" in metadata_content or "title" in metadata_content
                ), "Metadata should contain episode information"

    @pytest.mark.slow
    def test_run_pipeline_with_summaries(self, e2e_server):
        """Test run_pipeline() with summarization.

        This test is marked as slow because it tests an extended feature
        (summarization) that takes significant time (~278s).

        Note: Uses 'local' provider to avoid OpenAI API key requirement.
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                generate_metadata=True,  # Required for summaries
                generate_summaries=True,
                summary_provider="local",  # Use local provider (no API key needed)
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # Test default: bart-base
                transcribe_missing=True,
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify return values
            assert count > 0, "Should process at least one episode"
            assert isinstance(summary, str), "Summary should be a string"

            # Verify transcript file was created
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be created"

            # Verify metadata file was created (summaries are stored in metadata)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "At least one metadata file should be created"

    @pytest.mark.slow
    def test_run_pipeline_all_features(self, e2e_server):
        """Test run_pipeline() with all features enabled.

        This test is marked as slow because it tests all extended features together,
        not the core use case.
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                generate_metadata=True,
                metadata_format="json",
                generate_summaries=True,
                summary_provider="local",  # Use local provider (no API key needed)
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # Test default: bart-base
                transcribe_missing=True,
            )

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
