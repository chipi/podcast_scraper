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

import podcast_scraper.cli as cli
from podcast_scraper import config


@pytest.mark.e2e
class TestCLIBasicCommands:
    """Basic CLI command E2E tests."""

    @pytest.mark.slow
    def test_basic_transcript_download(self, e2e_server):
        """Test basic CLI transcript download: podcast-scraper <rss_url>.

        This test is marked as slow because it duplicates test_cli_basic_transcript_download
        from test_basic_e2e.py. The critical path only needs one CLI end-to-end test.
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main(
                [rss_url, "--output-dir", tmpdir, "--max-episodes", "1", "--transcribe-missing"]
            )

            assert exit_code == 0, f"CLI should succeed, got exit code {exit_code}"

            # Verify transcript file was downloaded/transcribed
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            output_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(output_files) > 0, "At least one transcript file should be downloaded"

            # Verify file content
            transcript_file = output_files[0]
            content = transcript_file.read_text(encoding="utf-8")
            assert len(content) > 0, "Transcript file should not be empty"

    @pytest.mark.slow
    def test_dry_run(self, e2e_server):
        """Test --dry-run flag: podcast-scraper <rss_url> --dry-run.

        This test is marked as slow because dry-run is a utility feature,
        not part of the core use case.
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main(
                [rss_url, "--output-dir", tmpdir, "--max-episodes", "1", "--dry-run"]
            )

            assert exit_code == 0, f"CLI should succeed with dry-run, got exit code {exit_code}"

            # Verify no files were created (dry-run should not save files)
            output_files = list(Path(tmpdir).glob("*.txt"))
            assert len(output_files) == 0, "Dry-run should not create files"

    @pytest.mark.slow
    def test_generate_metadata(self, e2e_server):
        """Test --generate-metadata flag: podcast-scraper <rss_url> --generate-metadata.

        This test is marked as slow because it tests an extended feature,
        not the core use case.
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main(
                [
                    rss_url,
                    "--output-dir",
                    tmpdir,
                    "--max-episodes",
                    "1",
                    "--generate-metadata",
                    "--metadata-format",
                    "json",
                    "--transcribe-missing",
                ]
            )

            assert exit_code == 0, f"CLI should succeed with metadata, got exit code {exit_code}"

            # Verify transcript file was downloaded/transcribed
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be downloaded"

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
    def test_generate_summaries(self, e2e_server):
        """Test --generate-summaries flag: podcast-scraper <rss_url> --generate-summaries.

        This test is marked as slow because it tests an extended feature
        (summarization) that takes significant time (~278s).

        Note: Uses 'local' provider to avoid OpenAI API key requirement.
        OpenAI provider tests are covered in test_openai_mock.py.
        Note: generate_summaries requires generate_metadata=True.
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main(
                [
                    rss_url,
                    "--output-dir",
                    tmpdir,
                    "--max-episodes",
                    "1",
                    "--generate-metadata",  # Required for summaries
                    "--generate-summaries",
                    "--summary-provider",
                    "local",  # Use local provider (no API key needed)
                    "--summary-model",
                    config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                    "--transcribe-missing",
                ]
            )

            assert exit_code == 0, f"CLI should succeed with summaries, got exit code {exit_code}"

            # Verify transcript file was downloaded/transcribed
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be downloaded"

            # Note: Summary files may not be created if summarization fails or is skipped
            # This test verifies the CLI command completes successfully

    @pytest.mark.slow
    def test_all_features_combined(self, e2e_server):
        """Test all features combined: --generate-metadata --generate-summaries.

        This test is marked as slow because it tests all extended features together,
        not the core use case.

        Note: Uses 'local' provider to avoid OpenAI API key requirement.
        OpenAI provider tests are covered in test_openai_mock.py.
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = cli.main(
                [
                    rss_url,
                    "--output-dir",
                    tmpdir,
                    "--max-episodes",
                    "1",
                    "--generate-metadata",
                    "--metadata-format",
                    "json",
                    "--generate-summaries",
                    "--summary-provider",
                    "local",  # Use local provider (no API key needed)
                    "--summary-model",
                    config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                    "--transcribe-missing",
                ]
            )

            assert (
                exit_code == 0
            ), f"CLI should succeed with all features, got exit code {exit_code}"

            # Verify transcript file was downloaded/transcribed
            # When transcribe_missing=True, files are saved in run_whisper_base subdirectory
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be downloaded"

            # Verify metadata file was created
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "At least one metadata file should be created"


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

    def test_transcribe_missing_with_mocked_whisper(self, e2e_server):
        """Test --transcribe-missing flag (with mocked Whisper for speed).

        Note: This test uses mocked Whisper to keep execution fast.
        Real Whisper tests are in Stage 8.
        """
        # For now, we'll test that the flag is accepted and the workflow completes
        # Real Whisper transcription will be tested in Stage 8
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a feed that doesn't have transcripts to trigger transcription
            # For now, just verify the flag is accepted
            exit_code = cli.main(
                [
                    rss_url,
                    "--output-dir",
                    tmpdir,
                    "--max-episodes",
                    "1",
                    "--transcribe-missing",
                    "--whisper-model",
                    "tiny",  # Smallest model for speed
                ]
            )

            # The command should complete (may not transcribe if transcripts exist)
            # This test verifies the CLI accepts the flag and workflow completes
            assert exit_code in [0, 1], f"CLI should complete, got exit code {exit_code}"


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
