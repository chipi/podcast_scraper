#!/usr/bin/env python3
"""True E2E tests using CLI as subprocess (like real user).

These tests execute the CLI as a subprocess (`python -m podcast_scraper.cli <args>`),
exactly as a user would. This ensures the CLI entry point, argument parsing,
progress reporting, and full application flow are tested together.

This complements direct-call E2E tests (which call cli.main() directly) by
testing the complete user experience from command line invocation.

All tests use real HTTP client (no mocking) and E2E server fixture.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import config

# Import cache helpers from integration tests
integration_dir = Path(__file__).parent.parent / "integration"
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))
from ml_model_cache_helpers import (  # noqa: E402
    require_transformers_model_cached,
)


def run_cli_subprocess(
    args: list[str],
    cwd: Path,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Execute CLI as subprocess.

    Args:
        args: CLI arguments (without 'python -m podcast_scraper.cli')
        cwd: Working directory for subprocess
        timeout: Timeout in seconds (default: 5 minutes)
        env: Environment variables (optional, merged with current env)

    Returns:
        CompletedProcess with returncode, stdout, stderr
    """
    cmd = [
        sys.executable,  # Use same Python interpreter
        "-m",
        "podcast_scraper.cli",
    ] + args

    # Merge provided env with current environment
    subprocess_env = os.environ.copy()
    if env:
        subprocess_env.update(env)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=timeout,
        env=subprocess_env,
    )


@pytest.fixture
def project_root() -> Path:
    """Fixture providing path to project root (for subprocess cwd).

    Returns:
        Path to project root directory
    """
    return Path(__file__).parent.parent.parent


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.critical_path
class TestCLISubprocessE2E:
    """True E2E tests using CLI as subprocess (like real user)."""

    @pytest.mark.critical_path
    def test_cli_minimal_via_subprocess(self, e2e_server, tmp_path, project_root):
        """Test minimal CLI usage: RSS URL + output dir via subprocess.

        This test exercises the ENTIRE application flow:
        - CLI entry point (python -m podcast_scraper.cli)
        - CLI argument parsing
        - Config building from CLI args
        - Progress reporting setup
        - Workflow execution
        - All components working together

        Uses subprocess to execute CLI exactly as a user would.
        """
        rss_url = e2e_server.urls.feed("podcast1_with_transcript")
        output_dir = tmp_path / "output"

        # Execute CLI as subprocess (like real user)
        result = run_cli_subprocess(
            [
                rss_url,
                "--output-dir",
                str(output_dir),
                "--max-episodes",
                "1",
            ],
            cwd=project_root,
        )

        # Verify CLI execution
        assert result.returncode == 0, (
            f"CLI should succeed. "
            f"returncode: {result.returncode}, "
            f"stdout: {result.stdout}, "
            f"stderr: {result.stderr}"
        )

        # Verify output files created
        transcript_files = list(output_dir.rglob("*.txt"))
        assert len(transcript_files) > 0, "Transcript files should be created"

    @pytest.mark.critical_path
    def test_cli_full_features_path1_via_subprocess(self, e2e_server, tmp_path, project_root):
        """Test Path 1 (transcript exists) with all features via CLI subprocess.

        This test validates the COMPLETE Path 1 of the critical path via subprocess:
        RSS → Parse → Download Transcript → NER → Summarization → Metadata → Files

        Uses podcast1_with_transcript which has a transcript URL, so Whisper is NOT needed.
        Uses real ML providers (local spaCy for NER, local transformers for summarization).
        Requires models to be pre-cached (skip if not available).
        Enables NER, summarization, and metadata to cover the full critical path.
        """
        # Require ML models to be cached (skip if not available)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1_with_transcript")
        output_dir = tmp_path / "output"

        # Execute CLI as subprocess with all features enabled
        result = run_cli_subprocess(
            [
                rss_url,
                "--output-dir",
                str(output_dir),
                "--max-episodes",
                "1",
                "--auto-speakers",  # Enable NER (speaker detection)
                "--generate-metadata",  # Required for summaries (summaries stored in metadata)
                "--generate-summaries",  # Enable summarization
                "--summary-model",
                config.TEST_DEFAULT_SUMMARY_MODEL,
            ],
            cwd=project_root,
        )

        # Verify CLI execution
        assert result.returncode == 0, (
            f"CLI should succeed. "
            f"returncode: {result.returncode}, "
            f"stdout: {result.stdout}, "
            f"stderr: {result.stderr}"
        )

        # Verify transcript files created
        transcript_files = list(output_dir.rglob("*.txt"))
        assert len(transcript_files) > 0, "Transcript files should be created"

        # Verify metadata file was created (indicates NER and summarization ran)
        metadata_files = list(output_dir.rglob("*.metadata.json"))
        assert (
            len(metadata_files) > 0
        ), "Metadata file should be created (indicates NER and summarization ran)"

        # Verify metadata contains NER and summarization results
        if metadata_files:
            import json

            with open(metadata_files[0], "r", encoding="utf-8") as f:
                metadata = json.load(f)
                assert "content" in metadata, "Metadata should have content section"
                # Should have speaker detection results (NER)
                assert (
                    "detected_hosts" in metadata["content"]
                    or "detected_guests" in metadata["content"]
                    or "speakers" in metadata["content"]
                ), "Metadata should contain speaker detection results (NER)"
                # Should have summary (summarization) - summary is top-level
                assert "summary" in metadata, "Metadata should contain summary"
                assert metadata["summary"] is not None, "Summary should not be None"

    @pytest.mark.critical_path
    def test_cli_full_features_path2_via_subprocess(self, e2e_server, tmp_path, project_root):
        """Test Path 2 (transcript missing, needs Whisper) with all features via CLI subprocess.

        This test validates the COMPLETE Path 2 of the critical path via subprocess:
        RSS → Parse → Download Audio → Whisper Transcription → NER → Summarization
        → Metadata → Files

        Uses podcast1 (without transcript URL), so Whisper transcription is required.
        Uses real ML providers (Whisper, spaCy, Transformers).
        Requires models to be pre-cached (skip if not available).
        Enables NER, summarization, and metadata to cover the full critical path.
        """
        # Require ML models to be cached (skip if not available)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1")
        output_dir = tmp_path / "output"

        # Execute CLI as subprocess with all features enabled
        result = run_cli_subprocess(
            [
                rss_url,
                "--output-dir",
                str(output_dir),
                "--max-episodes",
                "1",
                "--transcribe-missing",  # Enable Whisper transcription
                "--whisper-model",
                config.TEST_DEFAULT_WHISPER_MODEL,
                "--auto-speakers",  # Enable NER (speaker detection)
                "--generate-metadata",  # Required for summaries (summaries stored in metadata)
                "--generate-summaries",  # Enable summarization
                "--summary-model",
                config.TEST_DEFAULT_SUMMARY_MODEL,
            ],
            cwd=project_root,
        )

        # Verify CLI execution
        assert result.returncode == 0, (
            f"CLI should succeed. "
            f"returncode: {result.returncode}, "
            f"stdout: {result.stdout}, "
            f"stderr: {result.stderr}"
        )

        # Verify transcript files created
        transcript_files = list(output_dir.rglob("*.txt"))
        assert len(transcript_files) > 0, "Transcript files should be created"


@pytest.mark.e2e
@pytest.mark.slow
class TestCLISubprocessConfigFile:
    """CLI config file tests via subprocess."""

    def test_cli_config_file_json_via_subprocess(self, e2e_server, tmp_path, project_root):
        """Test CLI with JSON config file via subprocess."""
        rss_url = e2e_server.urls.feed("podcast1_with_transcript")
        config_file = tmp_path / "config.json"
        output_dir = tmp_path / "output"

        # Create config file
        import json

        config_data = {
            "rss": rss_url,
            "output_dir": str(output_dir),
            "max_episodes": 1,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        # Execute CLI as subprocess with config file
        result = run_cli_subprocess(
            ["--config", str(config_file)],
            cwd=project_root,
        )

        # Verify CLI execution
        assert result.returncode == 0, (
            f"CLI should succeed with JSON config. "
            f"returncode: {result.returncode}, "
            f"stderr: {result.stderr}"
        )

        # Verify output files created
        transcript_files = list(output_dir.rglob("*.txt"))
        assert len(transcript_files) > 0, "Transcript files should be created"

    def test_cli_config_file_yaml_via_subprocess(self, e2e_server, tmp_path, project_root):
        """Test CLI with YAML config file via subprocess."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not available")

        rss_url = e2e_server.urls.feed("podcast1_with_transcript")
        config_file = tmp_path / "config.yaml"
        output_dir = tmp_path / "output"

        # Create config file
        config_data = {
            "rss": rss_url,
            "output_dir": str(output_dir),
            "max_episodes": 1,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # Execute CLI as subprocess with config file
        result = run_cli_subprocess(
            ["--config", str(config_file)],
            cwd=project_root,
        )

        # Verify CLI execution
        assert result.returncode == 0, (
            f"CLI should succeed with YAML config. "
            f"returncode: {result.returncode}, "
            f"stderr: {result.stderr}"
        )

        # Verify output files created
        transcript_files = list(output_dir.rglob("*.txt"))
        assert len(transcript_files) > 0, "Transcript files should be created"

    def test_cli_overrides_config_via_subprocess(self, e2e_server, tmp_path, project_root):
        """Test CLI arguments override config file values via subprocess."""
        rss_url = e2e_server.urls.feed("podcast1_with_transcript")
        config_file = tmp_path / "config.json"
        output_dir = tmp_path / "output"

        # Create config file with max_episodes=5
        import json

        config_data = {
            "rss": rss_url,
            "output_dir": str(output_dir),
            "max_episodes": 5,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        # Execute CLI as subprocess with config file + CLI override
        result = run_cli_subprocess(
            ["--config", str(config_file), "--max-episodes", "1"],
            cwd=project_root,
        )

        # Verify CLI execution
        assert result.returncode == 0, (
            f"CLI should succeed. " f"returncode: {result.returncode}, " f"stderr: {result.stderr}"
        )

        # Verify only 1 episode processed (CLI override worked)
        transcript_files = list(output_dir.rglob("*.txt"))
        assert len(transcript_files) == 1, "CLI should override config max_episodes"


@pytest.mark.e2e
class TestCLISubprocessArgumentParsing:
    """CLI argument parsing tests via subprocess."""

    def test_cli_version_flag_via_subprocess(self, project_root):
        """Test CLI --version flag via subprocess."""
        result = run_cli_subprocess(
            ["--version"],
            cwd=project_root,
        )

        # Version should print and exit with code 0
        assert result.returncode == 0, (
            f"Version flag should succeed. "
            f"returncode: {result.returncode}, "
            f"stderr: {result.stderr}"
        )
        # Version output should contain podcast_scraper
        assert (
            "podcast_scraper" in result.stdout.lower() or "podcast_scraper" in result.stderr.lower()
        ), (
            f"Version output should contain 'podcast_scraper'. "
            f"stdout: {result.stdout}, "
            f"stderr: {result.stderr}"
        )


@pytest.mark.e2e
class TestCLISubprocessErrorHandling:
    """CLI error handling tests via subprocess."""

    def test_cli_invalid_rss_url_via_subprocess(self, e2e_server, tmp_path, project_root):
        """Test CLI with invalid RSS URL via subprocess."""
        invalid_url = e2e_server.urls.base() + "/nonexistent/feed.xml"

        result = run_cli_subprocess(
            [
                invalid_url,
                "--output-dir",
                str(tmp_path),
                "--max-episodes",
                "1",
            ],
            cwd=project_root,
        )

        # Should fail gracefully
        assert result.returncode != 0, (
            f"CLI should fail with invalid RSS URL. "
            f"returncode: {result.returncode}, "
            f"stdout: {result.stdout}, "
            f"stderr: {result.stderr}"
        )
        # Should have error message
        assert (
            "error" in result.stderr.lower()
            or "failed" in result.stderr.lower()
            or "not found" in result.stderr.lower()
        ), f"Should have error message in stderr: {result.stderr}"

    def test_cli_missing_config_file_via_subprocess(self, tmp_path, project_root):
        """Test CLI with non-existent config file via subprocess."""
        non_existent_config = tmp_path / "nonexistent.json"

        result = run_cli_subprocess(
            ["--config", str(non_existent_config)],
            cwd=project_root,
        )

        # Should fail gracefully
        assert result.returncode != 0, (
            f"CLI should fail with missing config file. "
            f"returncode: {result.returncode}, "
            f"stderr: {result.stderr}"
        )

    def test_cli_invalid_config_file_via_subprocess(self, tmp_path, project_root):
        """Test CLI with invalid config file via subprocess."""
        invalid_config = tmp_path / "invalid.json"

        # Create invalid JSON file
        with open(invalid_config, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")

        result = run_cli_subprocess(
            ["--config", str(invalid_config)],
            cwd=project_root,
        )

        # Should fail gracefully
        assert result.returncode != 0, (
            f"CLI should fail with invalid config file. "
            f"returncode: {result.returncode}, "
            f"stderr: {result.stderr}"
        )

    def test_cli_missing_required_args_via_subprocess(self, project_root):
        """Test CLI with missing required arguments via subprocess."""
        result = run_cli_subprocess(
            ["--output-dir", "/tmp/test"],  # Missing RSS URL
            cwd=project_root,
        )

        # Should fail gracefully
        assert result.returncode != 0, (
            f"CLI should fail with missing required args. "
            f"returncode: {result.returncode}, "
            f"stderr: {result.stderr}"
        )
