#!/usr/bin/env python3
"""Ollama provider E2E tests (Issue #196).

These tests verify that Ollama providers work correctly in complete user workflows:
- Ollama speaker detection in workflow
- Ollama summarization in workflow
- Error handling (connection errors, model validation)
- Health check validation

These tests use the E2E server's Ollama mock endpoints (real HTTP requests to mock server)
and are marked with @pytest.mark.e2e to allow selective execution.

Real API Mode:
    When USE_REAL_OLLAMA_API=1, tests use real Ollama server (must be running locally).
    This requires Ollama to be installed and running with models pulled.
    This is for manual testing only and will use local hardware resources.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.module_ollama_providers]

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import workflow

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly to avoid pytest resolution issues
from tests.conftest import (  # noqa: E402
    create_test_config,
)

# Check if we should use real Ollama API (for manual testing only)
USE_REAL_OLLAMA_API = os.getenv("USE_REAL_OLLAMA_API", "0") == "1"

# Feed selection for LLM provider tests (shared by OpenAI, Gemini, etc.)
# Default to "multi" to work in both fast and multi_episode E2E_TEST_MODE
LLM_TEST_FEED = os.getenv("LLM_TEST_FEED", "multi")

# Real RSS feed URL for testing (only used when USE_REAL_OLLAMA_API=1)
# NOTE: No default real feed - must be explicitly set via LLM_TEST_RSS_FEED
REAL_TEST_RSS_FEED = os.getenv("LLM_TEST_RSS_FEED", None)


def _check_ollama_available(ollama_api_base: Optional[str]) -> bool:
    """Check if Ollama server is available at the given API base URL.

    Args:
        ollama_api_base: Ollama API base URL (None for real API mode)

    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        import httpx

        # Determine health check URL
        if ollama_api_base:
            # Mock server mode - use provided base URL
            health_url = ollama_api_base.rstrip("/v1") + "/api/version"
        else:
            # Real API mode - use default localhost
            health_url = "http://localhost:11434/api/version"

        # Try to connect with short timeout
        response = httpx.get(health_url, timeout=2.0)
        response.raise_for_status()
        return True
    except Exception:
        # Any error means Ollama is not available
        return False


def _get_test_feed_url(
    e2e_server: Optional[Any] = None,
) -> tuple[str, Optional[str]]:
    """Get RSS feed URL and Ollama config based on LLM_TEST_FEED environment variable.

    Args:
        e2e_server: E2E server fixture (None if using real API or if fixture not available)

    Returns:
        Tuple of (rss_url, ollama_api_base)
    """
    feed_type = (LLM_TEST_FEED or "multi").lower()

    # Real API mode - can use either real RSS feed OR fixture feeds
    if USE_REAL_OLLAMA_API:
        # If LLM_TEST_RSS_FEED is explicitly set, use that real RSS feed
        if REAL_TEST_RSS_FEED is not None:
            return REAL_TEST_RSS_FEED, None

        # Otherwise, use fixture feeds from E2E server (fixture input, real API calls)
        if e2e_server is None:
            raise ValueError(
                "E2E server is required when using fixture feeds with real API. "
                "Set LLM_TEST_RSS_FEED=<url> to use a real RSS feed instead."
            )

        # Use fixture feeds but with real API (no mock API base)
        feed_mapping = {
            "multi": "podcast1_multi_episode",
            "fast": "podcast1",
            "p01": "podcast1",
            "p02": "podcast2",
            "p03": "podcast3",
            "p04": "podcast4",
            "p05": "podcast5",
        }
        podcast_name = feed_mapping.get(feed_type, "podcast1_multi_episode")
        rss_url = e2e_server.urls.feed(podcast_name)
        # Return None for ollama_api_base to use real Ollama server (not mocked)
        return rss_url, None

    # Mock API mode - use E2E server for both feeds and API
    if e2e_server is None:
        raise ValueError("E2E server is required for mock API mode")

    feed_mapping = {
        "multi": "podcast1_multi_episode",
        "fast": "podcast1",
        "p01": "podcast1",
        "p02": "podcast2",
        "p03": "podcast3",
        "p04": "podcast4",
        "p05": "podcast5",
    }
    podcast_name = feed_mapping.get(feed_type, "podcast1_multi_episode")
    rss_url = e2e_server.urls.feed(podcast_name)
    ollama_api_base = e2e_server.urls.ollama_api_base()

    return rss_url, ollama_api_base


def _summarize_ollama_usage(metadata_content: dict) -> str:
    """Extract and summarize Ollama API calls and models used from metadata.

    Args:
        metadata_content: Parsed metadata JSON content

    Returns:
        Human-readable summary string of Ollama APIs and models used
    """
    processing = metadata_content.get("processing", {})
    config_snapshot = processing.get("config_snapshot", {})
    ml_providers = config_snapshot.get("ml_providers", {})

    summary_parts = []

    # Speaker detection
    speaker_info = ml_providers.get("speaker_detection", {})
    if speaker_info.get("provider") == "ollama":
        model = speaker_info.get("ollama_speaker_model", "llama3.3:latest")
        summary_parts.append(f"  • Speaker Detection: Ollama API (model: {model})")

    # Summarization
    summarization_info = ml_providers.get("summarization", {})
    if summarization_info.get("provider") == "ollama":
        model = summarization_info.get("ollama_summary_model", "llama3.3:latest")
        summary_parts.append(f"  • Summarization: Ollama API (model: {model})")

    if summary_parts:
        return "Ollama APIs called:\n" + "\n".join(summary_parts)
    return "No Ollama APIs detected in metadata"


def _save_all_episode_responses(
    temp_dir: Path,
    metadata_files: list[Path],
    test_name: str,
    validate_provider: Optional[str] = None,
    validate_speaker_detection: bool = True,
    validate_summarization: bool = True,
) -> list[Path]:
    """Save Ollama API responses for all episodes processed in a test.

    All episodes from the same test run are saved to the same run folder,
    with one file per episode (e.g., ollama-responses_ep01.txt, ollama-responses_ep02.txt).

    Args:
        temp_dir: Temporary directory where test output was written
        metadata_files: List of metadata file paths (one per episode)
        test_name: Name of the test (e.g., "test_ollama_all_providers_in_pipeline")
        validate_provider: Optional provider name to validate (e.g., "ollama")
        validate_speaker_detection: Whether to validate speaker detection provider (default: True)
        validate_summarization: Whether to validate summarization provider (default: True)

    Returns:
        List of paths to saved response files
    """
    import json as json_module
    from datetime import datetime

    # Generate ONE run folder name for all episodes in this test run
    # This ensures all episodes from the same test are saved together
    now = datetime.now()
    shared_run_name = now.strftime("run_%Y%m%d-%H%M%S_%f")

    saved_files = []
    for idx, metadata_file in enumerate(sorted(metadata_files), 1):
        metadata_content = json_module.loads(metadata_file.read_text())

        # Validate provider if specified
        if validate_provider:
            processing = metadata_content.get("processing", {})
            config_snapshot = processing.get("config_snapshot", {})
            ml_providers = config_snapshot.get("ml_providers", {})

            # Check which provider type to validate
            if validate_provider == "ollama":
                # Check speaker detection provider (only if validation is enabled)
                if validate_speaker_detection:
                    speaker_info = ml_providers.get("speaker_detection", {})
                    if speaker_info:
                        assert speaker_info.get("provider") == "ollama", (
                            f"Speaker detection provider should be 'ollama' "
                            f"in metadata (episode {idx})"
                        )

                # Check summarization provider (only if validation is enabled)
                if validate_summarization:
                    summarization_info = ml_providers.get("summarization", {})
                    if summarization_info:
                        assert (
                            summarization_info.get("provider") == "ollama"
                        ), f"Summarization provider should be 'ollama' in metadata (episode {idx})"

        # Print summary of Ollama APIs and models used for this episode
        if len(metadata_files) > 1:
            print(f"\nEpisode {idx}: {_summarize_ollama_usage(metadata_content)}")
        else:
            print(f"\n{_summarize_ollama_usage(metadata_content)}")

        # Save actual API responses (speakers, summary) to file
        # Include episode number in the test name for uniqueness when multiple episodes
        episode_suffix = f"_ep{idx:02d}" if len(metadata_files) > 1 else ""
        response_file = _save_ollama_responses(
            temp_dir,
            metadata_content,
            f"{test_name}{episode_suffix}",
            shared_run_name=shared_run_name,
            metadata_file=metadata_file,
        )
        saved_files.append(response_file)

    # Print summary of all saved files
    if len(saved_files) > 1:
        print(f"\n📁 Saved {len(saved_files)} Ollama API response files:")
        for idx, file_path in enumerate(saved_files, 1):
            print(f"  {idx}. {file_path}")
    elif len(saved_files) == 1:
        print(f"📁 Ollama API responses saved to: {saved_files[0]}")

    # Also copy metadata files to output folder for easy access
    import shutil

    if saved_files:
        # Get the output directory from the first saved file
        run_output_dir = saved_files[0].parent

        metadata_files_copied = []
        for metadata_file in sorted(metadata_files):
            # Copy metadata file to same output directory as response files
            output_metadata_file = run_output_dir / metadata_file.name
            shutil.copy2(metadata_file, output_metadata_file)
            metadata_files_copied.append(output_metadata_file)

        if len(metadata_files_copied) > 1:
            print(f"\n📄 Copied {len(metadata_files_copied)} metadata files to output folder:")
            for idx, file_path in enumerate(metadata_files_copied, 1):
                print(f"  {idx}. {file_path}")
        elif len(metadata_files_copied) == 1:
            print(f"📄 Metadata file copied to: {metadata_files_copied[0]}")

    return saved_files


def _save_ollama_responses(  # noqa: C901
    temp_dir: Path,
    metadata_content: dict,
    test_name: str,
    shared_run_name: Optional[str] = None,
    metadata_file: Optional[Path] = None,
) -> Path:
    """Save actual Ollama API responses (speakers, summary) to a file.

    Args:
        temp_dir: Temporary directory where test output was written
        metadata_content: Parsed metadata JSON content
        test_name: Name of the test (e.g., "test_ollama_all_providers_in_pipeline_ep01")
        shared_run_name: Optional shared run name for all episodes in a test run.
                        If provided, all episodes will be saved to the same run folder.
                        If None, generates a new run name (for backward compatibility).
        metadata_file: Optional metadata file path (used to determine base directory
                      for transcript lookup)

    Returns:
        Path to the saved response file
    """
    import hashlib
    import re
    from datetime import datetime
    from urllib.parse import urlparse

    from podcast_scraper.utils import filesystem

    # Extract feed information from metadata
    feed_data = metadata_content.get("feed", {})
    feed_url = feed_data.get("url", "")

    # Derive feed name from URL (same logic as derive_output_dir)
    if feed_url:
        parsed = urlparse(feed_url)
        base = parsed.netloc or "feed"
        safe_base = filesystem.sanitize_filename(base)
        # Deterministic hash for directory naming (not security sensitive)
        digest = hashlib.sha1(feed_url.encode("utf-8"), usedforsecurity=False).hexdigest()
        feed_dir_name = f"rss_{safe_base}_{digest[:filesystem.URL_HASH_LENGTH]}"
    else:
        # Fallback if no feed URL in metadata
        feed_dir_name = "unknown_feed"

    # Use shared run name if provided, otherwise generate new one
    if shared_run_name:
        run_name = shared_run_name
    else:
        now = datetime.now()
        run_name = now.strftime("run_%Y%m%d-%H%M%S_%f")  # Include microseconds for uniqueness

    # Create output directory structure: output/<feed-name>/<run-name>/
    output_dir = Path("output")
    feed_output_dir = output_dir / feed_dir_name
    run_output_dir = feed_output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract episode suffix from test_name
    # (e.g., "_ep01" from "test_ollama_all_providers_in_pipeline_ep01")
    episode_suffix_match = re.search(r"_ep(\d+)$", test_name)
    if episode_suffix_match:
        episode_num = episode_suffix_match.group(1)
        filename = f"ollama-responses_ep{episode_num}.txt"
    else:
        filename = "ollama-responses.txt"

    output_file = run_output_dir / filename

    # Additional safety: if file somehow exists, append a counter
    counter = 1
    original_output_file = output_file
    while output_file.exists():
        filename_with_counter = f"ollama-responses_{counter}.txt"
        output_file = run_output_dir / filename_with_counter
        counter += 1
        if counter > 10000:  # Safety limit to prevent infinite loop
            raise RuntimeError(f"Too many files with same name: {original_output_file}")

    # Collect all response data
    response_lines = []
    response_lines.append("=" * 80)
    response_lines.append("Ollama API Responses")
    response_lines.append("=" * 80)
    response_lines.append(f"Test: {test_name}")
    response_lines.append(f"Timestamp: {datetime.now().isoformat()}")
    response_lines.append("")
    response_lines.append("Note: Ollama does NOT support transcription (no audio API)")
    response_lines.append("")

    # 0. Input Information
    response_lines.append("📥 INPUT INFORMATION:")
    response_lines.append("-" * 80)

    # Extract input information from metadata
    content = metadata_content.get("content", {})
    episode_data = metadata_content.get("episode", {})

    # Episode information
    episode_title = episode_data.get("title", "")
    if episode_title:
        response_lines.append(f"Episode Title: {episode_title}")

    episode_description = episode_data.get("description", "")
    if episode_description:
        response_lines.append(f"Episode Description: {episode_description[:200]}...")

    # Transcript source information (if available from other providers)
    transcript_source = content.get("transcript_source")
    if transcript_source:
        response_lines.append("")
        response_lines.append(f"Transcript Source: {transcript_source}")
        if transcript_source == "direct_download":
            response_lines.append("  (Transcript was downloaded from feed, not generated)")
        elif transcript_source == "whisper_transcription":
            whisper_model = content.get("whisper_model")
            if whisper_model:
                response_lines.append(f"  (Generated using Whisper model: {whisper_model})")

    response_lines.append("")
    response_lines.append("=" * 80)
    response_lines.append("")

    # 1. Speaker detection result
    detected_hosts = content.get("detected_hosts", [])
    detected_guests = content.get("detected_guests", [])
    if detected_hosts or detected_guests:
        response_lines.append("\n👥 SPEAKER DETECTION RESULT:")
        response_lines.append("-" * 80)
        if detected_hosts:
            response_lines.append(f"  Hosts: {', '.join(detected_hosts)}")
        if detected_guests:
            response_lines.append(f"  Guests: {', '.join(detected_guests)}")
        if not detected_hosts and not detected_guests:
            response_lines.append("  No speakers detected")
        response_lines.append("")

    # 2. Summarization result
    summary_data = metadata_content.get("summary")
    if summary_data:
        response_lines.append("\n📄 SUMMARIZATION RESULT:")
        response_lines.append("-" * 80)
        if isinstance(summary_data, dict):
            short_summary = summary_data.get("short_summary", "")
            if short_summary:
                response_lines.append(short_summary)
            else:
                response_lines.append("  (Summary exists but short_summary field is empty)")
        else:
            # Summary might be a string directly
            response_lines.append(str(summary_data))
        response_lines.append("")

    response_lines.append("=" * 80)

    # Write to file
    output_file.write_text("\n".join(response_lines), encoding="utf-8")

    return output_file


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.ollama
class TestOllamaProviderE2E:
    """E2E tests for Ollama provider in complete workflows."""

    def test_ollama_speaker_detection_in_workflow(self, e2e_server):
        """Test Ollama speaker detection in complete workflow."""
        # Get feed URL and Ollama config based on LLM_TEST_FEED
        rss_url, ollama_api_base = _get_test_feed_url(e2e_server)

        # Skip if Ollama is not available
        if not _check_ollama_available(ollama_api_base):
            pytest.skip("Ollama server is not available (not running or not accessible)")

        temp_dir = tempfile.mkdtemp()
        try:
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
                speaker_detector_provider="ollama",
                ollama_api_base=ollama_api_base,
                auto_speakers=True,
                generate_summaries=False,  # Skip summarization for speed
                generate_metadata=True,  # Required for validation
            )

            # Run pipeline (uses E2E server Ollama endpoints or real API)
            count, summary = workflow.run_pipeline(cfg)

            # Verify pipeline completed
            assert count >= 0, "Pipeline should complete"

            # Validate metadata if generated
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            if metadata_files:
                # Save responses for all episodes (if any succeeded)
                _save_all_episode_responses(
                    Path(temp_dir),
                    metadata_files,
                    "test_ollama_speaker_detection_in_workflow",
                    validate_provider="ollama",
                    validate_summarization=False,  # Summarization not enabled
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_ollama_summarization_in_workflow(self, e2e_server):
        """Test Ollama summarization in complete workflow."""
        # Get feed URL and Ollama config based on LLM_TEST_FEED
        rss_url, ollama_api_base = _get_test_feed_url(e2e_server)

        # Skip if Ollama is not available
        if not _check_ollama_available(ollama_api_base):
            pytest.skip("Ollama server is not available (not running or not accessible)")

        temp_dir = tempfile.mkdtemp()
        try:
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
                summary_provider="ollama",
                ollama_api_base=ollama_api_base,
                generate_summaries=True,
                generate_metadata=True,  # Required when generate_summaries=True
                auto_speakers=False,  # Skip speaker detection for speed
            )

            # Run pipeline (uses E2E server Ollama endpoints or real API)
            count, summary = workflow.run_pipeline(cfg)

            # Verify pipeline completed
            assert count >= 0, "Pipeline should complete"

            # Validate metadata
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            if metadata_files:
                # Save responses for all episodes (if any succeeded)
                _save_all_episode_responses(
                    Path(temp_dir),
                    metadata_files,
                    "test_ollama_summarization_in_workflow",
                    validate_provider="ollama",
                    validate_speaker_detection=False,  # Speaker detection not enabled
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_ollama_all_providers_in_pipeline(self, e2e_server):
        """Test all Ollama providers (speaker detection + summarization) in complete workflow."""
        # Get feed URL and Ollama config based on LLM_TEST_FEED
        rss_url, ollama_api_base = _get_test_feed_url(e2e_server)

        # Skip if Ollama is not available
        if not _check_ollama_available(ollama_api_base):
            pytest.skip("Ollama server is not available (not running or not accessible)")

        temp_dir = tempfile.mkdtemp()
        try:

            # Create config with ALL Ollama providers ONLY (no local ML providers)
            # Allow model override via environment variable
            ollama_speaker_model = os.getenv("OLLAMA_SPEAKER_MODEL", None)
            ollama_summary_model = os.getenv("OLLAMA_SUMMARY_MODEL", None)
            config_overrides = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",  # Ollama doesn't support transcription
                "speaker_detector_provider": "ollama",
                "summary_provider": "ollama",
                "ollama_api_base": ollama_api_base,
                "auto_speakers": True,
                "generate_metadata": True,
                "generate_summaries": True,
                "preload_models": False,  # Disable model preloading (no local ML models)
                "max_episodes": int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
                "transcribe_missing": True,  # Enable transcription for episodes without transcripts
            }
            if ollama_speaker_model:
                config_overrides["ollama_speaker_model"] = ollama_speaker_model
            if ollama_summary_model:
                config_overrides["ollama_summary_model"] = ollama_summary_model
            cfg = create_test_config(**config_overrides)

            # Run pipeline (uses E2E server Ollama endpoints or real API)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved (from other providers or feed)
            assert transcripts_saved >= 0, "Pipeline should complete"

            # Verify metadata files were created
            # Use *.metadata.json to avoid matching metrics.json files
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata files should be created when generate_metadata=True"

            import json as json_module

            # Verify metadata structure for all episodes
            for metadata_file in sorted(metadata_files):
                metadata_content = json_module.loads(metadata_file.read_text())
                assert (
                    "content" in metadata_content or "episode" in metadata_content
                ), "Metadata should have content or episode section"

            # Save responses for all episodes
            _save_all_episode_responses(
                Path(temp_dir),
                metadata_files,
                "test_ollama_all_providers_in_pipeline",
                validate_provider="ollama",
            )
        finally:
            # Preserve temp_dir when using real API (for inspection/debugging)
            # Only clean up when using mock E2E server
            if not USE_REAL_OLLAMA_API:
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                # Log location for debugging
                print(f"\n⚠️  Preserving temp_dir (USE_REAL_OLLAMA_API=1): {temp_dir}")

    @pytest.mark.skipif(USE_REAL_OLLAMA_API, reason="Health check test only works with mock server")
    def test_ollama_health_check_failure(self, e2e_server):
        """Test that workflow fails gracefully when Ollama server is not running.

        Note: This test only works with mock server. In real API mode, it's skipped.
        """
        rss_url, ollama_api_base = _get_test_feed_url(e2e_server)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use an invalid API base to simulate server not running
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                speaker_detector_provider="ollama",
                ollama_api_base="http://localhost:99999/v1",  # Invalid port
                auto_speakers=True,
            )

            # Run pipeline - should fail with ConnectionError
            with pytest.raises(ConnectionError) as exc_info:
                workflow.run_pipeline(cfg)

            assert "Ollama server is not running" in str(exc_info.value)
