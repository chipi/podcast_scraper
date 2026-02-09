#!/usr/bin/env python3
"""Mistral provider E2E tests (Issue #106).

These tests verify that Mistral providers work correctly in complete user workflows:
- Mistral transcription in workflow
- Mistral speaker detection in workflow
- Mistral summarization in workflow
- Error handling (API errors, rate limiting, retries)

These tests use the E2E server's Mistral mock endpoints (real HTTP requests to mock server)
and are marked with @pytest.mark.e2e to allow selective execution.

Real API Mode:
    When USE_REAL_MISTRAL_API=1, tests use real Mistral API endpoints and real RSS feeds.
    This is for manual testing only and will incur API costs.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.module_mistral_providers]

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

# Check if we should use real Mistral API (for manual testing only)
USE_REAL_MISTRAL_API = os.getenv("USE_REAL_MISTRAL_API", "0") == "1"

# Feed selection for LLM provider tests (shared by OpenAI, Gemini, Mistral, etc.)
# Default to "multi" to work in both fast and multi_episode E2E_TEST_MODE
LLM_TEST_FEED = os.getenv("LLM_TEST_FEED", "multi")

# Real RSS feed URL for testing (only used when USE_REAL_MISTRAL_API=1)
# NOTE: No default real feed - must be explicitly set via LLM_TEST_RSS_FEED
REAL_TEST_RSS_FEED = os.getenv("LLM_TEST_RSS_FEED", None)


def _get_test_feed_url(
    e2e_server: Optional[Any] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Get RSS feed URL and Mistral config based on LLM_TEST_FEED environment variable.

    Args:
        e2e_server: E2E server fixture (None if using real API or if fixture not available)

    Returns:
        Tuple of (rss_url, mistral_api_base, mistral_api_key)
    """
    feed_type = (LLM_TEST_FEED or "multi").lower()

    # Real API mode - can use either real RSS feed OR fixture feeds
    if USE_REAL_MISTRAL_API:
        # If LLM_TEST_RSS_FEED is explicitly set, use that real RSS feed
        if REAL_TEST_RSS_FEED is not None:
            return REAL_TEST_RSS_FEED, None, None

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
        return rss_url, None, None

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
    mistral_api_base = e2e_server.urls.mistral_api_base()
    mistral_api_key = "test-dummy-key-for-e2e-tests"

    return rss_url, mistral_api_base, mistral_api_key


def _summarize_mistral_usage(metadata_content: dict) -> str:
    """Extract and summarize Mistral API calls and models used from metadata.

    Args:
        metadata_content: Parsed metadata JSON content

    Returns:
        Human-readable summary string of Mistral APIs and models used
    """
    processing = metadata_content.get("processing", {})
    config_snapshot = processing.get("config_snapshot", {})
    ml_providers = config_snapshot.get("ml_providers", {})

    summary_parts = []

    # Transcription
    transcription_info = ml_providers.get("transcription", {})
    if transcription_info.get("provider") == "mistral":
        model = transcription_info.get("mistral_model", "voxtral-mini-latest")
        summary_parts.append(f"  • Transcription: Mistral Voxtral API (model: {model})")

    # Speaker detection
    speaker_info = ml_providers.get("speaker_detection", {})
    if speaker_info.get("provider") == "mistral":
        model = speaker_info.get("mistral_model", "mistral-small-latest")
        summary_parts.append(f"  • Speaker Detection: Mistral Chat API (model: {model})")

    # Summarization
    summarization_info = ml_providers.get("summarization", {})
    if summarization_info.get("provider") == "mistral":
        model = summarization_info.get("mistral_model", "mistral-small-latest")
        summary_parts.append(f"  • Summarization: Mistral Chat API (model: {model})")

    if summary_parts:
        return "Mistral APIs called:\n" + "\n".join(summary_parts)
    return "No Mistral APIs detected in metadata"


def _save_all_episode_responses(
    temp_dir: Path,
    metadata_files: list[Path],
    test_name: str,
    validate_provider: Optional[str] = None,
    validate_transcription: bool = True,
    validate_speaker_detection: bool = True,
    validate_summarization: bool = True,
) -> list[Path]:
    """Save Mistral API responses for all episodes processed in a test.

    All episodes from the same test run are saved to the same run folder,
    with one file per episode (e.g., mistral-responses_ep01.txt, mistral-responses_ep02.txt).

    Args:
        temp_dir: Temporary directory where test output was written
        metadata_files: List of metadata file paths (one per episode)
        test_name: Name of the test (e.g., "test_mistral_full_pipeline")
        validate_provider: Optional provider name to validate (e.g., "mistral")
        validate_transcription: Whether to validate transcription provider (default: True)
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

        # Validate provider if requested
        if validate_provider:
            processing = metadata_content.get("processing", {})
            config_snapshot = processing.get("config_snapshot", {})
            ml_providers = config_snapshot.get("ml_providers", {})

            if validate_transcription:
                transcription_info = ml_providers.get("transcription", {})
                assert transcription_info.get("provider") == validate_provider, (
                    f"Transcription provider should be '{validate_provider}' "
                    f"in metadata (episode {idx})"
                )

            if validate_speaker_detection:
                speaker_info = ml_providers.get("speaker_detection", {})
                assert speaker_info.get("provider") == validate_provider, (
                    f"Speaker detection provider should be '{validate_provider}' "
                    f"in metadata (episode {idx})"
                )

            if validate_summarization:
                summarization_info = ml_providers.get("summarization", {})
                assert summarization_info.get("provider") == validate_provider, (
                    f"Summarization provider should be '{validate_provider}' "
                    f"in metadata (episode {idx})"
                )

        # Print summary of Mistral APIs and models used for this episode
        if len(metadata_files) > 1:
            print(f"\nEpisode {idx}: {_summarize_mistral_usage(metadata_content)}")
        else:
            print(f"\n{_summarize_mistral_usage(metadata_content)}")

        # Save actual API responses (transcript, speakers, summary) to file
        # Include episode number in the test name for uniqueness when multiple episodes
        episode_suffix = f"_ep{idx:02d}" if len(metadata_files) > 1 else ""
        response_file = _save_mistral_responses(
            temp_dir,
            metadata_content,
            f"{test_name}{episode_suffix}",
            shared_run_name=shared_run_name,
            metadata_file=metadata_file,
        )
        saved_files.append(response_file)

    # Print summary of all saved files
    if len(saved_files) > 1:
        print(f"\n📁 Saved {len(saved_files)} Mistral API response files:")
        for idx, file_path in enumerate(saved_files, 1):
            print(f"  {idx}. {file_path}")
    elif len(saved_files) == 1:
        print(f"📁 Mistral API responses saved to: {saved_files[0]}")

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


def _save_mistral_responses(  # noqa: C901
    temp_dir: Path,
    metadata_content: dict,
    test_name: str,
    shared_run_name: Optional[str] = None,
    metadata_file: Optional[Path] = None,
) -> Path:
    """Save actual Mistral API responses (transcript, speakers, summary) to a file.

    Args:
        temp_dir: Temporary directory where test output was written
        metadata_content: Parsed metadata JSON content
        test_name: Name of the test (e.g., "test_mistral_full_pipeline_ep01")
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

    # Create output directory structure: .test_outputs/e2e/<feed-name>/<run-name>/
    output_dir = Path(".test_outputs/e2e")
    feed_output_dir = output_dir / feed_dir_name
    run_output_dir = feed_output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract episode suffix from test_name (e.g., "_ep01" from "test_mistral_full_pipeline_ep01")
    episode_suffix_match = re.search(r"_ep(\d+)$", test_name)
    if episode_suffix_match:
        episode_num = episode_suffix_match.group(1)
        filename = f"mistral-responses_ep{episode_num}.txt"
    else:
        filename = "mistral-responses.txt"

    output_file = run_output_dir / filename

    # Additional safety: if file somehow exists, append a counter
    counter = 1
    original_output_file = output_file
    while output_file.exists():
        filename_with_counter = f"mistral-responses_{counter}.txt"
        output_file = run_output_dir / filename_with_counter
        counter += 1
        if counter > 10000:  # Safety limit to prevent infinite loop
            raise RuntimeError(f"Too many files with same name: {original_output_file}")

    # Collect all response data
    response_lines = []
    response_lines.append("=" * 80)
    response_lines.append("Mistral API Responses")
    response_lines.append("=" * 80)
    response_lines.append(f"Test: {test_name}")
    response_lines.append(f"Timestamp: {datetime.now().isoformat()}")
    response_lines.append("")

    # 0. Input Information
    response_lines.append("📥 INPUT INFORMATION:")
    response_lines.append("-" * 80)

    # Extract input information from metadata
    content = metadata_content.get("content", {})
    episode_data = metadata_content.get("episode", {})

    # Media file information
    media_url = content.get("media_url") or episode_data.get("media_url")
    if media_url:
        response_lines.append(f"Media URL: {media_url}")
        # Extract filename from URL
        try:
            from urllib.parse import unquote, urlparse

            parsed_url = urlparse(media_url)
            filename = unquote(parsed_url.path.split("/")[-1])
            if filename:
                response_lines.append(f"Media Filename: {filename}")
        except Exception:
            pass

    # Transcript URLs (if available from feed)
    transcript_infos = content.get("transcript_urls", [])
    if transcript_infos:
        response_lines.append("")
        response_lines.append("Available Transcript URLs (from feed):")
        for idx, transcript_info in enumerate(transcript_infos, 1):
            transcript_url = (
                transcript_info.get("url", "")
                if isinstance(transcript_info, dict)
                else str(transcript_info)
            )
            transcript_type = (
                transcript_info.get("type", "unknown")
                if isinstance(transcript_info, dict)
                else "unknown"
            )
            response_lines.append(f"  {idx}. {transcript_url} (type: {transcript_type})")
    else:
        response_lines.append("")
        response_lines.append("Available Transcript URLs (from feed): None")

    # Transcript source information
    transcript_source = content.get("transcript_source")
    if transcript_source:
        response_lines.append("")
        response_lines.append(f"Transcript Source: {transcript_source}")
        if transcript_source == "direct_download":
            response_lines.append("  (Transcript was downloaded from feed, not generated)")
        elif transcript_source == "mistral_transcription":
            mistral_model = content.get("mistral_model")
            if mistral_model:
                response_lines.append(f"  (Generated using Mistral Voxtral model: {mistral_model})")

    response_lines.append("")
    response_lines.append("=" * 80)
    response_lines.append("")

    # 1. Transcription result - read from transcript file (like OpenAI/Gemini)
    # Use transcript_file_path from metadata (most reliable)
    transcript_file = None
    content = metadata_content.get("content", {})
    transcript_file_path = content.get("transcript_file_path")

    # Determine base directory for transcript lookup
    # If metadata_file is provided, use its parent directory (handles run subdirectories)
    # Otherwise, use temp_dir directly
    if metadata_file:
        # metadata_file is like temp_dir/run_XXX/metadata/episode.metadata.json
        # Base dir should be temp_dir/run_XXX/ (parent of metadata/)
        base_dir = metadata_file.parent.parent
    else:
        base_dir = temp_dir
    if transcript_file_path:
        # transcript_file_path is relative to output_dir
        # Try direct path first
        transcript_file = base_dir / transcript_file_path
        if not transcript_file.exists():
            # Try transcripts/ subdirectory if path doesn't include it
            transcript_file = base_dir / "transcripts" / Path(transcript_file_path).name
            if not transcript_file.exists():
                transcript_file = None  # Reset to None so fallback can run

    # Fallback: try to find transcript file by episode index and title
    if transcript_file is None:
        episode_data = metadata_content.get("episode", {})
        episode_idx = episode_data.get("idx")
        episode_title = episode_data.get("title", "")

        if episode_idx is not None:
            # Transcript files are named like "0001 - Episode Title.txt"
            # They're in transcripts/ subdirectory
            transcripts_dir = base_dir / "transcripts"
            if transcripts_dir.exists():
                # Look in transcripts/ subdirectory first
                all_transcript_files = list(transcripts_dir.glob("*.txt"))
            else:
                # Fallback: search recursively
                all_transcript_files = list(base_dir.rglob("*.txt"))

            # Filter out cleaned transcripts and metadata files
            candidate_files = [
                f
                for f in all_transcript_files
                if "cleaned" not in f.name
                and "metadata" not in f.name
                and "response" not in f.name.lower()
                and "mistral" not in f.name.lower()
            ]

            # Try to match by episode index (format: "0001 - Title.txt")
            idx_str = f"{episode_idx:04d}"
            for candidate in candidate_files:
                if candidate.name.startswith(idx_str):
                    transcript_file = candidate
                    break

            # If not found by index, try to match by title (safe version)
            if transcript_file is None and episode_title:
                # Create a safe version of the title (similar to how filesystem does it)
                import re

                title_safe = re.sub(r"[^\w\s-]", "", episode_title).strip()
                title_safe = re.sub(r"[-\s]+", "_", title_safe)
                for candidate in candidate_files:
                    if title_safe.lower() in candidate.name.lower():
                        transcript_file = candidate
                        break

        # Fallback: if we couldn't match by episode, use first available transcript
        if transcript_file is None:
            # Look in transcripts/ subdirectory first
            transcripts_dir = base_dir / "transcripts"
            if transcripts_dir.exists():
                all_transcript_files = list(transcripts_dir.glob("*.txt"))
            else:
                all_transcript_files = list(base_dir.rglob("*.txt"))
            candidate_files = [
                f
                for f in all_transcript_files
                if "cleaned" not in f.name
                and "metadata" not in f.name
                and "response" not in f.name.lower()
                and "mistral" not in f.name.lower()
            ]
            if candidate_files:
                transcript_file = candidate_files[0]

    if transcript_file and transcript_file.exists():
        transcript_text = transcript_file.read_text(encoding="utf-8")
        response_lines.append("\n📝 TRANSCRIPTION RESULT:")
        response_lines.append("-" * 80)
        response_lines.append(f"Source file: {transcript_file.name}")
        response_lines.append(f"Total length: {len(transcript_text)} characters")
        response_lines.append("")
        response_lines.append(transcript_text)
        response_lines.append("")
    else:
        response_lines.append("\n📝 TRANSCRIPTION RESULT:")
        response_lines.append("-" * 80)
        response_lines.append("(No transcript file found)")
        response_lines.append("")

    # 2. Speaker Detection result
    content = metadata_content.get("content", {})
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
    else:
        response_lines.append("\n👥 SPEAKER DETECTION RESULT:")
        response_lines.append("-" * 80)
        response_lines.append("  No speakers detected")
        response_lines.append("")

    # 3. Summarization result
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
    else:
        response_lines.append("\n📄 SUMMARIZATION RESULT:")
        response_lines.append("-" * 80)
        response_lines.append("  (No summary available)")
        response_lines.append("")

    # Write to file
    output_file.write_text("\n".join(response_lines), encoding="utf-8")
    return output_file


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.mistral
class TestMistralProviderE2E:
    """Test Mistral providers in integration workflows using E2E server."""

    def test_mistral_full_pipeline(self, e2e_server: Optional[Any]):
        """Test all Mistral providers together in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Mistral config based on LLM_TEST_FEED
            rss_url, mistral_api_base, mistral_api_key = _get_test_feed_url(e2e_server)

            # Get max_episodes from environment variable (for real feeds) or use default
            max_episodes = int(os.getenv("LLM_TEST_MAX_EPISODES", "1"))

            # Create config with ALL Mistral providers ONLY (no local ML providers)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="mistral",
                speaker_detector_provider="mistral",
                summary_provider="mistral",
                mistral_api_key=mistral_api_key,
                mistral_api_base=mistral_api_base,
                auto_speakers=True,
                generate_metadata=True,
                generate_summaries=True,
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=max_episodes,
            )

            # Run pipeline (uses E2E server Mistral endpoints or real API)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

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
                "test_mistral_full_pipeline",
                validate_provider="mistral",
            )
        finally:
            # Preserve temp_dir when using real API (for inspection/debugging)
            # Only clean up when using mock E2E server
            if not USE_REAL_MISTRAL_API:
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                print(f"\n🔍 Preserving temp_dir for inspection: {temp_dir}")

    def test_mistral_transcription_in_pipeline(self, e2e_server: Optional[Any]):
        """Test Mistral transcription provider in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Mistral config based on LLM_TEST_FEED
            rss_url, mistral_api_base, mistral_api_key = _get_test_feed_url(e2e_server)

            # Get max_episodes from environment variable (for real feeds) or use default
            max_episodes = int(os.getenv("LLM_TEST_MAX_EPISODES", "1"))

            # Create config with Mistral transcription ONLY (no other providers)
            # Always pass API key/base (even if None) so field validators can load from env
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="mistral",
                speaker_detector_provider="mistral",  # Explicitly set to avoid spaCy default
                summary_provider="mistral",  # Explicitly set to avoid transformers default
                mistral_api_key=mistral_api_key,  # None means load from env via field validator
                mistral_api_base=mistral_api_base,  # None means load from env via field validator
                transcribe_missing=True,  # Enable transcription
                generate_metadata=True,
                generate_summaries=True,  # Enable summarization
                auto_speakers=True,  # Enable speaker detection
                preload_models=False,  # Disable model preloading (no local ML models)
                max_episodes=max_episodes,
            )

            # Run pipeline (uses E2E server Mistral endpoints, no direct mocking)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify transcript files were created
            transcript_files = list(Path(temp_dir).rglob("*.txt")) + list(
                Path(temp_dir).rglob("*.vtt")
            )
            assert len(transcript_files) >= 1, "Should have created at least one transcript file"

            # Verify Mistral transcription provider was used (check metadata)
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            if len(metadata_files) > 0:
                import json as json_module

                # Generate ONE run folder name for all episodes in this test run
                from datetime import datetime

                now = datetime.now()
                shared_run_name = now.strftime("run_%Y%m%d-%H%M%S_%f")

                # Process all metadata files (one per episode)
                saved_files = []
                for idx, metadata_file in enumerate(sorted(metadata_files), 1):
                    metadata_content = json_module.loads(metadata_file.read_text())
                    processing = metadata_content.get("processing", {})
                    config_snapshot = processing.get("config_snapshot", {})
                    ml_providers = config_snapshot.get("ml_providers", {})
                    transcription_info = ml_providers.get("transcription", {})
                    assert (
                        transcription_info.get("provider") == "mistral"
                    ), f"Transcription provider should be 'mistral' in metadata (episode {idx})"

                    # Print summary of Mistral APIs and models used for this episode
                    print(f"\nEpisode {idx}: {_summarize_mistral_usage(metadata_content)}")

                    # Save actual API responses (transcript, speakers, summary) to file
                    # Include episode number in the test name for uniqueness
                    episode_suffix = f"_ep{idx:02d}" if len(metadata_files) > 1 else ""
                    response_file = _save_mistral_responses(
                        Path(temp_dir),
                        metadata_content,
                        f"test_mistral_transcription_in_pipeline{episode_suffix}",
                        shared_run_name=shared_run_name,
                        metadata_file=metadata_file,
                    )
                    saved_files.append(response_file)

                # Print summary of all saved files
                if len(saved_files) > 1:
                    print(f"\n📁 Saved {len(saved_files)} Mistral API response files:")
                    for idx, file_path in enumerate(saved_files, 1):
                        print(f"  {idx}. {file_path}")
                else:
                    print(f"📁 Mistral API responses saved to: {saved_files[0]}")
        finally:
            # Preserve temp_dir when using real API (for inspection/debugging)
            # Only clean up when using mock E2E server
            if not USE_REAL_MISTRAL_API:
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                # Log location for debugging
                print(f"\n⚠️  Preserving temp_dir (USE_REAL_MISTRAL_API=1): {temp_dir}")

    def test_mistral_speaker_detection_in_pipeline(self, e2e_server: Optional[Any]):
        """Test Mistral speaker detection provider in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Mistral config based on LLM_TEST_FEED
            rss_url, mistral_api_base, mistral_api_key = _get_test_feed_url(e2e_server)

            # Create config with Mistral speaker detection and transcription ONLY (no summarization)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="mistral",
                speaker_detector_provider="mistral",
                summary_provider="mistral",  # Explicitly set to avoid transformers default
                mistral_api_key=mistral_api_key,
                mistral_api_base=mistral_api_base,
                auto_speakers=True,
                generate_metadata=True,
                generate_summaries=False,  # Disable summarization
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses Mistral ONLY, no local ML providers)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved (transcript file must exist for metadata)
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            # Use *.metadata.json to avoid matching metrics.json files
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata files should be created when generate_metadata=True"

            import json as json_module

            # Verify speaker detection results for all episodes
            for metadata_file in sorted(metadata_files):
                metadata_content = json_module.loads(metadata_file.read_text())
                content = metadata_content.get("content", {})
                assert (
                    "detected_hosts" in content
                    or "detected_guests" in content
                    or "detected_hosts" in metadata_content
                    or "detected_guests" in metadata_content
                ), (
                    "Speaker detection results (detected_hosts/detected_guests) "
                    "should be in metadata"
                )

            # Save responses for all episodes
            _save_all_episode_responses(
                Path(temp_dir),
                metadata_files,
                "test_mistral_speaker_detection_in_pipeline",
                validate_provider="mistral",
                validate_summarization=False,  # Summarization not enabled in this test
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_mistral_summarization_in_pipeline(self, e2e_server: Optional[Any]):
        """Test Mistral summarization provider in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Mistral config based on LLM_TEST_FEED
            rss_url, mistral_api_base, mistral_api_key = _get_test_feed_url(e2e_server)

            # Create config with Mistral summarization and transcription ONLY (no speaker detection)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="mistral",
                speaker_detector_provider="mistral",  # Explicitly set to avoid spaCy default
                summary_provider="mistral",
                mistral_api_key=mistral_api_key,
                mistral_api_base=mistral_api_base,
                generate_metadata=True,
                generate_summaries=True,
                auto_speakers=False,  # Disable speaker detection
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses Mistral ONLY, no local ML providers)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved (transcript file must exist for summarization)
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            # Use *.metadata.json to avoid matching metrics.json files
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata files should be created when generate_metadata=True"

            import json as json_module

            # Verify all episodes have summaries
            for metadata_file in sorted(metadata_files):
                metadata_content = json_module.loads(metadata_file.read_text())
                assert (
                    "summary" in metadata_content
                ), "Summary should exist in metadata when generate_summaries=True"
                summary_data = metadata_content.get("summary", {})
                if isinstance(summary_data, dict):
                    short_summary = summary_data.get("short_summary", "")
                    assert len(short_summary) > 0, "Summary should not be empty"
                else:
                    assert len(str(summary_data)) > 0, "Summary should not be empty"

            # Save responses for all episodes
            _save_all_episode_responses(
                Path(temp_dir),
                metadata_files,
                "test_mistral_summarization_in_pipeline",
                validate_provider="mistral",
                validate_speaker_detection=False,  # Speaker detection not enabled in this test
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_mistral_hybrid_cleaning_strategy(self, e2e_server: Optional[Any]):
        """Test Mistral provider with hybrid cleaning strategy (pattern + conditional LLM).

        This test verifies that the hybrid cleaning strategy works correctly:
        - Pattern-based cleaning always runs first
        - LLM cleaning is conditionally applied when heuristics indicate it's needed
        - Provider's clean_transcript() method is called when needed
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and Mistral config based on LLM_TEST_FEED
            rss_url, mistral_api_base, mistral_api_key = _get_test_feed_url(e2e_server)

            # Create config with Mistral summarization and hybrid cleaning strategy
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="mistral",
                speaker_detector_provider="mistral",
                summary_provider="mistral",
                mistral_api_key=mistral_api_key,
                mistral_api_base=mistral_api_base,
                auto_speakers=False,  # Disable speaker detection to focus on cleaning
                generate_metadata=True,
                generate_summaries=True,
                preload_models=False,
                transcribe_missing=True,
                transcript_cleaning_strategy="hybrid",  # Explicitly set hybrid strategy
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Verify config has hybrid cleaning strategy
            assert (
                cfg.transcript_cleaning_strategy == "hybrid"
            ), "Config should have hybrid cleaning strategy"

            # Run pipeline
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) > 0
            ), "Metadata files should be created when generate_metadata=True"

            import json as json_module

            # Verify summaries were generated (cleaning is used before summarization)
            for metadata_file in sorted(metadata_files):
                metadata_content = json_module.loads(metadata_file.read_text())
                assert (
                    "summary" in metadata_content
                ), "Summary should exist in metadata when generate_summaries=True"

                # Verify summary is valid (if cleaning failed, summarization might fail)
                summary_data = metadata_content.get("summary", {})
                if isinstance(summary_data, dict):
                    assert (
                        "bullets" in summary_data
                    ), "Summary should have bullets field (normalized schema)"
                    assert len(summary_data["bullets"]) > 0, "bullets should not be empty"

            # Verify cleaned transcript files were created (if save_cleaned_transcript is enabled)
            # Note: save_cleaned_transcript defaults to False, so we check if any exist
            cleaned_files = list(Path(temp_dir).rglob("*.cleaned.txt"))
            # If cleaned files exist, verify they're different from original transcripts
            if cleaned_files:
                transcript_files = list(Path(temp_dir).rglob("*.txt"))
                # Filter out cleaned files from transcript files
                original_transcripts = [f for f in transcript_files if "cleaned" not in f.name]
                if original_transcripts and cleaned_files:
                    # Verify cleaned files are different (shorter or different content)
                    original_text = original_transcripts[0].read_text(encoding="utf-8")
                    cleaned_text = cleaned_files[0].read_text(encoding="utf-8")
                    # Cleaned text should be different (may be shorter due to cleaning)
                    assert (
                        cleaned_text != original_text
                    ), "Cleaned transcript should be different from original"

            # Save responses for inspection
            _save_all_episode_responses(
                Path(temp_dir),
                metadata_files,
                "test_mistral_hybrid_cleaning_strategy",
                validate_provider="mistral",
                validate_speaker_detection=False,  # Speaker detection not enabled
            )
        finally:
            # Preserve temp_dir when using real API (for inspection/debugging)
            # Only clean up when using mock E2E server
            if not USE_REAL_MISTRAL_API:
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                print(f"\n🔍 Preserving temp_dir for inspection: {temp_dir}")
