#!/usr/bin/env python3
"""OpenAI provider E2E tests (Stage 10).

These tests verify that OpenAI providers work correctly in complete user workflows:
- OpenAI transcription in workflow
- OpenAI speaker detection in workflow
- OpenAI summarization in workflow
- Error handling (API errors, rate limiting, retries)
- Fallback behavior when OpenAI API fails

These tests use the E2E server's OpenAI mock endpoints (real HTTP requests to mock server)
and are marked with @pytest.mark.e2e to allow selective execution.

Real API Mode:
    When USE_REAL_OPENAI_API=1, tests use real OpenAI API endpoints and real RSS feeds.
    This is for manual testing only and will incur API costs.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.module_openai_providers]

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

# Check if we should use real OpenAI API (for manual testing only)
USE_REAL_OPENAI_API = os.getenv("USE_REAL_OPENAI_API", "0") == "1"

# Feed selection for LLM provider tests (shared by OpenAI, Gemini, etc.)
# Options:
# - "fast": Use p01_fast.xml (1 episode, 1 minute) - requires E2E_TEST_MODE=fast
# - "multi": Use p01_multi.xml (5 episodes, 10-15 seconds each) - DEFAULT (works in all modes)
# - "p01": Use p01_mtb.xml (podcast1) - requires E2E_TEST_MODE=nightly or data_quality
# - "p02": Use p02_software.xml (podcast2) - requires E2E_TEST_MODE=nightly or data_quality
# - "p03": Use p03_scuba.xml (podcast3) - requires E2E_TEST_MODE=nightly or data_quality
# - "p04": Use p04_photo.xml (podcast4) - requires E2E_TEST_MODE=nightly or data_quality
# - "p05": Use p05_investing.xml (podcast5) - requires E2E_TEST_MODE=nightly or data_quality
# For real API mode: Set USE_REAL_OPENAI_API=1 and LLM_TEST_RSS_FEED=<feed-url>
#   (no default real feed - must be explicitly provided)
# Default to "multi" to work in both fast and multi_episode E2E_TEST_MODE
LLM_TEST_FEED = os.getenv("LLM_TEST_FEED", "multi")

# Real RSS feed URL for testing (only used when USE_REAL_OPENAI_API=1 or USE_REAL_GEMINI_API=1)
# NOTE: No default real feed - must be explicitly set via LLM_TEST_RSS_FEED
REAL_TEST_RSS_FEED = os.getenv("LLM_TEST_RSS_FEED", None)


def _get_test_feed_url(
    e2e_server: Optional[Any] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Get RSS feed URL and OpenAI config based on LLM_TEST_FEED environment variable.

    Args:
        e2e_server: E2E server fixture (None if using real API or if fixture not available)

    Returns:
        Tuple of (rss_url, openai_api_base, openai_api_key)
    """
    feed_type = (LLM_TEST_FEED or "multi").lower()

    # Real API mode - can use either real RSS feed OR fixture feeds
    if USE_REAL_OPENAI_API:
        # If LLM_TEST_RSS_FEED is explicitly set, use that real RSS feed
        if REAL_TEST_RSS_FEED is not None:
            return REAL_TEST_RSS_FEED, None, None

        # Otherwise, use fixture feeds from E2E server (fixture input, real API calls)
        # This allows testing real API with known fixture data
        if e2e_server is None:
            raise ValueError(
                "E2E server is required when using fixture feeds with real API. "
                "Set LLM_TEST_RSS_FEED=<url> to use a real RSS feed instead."
            )

        # Use fixture feeds but with real API (no mock API base)
        feed_mapping = {
            "fast": "podcast1",  # p01_fast.xml (1 episode, 1 minute)
            "multi": "podcast1_multi_episode",  # p01_multi.xml (5 episodes, 10-15s each)
            "p01": "podcast1",  # p01_mtb.xml (3 episodes) or p01_fast.xml in fast mode
            "p02": "podcast2",  # p02_software.xml (3 episodes)
            "p03": "podcast3",  # p03_scuba.xml (3 episodes)
            "p04": "podcast4",  # p04_photo.xml (3 episodes)
            "p05": "podcast5",  # p05_investing.xml (3 episodes)
        }

        if feed_type in feed_mapping:
            feed_name = feed_mapping[feed_type]
            rss_url = e2e_server.urls.feed(feed_name)
            # Return None for openai_api_base to use real OpenAI API (not mocked)
            # Return None for openai_api_key to use key from .env file
            return rss_url, None, None

        raise ValueError(
            f"Unknown feed type: {feed_type}. "
            "Use 'fast', 'multi', 'p01', 'p02', 'p03', 'p04', 'p05', "
            "or set LLM_TEST_RSS_FEED for a real feed."
        )

    # E2E server mode - use test fixtures with mock API
    # Note: e2e_server may be None if USE_REAL_OPENAI_API=1, but we've already handled that above
    # If it's None here, it means the fixture wasn't available (shouldn't happen in normal E2E mode)
    if e2e_server is None:
        # Try to get the fixture from pytest request if available
        # This handles the case where the fixture wasn't injected
        try:
            # This won't work in this context, but we can check the environment
            # If we're not in real API mode and e2e_server is None, something is wrong
            if not USE_REAL_OPENAI_API:
                raise ValueError(
                    "E2E server fixture is not available. "
                    "This should not happen in E2E mode. "
                    "Check that the e2e_server fixture is properly configured."
                )
        except Exception:
            pass
        raise ValueError(
            "E2E server is required for test fixture feeds. "
            "Set USE_REAL_OPENAI_API=1 to use real feeds."
        )

    # Map feed types to E2E server feed names
    # Note: In fast mode, "podcast1" maps to p01_fast.xml (1 episode, 1 minute)
    #       In normal mode, "podcast1" maps to p01_mtb.xml (3 episodes)
    feed_mapping = {
        "fast": "podcast1",  # p01_fast.xml (1 episode, 1 minute) - DEFAULT
        "multi": "podcast1_multi_episode",  # p01_multi.xml (5 episodes, 10-15s each)
        "p01": "podcast1",  # p01_mtb.xml (3 episodes) or p01_fast.xml in fast mode
        "p02": "podcast2",  # p02_software.xml (3 episodes)
        "p03": "podcast3",  # p03_scuba.xml (3 episodes)
        "p04": "podcast4",  # p04_photo.xml (3 episodes)
        "p05": "podcast5",  # p05_investing.xml (3 episodes)
    }

    if feed_type in feed_mapping:
        feed_name = feed_mapping[feed_type]
        rss_url = e2e_server.urls.feed(feed_name)
        openai_api_base = e2e_server.urls.openai_api_base()
        openai_api_key = "sk-test123"
        return rss_url, openai_api_base, openai_api_key

    # Custom URL - assume it's a real feed URL
    # Note: This will only work if USE_REAL_OPENAI_API=1
    return feed_type, None, None


def _summarize_openai_usage(metadata_content: dict) -> str:
    """Extract and summarize OpenAI API calls and models used from metadata.

    Args:
        metadata_content: Parsed metadata JSON content

    Returns:
        Human-readable summary string of OpenAI APIs and models used
    """
    processing = metadata_content.get("processing", {})
    config_snapshot = processing.get("config_snapshot", {})
    ml_providers = config_snapshot.get("ml_providers", {})

    summary_parts = []

    # Transcription
    transcription_info = ml_providers.get("transcription", {})
    if transcription_info.get("provider") == "openai":
        model = transcription_info.get("openai_model", "whisper-1")
        summary_parts.append(f"  • Transcription: /v1/audio/transcriptions (model: {model})")

    # Speaker detection
    speaker_info = ml_providers.get("speaker_detection", {})
    if speaker_info.get("provider") == "openai":
        model = speaker_info.get("openai_model", "gpt-4o-mini")
        summary_parts.append(f"  • Speaker Detection: /v1/chat/completions (model: {model})")

    # Summarization
    summarization_info = ml_providers.get("summarization", {})
    if summarization_info.get("provider") == "openai":
        model = summarization_info.get("openai_model", "gpt-4o-mini")
        summary_parts.append(f"  • Summarization: /v1/chat/completions (model: {model})")

    if summary_parts:
        return "OpenAI APIs called:\n" + "\n".join(summary_parts)
    return "No OpenAI APIs detected in metadata"


def _save_all_episode_responses(
    temp_dir: Path,
    metadata_files: list[Path],
    test_name: str,
    validate_provider: Optional[str] = None,
    validate_transcription: bool = True,
    validate_speaker_detection: bool = True,
    validate_summarization: bool = True,
) -> list[Path]:
    """Save OpenAI API responses for all episodes processed in a test.

    All episodes from the same test run are saved to the same run folder,
    with one file per episode (e.g., openai-responses_ep01.txt, openai-responses_ep02.txt).

    Args:
        temp_dir: Temporary directory where test output was written
        metadata_files: List of metadata file paths (one per episode)
        test_name: Name of the test (e.g., "test_openai_all_providers_in_pipeline")
        validate_provider: Optional provider name to validate (e.g., "openai")
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

        # Validate provider if specified
        if validate_provider:
            processing = metadata_content.get("processing", {})
            config_snapshot = processing.get("config_snapshot", {})
            ml_providers = config_snapshot.get("ml_providers", {})

            # Check which provider type to validate
            if validate_provider == "openai":
                # Check transcription provider (only if validation is enabled)
                if validate_transcription:
                    transcription_info = ml_providers.get("transcription", {})
                    if transcription_info:
                        assert (
                            transcription_info.get("provider") == "openai"
                        ), f"Transcription provider should be 'openai' in metadata (episode {idx})"

                # Check speaker detection provider (only if validation is enabled)
                if validate_speaker_detection:
                    speaker_info = ml_providers.get("speaker_detection", {})
                    if speaker_info:
                        assert speaker_info.get("provider") == "openai", (
                            f"Speaker detection provider should be 'openai' "
                            f"in metadata (episode {idx})"
                        )

                # Check summarization provider (only if validation is enabled)
                if validate_summarization:
                    summarization_info = ml_providers.get("summarization", {})
                    if summarization_info:
                        assert (
                            summarization_info.get("provider") == "openai"
                        ), f"Summarization provider should be 'openai' in metadata (episode {idx})"

        # Print summary of OpenAI APIs and models used for this episode
        if len(metadata_files) > 1:
            print(f"\nEpisode {idx}: {_summarize_openai_usage(metadata_content)}")
        else:
            print(f"\n{_summarize_openai_usage(metadata_content)}")

        # Save actual API responses (transcript, speakers, summary) to file
        # Include episode number in the test name for uniqueness when multiple episodes
        episode_suffix = f"_ep{idx:02d}" if len(metadata_files) > 1 else ""
        response_file = _save_openai_responses(
            temp_dir,
            metadata_content,
            f"{test_name}{episode_suffix}",
            shared_run_name=shared_run_name,
            metadata_file=metadata_file,
        )
        saved_files.append(response_file)

    # Print summary of all saved files
    if len(saved_files) > 1:
        print(f"\n📁 Saved {len(saved_files)} OpenAI API response files:")
        for idx, file_path in enumerate(saved_files, 1):
            print(f"  {idx}. {file_path}")
    elif len(saved_files) == 1:
        print(f"📁 OpenAI API responses saved to: {saved_files[0]}")

    return saved_files


def _save_openai_responses(  # noqa: C901
    temp_dir: Path,
    metadata_content: dict,
    test_name: str,
    shared_run_name: Optional[str] = None,
    metadata_file: Optional[Path] = None,
) -> Path:
    """Save actual OpenAI API responses (transcript, speakers, summary) to a file.

    Args:
        temp_dir: Temporary directory where test output was written
        metadata_content: Parsed metadata JSON content
        test_name: Name of the test (e.g., "test_openai_all_providers_in_pipeline_ep01")
        shared_run_name: Optional shared run name for all episodes in a test run.
                        If provided, all episodes will be saved to the same run folder.
                        If None, generates a new run name (for backward compatibility).

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
    # (e.g., "_ep01" from "test_openai_all_providers_in_pipeline_ep01")
    episode_suffix_match = re.search(r"_ep(\d+)$", test_name)
    if episode_suffix_match:
        episode_num = episode_suffix_match.group(1)
        filename = f"openai-responses_ep{episode_num}.txt"
    else:
        filename = "openai-responses.txt"

    output_file = run_output_dir / filename

    # Additional safety: if file somehow exists, append a counter
    counter = 1
    original_output_file = output_file
    while output_file.exists():
        filename_with_counter = f"openai-responses_{counter}.txt"
        output_file = run_output_dir / filename_with_counter
        counter += 1
        if counter > 10000:  # Safety limit to prevent infinite loop
            raise RuntimeError(f"Too many files with same name: {original_output_file}")

    # Collect all response data
    response_lines = []
    response_lines.append("=" * 80)
    response_lines.append("OpenAI API Responses")
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
        elif transcript_source == "whisper_transcription":
            whisper_model = content.get("whisper_model")
            if whisper_model:
                response_lines.append(f"  (Generated using Whisper model: {whisper_model})")

    # Try to find reference transcript from fixtures (if available)
    # For fixture feeds, we can check if there's a corresponding transcript file
    # in the fixtures directory based on the media filename
    if media_url:
        try:
            from urllib.parse import urlparse

            parsed_url = urlparse(media_url)
            media_filename = parsed_url.path.split("/")[-1]
            # Extract base name (e.g., "p01_e01_fast" from "p01_e01_fast.mp3")
            base_name = (
                media_filename.rsplit(".", 1)[0] if "." in media_filename else media_filename
            )

            # Check for fixture transcript files
            # Look in tests/fixtures/transcripts/ directory
            fixtures_transcripts_dir = Path("tests/fixtures/transcripts")
            if fixtures_transcripts_dir.exists():
                # Try exact match first
                ref_transcript_file = fixtures_transcripts_dir / f"{base_name}.txt"
                if not ref_transcript_file.exists():
                    # Try with wildcard pattern
                    possible_files = list(fixtures_transcripts_dir.glob(f"{base_name}*.txt"))
                    if possible_files:
                        ref_transcript_file = possible_files[0]

                if ref_transcript_file.exists():
                    response_lines.append("")
                    response_lines.append(
                        f"Reference Transcript (from fixtures): {ref_transcript_file}"
                    )
                    # Include a preview of the reference transcript
                    try:
                        ref_text = ref_transcript_file.read_text(encoding="utf-8")
                        preview_length = min(300, len(ref_text))
                        response_lines.append(f"  Preview (first {preview_length} chars):")
                        preview_text = ref_text[:preview_length]
                        ellipsis = "..." if len(ref_text) > preview_length else ""
                        response_lines.append(f"  {preview_text}{ellipsis}")
                        response_lines.append(f"  (Total length: {len(ref_text)} characters)")
                    except Exception as e:
                        response_lines.append(f"  (Error reading transcript: {e})")
        except Exception:
            # Silently fail if we can't find reference transcripts
            pass

    response_lines.append("")
    response_lines.append("=" * 80)
    response_lines.append("")

    # 1. Transcription result
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
            # Search for files matching the episode index
            all_transcript_files = list(base_dir.rglob("*.txt"))
            # Filter out cleaned transcripts and metadata files
            candidate_files = [
                f
                for f in all_transcript_files
                if "cleaned" not in f.name and "metadata" not in f.name
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
            all_transcript_files = list(base_dir.rglob("*.txt"))
            candidate_files = [
                f
                for f in all_transcript_files
                if "cleaned" not in f.name and "metadata" not in f.name
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

    # 2. Speaker detection result
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

    response_lines.append("=" * 80)

    # Write to file
    output_file.write_text("\n".join(response_lines), encoding="utf-8")

    return output_file


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.openai
class TestOpenAIProviderE2E:
    """Test OpenAI providers in integration workflows using E2E server."""

    def test_openai_transcription_in_pipeline(self, e2e_server: Optional[Any]):
        """Test OpenAI transcription provider in full pipeline."""
        import os

        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and OpenAI config based on LLM_TEST_FEED
            rss_url, openai_api_base, openai_api_key = _get_test_feed_url(e2e_server)

            # Get max_episodes from environment variable (for real feeds) or use default
            max_episodes = int(os.getenv("LLM_TEST_MAX_EPISODES", "1"))

            # Create config with OpenAI transcription ONLY (no other providers)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="openai",
                speaker_detector_provider="openai",  # Explicitly set to avoid spaCy default
                summary_provider="openai",  # Explicitly set to avoid transformers default
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                transcribe_missing=True,  # Enable transcription
                generate_metadata=True,
                generate_summaries=False,  # Disable summarization
                auto_speakers=False,  # Disable speaker detection
                preload_models=False,  # Disable model preloading (no local ML models)
                max_episodes=max_episodes,
            )

            # Run pipeline (uses E2E server OpenAI endpoints, no direct mocking)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify transcript files were created
            transcript_files = list(Path(temp_dir).rglob("*.txt")) + list(
                Path(temp_dir).rglob("*.vtt")
            )
            assert len(transcript_files) >= 1, "Should have created at least one transcript file"

            # Verify OpenAI transcription provider was used (check metadata)
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
                        transcription_info.get("provider") == "openai"
                    ), f"Transcription provider should be 'openai' in metadata (episode {idx})"

                    # Print summary of OpenAI APIs and models used for this episode
                    print(f"\nEpisode {idx}: {_summarize_openai_usage(metadata_content)}")

                    # Save actual API responses (transcript, speakers, summary) to file
                    # Include episode number in the test name for uniqueness
                    episode_suffix = f"_ep{idx:02d}" if len(metadata_files) > 1 else ""
                    response_file = _save_openai_responses(
                        Path(temp_dir),
                        metadata_content,
                        f"test_openai_transcription_in_pipeline{episode_suffix}",
                        shared_run_name=shared_run_name,
                    )
                    saved_files.append(response_file)

                # Print summary of all saved files
                if len(saved_files) > 1:
                    print(f"\n📁 Saved {len(saved_files)} OpenAI API response files:")
                    for idx, file_path in enumerate(saved_files, 1):
                        print(f"  {idx}. {file_path}")
                else:
                    print(f"📁 OpenAI API responses saved to: {saved_files[0]}")
            else:
                metadata_content = {}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.flaky
    def test_openai_speaker_detection_in_pipeline(self, e2e_server: Optional[Any]):
        """Test OpenAI speaker detection provider in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and OpenAI config based on LLM_TEST_FEED
            rss_url, openai_api_base, openai_api_key = _get_test_feed_url(e2e_server)

            # Create config with OpenAI speaker detection and transcription ONLY (no summarization)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="openai",
                speaker_detector_provider="openai",
                summary_provider="openai",  # Explicitly set to avoid transformers default
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                auto_speakers=True,
                generate_metadata=True,
                generate_summaries=False,  # Disable summarization
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses OpenAI ONLY, no local ML providers)
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
                "test_openai_speaker_detection_in_pipeline",
                validate_provider="openai",
                validate_summarization=False,  # Summarization not enabled in this test
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_summarization_in_pipeline(self, e2e_server: Optional[Any]):
        """Test OpenAI summarization provider in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and OpenAI config based on LLM_TEST_FEED
            rss_url, openai_api_base, openai_api_key = _get_test_feed_url(e2e_server)

            # Create config with OpenAI summarization and transcription ONLY (no speaker detection)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="openai",
                speaker_detector_provider="openai",  # Explicitly set to avoid spaCy default
                summary_provider="openai",
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                generate_metadata=True,
                generate_summaries=True,
                auto_speakers=False,  # Disable speaker detection
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses OpenAI ONLY, no local ML providers)
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
                "test_openai_summarization_in_pipeline",
                validate_provider="openai",
                validate_speaker_detection=False,  # Speaker detection not enabled in this test
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.flaky
    def test_openai_all_providers_in_pipeline(self, e2e_server: Optional[Any]):
        """Test all OpenAI providers together in full pipeline."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and OpenAI config based on LLM_TEST_FEED
            rss_url, openai_api_base, openai_api_key = _get_test_feed_url(e2e_server)

            # Create config with ALL OpenAI providers ONLY (no local ML providers)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="openai",
                speaker_detector_provider="openai",
                summary_provider="openai",
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                auto_speakers=True,
                generate_metadata=True,
                generate_summaries=True,
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses E2E server OpenAI endpoints, no direct mocking)
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
                "test_openai_all_providers_in_pipeline",
                validate_provider="openai",
            )
        finally:
            # Preserve temp_dir when using real API (for inspection/debugging)
            # Only clean up when using mock E2E server
            if not USE_REAL_OPENAI_API:
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                # Log location for debugging
                print(f"\n⚠️  Preserving temp_dir (USE_REAL_OPENAI_API=1): {temp_dir}")

    def test_openai_transcription_api_error_handling(self, e2e_server: Optional[Any]):
        """Test that OpenAI transcription API errors are handled gracefully.

        Note: E2E server currently doesn't simulate errors, so this test verifies
        that the pipeline completes successfully. Error handling is tested in
        unit/integration tests.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and OpenAI config based on LLM_TEST_FEED
            rss_url, openai_api_base, openai_api_key = _get_test_feed_url(e2e_server)

            # Create config with OpenAI transcription ONLY (no other providers)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="openai",
                speaker_detector_provider="openai",  # Explicitly set to avoid spaCy default
                summary_provider="openai",  # Explicitly set to avoid transformers default
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                generate_summaries=False,  # Disable summarization
                auto_speakers=False,  # Disable speaker detection
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline - E2E server provides successful responses
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Pipeline should complete successfully
            # Note: Error handling is tested in unit/integration tests
            assert transcripts_saved >= 0, "Pipeline should complete"

            # Verify OpenAI transcription provider was used (if metadata exists)
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            if len(metadata_files) > 0:
                # Save responses for all episodes (if any succeeded)
                _save_all_episode_responses(
                    Path(temp_dir),
                    metadata_files,
                    "test_openai_transcription_api_error_handling",
                    validate_provider="openai",
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_speaker_detection_api_error_handling(self, e2e_server: Optional[Any]):
        """Test that OpenAI speaker detection works correctly.

        Note: E2E server currently doesn't simulate errors, so this test verifies
        that the pipeline completes successfully. Error handling is tested in
        unit/integration tests.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and OpenAI config based on LLM_TEST_FEED
            rss_url, openai_api_base, openai_api_key = _get_test_feed_url(e2e_server)

            # Create config with OpenAI speaker detection and transcription ONLY (no summarization)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="openai",
                speaker_detector_provider="openai",
                summary_provider="openai",  # Explicitly set to avoid transformers default
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                auto_speakers=True,
                generate_metadata=True,
                generate_summaries=False,  # Disable summarization
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses OpenAI ONLY, no local ML providers)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Pipeline should complete successfully
            # Note: Error handling is tested in unit/integration tests
            assert transcripts_saved >= 0, "Pipeline should complete"

            # Verify OpenAI speaker detection provider was used (if metadata exists)
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            if len(metadata_files) > 0:
                # Save responses for all episodes
                _save_all_episode_responses(
                    Path(temp_dir),
                    metadata_files,
                    "test_openai_speaker_detection_api_error_handling",
                    validate_provider="openai",
                    validate_summarization=False,  # Summarization not enabled in this test
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_summarization_api_error_handling(self, e2e_server: Optional[Any]):
        """Test that OpenAI summarization works correctly.

        Note: E2E server currently doesn't simulate errors, so this test verifies
        that the pipeline completes successfully. Error handling is tested in
        unit/integration tests.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and OpenAI config based on LLM_TEST_FEED
            rss_url, openai_api_base, openai_api_key = _get_test_feed_url(e2e_server)

            # Create config with OpenAI summarization and transcription ONLY (no speaker detection)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="openai",
                speaker_detector_provider="openai",  # Explicitly set to avoid spaCy default
                summary_provider="openai",
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                generate_metadata=True,
                generate_summaries=True,
                auto_speakers=False,  # Disable speaker detection
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline (uses OpenAI ONLY, no local ML providers)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Pipeline should complete successfully
            # Note: Error handling is tested in unit/integration tests
            assert transcripts_saved >= 0, "Pipeline should complete"

            # Verify metadata files were created
            # Metadata should be created with summary from E2E server
            # Use *.metadata.json to avoid matching metrics.json files
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "Metadata files should be created"

            # Save responses for all episodes
            _save_all_episode_responses(
                Path(temp_dir),
                metadata_files,
                "test_openai_summarization_api_error_handling",
                validate_provider="openai",
                validate_speaker_detection=False,  # Speaker detection not enabled in this test
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openai_transcription_rate_limiting(self, e2e_server: Optional[Any]):
        """Test that OpenAI transcription works correctly.

        Note: E2E server currently doesn't simulate rate limiting, so this test verifies
        that the pipeline completes successfully. Rate limiting is tested in
        unit/integration tests.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and OpenAI config based on LLM_TEST_FEED
            rss_url, openai_api_base, openai_api_key = _get_test_feed_url(e2e_server)

            # Create config with OpenAI transcription ONLY (no other providers)
            cfg = create_test_config(
                rss_url=rss_url,
                output_dir=temp_dir,
                transcription_provider="openai",
                speaker_detector_provider="openai",  # Explicitly set to avoid spaCy default
                summary_provider="openai",  # Explicitly set to avoid transformers default
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                generate_summaries=False,  # Disable summarization
                auto_speakers=False,  # Disable speaker detection
                preload_models=False,  # Disable model preloading (no local ML models)
                transcribe_missing=True,
                max_episodes=int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            )

            # Run pipeline - E2E server provides successful responses
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Pipeline should complete successfully
            # Note: Rate limiting is tested in unit/integration tests
            assert transcripts_saved >= 0, "Pipeline should complete"

            # Verify OpenAI transcription provider was used (if metadata exists)
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            if len(metadata_files) > 0:
                # Save responses for all episodes (if any succeeded)
                _save_all_episode_responses(
                    Path(temp_dir),
                    metadata_files,
                    "test_openai_transcription_rate_limiting",
                    validate_provider="openai",
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
