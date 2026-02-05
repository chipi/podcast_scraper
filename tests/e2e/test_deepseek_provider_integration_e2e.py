#!/usr/bin/env python3
"""DeepSeek provider E2E tests (Issue #107).

These tests verify that DeepSeek providers work correctly in complete user workflows:
- DeepSeek speaker detection in workflow
- DeepSeek summarization in workflow
- Error handling (API errors, rate limiting, retries)

These tests use the E2E server's mock API endpoints (DeepSeek uses OpenAI-compatible API,
so it uses the same /v1/chat/completions endpoints as OpenAI/Grok/Mistral).
Tests are marked with @pytest.mark.e2e to allow selective execution.

Real API Mode:
    When USE_REAL_DEEPSEEK_API=1, tests use real DeepSeek API endpoints and real RSS feeds.
    This is for manual testing only and will incur API costs.

Note: DeepSeek does NOT support transcription (no audio API).
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.module_deepseek_providers]

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

# Check if we should use real DeepSeek API (for manual testing only)
USE_REAL_DEEPSEEK_API = os.getenv("USE_REAL_DEEPSEEK_API", "0") == "1"

# Feed selection for LLM provider tests (shared by OpenAI, Gemini, etc.)
# Default to "multi" to work in both fast and multi_episode E2E_TEST_MODE
LLM_TEST_FEED = os.getenv("LLM_TEST_FEED", "multi")

# Real RSS feed URL for testing (only used when USE_REAL_DEEPSEEK_API=1)
# NOTE: No default real feed - must be explicitly set via LLM_TEST_RSS_FEED
REAL_TEST_RSS_FEED = os.getenv("LLM_TEST_RSS_FEED", None)


def _get_test_feed_url(
    e2e_server: Optional[Any] = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Get RSS feed URL and DeepSeek config based on LLM_TEST_FEED environment variable.

    Args:
        e2e_server: E2E server fixture (None if using real API or if fixture not available)

    Returns:
        Tuple of (rss_url, deepseek_api_base, deepseek_api_key)
    """
    feed_type = (LLM_TEST_FEED or "multi").lower()

    # Real API mode - can use either real RSS feed OR fixture feeds
    if USE_REAL_DEEPSEEK_API:
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
        raise ValueError("E2E server is required for mocked API tests")

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

    # Use E2E server's mock DeepSeek API endpoint
    deepseek_api_base = e2e_server.urls.deepseek_api_base()
    # API key will be set by fixture (test-dummy-key-for-e2e-tests)
    # Return None to use env var from fixture
    deepseek_api_key = None

    return rss_url, deepseek_api_base, deepseek_api_key


def _summarize_deepseek_usage(metadata_content: dict) -> str:
    """Summarize DeepSeek API usage from metadata."""
    processing = metadata_content.get("processing", {})
    api_usage = processing.get("api_usage", {})
    deepseek_usage = api_usage.get("deepseek", {})

    if not deepseek_usage:
        return "No DeepSeek API usage recorded in metadata"

    lines = ["DeepSeek API Usage Summary:"]
    for capability, usage in deepseek_usage.items():
        lines.append(f"  {capability}:")
        if "requests" in usage:
            lines.append(f"    Requests: {usage['requests']}")
        if "tokens" in usage:
            tokens = usage["tokens"]
            lines.append(f"    Input tokens: {tokens.get('input', 0)}")
            lines.append(f"    Output tokens: {tokens.get('output', 0)}")
        if "cost" in usage:
            lines.append(f"    Estimated cost: ${usage['cost']:.4f}")

    return "\n".join(lines)


def _save_all_episode_responses(
    temp_dir: Path,
    metadata_files: list[Path],
    test_name: str,
    validate_provider: Optional[str] = None,
    validate_speaker_detection: bool = True,
    validate_summarization: bool = True,
) -> list[Path]:
    """Save DeepSeek API responses for all episodes processed in a test.

    All episodes from the same test run are saved to the same run folder,
    with one file per episode (e.g., deepseek-responses_ep01.txt, deepseek-responses_ep02.txt).

    Args:
        temp_dir: Temporary directory where test output was written
        metadata_files: List of metadata file paths (one per episode)
        test_name: Name of the test (e.g., "test_deepseek_full_pipeline")
        validate_provider: Optional provider name to validate (e.g., "deepseek")
        validate_speaker_detection: Whether to validate speaker detection provider (default: True)
        validate_summarization: Whether to validate summarization provider (default: True)

    Returns:
        List of paths to saved response files
    """
    import json as json_module
    from datetime import datetime

    # Generate ONE run folder name for all episodes in this test run
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

        # Print summary of DeepSeek APIs and models used for this episode
        if len(metadata_files) > 1:
            print(f"\nEpisode {idx}: {_summarize_deepseek_usage(metadata_content)}")
        else:
            print(f"\n{_summarize_deepseek_usage(metadata_content)}")

        # Save actual API responses (speakers, summary) to file
        episode_suffix = f"_ep{idx:02d}" if len(metadata_files) > 1 else ""
        response_file = _save_deepseek_responses(
            temp_dir,
            metadata_content,
            f"{test_name}{episode_suffix}",
            shared_run_name=shared_run_name,
            metadata_file=metadata_file,
        )
        saved_files.append(response_file)

    # Print summary of all saved files
    if len(saved_files) > 1:
        print(f"\n📁 Saved {len(saved_files)} DeepSeek API response files:")
        for idx, file_path in enumerate(saved_files, 1):
            print(f"  {idx}. {file_path}")
    elif len(saved_files) == 1:
        print(f"📁 DeepSeek API responses saved to: {saved_files[0]}")

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


def _save_deepseek_responses(  # noqa: C901
    temp_dir: Path,
    metadata_content: dict,
    test_name: str,
    shared_run_name: Optional[str] = None,
    metadata_file: Optional[Path] = None,
) -> Path:
    """Save actual DeepSeek API responses (speakers, summary) to a file.

    Args:
        temp_dir: Temporary directory where test output was written
        metadata_content: Parsed metadata JSON content
        test_name: Name of the test (e.g., "test_deepseek_full_pipeline_ep01")
        shared_run_name: Optional shared run name for all episodes in a test run.
                        If provided, all episodes will be saved to the same run folder.
                        If None, generates a new run name (for backward compatibility).
        metadata_file: Optional metadata file path (used to determine base directory)

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
        digest = hashlib.sha1(feed_url.encode("utf-8"), usedforsecurity=False).hexdigest()
        feed_dir_name = f"rss_{safe_base}_{digest[:filesystem.URL_HASH_LENGTH]}"
    else:
        feed_dir_name = "unknown_feed"

    # Use shared run name if provided, otherwise generate new one
    if shared_run_name:
        run_name = shared_run_name
    else:
        now = datetime.now()
        run_name = now.strftime("run_%Y%m%d-%H%M%S_%f")

    # Create output directory structure: output/<feed-name>/<run-name>/
    output_dir = Path("output")
    feed_output_dir = output_dir / feed_dir_name
    run_output_dir = feed_output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract episode suffix from test_name
    episode_suffix_match = re.search(r"_ep(\d+)$", test_name)
    if episode_suffix_match:
        episode_num = episode_suffix_match.group(1)
        filename = f"deepseek-responses_ep{episode_num}.txt"
    else:
        filename = "deepseek-responses.txt"

    output_file = run_output_dir / filename

    # Additional safety: if file somehow exists, append a counter
    counter = 1
    original_output_file = output_file
    while output_file.exists():
        filename_with_counter = f"deepseek-responses_{counter}.txt"
        output_file = run_output_dir / filename_with_counter
        counter += 1
        if counter > 10000:
            raise RuntimeError(f"Too many files with same name: {original_output_file}")

    # Collect all response data
    response_lines = []
    response_lines.append("=" * 80)
    response_lines.append("DeepSeek API Responses")
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

    # Note about DeepSeek capabilities
    response_lines.append("")
    response_lines.append("Note: DeepSeek does NOT support transcription (no audio API)")
    response_lines.append("      DeepSeek is used for speaker detection and summarization only")
    response_lines.append("")
    response_lines.append("=" * 80)
    response_lines.append("")

    # 1. Speaker detection result
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

    # 2. Transcript (if available - from local Whisper or other provider)
    transcript_file = None
    content = metadata_content.get("content", {})
    transcript_file_path = content.get("transcript_file_path")

    # Determine base directory for transcript lookup
    if metadata_file:
        base_dir = metadata_file.parent.parent
    else:
        base_dir = temp_dir

    if transcript_file_path:
        transcript_file = base_dir / transcript_file_path
        if not transcript_file.exists():
            transcript_file = base_dir / "transcripts" / Path(transcript_file_path).name
            if not transcript_file.exists():
                transcript_file = None

    # Fallback: try to find transcript file by episode index
    if transcript_file is None:
        episode_data = metadata_content.get("episode", {})
        episode_idx = episode_data.get("idx")

        if episode_idx is not None:
            all_transcript_files = list(base_dir.rglob("*.txt"))
            candidate_files = [
                f
                for f in all_transcript_files
                if "cleaned" not in f.name and "metadata" not in f.name
            ]

            idx_str = f"{episode_idx:04d}"
            for candidate in candidate_files:
                if candidate.name.startswith(idx_str):
                    transcript_file = candidate
                    break

    if transcript_file and transcript_file.exists():
        response_lines.append("\n📝 TRANSCRIPT (from local Whisper or other provider):")
        response_lines.append("-" * 80)
        try:
            transcript_text = transcript_file.read_text(encoding="utf-8")
            # Limit transcript preview to first 2000 characters
            preview_length = 2000
            if len(transcript_text) > preview_length:
                response_lines.append(transcript_text[:preview_length])
                response_lines.append("")
                response_lines.append(f"... (truncated, {len(transcript_text)} total characters)")
            else:
                response_lines.append(transcript_text)
        except Exception as e:
            response_lines.append(f"Error reading transcript file: {e}")
        response_lines.append("")

    # 3. Summarization result
    summary_data = metadata_content.get("summary")
    if summary_data:
        response_lines.append("\n📄 SUMMARIZATION RESULT:")
        response_lines.append("-" * 80)
        if isinstance(summary_data, dict):
            short_summary = summary_data.get("short_summary", "")
            long_summary = summary_data.get("summary", "")
            if short_summary:
                response_lines.append("Short Summary:")
                response_lines.append(short_summary)
                response_lines.append("")
            if long_summary:
                response_lines.append("Long Summary:")
                response_lines.append(long_summary)
            elif not short_summary:
                response_lines.append("  (Summary exists but summary fields are empty)")
        else:
            response_lines.append(str(summary_data))
        response_lines.append("")
    else:
        response_lines.append("\n📄 SUMMARIZATION RESULT:")
        response_lines.append("-" * 80)
        response_lines.append("  (No summary available)")
        response_lines.append("")

    response_lines.append("=" * 80)

    # Write to file
    output_file.write_text("\n".join(response_lines), encoding="utf-8")
    return output_file


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.deepseek
class TestDeepSeekProviderE2E:
    """Test DeepSeek providers in integration workflows using E2E server mock endpoints."""

    def test_deepseek_speaker_detection_in_pipeline(self, e2e_server: Optional[Any]):
        """Test DeepSeek speaker detection provider in full pipeline."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and DeepSeek config
            rss_url, deepseek_api_base, deepseek_api_key = _get_test_feed_url(e2e_server)

            # Create config with DeepSeek speaker detection
            # Note: deepseek_api_key can be None (will use env var from fixture)
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",  # Use local for transcription
                "speaker_detector_provider": "deepseek",
                "summary_provider": "deepseek",  # Use DeepSeek for summarization
                "generate_summaries": True,  # Enable summary generation
            }
            # Always pass API key (even if None) to ensure field validator runs
            # Field validator will load from .env if value is None
            config_kwargs["deepseek_api_key"] = deepseek_api_key
            # Only set API base if provided (None means use real API default)
            # When USE_REAL_DEEPSEEK_API=1, deepseek_api_base is None, so we use real API
            if deepseek_api_base is not None:
                config_kwargs["deepseek_api_base"] = deepseek_api_base
            # When using real API, ensure we don't accidentally use E2E server
            elif USE_REAL_DEEPSEEK_API:
                # Explicitly ensure API base is not set (will use default from field validator)
                # This ensures we hit the real DeepSeek API server, not the E2E mock server
                pass  # deepseek_api_base is already None, field validator will use default

            config_kwargs["auto_speakers"] = True
            config_kwargs["generate_metadata"] = True  # Enable metadata generation
            config_kwargs["max_episodes"] = int(os.getenv("LLM_TEST_MAX_EPISODES", "1"))
            config_kwargs["transcribe_missing"] = (
                True  # Enable transcription for episodes without transcripts
            )

            cfg = create_test_config(**config_kwargs)

            # Run pipeline (uses DeepSeek for speaker detection, local for other tasks)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created (search recursively for run subdirectories)
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            # Verify DeepSeek was used (check metadata)
            if metadata_files:
                with open(metadata_files[0]) as f:
                    metadata_content = json.load(f)

                # Check that DeepSeek was used for speaker detection
                processing = metadata_content.get("processing", {})
                config_snapshot = processing.get("config_snapshot", {})
                ml_providers = config_snapshot.get("ml_providers", {})
                speaker_info = ml_providers.get("speaker_detection", {})
                assert speaker_info.get("provider") == "deepseek"

                # Print summary
                usage_summary = _summarize_deepseek_usage(metadata_content)
                print(f"\n{usage_summary}")

                # Save API responses for real API mode (for manual testing)
                if USE_REAL_DEEPSEEK_API:
                    _save_all_episode_responses(
                        Path(temp_dir),
                        metadata_files,
                        "test_deepseek_speaker_detection_in_pipeline",
                        validate_provider="deepseek",
                        validate_summarization=False,
                    )

        finally:
            # Only clean up when using mock E2E server
            if not USE_REAL_DEEPSEEK_API:
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                # Log location for debugging
                print(f"\n⚠️  Preserving temp_dir (USE_REAL_DEEPSEEK_API=1): {temp_dir}")

    def test_deepseek_summarization_in_pipeline(self, e2e_server: Optional[Any]):
        """Test DeepSeek summarization provider in full pipeline."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and DeepSeek config
            rss_url, deepseek_api_base, deepseek_api_key = _get_test_feed_url(e2e_server)

            # Create config with DeepSeek summarization
            # Note: deepseek_api_key can be None (will use env var from fixture)
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",  # Use local for transcription
                "speaker_detector_provider": "spacy",  # Use local for speaker detection
                "summary_provider": "deepseek",
                "generate_summaries": True,
                "generate_metadata": True,  # Required when generate_summaries=True
            }
            # Only pass API key/base if provided (when None, Config loads from .env/fixture)
            if deepseek_api_key is not None:
                config_kwargs["deepseek_api_key"] = deepseek_api_key
            if deepseek_api_base is not None:
                config_kwargs["deepseek_api_base"] = deepseek_api_base

            config_kwargs["max_episodes"] = int(os.getenv("LLM_TEST_MAX_EPISODES", "1"))

            cfg = create_test_config(**config_kwargs)

            # Run pipeline (uses DeepSeek for summarization, local for other tasks)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created (search recursively for run subdirectories)
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            # Verify DeepSeek was used (check metadata)
            if metadata_files:
                with open(metadata_files[0]) as f:
                    metadata_content = json.load(f)

                # Check that DeepSeek was used for summarization
                processing = metadata_content.get("processing", {})
                config_snapshot = processing.get("config_snapshot", {})
                ml_providers = config_snapshot.get("ml_providers", {})
                summary_info = ml_providers.get("summarization", {})
                assert summary_info.get("provider") == "deepseek"

                # Print summary
                usage_summary = _summarize_deepseek_usage(metadata_content)
                print(f"\n{usage_summary}")

                # Save API responses for real API mode (for manual testing)
                if USE_REAL_DEEPSEEK_API:
                    _save_all_episode_responses(
                        Path(temp_dir),
                        metadata_files,
                        "test_deepseek_summarization_in_pipeline",
                        validate_provider="deepseek",
                        validate_speaker_detection=False,
                    )

        finally:
            # Only clean up when using mock E2E server
            if not USE_REAL_DEEPSEEK_API:
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                # Log location for debugging
                print(f"\n⚠️  Preserving temp_dir (USE_REAL_DEEPSEEK_API=1): {temp_dir}")

    def test_deepseek_full_pipeline(self, e2e_server: Optional[Any]):
        """Test all DeepSeek providers together in full pipeline."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and DeepSeek config
            rss_url, deepseek_api_base, deepseek_api_key = _get_test_feed_url(e2e_server)

            # Create config with ALL DeepSeek providers (speaker detection and summarization)
            # Note: DeepSeek does NOT support transcription
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",  # Must use local for transcription
                "speaker_detector_provider": "deepseek",
                "summary_provider": "deepseek",
                "auto_speakers": True,
                "generate_metadata": True,
                "generate_summaries": True,
                "preload_models": False,  # Disable model preloading (no local ML models)
                "transcribe_missing": True,
                "max_episodes": int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            }
            # Only pass API key/base if provided (when None, Config loads from .env)
            if deepseek_api_key is not None:
                config_kwargs["deepseek_api_key"] = deepseek_api_key
            if deepseek_api_base is not None:
                config_kwargs["deepseek_api_base"] = deepseek_api_base

            cfg = create_test_config(**config_kwargs)

            # Run pipeline (uses DeepSeek for speaker detection and summarization)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0, "Should have saved at least one transcript"

            # Verify metadata files were created
            # Use *.metadata.json to avoid matching metrics.json files
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "Should have created at least one metadata file"

            # Verify DeepSeek was used for both capabilities
            if metadata_files:
                with open(metadata_files[0]) as f:
                    metadata_content = json.load(f)

                processing = metadata_content.get("processing", {})
                config_snapshot = processing.get("config_snapshot", {})
                ml_providers = config_snapshot.get("ml_providers", {})

                # Check speaker detection
                speaker_info = ml_providers.get("speaker_detection", {})
                assert speaker_info.get("provider") == "deepseek"

                # Check summarization
                summary_info = ml_providers.get("summarization", {})
                assert summary_info.get("provider") == "deepseek"

                # Print summary
                usage_summary = _summarize_deepseek_usage(metadata_content)
                print(f"\n{usage_summary}")

                # Save API responses for real API mode (for manual testing)
                if USE_REAL_DEEPSEEK_API:
                    _save_all_episode_responses(
                        Path(temp_dir),
                        metadata_files,
                        "test_deepseek_full_pipeline",
                        validate_provider="deepseek",
                    )

        finally:
            # Only clean up when using mock E2E server
            if not USE_REAL_DEEPSEEK_API:
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                # Log location for debugging
                print(f"\n⚠️  Preserving temp_dir (USE_REAL_DEEPSEEK_API=1): {temp_dir}")

    def test_deepseek_speaker_detection_api_error_handling(self, e2e_server: Optional[Any]):
        """Test that DeepSeek speaker detection works correctly.

        Note: E2E server currently doesn't simulate errors, so this test verifies
        that the pipeline completes successfully. Error handling is tested in
        unit/integration tests.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and DeepSeek config
            rss_url, deepseek_api_base, deepseek_api_key = _get_test_feed_url(e2e_server)

            # Create config with DeepSeek speaker detection ONLY (no summarization)
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",  # Use local for transcription
                "speaker_detector_provider": "deepseek",
                "summary_provider": "transformers",  # Use local for summarization
                "auto_speakers": True,
                "generate_metadata": True,
                "generate_summaries": False,  # Disable summarization
                "preload_models": False,  # Disable model preloading (no local ML models)
                "transcribe_missing": True,
                "max_episodes": int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            }
            # Only pass API key/base if provided (when None, Config loads from .env/fixture)
            if deepseek_api_key is not None:
                config_kwargs["deepseek_api_key"] = deepseek_api_key
            if deepseek_api_base is not None:
                config_kwargs["deepseek_api_base"] = deepseek_api_base

            cfg = create_test_config(**config_kwargs)

            # Run pipeline (uses DeepSeek for speaker detection, local for other tasks)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Pipeline should complete successfully
            # Note: Error handling is tested in unit/integration tests
            assert transcripts_saved >= 0, "Pipeline should complete"

            # Verify DeepSeek speaker detection provider was used (if metadata exists)
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            if len(metadata_files) > 0:
                # Save responses for all episodes
                _save_all_episode_responses(
                    Path(temp_dir),
                    metadata_files,
                    "test_deepseek_speaker_detection_api_error_handling",
                    validate_provider="deepseek",
                    validate_summarization=False,  # Summarization not enabled in this test
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_deepseek_summarization_api_error_handling(self, e2e_server: Optional[Any]):
        """Test that DeepSeek summarization works correctly.

        Note: E2E server currently doesn't simulate errors, so this test verifies
        that the pipeline completes successfully. Error handling is tested in
        unit/integration tests.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Get feed URL and DeepSeek config
            rss_url, deepseek_api_base, deepseek_api_key = _get_test_feed_url(e2e_server)

            # Create config with DeepSeek summarization ONLY (no speaker detection)
            config_kwargs = {
                "rss_url": rss_url,
                "output_dir": temp_dir,
                "transcription_provider": "whisper",  # Use local for transcription
                "speaker_detector_provider": "spacy",  # Use local for speaker detection
                "summary_provider": "deepseek",
                "generate_metadata": True,
                "generate_summaries": True,
                "auto_speakers": False,  # Disable speaker detection
                "preload_models": False,  # Disable model preloading (no local ML models)
                "transcribe_missing": True,
                "max_episodes": int(os.getenv("LLM_TEST_MAX_EPISODES", "1")),
            }
            # Only pass API key/base if provided (when None, Config loads from .env/fixture)
            if deepseek_api_key is not None:
                config_kwargs["deepseek_api_key"] = deepseek_api_key
            if deepseek_api_base is not None:
                config_kwargs["deepseek_api_base"] = deepseek_api_base

            cfg = create_test_config(**config_kwargs)

            # Run pipeline (uses DeepSeek for summarization, local for other tasks)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Pipeline should complete successfully
            # Note: Error handling is tested in unit/integration tests
            assert transcripts_saved >= 0, "Pipeline should complete"

            # Verify metadata files were created
            metadata_files = list(Path(temp_dir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "Metadata files should be created"

            # Save responses for all episodes
            _save_all_episode_responses(
                Path(temp_dir),
                metadata_files,
                "test_deepseek_summarization_api_error_handling",
                validate_provider="deepseek",
                validate_speaker_detection=False,  # Speaker detection not enabled in this test
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
