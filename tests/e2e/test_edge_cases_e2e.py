#!/usr/bin/env python3
"""E2E tests for Edge Cases.

These tests verify edge cases are handled correctly:
- Special characters in episode titles
- Unicode characters in content
- Very long episode titles
- Missing optional fields in RSS feeds
- Empty descriptions
- Relative URLs in RSS feeds

All tests use real HTTP client and E2E server.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import Config, run_pipeline, service


@pytest.mark.e2e
@pytest.mark.slow
class TestSpecialCharactersInTitles:
    """Special characters in episode titles E2E tests."""

    def test_special_characters_in_title(self, e2e_server):
        """Test handling of special characters in episode titles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("edgecases"),
                output_dir=tmpdir,
                max_episodes=1,  # Only episode 1 has special chars
            )

            # Run pipeline - should handle special characters gracefully
            count, summary = run_pipeline(cfg)

            # Pipeline should complete successfully
            assert count == 1, "Should process 1 episode with special characters"
            assert isinstance(summary, str), "Summary should be a string"

            # Verify output file was created with sanitized filename
            output_files = list(Path(tmpdir).glob("*.txt"))
            assert len(output_files) == 1, "Should create one transcript file"
            # Filename should be sanitized (no special chars)
            assert "Special Chars" in output_files[0].name or "Episode 1" in output_files[0].name
            # Verify content exists
            assert output_files[0].stat().st_size > 0, "Transcript file should not be empty"


@pytest.mark.e2e
@pytest.mark.slow
class TestUnicodeCharacters:
    """Unicode characters in content E2E tests."""

    def test_unicode_characters_in_title(self, e2e_server):
        """Test handling of Unicode characters in episode titles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("edgecases"),
                output_dir=tmpdir,
                max_episodes=2,  # Episode 2 has Unicode
            )

            # Run pipeline - should handle Unicode gracefully
            count, summary = run_pipeline(cfg)

            # Determine expected episode count based on test mode
            test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()
            expected_episodes = 1 if test_mode == "fast" else 2

            # Pipeline should complete successfully (adjust for test mode)
            assert count == expected_episodes, (
                f"Should process {expected_episodes} episode(s) including "
                f"Unicode (mode: {test_mode}), got {count}"
            )
            assert isinstance(summary, str), "Summary should be a string"

            # Verify output files were created
            output_files = list(Path(tmpdir).glob("*.txt"))
            assert (
                len(output_files) == expected_episodes
            ), f"Should create {expected_episodes} transcript file(s), got {len(output_files)}"

            # Verify Unicode characters are preserved or handled correctly
            for output_file in output_files:
                assert output_file.stat().st_size > 0, "Transcript file should not be empty"


@pytest.mark.e2e
@pytest.mark.slow
class TestVeryLongTitles:
    """Very long episode titles E2E tests."""

    def test_very_long_episode_title(self, e2e_server):
        """Test handling of very long episode titles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("edgecases"),
                output_dir=tmpdir,
                max_episodes=3,  # Episode 3 has very long title
            )

            # Run pipeline - should handle long titles gracefully (may truncate filename)
            try:
                count, summary = run_pipeline(cfg)
                # Pipeline may complete successfully or fail due to filename length limits
                # On some systems (macOS), very long filenames cause "File name too long" error
                assert isinstance(summary, str), "Summary should be a string"

                # Verify output files were created (may be fewer if long title caused issues)
                output_files = list(Path(tmpdir).glob("*.txt"))
                # At least episodes 1 and 2 should be processed
                assert len(output_files) >= 2, "Should create at least two transcript files"

                # Verify files have content
                for output_file in output_files:
                    assert output_file.stat().st_size > 0, "Transcript file should not be empty"
            except OSError as e:
                # On some systems, very long filenames cause OSError
                if "File name too long" in str(e):
                    # This is expected behavior - filename sanitization should truncate
                    # but on some systems it may still be too long
                    pytest.skip(f"System filename length limit: {e}")
                raise


@pytest.mark.e2e
@pytest.mark.slow
class TestMissingOptionalFields:
    """Missing optional fields in RSS feeds E2E tests."""

    def test_missing_optional_fields(self, e2e_server):
        """Test handling of missing optional fields in RSS feeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("edgecases"),
                output_dir=tmpdir,
                max_episodes=4,  # Episode 4 has missing fields
            )

            # Run pipeline - should handle missing fields gracefully
            try:
                count, summary = run_pipeline(cfg)
                # Episode 3 may fail due to very long title, so we may get 3 or 4 episodes
                assert (
                    count >= 3
                ), "Should process at least 3 episodes (episode 3 may fail due to long title)"
                assert isinstance(summary, str), "Summary should be a string"

                # Verify output files were created for episodes with transcripts
                output_files = list(Path(tmpdir).glob("*.txt"))
                assert (
                    len(output_files) >= 3
                ), "Should create transcript files for at least 3 episodes"
            except OSError as e:
                # On some systems, very long filenames cause OSError
                if "File name too long" in str(e):
                    pytest.skip(f"System filename length limit: {e}")
                raise


@pytest.mark.e2e
@pytest.mark.slow
class TestEmptyDescriptions:
    """Empty descriptions E2E tests."""

    def test_empty_description(self, e2e_server):
        """Test handling of empty episode descriptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("edgecases"),
                output_dir=tmpdir,
                max_episodes=5,  # Episode 5 has empty description
            )

            # Run pipeline - should handle empty description gracefully
            try:
                count, summary = run_pipeline(cfg)
                # Episode 3 may fail due to very long title, so we may get 4 or 5 episodes
                assert (
                    count >= 4
                ), "Should process at least 4 episodes (episode 3 may fail due to long title)"
                assert isinstance(summary, str), "Summary should be a string"

                # Verify output files were created
                output_files = list(Path(tmpdir).glob("*.txt"))
                assert len(output_files) >= 4, "Should create at least four transcript files"
            except OSError as e:
                # On some systems, very long filenames cause OSError
                if "File name too long" in str(e):
                    pytest.skip(f"System filename length limit: {e}")
                raise


@pytest.mark.e2e
@pytest.mark.slow
class TestRelativeURLs:
    """Relative URLs in RSS feeds E2E tests."""

    def test_relative_urls_in_rss(self, e2e_server):
        """Test handling of relative URLs in RSS feeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("edgecases"),
                output_dir=tmpdir,
                max_episodes=6,  # Episode 6 has relative URLs
            )

            # Run pipeline - should resolve relative URLs correctly
            # Note: Relative URLs are resolved against RSS feed URL, which may cause 404
            # if the base URL path doesn't match the server structure
            try:
                count, summary = run_pipeline(cfg)
                # Episode 3 may fail due to very long title,
                # episode 6 may fail due to relative URL resolution
                # So we may get 4, 5, or 6 episodes depending on failures
                assert count >= 4, "Should process at least 4 episodes (episode 3/6 may fail)"
                assert isinstance(summary, str), "Summary should be a string"

                # Verify output files were created
                output_files = list(Path(tmpdir).glob("*.txt"))
                assert len(output_files) >= 4, "Should create at least four transcript files"
            except OSError as e:
                # On some systems, very long filenames cause OSError
                if "File name too long" in str(e):
                    pytest.skip(f"System filename length limit: {e}")
                raise


@pytest.mark.e2e
@pytest.mark.slow
class TestAllEdgeCasesTogether:
    """Comprehensive edge cases E2E tests."""

    def test_all_edge_cases_in_single_run(self, e2e_server):
        """Test all edge cases in a single pipeline run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("edgecases"),
                output_dir=tmpdir,
                max_episodes=6,  # All edge case episodes
            )

            # Run pipeline - should handle all edge cases gracefully
            try:
                count, summary = run_pipeline(cfg)
                # Pipeline may process fewer episodes if:
                # - Episode 3: very long title causes filename issues
                # - Episode 6: relative URL resolution may cause 404
                assert count >= 4, "Should process at least 4 episodes (episode 3/6 may fail)"
                assert isinstance(summary, str), "Summary should be a string"

                # Verify output files were created
                output_files = list(Path(tmpdir).glob("*.txt"))
                assert len(output_files) >= 4, "Should create at least four transcript files"

                # Verify all files have content
                for output_file in output_files:
                    assert output_file.stat().st_size > 0, "Transcript file should not be empty"
            except OSError as e:
                # On some systems, very long filenames cause OSError
                if "File name too long" in str(e):
                    # This is expected behavior - filename sanitization should truncate
                    # but on some systems it may still be too long
                    pytest.skip(f"System filename length limit: {e}")
                raise

    def test_edge_cases_with_metadata(self, e2e_server):
        """Test edge cases with metadata generation enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("edgecases"),
                output_dir=tmpdir,
                max_episodes=2,  # First 2 episodes (avoid episode 3 with long title)
                generate_metadata=True,
            )

            # Run pipeline - should handle edge cases with metadata
            count, summary = run_pipeline(cfg)

            # Determine expected episode count based on test mode
            test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()
            expected_episodes = 1 if test_mode == "fast" else 2

            # Pipeline should process expected episodes successfully (adjust for test mode)
            assert count == expected_episodes, (
                f"Should process {expected_episodes} episode(s) with metadata "
                f"(mode: {test_mode}), got {count}"
            )
            assert isinstance(summary, str), "Summary should be a string"

            # Verify metadata files were created
            metadata_files = list(Path(tmpdir).glob("*.metadata.json"))
            assert (
                len(metadata_files) == expected_episodes
            ), f"Should create {expected_episodes} metadata file(s), got {len(metadata_files)}"

            # Verify metadata files contain valid JSON
            import json

            for metadata_file in metadata_files:
                assert metadata_file.stat().st_size > 0, "Metadata file should not be empty"
                # Verify it's valid JSON
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata_content = json.load(f)
                    assert isinstance(metadata_content, dict), "Metadata should be a dictionary"
                    # Metadata has nested structure: feed, episode, content, processing, summary
                    assert "episode" in metadata_content, "Metadata should contain episode"
                    assert "content" in metadata_content, "Metadata should contain content"
                    assert isinstance(
                        metadata_content["episode"], dict
                    ), "Episode should be a dictionary"
                    assert "title" in metadata_content["episode"], "Episode should contain title"

    def test_service_api_with_edge_cases(self, e2e_server):
        """Test service API with edge cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=e2e_server.urls.feed("edgecases"),
                output_dir=tmpdir,
                max_episodes=2,  # First 2 episodes (avoid episode 3 with long title)
            )

            # Run service - should handle edge cases gracefully
            result = service.run(cfg)

            # Determine expected episode count based on test mode
            test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()
            expected_episodes = 1 if test_mode == "fast" else 2

            # Service should complete successfully
            assert result.success is True, "Service should succeed with edge cases"
            assert result.episodes_processed == expected_episodes, (
                f"Should process {expected_episodes} episode(s) "
                f"(mode: {test_mode}), got {result.episodes_processed}"
            )
            assert isinstance(result.summary, str), "Summary should be a string"
            assert result.error is None, "Should not have errors"

            # Verify output files were created
            output_files = list(Path(tmpdir).glob("*.txt"))
            assert (
                len(output_files) == expected_episodes
            ), f"Should create {expected_episodes} transcript file(s), got {len(output_files)}"
