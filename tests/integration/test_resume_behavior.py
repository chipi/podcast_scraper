#!/usr/bin/env python3
"""Integration tests for resume/incremental download behavior.

These tests verify that skip_existing and reuse_media flags work correctly
in component-level workflows (using workflow.run_pipeline()).

These tests use mocked HTTP responses and mocked providers for speed while
testing real file I/O and workflow orchestration.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper.rss import downloader
from podcast_scraper.workflow import orchestration as workflow

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly to avoid pytest resolution issues
import importlib.util

parent_conftest_path = tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

build_rss_xml_with_transcript = parent_conftest.build_rss_xml_with_transcript
create_rss_response = parent_conftest.create_rss_response
create_test_config = parent_conftest.create_test_config
create_transcript_response = parent_conftest.create_transcript_response


@pytest.mark.integration
class TestResumeBehaviorIntegration(unittest.TestCase):
    """Integration tests for resume/incremental download behavior."""

    def _mock_http_map(self, responses):
        """Create HTTP mock side effect function."""

        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            normalized = downloader.normalize_url(url)
            resp = responses.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect

    def test_multi_episode_skip_existing(self):
        """Test skip_existing with multiple episodes (integration-level).

        This test verifies that when skip_existing=True, episodes with
        existing transcripts are skipped and only new episodes are processed.
        """
        rss_url = "https://example.com/multi_episode_feed.xml"
        transcript_url1 = "https://example.com/ep1.txt"
        transcript_url2 = "https://example.com/ep2.txt"
        transcript_url3 = "https://example.com/ep3.txt"
        transcript_url4 = "https://example.com/ep4.txt"
        transcript_url5 = "https://example.com/ep5.txt"

        # Create RSS feed with 5 episodes, each with transcript URLs
        # Use proper podcast:transcript namespace
        rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">
    <channel>
        <title>Multi Episode Test Feed</title>
        <item>
            <title>Episode 1</title>
            <podcast:transcript url="{transcript_url1}" type="text/plain"/>
        </item>
        <item>
            <title>Episode 2</title>
            <podcast:transcript url="{transcript_url2}" type="text/plain"/>
        </item>
        <item>
            <title>Episode 3</title>
            <podcast:transcript url="{transcript_url3}" type="text/plain"/>
        </item>
        <item>
            <title>Episode 4</title>
            <podcast:transcript url="{transcript_url4}" type="text/plain"/>
        </item>
        <item>
            <title>Episode 5</title>
            <podcast:transcript url="{transcript_url5}" type="text/plain"/>
        </item>
    </channel>
</rss>"""

        transcript_text = "Test transcript content"
        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url1): create_transcript_response(
                transcript_text, transcript_url1
            ),
            downloader.normalize_url(transcript_url2): create_transcript_response(
                transcript_text, transcript_url2
            ),
            downloader.normalize_url(transcript_url3): create_transcript_response(
                transcript_text, transcript_url3
            ),
            downloader.normalize_url(transcript_url4): create_transcript_response(
                transcript_text, transcript_url4
            ),
            downloader.normalize_url(transcript_url5): create_transcript_response(
                transcript_text, transcript_url5
            ),
        }

        http_mock = self._mock_http_map(responses)

        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with tempfile.TemporaryDirectory() as tmpdir:
                # First run: Process first 2 episodes
                cfg1 = create_test_config(
                    rss_url=rss_url,
                    output_dir=tmpdir,
                    max_episodes=2,
                    skip_existing=False,
                    transcribe_missing=False,  # Use transcript downloads (faster)
                    generate_metadata=True,
                    generate_summaries=False,
                    auto_speakers=False,
                )

                # Run pipeline first time
                count1, summary1 = workflow.run_pipeline(cfg1)

                # Verify: 2 transcripts created
                transcript_files = list(Path(tmpdir).rglob("*.txt"))
                self.assertGreaterEqual(
                    len(transcript_files), 2, "First run should create at least 2 transcripts"
                )

                # Second run: Process all 5 episodes with skip_existing=True
                cfg2 = create_test_config(
                    rss_url=rss_url,
                    output_dir=tmpdir,
                    max_episodes=5,
                    skip_existing=True,  # Enable skip existing
                    transcribe_missing=False,
                    generate_metadata=True,
                    generate_summaries=False,
                    auto_speakers=False,
                )

                # Run pipeline second time
                count2, summary2 = workflow.run_pipeline(cfg2)

                # Verify: Total transcripts should be 5 (first 2 skipped, 3 new processed)
                transcript_files_after = list(Path(tmpdir).rglob("*.txt"))
                self.assertGreaterEqual(
                    len(transcript_files_after),
                    5,
                    "Second run should have 5 total transcripts",
                )

                # Verify: Count should reflect skipped episodes
                # (count2 should be less than 5 if some were skipped)
                self.assertGreaterEqual(
                    count2, 3, "Second run should process at least 3 new episodes"
                )

    def test_skip_existing_regenerate_summaries(self):
        """Test regenerating summaries on existing transcripts, skip if network unavailable.

        This test verifies that when skip_existing=True and generate_summaries=True,
        transcripts are skipped but summaries are still generated.
        """
        rss_url = "https://example.com/test_feed.xml"
        transcript_url1 = "https://example.com/ep1.txt"
        transcript_url2 = "https://example.com/ep2.txt"

        # Create RSS feed with 2 episodes, each with transcript URLs
        # Use proper podcast:transcript namespace
        rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">
    <channel>
        <title>Test Feed</title>
        <item>
            <title>Episode 1</title>
            <podcast:transcript url="{transcript_url1}" type="text/plain"/>
        </item>
        <item>
            <title>Episode 2</title>
            <podcast:transcript url="{transcript_url2}" type="text/plain"/>
        </item>
    </channel>
</rss>"""

        transcript_text = "Test transcript content for summarization"
        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url1): create_transcript_response(
                transcript_text, transcript_url1
            ),
            downloader.normalize_url(transcript_url2): create_transcript_response(
                transcript_text, transcript_url2
            ),
        }

        http_mock = self._mock_http_map(responses)

        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with tempfile.TemporaryDirectory() as tmpdir:
                # First run: Transcripts only (no summaries)
                cfg1 = create_test_config(
                    rss_url=rss_url,
                    output_dir=tmpdir,
                    max_episodes=2,
                    skip_existing=False,
                    transcribe_missing=False,
                    generate_metadata=False,  # No metadata initially
                    generate_summaries=False,  # No summaries initially
                    auto_speakers=False,
                )

                # Run pipeline first time
                count1, summary1 = workflow.run_pipeline(cfg1)

                # Verify: Transcripts exist, no metadata
                transcript_files = list(Path(tmpdir).rglob("*.txt"))
                self.assertGreaterEqual(len(transcript_files), 2, "Should have transcripts")

                metadata_files_before = list(Path(tmpdir).rglob("*.metadata.json"))
                self.assertEqual(
                    len(metadata_files_before),
                    0,
                    "First run should not create metadata files",
                )

                # Second run: skip_existing=True + generate_summaries=True
                cfg2 = create_test_config(
                    rss_url=rss_url,
                    output_dir=tmpdir,
                    max_episodes=2,
                    skip_existing=True,  # Skip existing transcripts
                    transcribe_missing=False,
                    generate_metadata=True,  # Generate metadata now
                    generate_summaries=True,  # Generate summaries now
                    auto_speakers=False,
                )

                # Run pipeline second time
                # Catch network errors (HuggingFace connection issues) and skip test
                try:
                    count2, summary2 = workflow.run_pipeline(cfg2)
                except (OSError, RuntimeError) as e:
                    error_msg = str(e).lower()
                    if (
                        "couldn't connect" in error_msg
                        or "huggingface.co" in error_msg
                        or "network" in error_msg
                    ):
                        import pytest

                        pytest.skip(f"Network unavailable or HuggingFace unreachable: {e}")
                    raise

                # Verify: Transcripts still exist (not regenerated)
                # Note: When generate_summaries=True, the workflow may process transcripts
                # for summary generation, but skip_existing should prevent re-downloading
                transcript_files_after = list(Path(tmpdir).rglob("*.txt"))
                # Transcript count should be at least the same (may be more if summaries
                # create additional files, but original transcripts should still exist)
                self.assertGreaterEqual(
                    len(transcript_files_after),
                    len(transcript_files),
                    "Transcript count should not decrease when skip_existing=True",
                )

                # Verify: Metadata files created (summaries generated)
                metadata_files_after = list(Path(tmpdir).rglob("*.metadata.json"))
                self.assertGreaterEqual(
                    len(metadata_files_after),
                    2,
                    "Second run should create metadata files with summaries",
                )
