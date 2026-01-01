#!/usr/bin/env python3
"""E2E Infrastructure Validation Tests.

These tests validate that the E2E test infrastructure is set up correctly:
- E2E HTTP server is running and serving files
- Fast mode RSS feed limiting works
- Fast mode episode limiting works
- Fast test fixtures (RSS, audio, transcript) exist and are correct
- Server URLs are accessible

These tests should run FIRST to ensure the E2E infrastructure is working
before running other E2E tests.
"""

import os
import sys
from pathlib import Path

import pytest
import requests

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)


@pytest.mark.e2e
@pytest.mark.critical_path
class TestE2EServerInfrastructure:
    """Test E2E server infrastructure setup."""

    def test_e2e_server_is_running(self, e2e_server):
        """Test that E2E server is running and accessible."""
        base_url = e2e_server.urls.base()

        # Test server is accessible
        response = requests.get(base_url, timeout=5)
        assert response.status_code in [
            200,
            404,
        ], f"Server should be accessible, got {response.status_code}"

    def test_fast_rss_feed_exists(self, e2e_server):
        """Test that fast RSS feed (p01_fast.xml) exists and is served."""
        # In fast mode, podcast1 should serve p01_fast.xml
        rss_url = e2e_server.urls.feed("podcast1")

        response = requests.get(rss_url, timeout=5)
        assert (
            response.status_code == 200
        ), f"Fast RSS feed should be accessible, got {response.status_code}"

        # Verify it's the fast feed (should contain "Fast Test" in title)
        content = response.text
        assert "Fast Test" in content or "p01_fast" in content, "Should serve fast RSS feed"
        assert "p01_e01_fast" in content, "Fast feed should contain fast episode"

    def test_fast_audio_file_exists(self, e2e_server):
        """Test that fast audio file (p01_e01_fast.mp3) exists and is served."""
        audio_url = e2e_server.urls.audio("p01_e01_fast")

        response = requests.get(audio_url, timeout=5, stream=True)
        assert (
            response.status_code == 200
        ), f"Fast audio file should be accessible, got {response.status_code}"

        # Verify it's actually an audio file (check Content-Type or file size)
        content_type = response.headers.get("Content-Type", "")
        assert (
            "audio" in content_type.lower() or "mpeg" in content_type.lower()
        ), f"Should be audio file, got Content-Type: {content_type}"

        # Verify file size (should be ~469KB for 1-minute audio)
        content_length = response.headers.get("Content-Length")
        if content_length:
            size = int(content_length)
            assert 400000 <= size <= 500000, f"Fast audio should be ~469KB, got {size} bytes"

    def test_fast_transcript_file_exists(self, e2e_server):
        """Test that fast transcript file (p01_e01_fast.txt) exists and is served."""
        transcript_url = e2e_server.urls.transcript("p01_e01_fast")

        response = requests.get(transcript_url, timeout=5)
        assert (
            response.status_code == 200
        ), f"Fast transcript file should be accessible, got {response.status_code}"

        # Verify it's actually text
        content = response.text
        assert len(content) > 0, "Transcript should not be empty"
        assert len(content) < 5000, "Fast transcript should be short (~1 minute of content)"

    def test_feed_limiting_in_fast_mode(self, e2e_server, request):
        """Test that feed limiting works in fast mode (only podcast1 available)."""
        # In fast mode, only podcast1 should be available
        podcast1_url = e2e_server.urls.feed("podcast1")
        response = requests.get(podcast1_url, timeout=5)
        assert response.status_code == 200, "podcast1 should be available in fast mode"

        # Other podcasts should return 404 in fast mode
        # Note: This test assumes we're in fast mode (not marked as slow)
        # If we're in slow mode, this test should be skipped or adjusted
        is_slow = request.node.get_closest_marker("slow") is not None
        if not is_slow:
            # Fast mode: other podcasts should be blocked
            podcast2_url = e2e_server.urls.feed("podcast2")
            response2 = requests.get(podcast2_url, timeout=5)
            # In fast mode, podcast2 should return 404
            assert (
                response2.status_code == 404
            ), f"In fast mode, podcast2 should return 404, got {response2.status_code}"

    def test_fast_rss_feed_has_correct_structure(self, e2e_server):
        """Test that fast RSS feed has correct structure (1 episode, 1 minute duration)."""
        rss_url = e2e_server.urls.feed("podcast1")

        response = requests.get(rss_url, timeout=5)
        assert response.status_code == 200

        content = response.text

        # Should have exactly 1 episode
        item_count = content.count("<item>")
        assert item_count == 1, f"Fast feed should have 1 episode, got {item_count}"

        # Should have 1-minute duration
        assert "00:01:00" in content or "1:00" in content, "Fast episode should be 1 minute"

        # Should reference fast audio file
        assert "p01_e01_fast.mp3" in content, "Should reference fast audio file"

    def test_fast_fixtures_exist_on_disk(self):
        """Test that fast test fixtures exist in the fixtures directory."""
        tests_dir = Path(__file__).parent.parent
        fixtures_dir = tests_dir / "fixtures"

        # Check fast RSS feed
        fast_rss = fixtures_dir / "rss" / "p01_fast.xml"
        assert fast_rss.exists(), f"Fast RSS feed should exist: {fast_rss}"

        # Check fast audio file
        fast_audio = fixtures_dir / "audio" / "p01_e01_fast.mp3"
        assert fast_audio.exists(), f"Fast audio file should exist: {fast_audio}"

        # Check file size (should be ~469KB)
        if fast_audio.exists():
            size = fast_audio.stat().st_size
            assert 400000 <= size <= 500000, f"Fast audio should be ~469KB, got {size} bytes"

        # Check fast transcript file
        fast_transcript = fixtures_dir / "transcripts" / "p01_e01_fast.txt"
        assert fast_transcript.exists(), f"Fast transcript file should exist: {fast_transcript}"

        # Check transcript length (should be short, ~1 minute of content)
        if fast_transcript.exists():
            content = fast_transcript.read_text(encoding="utf-8")
            assert len(content) > 0, "Fast transcript should not be empty"
            assert len(content) < 5000, "Fast transcript should be short (~1 minute)"

    def test_e2e_server_urls_helper(self, e2e_server):
        """Test that E2E server URLs helper works correctly."""
        urls = e2e_server.urls

        # Test feed URL
        feed_url = urls.feed("podcast1")
        assert feed_url.startswith("http://"), "Feed URL should be HTTP"
        assert (
            "podcast1" in feed_url or "feed.xml" in feed_url
        ), "Feed URL should reference podcast1"

        # Test audio URL
        audio_url = urls.audio("p01_e01_fast")
        assert audio_url.startswith("http://"), "Audio URL should be HTTP"
        assert (
            "p01_e01_fast" in audio_url or ".mp3" in audio_url
        ), "Audio URL should reference episode"

        # Test transcript URL
        transcript_url = urls.transcript("p01_e01_fast")
        assert transcript_url.startswith("http://"), "Transcript URL should be HTTP"
        assert (
            "p01_e01_fast" in transcript_url or ".txt" in transcript_url
        ), "Transcript URL should reference episode"
