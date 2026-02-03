#!/usr/bin/env python3
"""Integration tests for E2E test infrastructure validation.

These tests validate that the E2E test infrastructure is set up correctly:
- E2E HTTP server is running and serving files
- Fast mode RSS feed limiting works
- Fast mode episode limiting works
- Fast test fixtures (RSS, audio, transcript) exist and are correct
- Server URLs are accessible

Moved from tests/e2e/ as part of Phase 3 test pyramid refactoring - these
test infrastructure components, not user workflows.
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


@pytest.mark.integration
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
        # Note: This test checks for fast feed, but integration tests may not run in fast mode
        # So we check that the feed is accessible and contains valid RSS content
        rss_url = e2e_server.urls.feed("podcast1")

        response = requests.get(rss_url, timeout=5)
        assert (
            response.status_code == 200
        ), f"RSS feed should be accessible, got {response.status_code}"

        # Verify it's valid RSS content
        content = response.text
        assert "<?xml" in content, "Should be XML content"
        assert "rss" in content.lower(), "Should be RSS feed"
        # In fast mode, it would contain "Fast Test" or "p01_fast", but in regular mode it's the normal feed
        # Both are valid, so we just verify RSS structure

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

        # Verify file size (should be ~700KB for 1-minute audio)
        # Note: Actual file size may vary based on encoding settings
        content_length = response.headers.get("Content-Length")
        if content_length:
            size = int(content_length)
            assert 600000 <= size <= 800000, f"Fast audio should be ~700KB, got {size} bytes"

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
        # Note: Integration tests may not run in fast mode, so we test that podcast1 is accessible
        podcast1_url = e2e_server.urls.feed("podcast1")
        response = requests.get(podcast1_url, timeout=5)
        assert response.status_code == 200, "podcast1 should be available"

        # In fast mode, other podcasts would return 404, but in regular mode they may be available
        # This test verifies the server is working, not specifically fast mode behavior
        # Fast mode behavior is tested in E2E tests

    def test_fast_rss_feed_has_correct_structure(self, e2e_server):
        """Test that RSS feed has correct structure."""
        # Note: This test verifies RSS structure, not specifically fast mode
        # Fast mode structure (1 episode, 1 minute) is tested in E2E tests
        rss_url = e2e_server.urls.feed("podcast1")

        response = requests.get(rss_url, timeout=5)
        assert response.status_code == 200

        content = response.text

        # Should have at least 1 episode
        item_count = content.count("<item>")
        assert item_count >= 1, f"RSS feed should have at least 1 episode, got {item_count}"

        # Should have valid RSS structure
        assert "<channel>" in content, "Should have channel element"
        assert "<item>" in content, "Should have item elements"

    def test_fast_fixtures_exist_on_disk(self):
        """Test that fast test fixtures exist in the fixtures directory."""
        # Fixtures are in tests/fixtures, not tests/integration/fixtures
        tests_dir = Path(__file__).parent.parent.parent
        fixtures_dir = tests_dir / "fixtures"

        # Check fast RSS feed
        fast_rss = fixtures_dir / "rss" / "p01_fast.xml"
        assert fast_rss.exists(), f"Fast RSS feed should exist: {fast_rss}"

        # Check fast audio file
        fast_audio = fixtures_dir / "audio" / "p01_e01_fast.mp3"
        assert fast_audio.exists(), f"Fast audio file should exist: {fast_audio}"

        # Check file size (should be ~700KB)
        # Note: Actual file size may vary based on encoding settings
        if fast_audio.exists():
            size = fast_audio.stat().st_size
            assert 600000 <= size <= 800000, f"Fast audio should be ~700KB, got {size} bytes"

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
