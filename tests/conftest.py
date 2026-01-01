"""Shared fixtures and test utilities for podcast_scraper tests.

This module contains:
- Test constants
- Helper functions for creating test objects
- Mock classes and fixtures
- Shared test utilities
- Pytest hooks for validating marker behavior

All test files can import from this module using pytest's conftest.py mechanism.
"""

import argparse
import unittest.mock

# Bandit: tests construct safe XML elements
import xml.etree.ElementTree as ET  # nosec B405

import pytest

from podcast_scraper import config, models

# Test constants
TEST_BASE_URL = "https://example.com"
TEST_FEED_URL = "https://example.com/feed.xml"
TEST_PATH = "/path"
TEST_FULL_URL = f"{TEST_BASE_URL}{TEST_PATH}"
TEST_TRANSCRIPT_URL = f"{TEST_BASE_URL}/transcript.vtt"
TEST_TRANSCRIPT_URL_SRT = f"{TEST_BASE_URL}/transcript.srt"
TEST_MEDIA_URL = f"{TEST_BASE_URL}/episode.mp3"
TEST_RELATIVE_TRANSCRIPT = "transcripts/ep1.vtt"
TEST_RELATIVE_MEDIA = "episodes/ep1.mp3"
TEST_EPISODE_TITLE = "Episode Title"
TEST_EPISODE_TITLE_SPECIAL = "Episode: Title/With\\Special*Chars?"
TEST_FEED_TITLE = "Test Feed"
TEST_OUTPUT_DIR = "output"
TEST_CUSTOM_OUTPUT_DIR = "my_output"
TEST_RUN_ID = "test_run"
TEST_MEDIA_TYPE_MP3 = "audio/mpeg"
TEST_MEDIA_TYPE_M4A = "audio/m4a"
TEST_TRANSCRIPT_TYPE_VTT = "text/vtt"
TEST_TRANSCRIPT_TYPE_SRT = "text/srt"
TEST_CONTENT_TYPE_VTT = "text/vtt"
TEST_CONTENT_TYPE_SRT = "text/srt"


# Test helper functions
def create_test_args(**overrides):
    """Create test argparse.Namespace with defaults.

    Args:
        **overrides: Fields to override from defaults

    Returns:
        argparse.Namespace object with test defaults
    """
    defaults = {
        "rss": TEST_FEED_URL,
        "max_episodes": None,
        "timeout": 30,
        "delay_ms": 0,
        "transcribe_missing": False,
        "whisper_model": config.TEST_DEFAULT_WHISPER_MODEL,
        "screenplay": False,
        "screenplay_gap": 1.25,
        "num_speakers": 2,
        "speaker_names": "",
        "run_id": None,
        "log_level": "INFO",
        "workers": 1,
        "output_dir": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def create_test_config(**overrides):
    """Create test Config object with defaults.

    Args:
        **overrides: Fields to override from defaults

    Returns:
        config.Config object with test defaults
    """
    defaults = {
        "rss_url": TEST_FEED_URL,
        "output_dir": ".",
        "max_episodes": None,
        "user_agent": "test-agent",
        "timeout": 30,
        "delay_ms": 0,
        "prefer_types": [],
        "transcribe_missing": False,
        "whisper_model": config.TEST_DEFAULT_WHISPER_MODEL,
        "screenplay": False,
        "screenplay_gap_s": 1.0,
        "screenplay_num_speakers": 2,
        "screenplay_speaker_names": [],
        "run_id": None,
        "log_level": "INFO",
        "workers": 1,
        "skip_existing": False,
        "clean_output": False,
    }
    defaults.update(overrides)

    # Auto-enable generate_metadata if generate_summaries is True
    # (required by cross-field validation)
    if overrides.get("generate_summaries") and "generate_metadata" not in overrides:
        defaults["generate_metadata"] = True

    return config.Config(**defaults)


def create_test_feed(**overrides):
    """Create test RssFeed object with defaults.

    Args:
        **overrides: Fields to override from defaults

    Returns:
        models.RssFeed object with test defaults
    """
    defaults = {
        "title": TEST_FEED_TITLE,
        "items": [],
        "base_url": TEST_BASE_URL,
        "authors": ["Test Host"],
    }
    defaults.update(overrides)
    return models.RssFeed(**defaults)


def create_test_episode(**overrides):
    """Create test Episode object with defaults.

    Args:
        **overrides: Fields to override from defaults

    Returns:
        models.Episode object with test defaults
    """
    defaults = {
        "idx": 1,
        "title": TEST_EPISODE_TITLE,
        "title_safe": "Episode_Title",
        "item": ET.Element("item"),
        "transcript_urls": [(TEST_TRANSCRIPT_URL, TEST_TRANSCRIPT_TYPE_VTT)],
        "media_url": TEST_MEDIA_URL,
        "media_type": TEST_MEDIA_TYPE_MP3,
    }
    defaults.update(overrides)
    return models.Episode(**defaults)


def build_rss_xml_with_transcript(title, transcript_url, transcript_type="text/plain"):
    """Build RSS XML with transcript.

    Args:
        title: Feed title
        transcript_url: Transcript URL
        transcript_type: Transcript type

    Returns:
        RSS XML string
    """
    return f"""<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>{title}</title>
    <item>
      <title>Episode 1</title>
      <podcast:transcript url="{transcript_url}" type="{transcript_type}" />
    </item>
  </channel>
</rss>""".strip()


def build_rss_xml_with_media(title, media_url, media_type="audio/mpeg"):
    """Build RSS XML with media enclosure.

    Args:
        title: Feed title
        media_url: Media URL
        media_type: Media type

    Returns:
        RSS XML string
    """
    return f"""<?xml version='1.0'?>
<rss>
  <channel>
    <title>{title}</title>
    <item>
      <title>Episode 1</title>
      <enclosure url="{media_url}" type="{media_type}" />
    </item>
  </channel>
</rss>""".strip()


def build_rss_xml_with_speakers(title, authors=None, items=None):
    """Build RSS XML with speaker information.

    Args:
        title: Feed title
        authors: List of author names
        items: List of item dictionaries with title and description

    Returns:
        RSS XML string
    """
    author_tags = ""
    if authors:
        for author in authors:
            author_tags += f"    <author>{author}</author>\n"

    items_xml = ""
    if items:
        for item in items:
            item_title = item.get("title", "Episode")
            item_desc = item.get("description", "")
            items_xml += f"""    <item>
      <title>{item_title}</title>
      <description>{item_desc}</description>
    </item>
"""

    return f"""<?xml version='1.0'?>
<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>{title}</title>
{author_tags}{items_xml}  </channel>
</rss>""".strip()


def create_rss_response(rss_xml, url):
    """Create MockHTTPResponse for RSS feed.

    Args:
        rss_xml: RSS XML string
        url: Feed URL

    Returns:
        MockHTTPResponse object
    """
    return MockHTTPResponse(
        content=rss_xml.encode("utf-8"),
        url=url,
        headers={"Content-Type": "application/rss+xml"},
    )


def create_transcript_response(transcript_text, url, content_type="text/plain"):
    """Create MockHTTPResponse for transcript.

    Args:
        transcript_text: Transcript text content
        url: Transcript URL
        content_type: Content type header

    Returns:
        MockHTTPResponse object
    """
    return MockHTTPResponse(
        url=url,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(transcript_text.encode("utf-8"))),
        },
        chunks=[transcript_text.encode("utf-8")],
    )


def create_media_response(media_bytes, url, content_type="audio/mpeg"):
    """Create MockHTTPResponse for media file.

    Args:
        media_bytes: Media file bytes
        url: Media URL
        content_type: Content type header

    Returns:
        MockHTTPResponse object
    """
    return MockHTTPResponse(
        url=url,
        headers={"Content-Type": content_type, "Content-Length": str(len(media_bytes))},
        chunks=[media_bytes],
    )


def create_mock_spacy_model(entities=None):
    """Create mock spaCy model with entities.

    Args:
        entities: List of (text, label, score) tuples, or None for empty model

    Returns:
        Mock spaCy NLP model
    """
    mock_nlp = unittest.mock.MagicMock()
    mock_doc = unittest.mock.MagicMock()
    if entities:
        mock_ents = []
        for ent_text, label, score in entities:
            mock_ent = unittest.mock.MagicMock()
            mock_ent.text = ent_text
            mock_ent.label_ = label
            mock_ent.score = score
            mock_ents.append(mock_ent)
        mock_doc.ents = mock_ents
    else:
        mock_doc.ents = []
    mock_nlp.return_value = mock_doc
    return mock_nlp


class MockHTTPResponse:
    """Simple mock for HTTP responses used in integration-style tests."""

    def __init__(self, *, content=b"", url="", headers=None, chunks=None):
        self.content = content
        self.url = url
        self.headers = headers or {}
        self._chunks = chunks if chunks is not None else [content]

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1):
        for chunk in self._chunks:
            yield chunk

    def close(self):
        return None


def pytest_collection_modifyitems(config, items):
    """Validate that markers are working correctly.

    This hook checks that when running with explicit markers (e.g., -m integration),
    tests with those markers are actually collected. This helps catch configuration
    bugs like marker conflicts in addopts.
    """
    marker_expr = config.getoption("-m", default=None)

    # Only validate if an explicit marker expression is provided
    if marker_expr:
        # Check for integration marker
        if marker_expr == "integration":
            integration_items = [item for item in items if item.get_closest_marker("integration")]
            if not integration_items:
                pytest.fail(
                    "ERROR: Running with -m integration but no integration tests collected! "
                    "Check that:\n"
                    "  1. Tests have @pytest.mark.integration decorator\n"
                    "  2. addopts in pyproject.toml doesn't conflict with -m flags\n"
                    "  3. Marker configuration is correct"
                )

        # Check for e2e marker
        elif marker_expr == "e2e":
            e2e_items = [item for item in items if item.get_closest_marker("e2e")]
            if not e2e_items:
                pytest.fail(
                    "ERROR: Running with -m e2e but no e2e tests collected! "
                    "Check that:\n"
                    "  1. Tests have @pytest.mark.e2e decorator\n"
                    "  2. addopts in pyproject.toml doesn't conflict with -m flags\n"
                    "  3. Marker configuration is correct"
                )

        # Check for "not network" marker (common in test-all)
        elif marker_expr == "not network":
            # Should collect tests that don't have network marker
            non_network_items = [item for item in items if not item.get_closest_marker("network")]
            if not non_network_items:
                pytest.fail(
                    "ERROR: Running with -m 'not network' but no tests collected! "
                    "Check marker configuration."
                )
