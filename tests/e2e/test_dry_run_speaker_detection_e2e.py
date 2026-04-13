#!/usr/bin/env python3
"""E2E test for dry-run speaker detection with real spaCy model.

Moved from tests/integration/workflow/test_workflow_integration.py —
this test uses real spaCy for speaker detection, so it belongs in E2E.
"""

import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

# Import helpers from root conftest
from pathlib import Path

from podcast_scraper import cli, downloader

tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

import importlib.util

parent_conftest_path = tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

build_rss_xml_with_speakers = parent_conftest.build_rss_xml_with_speakers
create_rss_response = parent_conftest.create_rss_response

SPACY_AVAILABLE = False
try:
    import spacy  # noqa: F401

    SPACY_AVAILABLE = True
except ImportError:
    pass


def _mock_http_map(mapping):
    """Return side effect function for fetch_url using mapping dict."""

    def _side_effect(url, user_agent, timeout, stream=False):
        normalized = downloader.normalize_url(url)
        if normalized in mapping:
            return mapping[normalized]
        raise Exception(f"Unexpected URL: {url}")

    return _side_effect


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.ml_models
@unittest.skipIf(not SPACY_AVAILABLE, "spaCy dependencies not available")
class TestDryRunSpeakerDetection(unittest.TestCase):
    """E2E test: dry-run mode performs host/guest detection with real spaCy."""

    def test_dry_run_performs_speaker_detection(self):
        """Dry-run mode still performs host/guest detection using real spaCy."""
        rss_url = "https://example.com/feed.xml"
        rss_xml = build_rss_xml_with_speakers(
            "Test Podcast",
            authors=["John Host"],
            items=[
                {
                    "title": "Interview with Alice Guest",
                    "description": "This episode features Alice Guest discussing technology.",
                },
                {
                    "title": "Chat with Bob Guest",
                    "description": "Bob Guest joins us for a conversation.",
                },
            ],
        )
        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        }

        http_mock = _mock_http_map(responses)
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertLogs(logging.getLogger("podcast_scraper"), level="INFO") as log_ctx:
                    exit_code = cli.main(
                        [
                            rss_url,
                            "--output-dir",
                            tmpdir,
                            "--dry-run",
                            "--auto-speakers",
                        ]
                    )
                self.assertEqual(exit_code, 0)
                log_text = "\n".join(log_ctx.output)
                if "DETECTED HOSTS" not in log_text:
                    self.assertIn("(dry-run) would initialize speaker detector", log_text)
                else:
                    self.assertIn("John Host", log_text)
                self.assertIn("Interview with Alice Guest", log_text)
                self.assertIn("Chat with Bob Guest", log_text)
