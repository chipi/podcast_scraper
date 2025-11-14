#!/usr/bin/env python3
"""Tests for podcast_scraper package."""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

# Import shared test utilities from conftest
# Note: pytest automatically loads conftest.py, but we need explicit imports for unittest
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET  # nosec B405 - tests construct safe XML elements
from pathlib import Path
from unittest.mock import patch

import requests
from platformdirs import user_cache_dir, user_data_dir

import podcast_scraper
import podcast_scraper.cli as cli
from podcast_scraper import (
    config,
    downloader,
    episode_processor,
    filesystem,
    metadata,
    models,
    progress,
    rss_parser,
    speaker_detection,
    whisper_integration as whisper,
)

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import (  # noqa: F401, E402
    TEST_BASE_URL,
    TEST_CONTENT_TYPE_SRT,
    TEST_CONTENT_TYPE_VTT,
    TEST_CUSTOM_OUTPUT_DIR,
    TEST_EPISODE_TITLE,
    TEST_EPISODE_TITLE_SPECIAL,
    TEST_FEED_TITLE,
    TEST_FEED_URL,
    TEST_FULL_URL,
    TEST_MEDIA_TYPE_M4A,
    TEST_MEDIA_TYPE_MP3,
    TEST_MEDIA_URL,
    TEST_OUTPUT_DIR,
    TEST_PATH,
    TEST_RELATIVE_MEDIA,
    TEST_RELATIVE_TRANSCRIPT,
    TEST_RUN_ID,
    TEST_TRANSCRIPT_TYPE_SRT,
    TEST_TRANSCRIPT_TYPE_VTT,
    TEST_TRANSCRIPT_URL,
    TEST_TRANSCRIPT_URL_SRT,
    MockHTTPResponse,
    build_rss_xml_with_media,
    build_rss_xml_with_speakers,
    build_rss_xml_with_transcript,
    create_media_response,
    create_mock_spacy_model,
    create_rss_response,
    create_test_args,
    create_test_config,
    create_test_episode,
    create_test_feed,
    create_transcript_response,
)

# All test classes have been moved to feature-specific test files:
# - RSS parser tests: tests/test_rss_parser.py
# - Filesystem tests: tests/test_filesystem.py
# - Downloader tests: tests/test_downloader.py
# - CLI tests: tests/test_cli.py
# - Utilities tests: tests/test_utilities.py
# - Metadata tests: tests/test_metadata.py
# - Speaker detection tests: tests/test_speaker_detection.py
# - Integration/E2E tests: tests/test_integration.py
