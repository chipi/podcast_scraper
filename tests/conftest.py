"""Shared fixtures and test utilities for podcast_scraper tests.

This module contains:
- Test constants
- Helper functions for creating test objects
- Mock classes and fixtures
- Shared test utilities
- Pytest hooks for validating marker behavior

All test files can import from this module using pytest's conftest.py mechanism.
"""

# Disable tqdm progress bars in tests to prevent hangs with -s flag
# and pytest-xdist parallel execution. Must be set BEFORE any tqdm imports.
# See: https://github.com/chipi/podcast_scraper/issues/176
import os

os.environ["TQDM_DISABLE"] = "1"

# Force Hugging Face libraries to work offline (use only cached models)
# This prevents network access attempts that would fail with pytest-socket blocking
# Must be set BEFORE any transformers/huggingface_hub imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# DEBUG: Print cache state inside pytest (before any imports that might affect it)
import sys
from pathlib import Path

_hf_home = os.environ.get("HF_HOME", "NOT SET")
_hf_cache = os.environ.get("HF_HUB_CACHE", "NOT SET")
_cache_path = Path(_hf_cache) if _hf_cache != "NOT SET" else Path.home() / ".cache" / "huggingface" / "hub"
print(f"\n[conftest.py DEBUG] HF_HOME={_hf_home}", file=sys.stderr)
print(f"[conftest.py DEBUG] HF_HUB_CACHE={_hf_cache}", file=sys.stderr)
print(f"[conftest.py DEBUG] Cache path: {_cache_path}", file=sys.stderr)
print(f"[conftest.py DEBUG] Cache path exists: {_cache_path.exists()}", file=sys.stderr)
if _cache_path.exists():
    print(f"[conftest.py DEBUG] Contents: {list(_cache_path.iterdir())[:5]}", file=sys.stderr)
else:
    print(f"[conftest.py DEBUG] Cache directory does NOT exist!", file=sys.stderr)
    # Try to list parent directories to understand the issue
    _parent = _cache_path.parent
    print(f"[conftest.py DEBUG] Parent ({_parent}): exists={_parent.exists()}", file=sys.stderr)
    if _parent.exists():
        print(f"[conftest.py DEBUG] Parent contents: {list(_parent.iterdir())}", file=sys.stderr)
    _grandparent = _parent.parent
    print(f"[conftest.py DEBUG] Grandparent ({_grandparent}): exists={_grandparent.exists()}", file=sys.stderr)
    if _grandparent.exists():
        print(f"[conftest.py DEBUG] Grandparent contents: {list(_grandparent.iterdir())}", file=sys.stderr)

import argparse
import gc
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
        "whisper_model": config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
        "screenplay": False,
        "screenplay_gap_s": 1.0,
        "screenplay_num_speakers": 2,
        "screenplay_speaker_names": [],
        "run_id": None,
        "log_level": "INFO",
        "workers": 1,
        "skip_existing": False,
        "clean_output": False,
        # Summary models: use test defaults (small, fast) unless explicitly overridden
        # Tests that need to test production behavior can override with summary_model=None
        "summary_model": config.TEST_DEFAULT_SUMMARY_MODEL,  # Test default: bart-base
        "summary_reduce_model": config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,  # Test default: LED-base
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


def _is_unit_test_safe() -> bool:
    """Safely check if current test is a unit test without accessing request object.

    This function only uses environment variables to avoid hangs when -s flag is used.
    """
    import os

    test_name = os.environ.get("PYTEST_CURRENT_TEST", "")
    return "/unit/" in test_name


@pytest.fixture(autouse=True, scope="function")
def cleanup_ml_resources_after_test(request):
    """Ensure ML resources are cleaned up after each test.

    This fixture runs automatically after each test to:
    - Limit PyTorch thread pools to prevent excessive thread spawning
    - Force garbage collection to clean up any lingering model references
    - Help prevent memory leaks and thread accumulation in parallel test execution

    This is especially important when running tests in parallel with pytest-xdist,
    where multiple worker processes load ML models simultaneously.

    PyTorch/Transformers can spawn many threads per model, so we limit them:
    - OMP_NUM_THREADS: OpenMP threads (used by PyTorch)
    - MKL_NUM_THREADS: Intel MKL threads (if available)
    - TORCH_NUM_THREADS: PyTorch CPU threads

    Note: This fixture skips all logic for unit tests to avoid hangs.
    Unit tests don't load real ML models, so they don't need this cleanup.

    WARNING: When using pytest with `-s` or `--capture=no` flags, unit tests may
    hang due to pytest's fixture parameter resolution. This is a pytest behavior
    issue, not a test logic problem. Tests pass normally without these flags.

    WORKAROUND: Use `-v` (verbose) instead of `-s` for better output without hangs:
        pytest tests/unit/ -v  # Works fine
        pytest tests/unit/ -s  # May hang

    IMPORTANT: When using pytest with `-s` or `--capture=no` flags, accessing
    request.node attributes can hang. This fixture checks PYTEST_CURRENT_TEST
    environment variable FIRST (before accessing request) to avoid hangs.
    """
    import os
    import sys

    # CRITICAL: Check if unit test FIRST using safe method (env var only)
    # This MUST be done before accessing request or monkeypatch to avoid hangs with -s flag
    is_unit_test = _is_unit_test_safe()

    # For unit tests, exit immediately - don't even request monkeypatch fixture
    # This prevents pytest from trying to resolve fixture parameters which can hang
    if is_unit_test:
        yield
        return

    # For non-unit tests, we need monkeypatch, so request it now
    # This is safe because we've already determined it's not a unit test
    monkeypatch = request.getfixturevalue("monkeypatch")

    # Disable Hugging Face Hub progress bars to avoid misleading "Downloading" messages
    # when loading models from cache. This is especially important in test environments
    # where network is blocked and progress bars can be confusing.
    # Set this early before any transformers imports
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Limit PyTorch thread pools to prevent excessive thread spawning
    # Set to 1 thread per worker to minimize resource usage in parallel tests
    # This prevents PyTorch from spawning many threads per model
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")

    # Also set via monkeypatch to ensure it applies even if already imported
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("TORCH_NUM_THREADS", "1")

    # Try to set PyTorch thread count directly if already imported
    # Note: set_num_interop_threads can only be called once per process,
    # so we catch RuntimeError if it's already been set
    # Only check if torch is already imported (don't import it ourselves to save memory)
    # Skip this for unit tests to avoid any potential hangs
    # Reuse is_unit_test from earlier check
    if "torch" in sys.modules and not is_unit_test:
        try:
            import torch

            if hasattr(torch, "set_num_threads"):
                try:
                    torch.set_num_threads(1)
                except RuntimeError:
                    pass  # Already set, ignore
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass  # Already set or parallel work started, ignore
        except ImportError:
            pass  # PyTorch not available, skip

    # Reset preloaded ML provider global BEFORE test runs
    # This ensures each test starts with clean state and prevents cross-test interference
    # in parallel execution (pytest-xdist) where the global could be shared
    # See: https://github.com/chipi/podcast_scraper/issues/177 (flaky tests)
    try:
        from podcast_scraper import workflow

        workflow._preloaded_ml_provider = None
    except ImportError:
        pass  # workflow module not available

    # Run the test
    yield

    # Reset preloaded ML provider global AFTER test completes
    # This prevents one test's cleanup from affecting another test that's still running
    try:
        from podcast_scraper import workflow

        if workflow._preloaded_ml_provider is not None:
            try:
                workflow._preloaded_ml_provider.cleanup()
            except Exception:
                pass  # Ignore cleanup errors
            workflow._preloaded_ml_provider = None
    except ImportError:
        pass  # workflow module not available

    # After test completes, force garbage collection
    # This helps clean up any ML models that weren't explicitly cleaned up
    # and releases threads that might be holding references
    # Note: Single GC round is more efficient - multiple rounds can actually
    # cause memory fragmentation and keep objects alive longer
    # Skip gc.collect() for unit tests to avoid hangs from mock finalizers
    # Unit tests should clean up explicitly, and integration/E2E tests will benefit from GC
    # Only run GC for integration/E2E tests, skip for unit tests
    # Check test name from env var (safer than request.node)
    test_name_after = os.environ.get("PYTEST_CURRENT_TEST", "")
    if not is_unit_test and (
        "test_integration" in test_name_after or "test_e2e" in test_name_after
    ):
        try:
            gc.collect()
        except Exception:
            # Ignore any errors from GC (e.g., finalizer issues)
            pass
