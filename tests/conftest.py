"""Shared fixtures and test utilities for podcast_scraper tests.

This module contains:
- Test constants
- Helper functions for creating test objects
- Mock classes and fixtures
- Shared test utilities
- Pytest hooks for validating marker behavior

All test files can import from this module using pytest's conftest.py mechanism.
"""

# Suppress rich progress bars in tests to keep output clean
# Must be set BEFORE any rich imports
import os

os.environ["TERM"] = "dumb"  # Disable rich terminal features

# Force Hugging Face libraries to work offline (use only cached models)
# This prevents network access attempts that would fail with pytest-socket blocking
# Must be set BEFORE any transformers/huggingface_hub imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
        "output_dir": TEST_OUTPUT_DIR,
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
        # NER model: use test default (small, fast) explicitly for safety
        "ner_model": config.TEST_DEFAULT_NER_MODEL,  # Test default: en_core_web_sm
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


def cleanup_model(model):
    """Helper function to ensure a SummaryModel is properly cleaned up.

    This is a convenience function for tests that create SummaryModel instances
    directly. The automatic cleanup fixture will also clean up models, but
    explicit cleanup in tests is recommended for clarity and immediate memory
    release.

    Args:
        model: SummaryModel instance to clean up, or None (no-op if None)

    Example:
        def test_something():
            model = summarizer.SummaryModel(...)
            try:
                # test code
            finally:
                cleanup_model(model)  # Explicit cleanup
    """
    if model is None:
        return

    try:
        from podcast_scraper.providers.ml import summarizer

        summarizer.unload_model(model)
    except (ImportError, AttributeError):
        # ML modules not available (e.g., in unit tests without ML dependencies)
        pass
    except Exception:
        # Ignore cleanup errors (model may already be cleaned up)
        pass


def cleanup_provider(provider):
    """Helper function to ensure a provider is properly cleaned up.

    This is a convenience function for tests that create provider instances
    directly. The automatic cleanup fixture will also clean up providers, but
    explicit cleanup in tests is recommended for clarity and immediate memory
    release.

    Args:
        provider: Provider instance (MLProvider, etc.) to clean up, or None (no-op if None)

    Example:
        def test_something():
            provider = create_summarization_provider(cfg)
            try:
                # test code
            finally:
                cleanup_provider(provider)  # Explicit cleanup
    """
    if provider is None:
        return

    try:
        if hasattr(provider, "cleanup"):
            provider.cleanup()
    except Exception:
        # Ignore cleanup errors (provider may already be cleaned up)
        pass


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


# Removed automatic process cleanup (Issue #351) - it was over-engineered and caused
# problems with pytest-xdist workers. If you need to clean up leftover test processes,
# use the manual script: scripts/tools/cleanup_test_processes.sh
#
# Reasons for removal:
# 1. Unix/Linux only (uses pkill, doesn't work on Windows)
# 2. Risky - could kill current test run's workers
# 3. Unnecessary in CI (clean environments)
# 4. Unnecessary in local dev (users can manually clean up if needed)
# 5. Caused "OSError: cannot send (already closed?)" during pytest_sessionfinish


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


def _cleanup_ml_set_env_and_torch(monkeypatch) -> None:
    """Set HF hub and thread env vars; limit torch threads if already imported."""
    import os
    import sys

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS"):
        os.environ.setdefault(key, "1")
        monkeypatch.setenv(key, "1")
    if "torch" in sys.modules:
        try:
            import torch

            if hasattr(torch, "set_num_threads"):
                try:
                    torch.set_num_threads(1)
                except RuntimeError:
                    pass
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
        except ImportError:
            pass


def _cleanup_ml_reset_preloaded_before() -> None:
    """Reset workflow._preloaded_ml_provider to None before test."""
    try:
        from podcast_scraper import workflow

        workflow._preloaded_ml_provider = None
    except ImportError:
        pass


def _cleanup_ml_reset_preloaded_after() -> None:
    """Cleanup and reset workflow._preloaded_ml_provider after test."""
    try:
        from podcast_scraper import workflow

        if workflow._preloaded_ml_provider is not None:
            try:
                workflow._preloaded_ml_provider.cleanup()
            except Exception:
                pass
            workflow._preloaded_ml_provider = None
    except ImportError:
        pass


def _cleanup_ml_find_and_clean_models() -> None:
    """Find SummaryModel/MLProvider instances via gc and clean them (non-parallel only)."""
    import os

    if os.environ.get("PYTEST_XDIST_WORKER") is not None:
        return
    try:
        from podcast_scraper.providers.ml import summarizer
        from podcast_scraper.providers.ml.ml_provider import MLProvider

        all_objects = gc.get_objects()
        summary_models = [
            obj
            for obj in all_objects
            if isinstance(obj, summarizer.SummaryModel) and obj.model is not None
        ]
        providers = [
            obj for obj in all_objects if isinstance(obj, MLProvider) and obj.is_initialized
        ]
        for model in summary_models:
            try:
                summarizer.unload_model(model)
            except Exception:
                pass
        for provider in providers:
            try:
                from podcast_scraper import workflow

                if provider is not workflow._preloaded_ml_provider:
                    provider.cleanup()
            except Exception:
                pass
    except (ImportError, AttributeError):
        pass


def _cleanup_ml_gc_after_test() -> None:
    """Run GC (and optionally torch cache clear) after test for integration/e2e."""
    import os

    test_name = os.environ.get("PYTEST_CURRENT_TEST", "")
    if "test_integration" not in test_name and "test_e2e" not in test_name:
        return
    try:
        is_parallel = os.environ.get("PYTEST_XDIST_WORKER") is not None
        if is_parallel:
            for _ in range(3):
                gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
            except (ImportError, AttributeError):
                pass
        else:
            gc.collect()
    except Exception:
        pass


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
    if _is_unit_test_safe():
        yield
        return
    monkeypatch = request.getfixturevalue("monkeypatch")
    _cleanup_ml_set_env_and_torch(monkeypatch)
    _cleanup_ml_reset_preloaded_before()
    yield
    _cleanup_ml_reset_preloaded_after()
    _cleanup_ml_find_and_clean_models()
    _cleanup_ml_gc_after_test()
