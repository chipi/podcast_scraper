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
from pathlib import Path

os.environ["TERM"] = "dumb"  # Disable rich terminal features

# Prefer repo-local Hugging Face hub when ``make preload-ml-models`` has populated
# ``.cache/huggingface/hub`` (matches ``get_transformers_cache_dir`` tier-3). Without
# this, libraries default to ``~/.cache/...`` and offline tests miss the preload tree.
_project_hf_hub = Path(__file__).resolve().parents[1] / ".cache" / "huggingface" / "hub"
if _project_hf_hub.is_dir():
    os.environ.setdefault("HF_HUB_CACHE", str(_project_hf_hub))

# Force Hugging Face libraries to work offline (use only cached models)
# This prevents network access attempts that would fail with pytest-socket blocking
# Must be set BEFORE any transformers/huggingface_hub imports
# ML integration tests (summarization, GIL evidence stack) gate on cache probes such as
# tests.integration.ml_model_cache_helpers — not on disabling these flags.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Tests default to the `test_default` profile so model defaults stay cheap/fast.
# Replaces the old `_is_pytest_run()` + 24 `_get_default_*` env-detect
# machinery removed 2026-06-23 (see commit history).
#
# Profiles are now the source of truth: `config/profiles/test_default.yaml`
# pins all the test-tier cheap models that the old TEST_DEFAULT_* constants
# used to provide via runtime env-detection. Individual tests can still
# override per-field by passing explicit kwargs to Config(...), or override
# the profile itself by setting profile=... in their data dict.
os.environ.setdefault("PODCAST_SCRAPER_PROFILE", "test_default")

# Ollama unit tests replace sys.modules["httpx"] with a MagicMock at import time.
# Starlette's testclient defines WebSocketDenialResponse(httpx.Response, ...) at import;
# if that runs after httpx is mocked, Python raises a metaclass conflict. Preload once
# while the real httpx module is still in sys.modules (see tests/.../ollama/test_*.py).
try:
    import starlette.testclient  # noqa: F401
except ImportError:
    pass

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


# Imported HERE, at collection time, and deliberately not inside the fixture below.
#
# Several provider test modules install a module-level ``patch.dict(sys.modules, …)``; patch.dict
# restores its snapshot on exit, which DELETES every key added while it was active (see the long
# note in tests/integration/conftest.py). A module first imported inside that window is therefore
# evicted. For most pure-Python modules that is harmless — the next import re-executes them — but
# workflow.run_budget holds the process-wide spend ledger in a MODULE-LEVEL singleton, so a
# re-import silently swaps in a fresh ledger reading $0.00 spent. Importing at collection puts it
# in sys.modules before any test's patch window opens.
from podcast_scraper.workflow.run_budget import reset_run_budget as _reset_the_run_budget

#: LLM SDKs. NOT core dependencies — they live in the ``[llm]`` extra, so CI's unit job
#: (``pip install -e ".[dev]"``) does not have them and test modules legitimately stub them
#: there. The rule is therefore conditional: replacing one of these with a Mock is a defect only
#: when the real package IS installed, because then the Mock is shadowing something that works.
#: I previously read these as core dependencies, deleted the stubs on that basis, and broke CI's
#: unit job for five commits — the check below is written to make that specific error loud.
_CORE_SDKS = ("openai", "anthropic", "google.genai")


#: Modules that were genuinely importable when this conftest loaded — i.e. BEFORE any test
#: module had a chance to stub anything. A stub standing in for a name that is absent here is
#: legitimate (CI's unit job installs `.[dev]` only, so the whole [llm] extra is missing and the
#: stub is the only reason those modules import). A stub SHADOWING a name that is present here
#: is the #1799 bug.
def _really_installed(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


_REAL_AT_START: frozenset = frozenset(
    n
    for n in ("openai", "anthropic", "google.genai", "spacy", "torch", "transformers", "whisper")
    if _really_installed(n)
)


def _mocked_modules() -> set:
    """Names in ``sys.modules`` currently standing in for a real module via a Mock.

    ``sys.modules`` is mutated by ANY thread that imports, and this runs from a pytest hook while
    tests may still have executors alive —
    ``test_get_run_summary_returns_payload_after_executor_run`` does exactly that. Even
    ``list(sys.modules.items())`` raises
    ``RuntimeError: dictionary changed size during iteration`` when an import lands mid-copy, and
    from a hook that surfaces as pytest ``INTERNALERROR`` which kills the whole xdist worker —
    turning a diagnostic into a suite-wide outage. It did, on CI, in the run that added it.

    Retry the snapshot a few times, then give up and report nothing. A guard must never be the
    reason a run fails; the worst acceptable outcome is that it misses a stub this once.
    """
    import sys as _sys

    for _ in range(5):
        try:
            snapshot = list(_sys.modules.items())
        except RuntimeError:  # another thread imported mid-copy
            continue
        return {name for name, mod in snapshot if mod is not None and "Mock" in type(mod).__name__}
    return set()


#: module name -> the test that first introduced a Mock for it (for attribution only).
_STUB_INTRODUCED_BY: dict = {}


def pytest_runtest_logfinish(nodeid, location):
    """Remember which test first made each module a Mock, so the report can name a culprit."""
    for name in _mocked_modules():
        _STUB_INTRODUCED_BY.setdefault(name, nodeid)


def pytest_sessionfinish(session, exitstatus):
    """Fail the session if a module stub OUTLIVED THE WHOLE RUN (#1799).

    Stubbing an optional dependency is legitimate and common here — ``spacy``, ``torch``,
    ``transformers`` and the LLM SDKs are not installed in every environment, and
    ``INTEGRATION_TESTING_GUIDE.md`` documents the ``setUpModule``/``tearDownModule`` pattern for
    it. What is never legitimate is a stub still standing when the run ends, because from that
    point on an absent dependency looks present to everything that follows.

    SESSION SCOPE IS THE POINT, AND I GOT IT WRONG TWICE BEFORE LANDING HERE.
    v1 was an autouse fixture comparing around its own ``yield``: it flagged 12 tests that use
    ``monkeypatch.setitem`` correctly, because a fixture teardown can run before monkeypatch's
    undo. v2 moved to ``logstart``/``logfinish`` to bracket the whole test, and flagged 16 more
    that install a MODULE-scoped stub via ``setUpModule`` and remove it in ``tearDownModule`` —
    correct code, flagged on the module's first test. Both versions would have failed honest
    tests, which is worse than not checking at all.

    Only survival past the end of the session is unambiguous, and it is precisely what broke:
    a Mock left in ``sys.modules['spacy']`` with ``.load`` deleted made ``requires("spacy")``
    stop skipping and ``MLProvider.preload()`` die with ``AttributeError: load``, surfacing as an
    unrelated feed-error test two suites away.
    """
    # TWO OUTCOMES, because the two cases are not equally bad.
    #
    # SHADOWING an installed module -> FAIL. A Mock over a working package is the #1799 defect:
    # every test after it gets the Mock, and the symptom surfaces somewhere unrelated.
    #
    # Standing in for an ABSENT one -> report, do not fail. CI's unit job installs `.[dev]`, so
    # the whole [llm] extra is missing and those stubs are the only reason the modules import.
    # Failing there would make the suite unrunnable exactly where the stubs are load-bearing.
    # It is still worth printing: a leaked stub for an absent module is how the spaCy incident
    # started, and on a machine without spaCy this check can no longer prove it is gone.
    all_stubs = _mocked_modules()
    surviving = sorted(n for n in all_stubs if n in _REAL_AT_START)
    tolerated = sorted(n for n in all_stubs if n not in _REAL_AT_START)
    if tolerated:
        print(
            "\nNote: module stubs survived the run for packages that are NOT installed here: "
            + ", ".join(tolerated)
            + "\nThat is tolerated (the suite needs them), but it means this run could not prove "
            "those names are unstubbed. See #1799.",
            flush=True,
        )
    if not surviving:
        return
    lines = [
        f"  {name} (first seen in {_STUB_INTRODUCED_BY.get(name, 'unknown')})" for name in surviving
    ]
    print(
        "\nA module stub OUTLIVED THE RUN (#1799). These names still point at a Mock in\n"
        "sys.modules now that the session is over, so an absent dependency would look present\n"
        "to anything that ran after them:\n"
        + "\n".join(lines)
        + "\n\nScope the stub so it is removed on exit — `with patch.dict(sys.modules, {...}):`,\n"
        "`monkeypatch.setitem(...)`, or the setUpModule/tearDownModule pair documented in\n"
        "docs/guides/INTEGRATION_TESTING_GUIDE.md.",
        flush=True,
    )
    session.exitstatus = 1


def pytest_collection_finish(session):
    """Fail loudly if collection left a Mock standing in for a core SDK (#1799).

    THE FAILURE THIS PREVENTS. Four unit modules used to run
    ``patch.dict("sys.modules", {"openai": MagicMock()}).start()`` at import time with no
    matching ``.stop()`` — one of them said so in a comment and shipped anyway. pytest imports
    EVERY test module during collection, so the Mock was in place before
    ``tests/integration/utils/test_llm_resilience_mock_server.py`` was imported, and its
    module-level ``openai = pytest.importorskip("openai")`` bound the Mock. Eight of its tests
    then failed with ``'Mock' object is not subscriptable`` and ``DID NOT RAISE`` — symptoms
    pointing nowhere near the cause. CI never saw it because unit and integration run as
    separate jobs, so the two never shared a process: green by scheduling, not by isolation.

    WHY THIS HOOK, AND WHY IT FAILS RATHER THAN REPAIRS. By the time collection finishes the
    victim has already BOUND the Mock, so restoring ``sys.modules`` here would fix nothing while
    looking like it had — and a guard that appears to work is worse than none. Removing the
    Mock is the only real fix, and it is always available: these SDKs are core dependencies, so
    the stub they replace was never needed. Failing here names the offender at the moment it is
    introduced instead of surfacing as eight unrelated failures a suite away.
    """
    import sys as _sys

    poisoned = [
        name
        for name in _CORE_SDKS
        if name in _REAL_AT_START
        and (mod := _sys.modules.get(name)) is not None
        and "Mock" in type(mod).__name__
    ]
    if not poisoned:
        return
    raise pytest.UsageError(
        "A test module replaced a CORE SDK in sys.modules with a Mock and never restored it: "
        + ", ".join(poisoned)
        + ".\nThese are core dependencies (pyproject) and are always installed, so the stub is "
        "unnecessary — delete it rather than trying to scope it. `tearDownModule` does NOT work "
        "here: pytest imports every test module during collection, so later modules bind the "
        "Mock before any teardown runs. See #1799."
    )


@pytest.fixture(autouse=True)
def _reset_signal_context():
    """Clear the run/correlation context around every test (#1873).

    The observability context is process-global BY DESIGN — a pipeline run is its own
    subprocess, so a module global is the right scope in production. A pytest session is the
    one place that assumption breaks: without this, a test that sets a profile or feed leaks
    it into every later test, and assertions start depending on execution ORDER. It surfaced
    exactly that way — ``test_set_run_tag`` counted three sentry tags instead of one, and a
    search-log test found ASR routing on an unrelated event, both only in full-suite runs.
    """

    def _clear():
        try:
            from podcast_scraper.obs.events import clear_run_context

            clear_run_context()
        except Exception:  # noqa: BLE001 - never break collection on telemetry helpers
            pass
        try:
            from podcast_scraper.utils import correlation

            correlation.set_feed_id(None)
            correlation.set_profile(None)
            correlation.set_episode_id(None)
        except Exception:  # noqa: BLE001
            pass

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def _restore_podcast_scraper_profile_env():
    """Snapshot + restore ``PODCAST_SCRAPER_PROFILE`` around every test.

    Conftest sets this to ``test_default`` at module import so tests pick
    up the test_default profile by default (replaces the old env-detect
    machinery). Some
    tests (e.g. ``test_config.py::TestSummaryModeProfileDefaults``)
    intentionally mutate or pop this var to exercise other profile
    branches; without this fixture their teardown leaks state into the
    rest of the suite, silently flipping later tests off the
    test_default profile.
    """
    _prev = os.environ.get("PODCAST_SCRAPER_PROFILE")
    yield
    if _prev is None:
        os.environ.pop("PODCAST_SCRAPER_PROFILE", None)
    else:
        os.environ["PODCAST_SCRAPER_PROFILE"] = _prev


@pytest.fixture(autouse=True)
def _restore_hf_hub_cache_env():
    """Snapshot + restore ``HF_HUB_CACHE`` around every test.

    ``preload_evidence_models()`` intentionally sets ``os.environ["HF_HUB_CACHE"]`` to the
    active cache dir for the download helpers. When a test points that at a ``tmp_path``
    (e.g. test_model_loader*::test_downloads_uncached_models), the mutation LEAKS into the
    shared (xdist) worker process — ``get_transformers_cache_dir()`` then returns the dead
    tmp dir for every later test, so offline model loads (the search/two-tier suite) fail
    with ``LocalEntryNotFoundError``. Restoring per test isolates that side effect without
    changing the production behavior.
    """
    _prev = os.environ.get("HF_HUB_CACHE")
    yield
    if _prev is None:
        os.environ.pop("HF_HUB_CACHE", None)
    elif os.environ.get("HF_HUB_CACHE") != _prev:
        os.environ["HF_HUB_CACHE"] = _prev


@pytest.fixture(autouse=True)
def _restore_rss_cache_dir_env():
    """Snapshot + restore ``PODCAST_SCRAPER_RSS_CACHE_DIR`` around every test.

    Same shape as the HF_HUB_CACHE guard above, and for the same reason: production code sets
    this variable process-wide (``apply_session_rss_cache_env``), so any test that calls it leaks
    a cache dir into every later test. The consequence is quiet and confusing rather than loud —
    ``fetch_and_parse_feed`` consults ``feed_cache.read_cached_rss`` BEFORE the downloader, so a
    leaked cache serves stale feed XML and the downloader mock is never reached. That surfaced as
    three unrelated-looking failures (an episode count of 1 instead of 3, no hosts detected, and a
    fetch-failure test where the expected ValueError never raised) whenever the acceptance-script
    tests ran earlier in the session.

    The specific leak is fixed at its source; this is the guard that stops the class recurring.
    """
    _prev = os.environ.get("PODCAST_SCRAPER_RSS_CACHE_DIR")
    yield
    if _prev is None:
        os.environ.pop("PODCAST_SCRAPER_RSS_CACHE_DIR", None)
    elif os.environ.get("PODCAST_SCRAPER_RSS_CACHE_DIR") != _prev:
        os.environ["PODCAST_SCRAPER_RSS_CACHE_DIR"] = _prev


@pytest.fixture(autouse=True)
def _reset_run_budget():
    """Start every test with an empty, uncapped spend ledger.

    ``workflow.run_budget`` is a process-scoped singleton on purpose — the cap has to span the
    whole CLI invocation, since cli calls run_pipeline once per feed and a per-feed ledger is
    exactly the bug that let ~$48 through a $5 cap. Process-scoped means it survives between
    tests too, so a test that configures a cap and records spend would otherwise leave later
    tests running against an exhausted budget, and their selections would be refused for
    reasons having nothing to do with what they assert.
    """
    _reset_the_run_budget()
    yield
    _reset_the_run_budget()


@pytest.fixture(autouse=True)
def _isolate_audio_cache(tmp_path, monkeypatch):
    """Isolate the #947 GUID-keyed audio cache per test.

    ``audio_cache.resolve_cache_root`` defaults (cache enabled, not in-corpus) to a
    repo-relative global dir (``.cache/audio`` via ``DEFAULT_AUDIO_CACHE_DIR``). Without
    isolation, tests both **pollute** that shared cache and **read each other's**
    downloads — and a stale GUID-keyed hit silently masks failure-injection tests (the
    chaos 404-download test fetched a previously-cached episode instead of failing).
    Redirect the default to a per-test ``tmp_path`` (basename kept ``audio`` so the
    ``test_default_dir`` contract still holds); tests that pass an explicit
    ``audio_cache_dir`` are unaffected.
    """
    from podcast_scraper import config_constants

    monkeypatch.setattr(config_constants, "DEFAULT_AUDIO_CACHE_DIR", str(tmp_path / "audio"))


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
        # Keep metadata + artwork off unless a test opts in (production defaults are True).
        "generate_metadata": False,
        "download_podcast_artwork": False,
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

    if overrides.get("generate_kg") and "generate_metadata" not in overrides:
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

    def iter_bytes(self, chunk_size=1):
        # httpx.Response streaming API — the RSS downloader migrated to httpx
        # in #1194, so its production streaming loop calls ``iter_bytes``.
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
