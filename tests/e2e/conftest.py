"""Pytest configuration and fixtures for E2E tests.

This module provides:
- Network guard fixture (blocks external network calls)
- OpenAI mock server configuration (configures OpenAI providers to use E2E server)
- Shared E2E test utilities

Network Guard (NON-NEGOTIABLE):
- Blocks all external network calls (except localhost/127.0.0.1)
- Ensures all RSS and audio are served from local E2E HTTP server
- Fails hard if a real URL is hit
- SKIPPED when USE_REAL_OPENAI_API=1 (allows real API calls)

OpenAI Mocking:
- Configures OpenAI providers to use E2E server mock endpoints via OPENAI_API_BASE
- E2E server provides mock /v1/chat/completions and /v1/audio/transcriptions endpoints
- Tests use real HTTP requests to mock endpoints, testing full HTTP client →
  Network → Mock Server chain
- Prevents costs, rate limits, and flakiness while testing real HTTP client
  behavior
- SKIPPED when USE_REAL_OPENAI_API=1 (uses real OpenAI API)
"""

from __future__ import annotations

import os

import pytest

# Check if we should use real OpenAI API (for manual testing only)
# Set USE_REAL_OPENAI_API=1 to test with real API endpoints
USE_REAL_OPENAI_API = os.getenv("USE_REAL_OPENAI_API", "0") == "1"

# Set dummy OpenAI API key for all E2E tests (will use mocked provider)
# This is needed because config validation requires the key to be present
# even though we mock the OpenAI client
# SKIP if using real API (will use key from .env file)
if not USE_REAL_OPENAI_API and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-e2e-tests"

# Network guard: Block all external network calls (except localhost)
# This ensures E2E tests only use the local E2E HTTP server
# Note: pytest-socket is loaded via plugin system, not via pytest_plugins
# to avoid loading issues during pytest collection
try:
    import pytest_socket  # noqa: F401
except ImportError:
    pass  # pytest-socket not installed, will fail at runtime


@pytest.fixture(autouse=True, scope="function")
def block_external_network(socket_enabled, monkeypatch):
    """Block all external network calls in E2E tests (except localhost).

    This fixture is automatically applied to all E2E tests (autouse=True).
    It uses pytest-socket to block all socket connections except localhost/127.0.0.1.

    Additionally, it patches Whisper's model download mechanism to prevent
    automatic model downloads even if cache check is bypassed.

    Configuration:
        pytest-socket must be configured via command-line options:
        - --disable-socket: Blocks all sockets (socket_enabled=False)
        - --allow-hosts=127.0.0.1,localhost: Allows localhost connections only

    This ensures:
    - No accidental external network calls
    - All RSS and audio must be served from local mock server
    - Tests fail hard if a real URL is hit
    - Whisper cannot download models (must use pre-cached models)

    Note:
        This fixture requires pytest-socket to be installed.
        Socket blocking must be configured via pytest command-line:
        pytest --disable-socket --allow-hosts=127.0.0.1,localhost

        Or add to pyproject.toml pytest.ini_options.addopts for e2e tests.

    Real API Mode:
        When USE_REAL_OPENAI_API=1, this fixture is skipped to allow real API calls.
        This is for manual testing only and should NOT be used in CI.
    """
    # Skip network blocking if using real OpenAI API
    if USE_REAL_OPENAI_API:
        return

    # pytest-socket automatically blocks sockets when --disable-socket is used
    # and allows only specified hosts with --allow-hosts
    # The socket_enabled parameter is provided by pytest-socket
    # If socket_enabled is True, that means --disable-socket was not used
    if socket_enabled:
        pytest.fail(
            "Network guard is not active! E2E tests must run with --disable-socket. "
            "Use: pytest --disable-socket --allow-hosts=127.0.0.1,localhost"
        )

    # Block ML model downloads by patching network libraries
    # This prevents Whisper, spaCy, and Transformers from downloading models
    # even if cache check is bypassed

    # 1. Block urllib.request.urlopen (used by Whisper)
    try:
        import urllib.request

        original_urlopen = urllib.request.urlopen

        def block_urlopen(url, *args, **kwargs):
            """Block urllib.request.urlopen to prevent model downloads."""
            # Allow localhost URLs (for E2E server)
            url_str = str(url) if not isinstance(url, str) else url
            if "127.0.0.1" in url_str or "localhost" in url_str:
                return original_urlopen(url, *args, **kwargs)
            # Block all other URLs (including model downloads)
            raise RuntimeError(
                f"Network downloads are blocked in E2E tests. "
                f"Attempted to download from: {url_str}. "
                f"Models must be pre-cached. Run 'make preload-ml-models' to pre-cache models."
            )

        monkeypatch.setattr(urllib.request, "urlopen", block_urlopen)
    except ImportError:
        pass  # urllib not available (shouldn't happen in Python 3)

    # 2. Block requests library (used by Hugging Face Transformers)
    try:
        import requests

        original_get = requests.get
        original_post = requests.post

        def block_requests_get(url, *args, **kwargs):
            """Block requests.get to prevent model downloads."""
            url_str = str(url) if not isinstance(url, str) else url
            if "127.0.0.1" in url_str or "localhost" in url_str:
                return original_get(url, *args, **kwargs)
            raise RuntimeError(
                f"Network downloads are blocked in E2E tests. "
                f"Attempted to download from: {url_str}. "
                f"Models must be pre-cached. Run 'make preload-ml-models' to pre-cache models."
            )

        def block_requests_post(url, *args, **kwargs):
            """Block requests.post to prevent model downloads."""
            url_str = str(url) if not isinstance(url, str) else url
            if "127.0.0.1" in url_str or "localhost" in url_str:
                return original_post(url, *args, **kwargs)
            raise RuntimeError(
                f"Network downloads are blocked in E2E tests. "
                f"Attempted to download from: {url_str}. "
                f"Models must be pre-cached. Run 'make preload-ml-models' to pre-cache models."
            )

        monkeypatch.setattr(requests, "get", block_requests_get)
        monkeypatch.setattr(requests, "post", block_requests_post)
    except ImportError:
        pass  # requests not available, skip

    # 3. Block spaCy CLI download (uses subprocess, but we can block the subprocess call)
    try:
        import subprocess

        original_run = subprocess.run

        def block_spacy_download(*args, **kwargs):
            """Block subprocess calls that look like spaCy downloads."""
            # Check if this is a spaCy download command
            if args and len(args) > 0:
                cmd = args[0]
                if isinstance(cmd, (list, tuple)) and len(cmd) >= 3:
                    # Check for: ['python', '-m', 'spacy', 'download', ...]
                    if (
                        cmd[0].endswith("python")
                        and "-m" in cmd
                        and "spacy" in cmd
                        and "download" in cmd
                    ):
                        raise RuntimeError(
                            "spaCy model downloads are blocked in E2E tests. "
                            "Models must be pre-cached. "
                            "Run 'make preload-ml-models' to pre-cache models."
                        )
            # Allow other subprocess calls (for E2E server, etc.)
            return original_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", block_spacy_download)
    except ImportError:
        pass  # subprocess not available (shouldn't happen)


@pytest.fixture(autouse=True)
def configure_openai_mock_server(request, monkeypatch):
    """Configure OpenAI providers to use E2E server mock endpoints.

    This fixture automatically configures all OpenAI providers to use the E2E server's
    mock OpenAI API endpoints instead of the real OpenAI API. This allows E2E tests to
    use real HTTP requests to mock endpoints, testing the full HTTP client → Network →
    Mock Server → Response chain.

    The E2E server provides mock endpoints:
    - /v1/chat/completions: For summarization and speaker detection
    - /v1/audio/transcriptions: For transcription

    Note:
        This fixture is autouse=True, so it's automatically applied to all
        E2E tests. No need to explicitly use it in test functions.

    Real API Mode:
        When USE_REAL_OPENAI_API=1, this fixture is skipped to allow real API calls.
        This is for manual testing only and should NOT be used in CI.
    """
    # Skip E2E server configuration if using real OpenAI API
    # Also explicitly unset OPENAI_API_BASE to ensure real API is used
    if USE_REAL_OPENAI_API:
        # Explicitly unset OPENAI_API_BASE to ensure we use real OpenAI API
        # (not a mock server that might have been set in a previous test)
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        return

    # Get e2e_server fixture (may be skipped if USE_REAL_OPENAI_API=1)
    try:
        e2e_server = request.getfixturevalue("e2e_server")
    except pytest.FixtureLookupError:
        # E2E server not available (shouldn't happen in normal E2E mode)
        return

    # Set OPENAI_API_BASE environment variable to point to E2E server
    # This will be picked up by the Config model's field validator
    openai_api_base = e2e_server.urls.openai_api_base()
    monkeypatch.setenv("OPENAI_API_BASE", openai_api_base)


# Fixture to ensure OpenAI API key is set for all E2E tests
@pytest.fixture(autouse=True)
def ensure_openai_api_key(monkeypatch):
    """Ensure OpenAI API key is set for all E2E tests (will use mocked provider).

    Real API Mode:
        When USE_REAL_OPENAI_API=1, this fixture does NOT override the API key,
        allowing the real key from .env file to be used.
    """
    # Skip if using real API (will use key from .env file)
    if USE_REAL_OPENAI_API:
        return

    # Set dummy OpenAI API key (required for config validation)
    # The actual OpenAI client is mocked, so this key is never used
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-for-e2e-tests")


# Re-export helper functions from parent conftest for E2E tests
# This allows E2E tests to import from conftest without path manipulation
import sys
from pathlib import Path

# Add parent tests directory to path to import from parent conftest
parent_tests_dir = Path(__file__).parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

# Import helper functions from parent conftest
from conftest import (  # noqa: E402, F401
    build_rss_xml_with_media,
    build_rss_xml_with_speakers,
    build_rss_xml_with_transcript,
    create_media_response,
    create_test_config,
)

# Import E2E server fixture and handler
# Use absolute import to avoid relative import issues
try:
    from tests.e2e.fixtures.e2e_http_server import (  # noqa: F401, E402
        e2e_server,
        E2EHTTPRequestHandler,
    )
except ImportError:
    # Try relative import as fallback
    try:
        from .fixtures.e2e_http_server import (  # noqa: F401, E402
            e2e_server,
            E2EHTTPRequestHandler,
        )
    except ImportError:
        # If both fail, create a dummy fixture (shouldn't happen in normal operation)
        @pytest.fixture(scope="session")
        def e2e_server():
            """Dummy E2E server fixture (import failed)."""
            pytest.skip("E2E server fixture not available")

        E2EHTTPRequestHandler = None  # type: ignore


@pytest.fixture(autouse=True)
def configure_e2e_feed_limiting(request):
    """Configure E2E server to limit RSS feeds based on test run mode.

    Test run mode is determined by E2E_TEST_MODE environment variable:
    - "fast": Fast feed (1 episode, 1 minute) - used by make test-e2e-fast
    - "multi_episode": Multi-episode feed (5 episodes, 10-15 seconds each) - used by make test-e2e
    - "data_quality": All original mock data - used by make test-e2e-data-quality
    - "nightly": All podcasts p01-p05 (15 episodes) - used by make test-nightly
    - Default: "multi_episode" (if E2E_TEST_MODE not set, use multi-episode feed)

    This allows the same tests to run in different modes based on how they're invoked,
    rather than hardcoding feed selection per test.

    This fixture is automatically applied to all E2E tests (autouse=True).

    Real API Mode:
        When USE_REAL_OPENAI_API=1, this fixture is skipped (not needed for real API tests).
    """
    # Skip if using real OpenAI API
    if USE_REAL_OPENAI_API:
        yield
        return

    # Skip if E2EHTTPRequestHandler is not available (import failed)
    if E2EHTTPRequestHandler is None:
        yield
        return

    # Get e2e_server fixture (may be skipped if USE_REAL_OPENAI_API=1)
    try:
        request.getfixturevalue("e2e_server")
    except pytest.FixtureLookupError:
        # E2E server not available (shouldn't happen in normal E2E mode)
        yield
        return

    # Get test run mode from environment variable (set by Makefile)
    test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()

    # If test is marked as critical_path and E2E_TEST_MODE is not explicitly set,
    # default to fast mode (critical path tests need fast fixtures)
    is_critical_path = request.node.get_closest_marker("critical_path") is not None
    if test_mode == "multi_episode" and is_critical_path:
        test_mode = "fast"

    if test_mode == "data_quality":
        # Data Quality mode: Allow all podcasts (original mock data)
        E2EHTTPRequestHandler.set_allowed_podcasts(None)
        E2EHTTPRequestHandler.set_use_fast_fixtures(False)  # Use full fixtures
    elif test_mode == "nightly":
        # Nightly mode: Allow podcasts 1-5 (p01-p05) for comprehensive testing
        # Use full fixtures (not fast) for production-quality testing
        E2EHTTPRequestHandler.set_allowed_podcasts(
            {"podcast1", "podcast2", "podcast3", "podcast4", "podcast5"}
        )
        E2EHTTPRequestHandler.set_use_fast_fixtures(False)  # Use full fixtures
    elif test_mode == "fast":
        # Fast mode: Allow podcast1 (Path 2: Transcription),
        # podcast1_with_transcript (Path 1: Download), and podcast1_multi_episode
        # (multi-episode testing)
        # Uses fast fixtures (p01_fast.xml) with 1-minute episodes, or multi-episode
        # feed for multi-episode tests
        E2EHTTPRequestHandler.set_allowed_podcasts(
            {"podcast1", "podcast1_with_transcript", "podcast1_multi_episode"}
        )
        E2EHTTPRequestHandler.set_use_fast_fixtures(True)  # Use fast fixtures
    else:
        # Default/Multi-episode mode: Use multi-episode feed (5 short episodes for
        # multi-episode testing) and edgecases feed (for edge case tests)
        E2EHTTPRequestHandler.set_allowed_podcasts({"podcast1_multi_episode", "edgecases"})
        E2EHTTPRequestHandler.set_use_fast_fixtures(True)  # Use fast fixtures

    # Reset on test teardown to ensure clean state
    yield
    if E2EHTTPRequestHandler is not None:
        # Clear error behaviors only (don't reset allowed_podcasts - next test will set them)
        E2EHTTPRequestHandler.clear_all_error_behaviors()
        # Reset to default fast fixtures mode for next test
        E2EHTTPRequestHandler.set_use_fast_fixtures(True)


@pytest.fixture(autouse=True)
def limit_max_episodes_in_fast_mode(request, monkeypatch):
    """Automatically limit max_episodes based on test run mode.

    Test run mode is determined by E2E_TEST_MODE environment variable:
    - "fast": Limits to 1 episode - used by make test-e2e-fast
    - "multi_episode": No limitation (5 episodes) - used by make test-e2e
    - "data_quality": No limitation (3-5 episodes) - used by make test-e2e-data-quality
    - "nightly": No limitation (all 15 episodes across p01-p05) - used by make test-nightly
    - Default: "multi_episode" (if E2E_TEST_MODE not set, allow multiple episodes)

    This allows the same tests to run with different episode limits based on how they're invoked,
    rather than hardcoding limits per test.

    This fixture is automatically applied to all E2E tests (autouse=True).
    """
    # Get test run mode from environment variable (set by Makefile)
    test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()

    if test_mode == "fast":
        # Fast mode: Limit max_episodes to 1
        # Patch the workflow function that prepares episodes
        from podcast_scraper import workflow

        # Store original function
        original_prepare_episodes = workflow._prepare_episodes_from_feed

        def limited_prepare_episodes(feed, cfg):
            """Wrapper that limits max_episodes to 1 in fast mode."""

            # Since Config is frozen, we can't modify it directly
            # Instead, we'll create a wrapper config that limits episodes
            # by intercepting the max_episodes value before it's used
            # Create a mock config-like object that returns 1 for max_episodes
            # but passes through all other attributes
            class LimitedConfig:
                """Wrapper that limits max_episodes to 1 while preserving other config values."""

                def __init__(self, original_cfg):
                    self._original = original_cfg
                    # Override max_episodes to always be 1
                    self.max_episodes = 1

                def __getattr__(self, name):
                    # Delegate all other attributes to original config
                    return getattr(self._original, name)

            limited_cfg = LimitedConfig(cfg)

            # Call original function with limited config
            return original_prepare_episodes(feed, limited_cfg)

        # Apply monkeypatch
        monkeypatch.setattr(workflow, "_prepare_episodes_from_feed", limited_prepare_episodes)

        yield

        # Cleanup is automatic with monkeypatch context manager
    else:
        # Multi-episode/Data Quality mode: No limitation (can process multiple episodes)
        yield
        return


# REMOVED: mock_whisper_in_fast_mode fixture
# E2E tests should use real Whisper (no mocks). Tests that need mocked Whisper
# should be in integration tests, not E2E tests.
