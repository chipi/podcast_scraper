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

import importlib.util
import logging
import os

import pytest

logger = logging.getLogger(__name__)


def requires(*modules: str) -> "pytest.MarkDecorator":
    """Skip ONE e2e test when an optional ML/search module is not importable.

    Mirrors ``tests.integration.conftest.requires`` so the two tiers guard the same way. Legal
    here in a way it is not in unit tests (rule U1 bans ``importorskip`` there); ``tests/e2e``
    already uses ``pytest.importorskip`` at module scope for deepgram, fastapi and lancedb.

    Applied per TEST, never at module level. The four files using this hold 50 tests that pass
    without the ML stack and 15 that cannot: these run the FULL pipeline via the CLI, which
    reaches ``analyze_patterns`` -> ``_initialize_spacy`` -> ``import spacy`` at the point of use
    and dies there. A module-level guard would silence the 50 to quiet the 15 — the mistake the
    integration guards were careful to avoid.

    Inert in CI, which installs ``.[dev,ml,llm,search]``: the module is present and every guarded
    test runs. It only bites where the ML stack cannot be installed at all — macOS x86_64, where
    torch and torchcodec publish no wheels.

    NOTE this guards the TEST, not the product — and the two must not be confused when a guarded
    test starts failing. Ask which one is wrong:

    * If the capability is an ENHANCEMENT the pipeline can proceed without, a missing package is
      the PRODUCT's problem and it must degrade. That question is settled now: preload degrades
      (95be1ec1), speaker detection degrades at its point of use (processing.py, "speaker
      detection is an enhancement over episode metadata"), and feed host detection degrades too
      — it was the one site the sweep missed, and it killed the run at startup, before any
      episode, with a bare ``ModuleNotFoundError`` out of ``run_pipeline``.
    * If the capability IS the run — a transformers summary when you asked for one — the product
      is right to fail, and the TEST is what must declare the dependency. That is this guard.
    """
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"needs the [ml] extra — not importable: {', '.join(missing)}",
    )


# Check if we should use real OpenAI API (for manual testing only)
# Set USE_REAL_OPENAI_API=1 to test with real API endpoints
USE_REAL_OPENAI_API = os.getenv("USE_REAL_OPENAI_API", "0") == "1"

# Set dummy OpenAI API key for all E2E tests (will use mocked provider)
# This is needed because config validation requires the key to be present
# even though we mock the OpenAI client
# SKIP if using real API (will use key from .env file)
if not USE_REAL_OPENAI_API and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-e2e-tests"

# Check if we should use real Gemini API (for manual testing only)
USE_REAL_GEMINI_API = os.getenv("USE_REAL_GEMINI_API", "0") == "1"

# Set dummy Gemini API key for all E2E tests (will use mocked provider)
# This is needed because config validation requires the key to be present
# even though we mock the Gemini SDK
# SKIP if using real API (will use key from .env file)
if not USE_REAL_GEMINI_API and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = "test-dummy-key-for-e2e-tests"

# Check if we should use real Mistral API (for manual testing only)
USE_REAL_MISTRAL_API = os.getenv("USE_REAL_MISTRAL_API", "0") == "1"

# Set dummy Mistral API key for all E2E tests (will use mocked provider)
# This is needed because config validation requires the key to be present
# even though we mock the Mistral client
# SKIP if using real API (will use key from .env file)
if not USE_REAL_MISTRAL_API and "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = "test-dummy-key-for-e2e-tests"


# Set dummy DeepSeek API key for E2E tests (unless using real API)
USE_REAL_DEEPSEEK_API = os.getenv("USE_REAL_DEEPSEEK_API", "0") == "1"
if not USE_REAL_DEEPSEEK_API and "DEEPSEEK_API_KEY" not in os.environ:
    os.environ["DEEPSEEK_API_KEY"] = "test-dummy-key-for-e2e-tests"

# Set dummy Groq API key for E2E tests (unless using real API). Groq auth is warn-not-raise, so this
# is not strictly required, but it silences the "Groq API key not configured" warning every mock run
# would otherwise emit (note: groq is gsk_-prefixed, not sk-).
USE_REAL_GROQ_API = os.getenv("USE_REAL_GROQ_API", "0") == "1"
if not USE_REAL_GROQ_API and "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "gsk_e2e-key"

# Check if we should use real Ollama API (for manual testing only)
USE_REAL_OLLAMA_API = os.getenv("USE_REAL_OLLAMA_API", "0") == "1"
# Note: Ollama doesn't require an API key, but we set a dummy value for consistency
# with other providers (not actually used by Ollama)
if not USE_REAL_OLLAMA_API:
    # Ollama doesn't use API keys, but we set a dummy for consistency
    pass

# Set HF_HUB_CACHE to local cache if it exists (ensures consistent cache resolution)
# This must be set BEFORE any transformers imports happen
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
_local_cache = _project_root / ".cache" / "huggingface" / "hub"
if _local_cache.exists() and "HF_HUB_CACHE" not in os.environ:
    os.environ["HF_HUB_CACHE"] = str(_local_cache)

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


@pytest.fixture(autouse=True)
def configure_gemini_mock_server(request, monkeypatch):
    """Configure Gemini providers to use E2E server mock endpoints via fake SDK.

    This fixture automatically replaces the Gemini SDK's GenerativeModel class
    with a fake client that routes calls to the E2E server's mock Gemini API
    endpoints. This allows E2E tests to use real HTTP requests to mock endpoints,
    testing the full HTTP client → Network → Mock Server → Response chain.

    The E2E server provides mock endpoints:
    - /v1beta/models/{model}:generateContent: For transcription, summarization,
      and speaker detection

    Implementation:
        Uses a fake GenerativeModel class that intercepts SDK calls and routes
        them to the E2E mock server via HTTP, similar to how OpenAI tests work.

    Note:
        This fixture is autouse=True, so it's automatically applied to all
        E2E tests. No need to explicitly use it in test functions.

    Important:
        The Gemini SDK (google-genai) may not support custom base URLs directly.
        If the SDK doesn't support custom base URLs, tests will need to mock the SDK
        calls instead of using HTTP endpoints. This fixture sets GEMINI_API_BASE for
        documentation purposes, but actual mocking may need to be done at the SDK level.

    Real API Mode:
        When USE_REAL_GEMINI_API=1, this fixture is skipped to allow real API calls.
        This is for manual testing only and should NOT be used in CI.
    """
    # Check if we should use real Gemini API (for manual testing only)
    USE_REAL_GEMINI_API = os.getenv("USE_REAL_GEMINI_API", "0") == "1"

    # Skip E2E server configuration if using real Gemini API
    if USE_REAL_GEMINI_API:
        monkeypatch.delenv("GEMINI_API_BASE", raising=False)
        return

    # Get e2e_server fixture (may be skipped if USE_REAL_GEMINI_API=1)
    try:
        e2e_server = request.getfixturevalue("e2e_server")
    except pytest.FixtureLookupError:
        # E2E server not available (shouldn't happen in normal E2E mode)
        return

    # Set GEMINI_API_BASE environment variable to point to E2E server
    gemini_api_base = e2e_server.urls.gemini_api_base()
    monkeypatch.setenv("GEMINI_API_BASE", gemini_api_base)

    # Replace Gemini SDK's GenerativeModel with fake client that routes to E2E server
    # This allows tests to use real HTTP requests to mock endpoints
    try:
        from tests.fixtures.mock_server.gemini_mock_client import create_fake_gemini_client

        FakeGenerativeModel = create_fake_gemini_client(gemini_api_base)

        # The new google-genai API doesn't have GenerativeModel as a module attribute.
        # The provider code still uses genai.GenerativeModel(), so we need to add it.
        import google.genai as genai

        # Add GenerativeModel to the genai module so provider code can use it
        # Directly set the attribute (doesn't require it to exist first)
        setattr(genai, "GenerativeModel", FakeGenerativeModel)
        logger.debug(
            "Added/Replaced GenerativeModel in google.genai module with fake client pointing to %s",
            gemini_api_base,
        )

        # Also patch genai.configure() if it doesn't exist (new API doesn't have it)
        # The provider code calls genai.configure(api_key=...), so we need to provide it
        if not hasattr(genai, "configure"):

            def fake_configure(api_key: str, **kwargs):
                """Fake configure function that does nothing (API key handled by fake client)."""
                logger.debug("Fake genai.configure() called (no-op for E2E testing)")

            setattr(genai, "configure", fake_configure)
            logger.debug("Added fake genai.configure() function for E2E testing")

    except ImportError:
        # If fake client can't be imported, fall back to Python-level mocking
        # (This shouldn't happen in normal test runs)
        logger.warning("Could not import fake Gemini client, falling back to Python-level mocking")


@pytest.fixture(autouse=True)
def configure_mistral_mock_server(request, monkeypatch):
    """Configure Mistral providers to use E2E server mock endpoints via fake SDK.

    This fixture automatically replaces the Mistral SDK's Mistral class with a fake
    client that routes calls to the E2E server's mock Mistral API endpoints. This
    allows E2E tests to use real HTTP requests to mock endpoints, testing the full
    HTTP client → Network → Mock Server → Response chain.

    The E2E server provides mock endpoints:
    - /v1/chat/completions: For summarization and speaker detection
    - /v1/audio/transcriptions: For transcription

    Implementation:
        Uses a fake Mistral class that intercepts SDK calls and routes them to
        the E2E mock server via HTTP, similar to how Gemini tests work.

    Note:
        This fixture is autouse=True, so it's automatically applied to all
        E2E tests. No need to explicitly use it in test functions.

    Real API Mode:
        When USE_REAL_MISTRAL_API=1, this fixture is skipped to allow real API calls.
        This is for manual testing only and should NOT be used in CI.
    """
    # Check if we should use real Mistral API (for manual testing only)
    USE_REAL_MISTRAL_API = os.getenv("USE_REAL_MISTRAL_API", "0") == "1"

    # Skip E2E server configuration if using real Mistral API
    if USE_REAL_MISTRAL_API:
        monkeypatch.delenv("MISTRAL_API_BASE", raising=False)
        return

    # Get e2e_server fixture (may be skipped if USE_REAL_MISTRAL_API=1)
    try:
        e2e_server = request.getfixturevalue("e2e_server")
    except pytest.FixtureLookupError:
        # E2E server not available (shouldn't happen in normal E2E mode)
        return

    # Set MISTRAL_API_BASE environment variable to point to E2E server
    mistral_api_base = e2e_server.urls.mistral_api_base()
    monkeypatch.setenv("MISTRAL_API_BASE", mistral_api_base)

    # Replace Mistral SDK's Mistral class with fake client that routes to E2E server
    # This allows tests to use real HTTP requests to mock endpoints
    try:
        from tests.fixtures.mock_server.mistral_mock_client import create_fake_mistral_client

        FakeMistral = create_fake_mistral_client(mistral_api_base)
        monkeypatch.setattr(
            "podcast_scraper.providers.mistral.mistral_provider.Mistral", FakeMistral
        )
        logger.debug(
            "Replaced Mistral SDK Mistral with fake client pointing to %s",
            mistral_api_base,
        )
    except ImportError:
        # If fake client can't be imported, fall back to Python-level mocking
        # (This shouldn't happen in normal test runs)
        logger.warning("Could not import fake Mistral client, falling back to Python-level mocking")


@pytest.fixture(autouse=True)
def configure_grok_mock_server(request, monkeypatch):
    """Configure Grok providers to use E2E server mock endpoints.

    This fixture automatically configures all Grok providers to use the E2E server's
    mock Grok API endpoints instead of the real Grok API. Grok uses OpenAI-compatible
    API format, so it uses the same endpoints as OpenAI:
    - /v1/chat/completions: For summarization and speaker detection
    - Note: Grok does NOT support audio transcription

    This allows E2E tests to use real HTTP requests to mock endpoints, testing the full
    HTTP client → Network → Mock Server → Response chain.

    Note:
        This fixture is autouse=True, so it's automatically applied to all
        E2E tests. No need to explicitly use it in test functions.

    Real API Mode:
        When USE_REAL_GROK_API=1, this fixture is skipped to allow real API calls.
        This is for manual testing only and should NOT be used in CI.
    """
    # Check if we should use real Grok API (for manual testing only)
    USE_REAL_GROK_API = os.getenv("USE_REAL_GROK_API", "0") == "1"

    # Skip E2E server configuration if using real Grok API
    # Also explicitly unset GROK_API_BASE to ensure real API is used
    if USE_REAL_GROK_API:
        # Explicitly unset GROK_API_BASE to ensure we use real Grok API
        # (not a mock server that might have been set in a previous test)
        monkeypatch.delenv("GROK_API_BASE", raising=False)
        return

    # Get e2e_server fixture (may be skipped if USE_REAL_GROK_API=1)
    try:
        e2e_server = request.getfixturevalue("e2e_server")
    except pytest.FixtureLookupError:
        # E2E server not available (shouldn't happen in normal E2E mode)
        return

    # Set GROK_API_BASE environment variable to point to E2E server
    # This will be picked up by the Config model's field validator
    grok_api_base = e2e_server.urls.grok_api_base()
    monkeypatch.setenv("GROK_API_BASE", grok_api_base)


@pytest.fixture(autouse=True)
def configure_deepseek_mock_server(request, monkeypatch):
    """Configure DeepSeek providers to use E2E server mock endpoints.

    This fixture automatically configures all DeepSeek providers to use the E2E server's
    mock DeepSeek API endpoints instead of the real DeepSeek API. DeepSeek uses OpenAI-compatible
    API format, so it uses the same endpoints as OpenAI:
    - /v1/chat/completions: For summarization and speaker detection
    - Note: DeepSeek does NOT support audio transcription

    This allows E2E tests to use real HTTP requests to mock endpoints, testing the full
    HTTP client → Network → Mock Server → Response chain.

    Note:
        This fixture is autouse=True, so it's automatically applied to all
        E2E tests. No need to explicitly use it in test functions.

    Real API Mode:
        When USE_REAL_DEEPSEEK_API=1, this fixture is skipped to allow real API calls.
        This is for manual testing only and should NOT be used in CI.
    """
    # Check if we should use real DeepSeek API (for manual testing only)
    USE_REAL_DEEPSEEK_API = os.getenv("USE_REAL_DEEPSEEK_API", "0") == "1"

    # Skip E2E server configuration if using real DeepSeek API
    # Also explicitly unset DEEPSEEK_API_BASE to ensure real API is used
    if USE_REAL_DEEPSEEK_API:
        # Explicitly unset DEEPSEEK_API_BASE to ensure we use real DeepSeek API
        # (not a mock server that might have been set in a previous test)
        monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
        return

    # Get e2e_server fixture (may be skipped if USE_REAL_DEEPSEEK_API=1)
    try:
        e2e_server = request.getfixturevalue("e2e_server")
    except pytest.FixtureLookupError:
        # E2E server not available (shouldn't happen in normal E2E mode)
        return

    # Set DEEPSEEK_API_BASE environment variable to point to E2E server
    # This will be picked up by the Config model's field validator
    deepseek_api_base = e2e_server.urls.deepseek_api_base()
    monkeypatch.setenv("DEEPSEEK_API_BASE", deepseek_api_base)


@pytest.fixture(autouse=True)
def configure_groq_mock_server(request, monkeypatch):
    """Configure Groq providers to use E2E server mock endpoints.

    Groq (ADR-147) uses OpenAI-compatible API format AND is DUAL-USE (unlike deepseek/grok/qwen):
    the same base serves BOTH chat and Whisper transcription, so it reuses the same endpoints:
    - /v1/chat/completions: summarization / speaker detection / GI / KG
    - /v1/audio/transcriptions: whisper-large-v3-turbo transcription (Groq has NO diarization)

    This lets E2E tests exercise the full HTTP client -> Network -> Mock Server -> Response chain
    for both the LLM and transcription halves of the provider.

    Note:
        This fixture is autouse=True, so it's automatically applied to all E2E tests.

    Real API Mode:
        When USE_REAL_GROQ_API=1, this fixture is skipped to allow real API calls.
        This is for manual testing only and should NOT be used in CI.
    """
    # Check if we should use real Groq API (for manual testing only)
    USE_REAL_GROQ_API = os.getenv("USE_REAL_GROQ_API", "0") == "1"

    # Skip E2E server configuration if using real Groq API
    # Also explicitly unset GROQ_API_BASE to ensure real API is used
    if USE_REAL_GROQ_API:
        monkeypatch.delenv("GROQ_API_BASE", raising=False)
        return

    # Get e2e_server fixture (may be skipped if USE_REAL_GROQ_API=1)
    try:
        e2e_server = request.getfixturevalue("e2e_server")
    except pytest.FixtureLookupError:
        # E2E server not available (shouldn't happen in normal E2E mode)
        return

    # Set GROQ_API_BASE environment variable to point to E2E server
    # This will be picked up by the Config model's env loader (_load_string_env_var)
    groq_api_base = e2e_server.urls.groq_api_base()
    monkeypatch.setenv("GROQ_API_BASE", groq_api_base)


@pytest.fixture(autouse=True)
def configure_ollama_mock_server(request, monkeypatch):
    """Configure Ollama providers to use E2E server mock endpoints.

    This fixture automatically configures all Ollama providers to use the E2E server's
    mock Ollama API endpoints instead of the real Ollama server. Ollama uses OpenAI-compatible
    API format for chat completions, so it uses the same endpoints as OpenAI:
    - /v1/chat/completions: For summarization and speaker detection
    - Note: Ollama does NOT support audio transcription

    Additionally, the E2E server provides Ollama-specific endpoints:
    - /api/version: For health checks (validates server is running)
    - /api/tags: For model validation (lists available models)
    - /api/generate: For native warm-up POST (OllamaProvider.warmup / wait_until_ready)

    Static RSS/audio/transcript URLs on the E2E server support GET and HEAD (HEAD is
    used for media size checks and must not 404).

    This allows E2E tests to use real HTTP requests to mock endpoints, testing the full
    HTTP client → Network → Mock Server → Response chain.

    Note:
        This fixture is autouse=True, so it's automatically applied to all
        E2E tests. No need to explicitly use it in test functions.

    Real API Mode:
        When USE_REAL_OLLAMA_API=1, this fixture is skipped to allow real API calls.
        This is for manual testing only and should NOT be used in CI.
    """
    # Check if we should use real Ollama API (for manual testing only)
    USE_REAL_OLLAMA_API = os.getenv("USE_REAL_OLLAMA_API", "0") == "1"

    # Skip E2E server configuration if using real Ollama API
    # Also explicitly unset OLLAMA_API_BASE to ensure real API is used
    if USE_REAL_OLLAMA_API:
        # Explicitly unset OLLAMA_API_BASE to ensure we use real Ollama server
        # (not a mock server that might have been set in a previous test)
        monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
        return

    # Get e2e_server fixture (may be skipped if USE_REAL_OLLAMA_API=1)
    try:
        e2e_server = request.getfixturevalue("e2e_server")
    except pytest.FixtureLookupError:
        # E2E server not available (shouldn't happen in normal E2E mode)
        return

    # Set OLLAMA_API_BASE environment variable to point to E2E server
    # This will be picked up by the Config model's field validator
    # Note: Ollama provider uses this base URL for both:
    # - Health checks (/api/version) - removes /v1 suffix
    # - Model validation (/api/tags) - removes /v1 suffix
    # - API calls (/v1/chat/completions) - uses full base URL
    ollama_api_base = e2e_server.urls.ollama_api_base()
    monkeypatch.setenv("OLLAMA_API_BASE", ollama_api_base)
    logger.debug(
        "Configured Ollama to use E2E server at %s",
        ollama_api_base,
    )


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

# Import helper functions from parent conftest using absolute import
# Use tests.conftest to avoid ambiguity with other conftest files
from tests.conftest import (  # noqa: E402, F401
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

    # If test is marked as nightly and E2E_TEST_MODE is not explicitly set,
    # default to nightly mode (nightly tests need all podcasts)
    is_nightly = request.node.get_closest_marker("nightly") is not None
    if test_mode == "multi_episode" and is_nightly:
        test_mode = "nightly"

    # If test is marked as critical_path and E2E_TEST_MODE is not explicitly set,
    # default to fast mode (critical path tests need fast fixtures)
    is_critical_path = request.node.get_closest_marker("critical_path") is not None
    if test_mode == "multi_episode" and is_critical_path:
        test_mode = "fast"

    if test_mode == "nightly":
        # Nightly mode: Allow podcasts 1-5 (p01-p05) for comprehensive testing
        # plus episode-selection feed (p01_episode_selection.xml; GitHub #521).
        # Use full fixtures (not fast) for production-quality testing
        E2EHTTPRequestHandler.set_allowed_podcasts(
            {
                "podcast1",
                "podcast2",
                "podcast3",
                "podcast4",
                "podcast5",
                "podcast1_episode_selection",
            }
        )
        E2EHTTPRequestHandler.set_use_fast_fixtures(False)  # Use full fixtures
    elif test_mode == "fast":
        # Fast mode: Allow podcast1 (Path 2: Transcription),
        # podcast1_with_transcript (Path 1: Download), podcast1_multi_episode
        # (multi-episode testing), podcast9_solo (solo speaker testing),
        # podcast7_sustainability and podcast8_solar (Issue #283 threshold testing)
        # Uses fast fixtures (p01_fast.xml) with 1-minute episodes, or multi-episode
        # feed for multi-episode tests
        E2EHTTPRequestHandler.set_allowed_podcasts(
            {
                "podcast1",
                "podcast1_with_transcript",
                "podcast1_multi_episode",
                "podcast1_episode_selection",
                # GitHub #560 — multi-feed RSS error injection (resilience E2E)
                "podcast2",
                "podcast9_solo",
                "podcast7_sustainability",  # Issue #283: threshold testing
                "podcast8_solar",  # Issue #283: threshold testing
            }
        )
        E2EHTTPRequestHandler.set_use_fast_fixtures(True)  # Use fast fixtures
    else:
        # Default/Multi-episode mode: Use multi-episode feed (5 short episodes for
        # multi-episode testing), edgecases feed (for edge case tests), and
        # podcast7_sustainability/podcast8_solar (Issue #283 threshold testing)
        # Also include podcast1_with_transcript for config file tests and other tests
        # that need a feed with transcript URLs
        E2EHTTPRequestHandler.set_allowed_podcasts(
            {
                "podcast1_multi_episode",
                "podcast1_episode_selection",
                # For config file tests and tests needing transcript URLs
                "podcast1_with_transcript",
                "edgecases",
                "podcast7_sustainability",  # Issue #283: threshold testing
                "podcast8_solar",  # Issue #283: threshold testing
            }
        )
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
        from podcast_scraper.workflow.stages import scraping

        # Store original function
        original_prepare_episodes = scraping.prepare_episodes_from_feed

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
        monkeypatch.setattr(scraping, "prepare_episodes_from_feed", limited_prepare_episodes)

        yield

        # Cleanup is automatic with monkeypatch context manager
    else:
        # Multi-episode/Data Quality mode: No limitation (can process multiple episodes)
        yield
        return


# REMOVED: mock_whisper_in_fast_mode fixture
# E2E tests should use real Whisper (no mocks). Tests that need mocked Whisper
# should be in integration tests, not E2E tests.


# ---------------------------------------------------------------------------
# Cost-flow assertion helper (#650/#651)
# ---------------------------------------------------------------------------


def assert_cost_fields_populated(
    temp_dir,
    *,
    billable_stages=(),
    local_stages=(),
):
    """Assert the full cost chain flowed end-to-end for an E2E pipeline run.

    Reads every ``run_*/metrics.json`` under ``temp_dir`` and verifies that for
    each named stage the Metrics emission is consistent with the provider
    class driving that stage.

    Parameters
    ----------
    temp_dir : Path
        E2E test's tempdir root — contains ``run_*/metrics.json``.
    billable_stages : Sequence[str]
        Stages whose provider is a billable cloud LLM (openai, anthropic,
        gemini, mistral, deepseek, grok). Each MUST have ``<stage>_calls > 0``
        AND ``<stage>_cost_usd > 0``. Names without the ``llm_`` prefix —
        e.g. ``["summarization", "speaker_detection", "gi", "kg"]``.
    local_stages : Sequence[str]
        Stages whose provider is local / free (ollama, spaCy for speaker
        detection, local Whisper for transcription). For these the recorder
        either never fires (``calls == 0`` — non-LLM path like spaCy) or
        fires with a $0 YAML rate (``cost_usd == 0.0`` — ollama). Both
        outcomes are fine; what must NOT happen is ``cost_usd`` being
        dropped (None) when the recorder was invoked, or ``cost_usd`` being
        non-zero for a stage routed through a free provider.

    Why the split: a test like ``test_anthropic_full_pipeline`` uses Whisper
    for transcription (local) + Anthropic for speaker/summary (billable).
    Asserting "cost > 0 on every stage" would wrongly fail on the transcription
    row; asserting "cost == 0 on every stage" would wrongly pass on a
    regression that zeroed out the Anthropic summarization cost. Both
    invariants have to be stated explicitly.
    """
    import json as _json
    from pathlib import Path as _Path

    metrics_files = list(_Path(temp_dir).rglob("metrics.json"))
    assert metrics_files, (
        f"No metrics.json under {temp_dir} — pipeline may not have written " "run artifacts"
    )
    for m_path in metrics_files:
        try:
            m = _json.loads(m_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # pragma: no cover — surface loudly
            raise AssertionError(f"Cannot read {m_path}: {exc}") from exc

        for stage in billable_stages:
            calls_key = f"llm_{stage}_calls"
            cost_key = f"llm_{stage}_cost_usd"
            calls = m.get(calls_key)
            assert calls is not None and calls > 0, (
                f"{calls_key} expected > 0 in {m_path}, got {calls}. "
                "Provider recorder never fired — cost chain broken (#650/#651)."
            )
            cost = m.get(cost_key)
            assert cost is not None, (
                f"{cost_key} missing from {m_path} — Metrics.to_dict() dropped "
                "the field, or provider never passed cost_usd kwarg."
            )
            assert cost > 0.0, (
                f"{cost_key}={cost} in {m_path}. Expected > 0 — billable "
                "provider's call should have populated cost via "
                "calculate_provider_cost(). Check the wiring at the "
                f"provider's record_llm_{stage}_call(cost_usd=...) site."
            )

        for stage in local_stages:
            calls_key = f"llm_{stage}_calls"
            cost_key = f"llm_{stage}_cost_usd"
            # Recorder either never fired (non-LLM ML path, e.g. spaCy) or
            # fired with $0 pricing (ollama). Both are fine; what must not
            # happen is cost being non-zero for a free provider.
            cost = m.get(cost_key)
            assert cost is not None, (
                f"{cost_key} missing from {m_path} — Metrics.to_dict() dropped "
                "the field entirely. This is a serialisation bug, not a "
                "provider-wiring one."
            )
            assert cost == 0.0, (
                f"{cost_key}={cost} in {m_path}. Expected exactly 0.0 — "
                f"local / free provider shouldn't be accumulating cost. "
                "Either the pricing YAML has a non-zero rate for a local "
                "provider (ollama/whisper-local/spaCy should all be $0), "
                "or a billable provider's cost leaked into the wrong stage."
            )


# --- Retry-backoff sleep neutralizer (mirrors tests/integration/conftest.py) --------------
# The e2e suite has TWO retry mechanisms that wall-clock-wait real exponential backoff:
#   1. retry_with_metrics (LLM provider / cloud-resilience / fallback-chain tests)
#   2. the HTTP-download RetryTransport (rss/http_retry.py) used by the HTTP-500 / transient
#      download tests (test_http_behaviors_e2e / test_error_handling_e2e — ~247s EACH).
# Both sleep via a module-level `import time`, so we patch each module's `time` — scoped, so
# global time.sleep and everything else are untouched. The retry LOGIC still runs (N attempts →
# correct terminal error/behaviour), just instantly. e2e asserts behaviour + one-sided timing
# (`elapsed < N`), never "backoff took ≥ N", so faster is safe. See
# docs/wip/nightly-test-time-analysis.md.
import time as _e2e_real_time  # noqa: E402
from unittest.mock import patch as _e2e_patch  # noqa: E402


class _E2ENoBackoffSleep:
    def sleep(self, *_args, **_kwargs):
        return None

    def __getattr__(self, name):
        return getattr(_e2e_real_time, name)


@pytest.fixture(autouse=True)
def _instant_retry_backoff_e2e():
    """No-op both retry-backoff sleeps for the e2e suite (retry logic still runs)."""
    from podcast_scraper.rss import http_retry
    from podcast_scraper.utils import provider_metrics

    _noop = _E2ENoBackoffSleep()
    with (
        _e2e_patch.object(provider_metrics, "time", _noop),
        _e2e_patch.object(http_retry, "time", _noop),
    ):
        yield
