# Provider Implementation Guide

This comprehensive guide explains how to implement new providers for the podcast scraper.
It consolidates information from multiple guides and uses OpenAI as a complete example throughout.

## Overview

The podcast scraper uses a **protocol-based provider system** where each capability
(transcription, speaker detection, summarization) has a protocol interface that all providers must implement.
This design allows:

- **Pluggable implementations**: Swap providers via configuration
- **Type safety**: Protocols ensure consistent interfaces
- **Easy testing**: Mock providers for testing
- **Extensibility**: Add new providers without modifying core code

## Architecture

### Provider Types

1. **TranscriptionProvider**: Converts audio to text
2. **SpeakerDetector**: Detects speaker names from episode metadata
3. **SummarizationProvider**: Generates episode summaries

### Provider Structure

Each provider type follows this pattern:

```text
│   ├── base.py          # Protocol definition
│   ├── factory.py       # Factory function
│   └── {provider}_provider.py  # Implementation
```

- `transcription/openai_provider.py` - OpenAI Whisper API
- `speaker_detectors/openai_detector.py` - OpenAI GPT for speaker detection
- `summarization/openai_provider.py` - OpenAI GPT for summarization

## Step-by-Step Implementation

This section walks through implementing a provider using OpenAI as a complete example.

### Step 1: Understand the Protocol

First, examine the protocol interface in `{capability}/base.py`. For example, `TranscriptionProvider`:

```python
from typing import Protocol

class TranscriptionProvider(Protocol):
```python

    def initialize(self) -> None:
        """Initialize provider (load models, connect to API, etc.)."""
        ...

```python
        self,
        audio_path: str,
        language: str | None = None,
    ) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'es')

        Returns:
            Transcribed text as string
        """
        ...

- Method signatures must match the protocol exactly
- Return types must match protocol specifications

### Step 2: Implement the Provider Class

Create a new file `{capability}/{your_provider}_provider.py`. See the OpenAI transcription provider as a reference example:

**Reference Implementation**: [`src/podcast_scraper/transcription/openai_provider.py`](../../../src/podcast_scraper/transcription/openai_provider.py)

Key implementation patterns to follow:

1. **Configuration Validation**: Check required config fields in `__init__()`
2. **Custom Base URL Support**: Support `openai_api_base` for E2E testing
3. **Initialization Tracking**: Track `_initialized` state
4. **Error Handling**: Raise appropriate exceptions (`RuntimeError`, `ValueError`, `FileNotFoundError`)
5. **Logging**: Log important operations at appropriate levels

### Step 3: Register in Factory

Update `{capability}/factory.py` to include your provider:

```python

def create_transcription_provider(cfg: config.Config) -> TranscriptionProvider:
    """Create a transcription provider based on configuration.

    Args:
        cfg: Configuration object

    Returns:
        TranscriptionProvider instance

    Raises:
        ValueError: If provider type is not supported
    """
    provider_type = cfg.transcription_provider

    if provider_type == "whisper":
        from .whisper_provider import WhisperTranscriptionProvider
        return WhisperTranscriptionProvider(cfg)
    elif provider_type == "openai":

```python

        from .openai_provider import OpenAITranscriptionProvider
        return OpenAITranscriptionProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported transcription provider: {provider_type}. "
            "Supported providers: 'whisper', 'openai'."
        )

```

### Step 4: Add Configuration Support

Add your provider to the config `Literal` type and validators:

```python

        from .openai_provider import OpenAITranscriptionProvider
        return OpenAITranscriptionProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported transcription provider: {provider_type}. "
            "Supported providers: 'whisper', 'openai'."
        )

```

### Step 5: Add CLI Support (Optional)

Add CLI arguments for your provider.

### Step 6: Add E2E Server Mock Endpoint (For API Providers)

Add mock endpoint handlers to the E2E server for testing.

### Step 7: Update E2E Server URL Helper

Add a helper method to `E2EServerURLs` class for your API base URL.

**Reference**: See the E2E HTTP server implementation in `tests/e2e/fixtures/e2e_http_server.py`.

### Step 8: Write Tests

Write comprehensive unit, integration, and E2E tests.

## Testing Your Provider

> **See also:**
>
> - [Unit Testing Guide](UNIT_TESTING_GUIDE.md) - General unit test patterns and isolation
> - [Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md) - Integration test mocking guidelines
> - [E2E Testing Guide](E2E_TESTING_GUIDE.md) - E2E server and real ML model usage

Provider tests follow the standard test pyramid but require **provider-specific patterns** at each tier.

### E2E Server Mock Endpoints

For API providers, you'll need to add mock endpoint handlers. Here's an example for OpenAI:

```python

def do_POST(self):
    """Handle POST requests for OpenAI API endpoints."""
    path = self.path.split("?")[0]  # Remove query string

    # Route: OpenAI Whisper API endpoint
    if path == "/v1/audio/transcriptions":
        self._handle_audio_transcriptions()
        return

    # 404 for all other paths
    self.send_error(404, "OpenAI endpoint not found")

def _handle_audio_transcriptions(self):
    """Handle OpenAI audio transcriptions API requests."""
    try:
        # Parse multipart form data
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):

```text

            self.send_error(400, "Content-Type must be multipart/form-data")
            return

        # Read and parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Extract filename from multipart form data
        filename = "unknown_audio.mp3"
        # ... (filename extraction logic with validation) ...

        # Generate a realistic transcription response
        transcript = (
            f"This is a test transcription of {filename}. "
            "The audio contains spoken content that has been transcribed."
        )

        # Send response (text format)

```text

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(transcript)))
        self.end_headers()
        self.wfile.write(transcript.encode("utf-8"))

```python

    except Exception as e:
        self.send_error(500, f"Error handling audio transcriptions: {e}")

def _handle_chat_completions(self):
    """Handle OpenAI chat completions API requests."""
    try:
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        request_data = json.loads(body.decode("utf-8"))

        # Extract request details
        messages = request_data.get("messages", [])
        user_message = next((m for m in messages if m.get("role") == "user"), {})
        user_content = user_message.get("content", "")
        response_format = request_data.get("response_format", {})

        # Determine response type based on response_format

```text

        if response_format.get("type") == "json_object":
            # Speaker detection response
            response_data = {
                "id": "chatcmpl-test-speaker",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_data.get("model", "gpt-4o-mini"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "speakers": ["Host", "Guest"],
                                "hosts": ["Host"],
                                "guests": ["Guest"],
                            }),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            }
        else:

```text

            # Summarization response
            summary_length = min(200, len(user_content) // 10)
            summary = f"This is a test summary of the transcript. {user_content[:summary_length]}..."

```json

            response_data = {
                "id": "chatcmpl-test-summary",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_data.get("model", "gpt-4o-mini"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": summary},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            }

```text

        # Send response
        response_json = json.dumps(response_data)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        self.wfile.write(response_json.encode("utf-8"))

    except json.JSONDecodeError:
        self.send_error(400, "Invalid JSON in request body")
    except Exception as e:
        self.send_error(500, f"Error handling chat completions: {e}")

```python

**Key Points:**

- Handle different request types (e.g., speaker detection vs summarization)
- Include proper error handling and validation
- Use path traversal protection for any file operations

### Step 7: Update E2E Server URL Helper (Reference)

Add a helper method to `E2EServerURLs` class for your API base URL.

```python

class E2EServerURLs:
    """URL helper class for E2E server."""

    def openai_api_base(self) -> str:
        """Get OpenAI API base URL for E2E testing.

        Returns:
            TranscriptionProvider instance
        """
        # Implementation here

```

#### Unit Tests

**Location**: `tests/unit/podcast_scraper/test_openai_providers.py`

```python

"""Unit tests for OpenAI providers."""

import unittest
from unittest.mock import MagicMock, Mock, patch

from podcast_scraper import config
from podcast_scraper.transcription.factory import create_transcription_provider

class TestOpenAITranscriptionProvider(unittest.TestCase):
    """Test OpenAI transcription provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
        )

```python

    def test_provider_creation(self):
        """Test that provider can be created."""
        provider = create_transcription_provider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "OpenAITranscriptionProvider")

```python

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = create_transcription_provider(self.cfg)
        provider.initialize()
        self.assertTrue(provider.is_initialized)

```python

    @patch("podcast_scraper.transcription.openai_provider.OpenAI")
    def test_transcribe(self, mock_openai_class):
        """Test transcription."""
        # Setup mock
        mock_client = MagicMock()
        mock_transcript = Mock()
        mock_transcript.text = "Test transcription"
        mock_client.audio.transcriptions.create.return_value = mock_transcript
        mock_openai_class.return_value = mock_client

```python

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Create a temporary audio file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio")
            audio_path = f.name

        try:

        try:
            result = provider.transcribe(audio_path)
            self.assertEqual(result, "Test transcription")
            mock_client.audio.transcriptions.create.assert_called_once()
        finally:
            import os
            os.unlink(audio_path)

```python

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises error."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            # openai_api_key not set
        )
        with self.assertRaises(ValueError) as cm:
            create_transcription_provider(cfg)
        self.assertIn("OpenAI API key required", str(cm.exception))

```

#### Integration Tests

**Location**: `tests/integration/test_openai_providers.py`

```python

import pytest

from podcast_scraper import config
from podcast_scraper.transcription.factory import create_transcription_provider

@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.openai
class TestOpenAITranscriptionProviderIntegration(unittest.TestCase):
    """Integration tests for OpenAI transcription provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
            openai_api_base="http://localhost:8000/v1",  # Use E2E server
        )

```python

    def test_provider_works_with_e2e_server(self, e2e_server):
        """Test that provider works with E2E server mock endpoints."""
        # Update config to use E2E server
        self.cfg.openai_api_base = e2e_server.urls.openai_api_base()

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Create test audio file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio")
            audio_path = f.name

        try:
            result = provider.transcribe(audio_path)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
        finally:

```python

            import os
            os.unlink(audio_path)
            provider.cleanup()

```python

from podcast_scraper import workflow
from conftest import create_test_config

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.openai
class TestOpenAIProviderE2E:
    """Test OpenAI providers in integration workflows using E2E server."""

    def test_openai_transcription_in_pipeline(self, e2e_server):
        """Test OpenAI transcription provider in full pipeline."""
        import tempfile
        import shutil
        from pathlib import Path

        temp_dir = tempfile.mkdtemp()
        try:

```text

            # Create config with OpenAI transcription using E2E server
            cfg = create_test_config(
                rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
                output_dir=temp_dir,
                transcription_provider="openai",
                openai_api_key="sk-test123",
                openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
                transcribe_missing=True,
                generate_metadata=True,
                max_episodes=1,
            )

            # Run pipeline (uses E2E server OpenAI endpoints)
            transcripts_saved, summary = workflow.run_pipeline(cfg)

            # Verify transcripts were saved
            assert transcripts_saved > 0

```text

            # Verify transcript files were created
            transcript_files = list(Path(temp_dir).rglob("*.txt"))
            assert len(transcript_files) >= 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

```

#### E2E Server Endpoint Tests

**Location**: `tests/e2e/test_e2e_server.py`

```python

import requests
import pytest

@pytest.mark.e2e
class TestE2EServerOpenAIEndpoints:
    """Test that E2E server OpenAI mock endpoints work correctly."""

    def test_audio_transcriptions_endpoint(self, e2e_server):
        """Test that audio transcriptions endpoint works."""
        import tempfile

        openai_api_base = e2e_server.urls.openai_api_base()
        url = f"{openai_api_base}/audio/transcriptions"

        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data for testing")
            audio_path = f.name

```text

        try:
            # Create multipart form data request
            with open(audio_path, "rb") as audio_file:
                files = {"file": ("test_audio.mp3", audio_file, "audio/mpeg")}
                data = {"model": "whisper-1"}

                response = requests.post(url, files=files, data=data, timeout=5)
                assert response.status_code == 200
                assert "text/plain" in response.headers.get("Content-Type", "")
                assert len(response.text) > 0
                assert "test transcription" in response.text.lower()
        finally:
            import os
            os.unlink(audio_path)

```

## Testing Checklist

### Unit Test Checklist

- [ ] Error handling (missing API key, invalid inputs)
- [ ] Cleanup
- [ ] Configuration validation

### Integration Test Checklist

- [ ] Provider works with E2E server mock endpoints
- [ ] Provider switching (test with different providers)
- [ ] Error handling in workflow context

### E2E Tests

- [ ] Provider works in full pipeline
- [ ] Provider works with real HTTP client (E2E server)
- [ ] Multiple providers work together
- [ ] Error scenarios (API failures, rate limits)

### E2E Server Endpoint Test Checklist

- [ ] Mock endpoint returns correct format
- [ ] Mock endpoint handles different request types
- [ ] Mock endpoint error handling
- [ ] URL helper methods work correctly

## Configuration Reference

### Config Fields

For API providers, you typically need:

```python

# Provider selection

transcription_provider: Literal["whisper", "openai"] = "whisper"
speaker_detector_provider: Literal["ner", "openai"] = "ner"
summary_provider: Literal["local", "openai"] = "local"

# API credentials

openai_api_key: Optional[str] = None  # Loaded from OPENAI_API_KEY env var

# API base URL (for E2E testing)

openai_api_base: Optional[str] = None  # Defaults to OpenAI's production API

```

## Factory Registration

- [ ] Update `{capability}/factory.py` to include new provider
- [ ] Add provider to factory error message

### Configuration

- [ ] Add provider to `Literal` type in `config.py`
- [ ] Add field validator (if needed)
- [ ] Add API key validation (for API providers)
- [ ] Update config docstrings

### CLI Support (Optional)

- [ ] Add CLI argument to appropriate argument group
- [ ] Update `_build_config()` to include argument

### E2E Server Mocking (For API Providers)

- [ ] Add mock endpoint handler to `E2EHTTPRequestHandler`
- [ ] Implement `_handle_{endpoint}()` method
- [ ] Add route in `do_POST()` or `do_GET()`
- [ ] Add URL helper method to `E2EServerURLs`
- [ ] Test mock endpoint directly

### Testing

- [ ] Write unit tests (`tests/unit/podcast_scraper/test_{provider}_providers.py`)
- [ ] Write integration tests (`tests/integration/test_{provider}_providers.py`)
- [ ] Write E2E tests (`tests/e2e/test_{provider}_provider_integration_e2e.py`)
- [ ] Write E2E server endpoint tests (`tests/e2e/test_e2e_server.py`)
- [ ] All tests pass

### Documentation

- [ ] Update provider list in relevant docs
- [ ] Add examples to this guide (if needed)
- [ ] Update API reference docs

## Common Patterns

### Lazy Initialization

```python

def transcribe(self, audio_path: str, language: str | None = None) -> str:
    if not self._initialized:
        self.initialize()
    # ... rest of implementation

```

### Configuration Validation

```python

    if not cfg.api_key:
        raise ValueError("api_key required for CustomProvider")
    self.cfg = cfg

```

### Custom Base URL Support

```python

    client_kwargs: dict[str, Any] = {"api_key": cfg.api_key}
    if cfg.api_base:
        client_kwargs["base_url"] = cfg.api_base
    self.client = APIClient(**client_kwargs)

```

### Error Handling

```python

    if not self._initialized:
        raise RuntimeError("Provider not initialized")

    try:
        # API call
        return result
    except APIClientError as e:
        logger.error("API call failed: %s", e)
        raise ValueError(f"Transcription failed: {e}") from e

```

### Resource Management

- Implement `initialize()` and `cleanup()` methods
- Clean up resources in `cleanup()`
- Handle initialization failures gracefully
- Support re-initialization after cleanup

### Logging

- Use module-level logger: `logger = logging.getLogger(__name__)`
- Log initialization and cleanup
- Log errors with context
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

### Type Hints

- Use type hints for all methods
- Match protocol type hints exactly
- Use `Optional` for nullable parameters
- Use `Protocol` types for dependencies

### Test Coverage

- Test provider creation and initialization
- Test protocol method implementation
- Test error handling
- Test with E2E server mock endpoints
- Test in full pipeline context

## Migration from Direct Calls

If you're migrating existing code to use providers, see the migration patterns:

### Before (Direct Call)

```python

from podcast_scraper import whisper_integration as whisper

whisper_model = whisper.load_whisper_model(cfg)
result, elapsed = whisper.transcribe_with_whisper(whisper_model, audio_path, cfg)

```

### E2E Server Mocking

- **Mock Endpoints**: See `tests/e2e/fixtures/e2e_http_server.py` (lines 323-502)
- **URL Helpers**: See `tests/e2e/fixtures/e2e_http_server.py` (lines 32-92)

### Tests

- **Unit Tests**: `tests/unit/podcast_scraper/test_openai_providers.py`
- **Integration Tests**: `tests/integration/test_openai_providers.py`
- **E2E Tests**: `tests/e2e/test_openai_provider_integration_e2e.py`
- **E2E Server Tests**: `tests/e2e/test_e2e_server.py` (lines 111-261)

## Related Documentation

- [Protocol Extension Guide](./PROTOCOL_EXTENSION_GUIDE.md) - How to extend protocols
- [Testing Guide](./TESTING_GUIDE.md) - Quick reference and test execution
- [Unit Testing Guide](./UNIT_TESTING_GUIDE.md) - Unit test patterns
- [Integration Testing Guide](./INTEGRATION_TESTING_GUIDE.md) - Integration test guidelines
- [E2E Testing Guide](./E2E_TESTING_GUIDE.md) - E2E testing with real ML
- [Development Guide](./DEVELOPMENT_GUIDE.md) - Development workflow

## Questions?

- Check existing provider implementations (OpenAI, Whisper, NER) for examples
- Review protocol definitions in `{capability}/base.py`
- See test files for testing patterns
- Open an issue for questions or clarifications
