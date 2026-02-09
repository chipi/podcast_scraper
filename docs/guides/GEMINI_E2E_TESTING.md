# Gemini E2E Testing Guide

## Overview

Gemini E2E tests use a **fake SDK client** that routes calls to the E2E mock server via HTTP, similar to how OpenAI tests work. This provides consistent testing infrastructure across all API providers.

## How It Works

### Fake SDK Client

Instead of mocking individual SDK calls, we replace the Gemini SDK's `GenerativeModel` class with a fake client that:

1. **Intercepts SDK calls** - When code calls `genai.GenerativeModel().generate_content()`
2. **Routes to E2E server** - Makes HTTP POST requests to `http://127.0.0.1:8000/v1beta/models/{model}:generateContent`
3. **Returns fake responses** - Converts E2E server responses back to SDK-compatible format

### E2E Server Endpoints

The E2E server provides Gemini mock endpoints:

- **POST `/v1beta/models/{model}:generateContent`**
  - Handles transcription (multimodal with audio)
  - Handles speaker detection (JSON response)
  - Handles summarization (text response)

### Test Flow

```text
Test Code
  ↓
GeminiProvider.transcribe()
  ↓
genai.GenerativeModel()  ← Replaced with FakeGenerativeModel
  ↓
HTTP POST to E2E Server
  ↓
E2E Server Mock Endpoint
  ↓
Returns Mock Response
  ↓
FakeGeminiResponse (SDK-compatible)
  ↓
Test continues normally
```

## Configuration

### Automatic Setup

The `configure_gemini_mock_server` fixture in `tests/e2e/conftest.py` automatically:

1. Gets E2E server base URL
2. Creates fake client bound to that URL
3. Replaces `google.genai.GenerativeModel` with fake client
4. Sets `GEMINI_API_BASE` environment variable

### Manual Setup (if needed)

```python
from tests.e2e.fixtures.gemini_mock_client import create_fake_gemini_client

# In test setup
gemini_api_base = e2e_server.urls.gemini_api_base()
FakeGenerativeModel = create_fake_gemini_client(gemini_api_base)

# Replace SDK class
monkeypatch.setattr("google.genai.GenerativeModel", FakeGenerativeModel)
```

## Benefits Over Python-Level Mocking

| Aspect | Python Mocking | Fake SDK Client |
| --- | --- | --- |
| **HTTP Testing** | ❌ No HTTP layer | ✅ Real HTTP requests |
| **Network Testing** | ❌ Bypasses network | ✅ Tests full stack |
| **Consistency** | ❌ Different from OpenAI | ✅ Same as OpenAI |
| **Realism** | ❌ Only tests Python code | ✅ Tests HTTP + Python |
| **Maintenance** | ❌ Complex mocks | ✅ Single mock server |

## Example Test

```python
@pytest.mark.e2e
def test_gemini_transcription_in_pipeline(e2e_server):
    """Test Gemini transcription with fake SDK routing to E2E server."""
    # E2E server automatically configured via conftest fixture
    # Fake SDK client automatically replaces real SDK

    cfg = create_test_config(
        rss_url=e2e_server.urls.feed("podcast1"),
        transcription_provider="gemini",
        gemini_api_key="test-key",
        gemini_api_base=e2e_server.urls.gemini_api_base(),
    )

    # This will use fake SDK → E2E server (real HTTP)
    provider = create_transcription_provider(cfg)
    provider.initialize()

    transcript = provider.transcribe(audio_path)
    # Uses HTTP POST to E2E server, not real Google API
```

## Real API Mode

To test with real Gemini API (manual testing only):

```bash
USE_REAL_GEMINI_API=1 pytest tests/e2e/test_gemini_provider_integration_e2e.py
```

This disables the fake client and uses the real Gemini SDK.

## Implementation Details

### Fake Client Location

- **File**: `tests/e2e/fixtures/gemini_mock_client.py`
- **Class**: `FakeGenerativeModel`
- **Function**: `create_fake_gemini_client(base_url)`

### E2E Server Handler

- **File**: `tests/e2e/fixtures/e2e_http_server.py`
- **Method**: `_handle_gemini_generate_content()`
- **Endpoint**: `/v1beta/models/{model}:generateContent`

### Conftest Integration

- **File**: `tests/e2e/conftest.py`
- **Fixture**: `configure_gemini_mock_server`
- **Auto-applied**: Yes (autouse=True)

## Troubleshooting

### Fake Client Not Working

1. Check that E2E server is running: `e2e_server` fixture should be available
2. Verify Gemini endpoints exist: Check `e2e_http_server.py` for `_handle_gemini_generate_content`
3. Check logs: Fake client logs HTTP requests at DEBUG level

### HTTP Errors

- **404**: E2E server endpoint not found → Check endpoint path matches
- **500**: E2E server handler error → Check server logs
- **Connection refused**: E2E server not running → Check fixture setup

### Import Errors

If `google.genai` is not installed, the fake client will still work (it doesn't import the real SDK).
