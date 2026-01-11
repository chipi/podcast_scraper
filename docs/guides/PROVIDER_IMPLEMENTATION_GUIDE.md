# Provider Implementation Guide

This comprehensive guide explains how to implement new providers for the podcast scraper. It consolidates information from multiple guides and uses OpenAI as a complete example throughout.

## Overview

The podcast scraper uses a **protocol-based provider system** where each capability (transcription, speaker detection, summarization) has a protocol interface that all providers must implement.

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

### Unified Provider Pattern

As of v2.4.0, the project follows a **Unified Provider** pattern where a single class implementation handles multiple protocols using shared libraries or API clients.

- **`MLProvider`**: Unified local implementation using Whisper, spaCy, and Transformers.
- **`OpenAIProvider`**: Unified API implementation using OpenAI's various endpoints.

**File Structure:**

```text
src/podcast_scraper/
├── ml/
│   └── ml_provider.py       # Unified Local ML implementation
├── openai/
│   └── openai_provider.py   # Unified OpenAI API implementation
├── transcription/
│   ├── base.py              # Protocol definition
│   └── factory.py           # Factory logic
├── speaker_detectors/
│   ├── base.py              # Protocol definition
│   └── factory.py           # Factory logic
└── summarization/
    ├── base.py              # Protocol definition
    └── factory.py           # Factory logic
```

## Step-by-Step Implementation

### Step 1: Understand the Protocol

First, examine the protocol interface in `{capability}/base.py`. For example, `TranscriptionProvider`:

```python
from typing import Protocol

class TranscriptionProvider(Protocol):
    def initialize(self) -> None:
        """Initialize provider (load models, connect to API, etc.)."""
        ...

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> str:
        """Transcribe audio file to text."""
        ...
```

### Step 2: Implement the Provider Class

Create a new file for your provider. If your provider handles multiple capabilities, consider a unified structure like `openai/` or `ml/`.

**Reference Implementation**: `src/podcast_scraper/openai/openai_provider.py`

#### 1. Configuration Validation

Check required config fields in `__init__()`. API keys should be validated here.

#### 2. Thread Safety

Define `_requires_separate_instances` based on your implementation:

- `True`: For local ML models (HuggingFace/Whisper) that are not thread-safe.
- `False`: For API clients (OpenAI) that handle concurrent requests internally.

#### 3. Initialization Lifecycle

- **`__init__`**: Store configuration and initialize lightweight clients.
- **`initialize()`**: Load heavy resources (ML models) or perform network handshakes. This method must be idempotent.
- **Lazy Loading**: Call `initialize()` inside protocol methods if not already initialized.

#### 4. Error Handling

Use typed exceptions from `podcast_scraper.exceptions`:

- `ProviderConfigError`: For invalid configuration.
- `ProviderDependencyError`: For missing packages or models.
- `ProviderRuntimeError`: For API failures or inference errors.
- `ProviderNotInitializedError`: If a method is called before `initialize()`.

#### 5. Prompt Store (for LLMs)

Use the centralized `prompt_store` for LLM prompts:

```python
from ..prompt_store import render_prompt, get_prompt_metadata

# Render a versioned prompt
system_prompt = render_prompt("summarization/system_v1")
```

### Step 3: Register in Factory

Update the factory functions in `{capability}/factory.py` to include your new provider.

```python
def create_transcription_provider(cfg: config.Config) -> TranscriptionProvider:
    # ...
    if provider_type == "whisper":
        from ..ml.ml_provider import MLProvider
        return MLProvider(cfg)
    elif provider_type == "openai":
        from ..openai.openai_provider import OpenAIProvider
        return OpenAIProvider(cfg)
    # ...
```

---

## Testing Your Provider

### E2E Server Mock Endpoints

For API providers, you must add mock endpoint handlers to the E2E test server (`tests/e2e/fixtures/e2e_http_server.py`). This allows tests to run without real API keys or internet access.

### Testing Checklist

- [ ] **Unit Tests**: Test logic in isolation, mock all external dependencies.
- [ ] **Integration Tests**: Test provider with the real E2E server mock endpoints.
- [ ] **E2E Tests**: Test provider in the full pipeline context.
- [ ] **Resource Management**: Verify `cleanup()` properly unloads models or closes connections.

## Related Documentation

- [Protocol Extension Guide](./PROTOCOL_EXTENSION_GUIDE.md) - How to extend protocols
- [ML Provider Reference](./ML_PROVIDER_REFERENCE.md) - Details on local ML models
- [Development Guide](./DEVELOPMENT_GUIDE.md) - Development workflow
