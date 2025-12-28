# Custom Provider Guide

This guide explains how to create custom providers for the podcast scraper. Providers allow you to extend the system with new implementations for transcription, speaker detection, or summarization.

## Overview

The podcast scraper uses a **protocol-based provider system** where each capability (transcription, speaker detection, summarization) has a protocol interface that all providers must implement. This design allows:

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
podcast_scraper/
├── {capability}/
│   ├── __init__.py
│   ├── base.py          # Protocol definition
│   ├── factory.py       # Factory function
│   └── {provider}_provider.py  # Implementation
```

## Creating a Custom Provider

### Step 1: Understand the Protocol

First, examine the protocol interface in `{capability}/base.py`. For example, `TranscriptionProvider`:

```python
from typing import Protocol

class TranscriptionProvider(Protocol):
    def transcribe(
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
```

### Step 2: Implement the Provider Class

Create a new file `{capability}/{your_provider}_provider.py`:

```python
"""Your custom provider implementation."""

from __future__ import annotations

import logging
from typing import Optional

from .base import TranscriptionProvider
from .. import config

logger = logging.getLogger(__name__)


class YourCustomProvider:
    """Your custom transcription provider."""

    def __init__(self, cfg: config.Config):
        """Initialize provider with configuration.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self._initialized = False

    def initialize(self) -> None:
        """Initialize provider (load models, connect to API, etc.)."""
        if self._initialized:
            return

        logger.debug("Initializing YourCustomProvider")
        # Your initialization logic here
        self._initialized = True

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Optional language code

        Returns:
            Transcribed text
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        # Your transcription logic here
        return "transcribed text"

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._initialized = False
```

### Step 3: Register in Factory

Update `{capability}/factory.py` to include your provider:

```python
def create_transcription_provider(cfg: config.Config) -> TranscriptionProvider:
    """Create a transcription provider based on configuration."""
    provider_type = cfg.transcription_provider

    if provider_type == "whisper":
        from .whisper_provider import WhisperTranscriptionProvider
        return WhisperTranscriptionProvider(cfg)
    elif provider_type == "your_provider":
        from .your_provider_provider import YourCustomProvider
        return YourCustomProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported transcription provider: {provider_type}. "
            "Supported providers: 'whisper', 'your_provider'"
        )
```

### Step 4: Add Configuration Support

Update `config.py` to support your provider:

```python
transcription_provider: Literal["whisper", "your_provider"] = Field(
    default="whisper",
    alias="transcription_provider",
    description="Transcription provider type",
)
```

### Step 5: Write Tests

Create `tests/test_{your_provider}_provider.py`:

```python
"""Tests for YourCustomProvider."""

import unittest
from unittest.mock import Mock, patch

from podcast_scraper import config
from podcast_scraper.transcription.factory import create_transcription_provider


class TestYourCustomProvider(unittest.TestCase):
    """Test YourCustomProvider implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="your_provider",
        )

    def test_provider_creation(self):
        """Test that provider can be created."""
        provider = create_transcription_provider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "YourCustomProvider")

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = create_transcription_provider(self.cfg)
        provider.initialize()
        self.assertTrue(provider.is_initialized)

    def test_provider_transcribe(self):
        """Test transcription."""
        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Mock your transcription logic
        with patch.object(provider, "_your_transcription_method", return_value="test"):
            result = provider.transcribe("test_audio.mp3")
            self.assertEqual(result, "test")

    def test_provider_protocol_compliance(self):
        """Test that provider implements protocol."""
        provider = create_transcription_provider(self.cfg)

        # Verify required methods exist
        self.assertTrue(hasattr(provider, "transcribe"))

        # Verify method signature
        import inspect
        sig = inspect.signature(provider.transcribe)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("language", sig.parameters)
```

## Example Implementations

### Minimal Provider

A minimal provider that implements only the protocol:

```python
class MinimalProvider:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        return "minimal transcription"
```

### Full-Featured Provider

A provider with initialization, cleanup, and error handling:

```python
class FullFeaturedProvider:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self._client = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        # Initialize API client, load models, etc.
        self._client = YourAPIClient(api_key=self.cfg.api_key)
        self._initialized = True

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        try:
            return self._client.transcribe(audio_path, language=language)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise ValueError(f"Transcription failed: {e}") from e

    def cleanup(self) -> None:
        if self._client:
            self._client.close()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized
```

### Custom Config Provider

A provider that uses custom configuration fields:

```python
class CustomConfigProvider:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        # Access custom config fields
        self.api_endpoint = getattr(cfg, "custom_api_endpoint", "https://api.example.com")
        self.timeout = getattr(cfg, "custom_timeout", 30)
```

## Testing Requirements

### Unit Tests

- Test provider creation
- Test initialization
- Test protocol methods
- Test error handling
- Test cleanup

### Integration Tests

- Test provider works in workflow
- Test provider switching
- Test error handling in workflow

### Protocol Compliance Tests

- Verify all protocol methods are implemented
- Verify method signatures match protocol
- Verify return types match protocol

## Pull Request Process

When contributing a custom provider:

1. **Create Feature Branch**: `feature/add-{provider}-provider`
2. **Implement Provider**: Follow the steps above
3. **Add Tests**: Comprehensive test coverage (>80%)
4. **Update Documentation**: Add provider to relevant docs
5. **Update Config**: Add provider to config validation
6. **Submit PR**: Include:
   - Provider implementation
   - Tests
   - Documentation updates
   - Example usage
   - Migration guide (if replacing existing functionality)

## Best Practices

### Error Handling

- Always raise `RuntimeError` if provider not initialized
- Raise `ValueError` for invalid inputs
- Log errors before raising exceptions
- Provide helpful error messages

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
def __init__(self, cfg: config.Config):
    if not cfg.api_key:
        raise ValueError("api_key required for CustomProvider")
    self.cfg = cfg
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _transcribe_cached(self, audio_path: str) -> str:
    # Expensive transcription logic
    pass
```

## Questions?

- Check existing provider implementations for examples
- Review protocol definitions in `{capability}/base.py`
- See `tests/test_{capability}_provider.py` for test examples
- Open an issue for questions or clarifications
