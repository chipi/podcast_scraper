# Protocol Extension Guide

This guide explains how to add new methods to existing protocols or extend protocol functionality. This is useful when
adding new capabilities or features to providers.

## Overview

Protocols define the interface that all providers must implement. When you need to add new functionality, you may need
to extend protocols to include new methods. This guide shows how to do this correctly.

## Protocol Structure

Protocols are defined using Python's `Protocol` type from `typing`. Each capability has a protocol in
`{capability}/base.py`:

- `TranscriptionProvider` in `transcription/base.py`
- `SpeakerDetector` in `speaker_detectors/base.py`
- `SummarizationProvider` in `summarization/base.py`

## Step-by-Step: Adding a Method to a Protocol

### Step 1: Define the Method in the Protocol

Add the new method signature to the protocol class:

````python

# transcription/base.py

class TranscriptionProvider(Protocol):

    # ... existing methods ...

    def new_method(
        self,
        param1: str,
        param2: int | None = None,
    ) -> dict[str, object]:
        """New method description.

        Args:
            param1: Description of param1
            param2: Optional description of param2

        Returns:
            Dictionary with results

```text

        Raises:
            RuntimeError: If provider is not initialized
            ValueError: If operation fails
        """
        ...

```
## Example: WhisperTranscriptionProvider

```python

# transcription/whisper_provider.py

class WhisperTranscriptionProvider:

    # ... existing methods ...

    def new_method(
        self,
        param1: str,
        param2: int | None = None,
    ) -> dict[str, object]:
        """Implementation of new_method for Whisper provider."""
        if not self.is_initialized:
            raise RuntimeError("Provider not initialized")

        # Implementation here

        return {"result": "value"}

```python

# transcription/openai_provider.py

class OpenAITranscriptionProvider:

    # ... existing methods ...

    def new_method(
        self,
        param1: str,
        param2: int | None = None,
    ) -> dict[str, object]:
        """Implementation of new_method for OpenAI provider."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        # Implementation here (may differ from Whisper)

        return {"result": "value"}

```python

# transcription/factory.py

def create_transcription_provider(cfg: config.Config) -> TranscriptionProvider:
    """Create transcription provider."""
    provider_type = cfg.transcription_provider

    if provider_type == "whisper":
        from .whisper_provider import WhisperTranscriptionProvider
        return WhisperTranscriptionProvider(cfg)
    elif provider_type == "openai":
        from .openai_provider import OpenAITranscriptionProvider
        return OpenAITranscriptionProvider(cfg)

    # ... existing code ...

```python

# tests/unit/test_transcription_provider.py

class TestTranscriptionProvider(unittest.TestCase):
    def test_new_method(self):
        """Test new_method implementation."""
        provider = create_transcription_provider(cfg)
        provider.initialize()

        result = provider.new_method("test", param2=42)

        self.assertIsInstance(result, dict)
        self.assertIn("result", result)

```text

3. **API reference** - Add method to API docs
4. **Migration guide** - If migration is needed

---

## Example: Adding `transcribe_with_segments()` Method

This example shows how `transcribe_with_segments()` was added to `TranscriptionProvider`.

### Step 1: Protocol Definition

```python

# transcription/base.py

class TranscriptionProvider(Protocol):
    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> str:
        """Transcribe audio file to text."""
        ...

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> tuple[dict[str, object], float]:
        """Transcribe audio file and return full result with segments.

```text

        Returns:
            Tuple of (result_dict, elapsed_time)
        """

```python

        import time
        start_time = time.time()
        text = self.transcribe(audio_path, language)
        elapsed = time.time() - start_time
        return {"text": text, "segments": []}, elapsed

```

## WhisperTranscriptionProvider

```python

# transcription/whisper_provider.py

class WhisperTranscriptionProvider:
    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> tuple[dict[str, object], float]:
        """Transcribe with segments using Whisper."""
        if not self.is_initialized:
            raise RuntimeError("Provider not initialized")

        # Use existing whisper integration

        result_dict, elapsed = whisper_integration.transcribe_with_whisper(
            self._model, audio_path, self.cfg
        )
        return result_dict, elapsed

```python

# transcription/openai_provider.py

class OpenAITranscriptionProvider:
    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> tuple[dict[str, object], float]:
        """Transcribe with segments using OpenAI API."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        # Use OpenAI API

        result_dict, elapsed = self._transcribe_with_openai(audio_path, language)
        return result_dict, elapsed

```text

# episode_processor.py

if transcription_provider is not None:
    result_dict, elapsed = transcription_provider.transcribe_with_segments(
        audio_path, language=cfg.language
    )
    text = result_dict["text"]
    segments = result_dict["segments"]

```text

## 1. Provide Default Implementation When Possible

If a method can have a reasonable default implementation, provide it in the protocol:

```python

class TranscriptionProvider(Protocol):
    def optional_method(self, param: str) -> str:
        """Optional method with default implementation."""

        # Default implementation

        return f"default: {param}"

```bash

- ✅ Make new parameters optional with defaults
- ✅ Don't break existing method signatures
- ✅ Provide migration path for old code

### 3. Document Thoroughly

Always document:

- Method purpose and usage
- Parameters and return types
- Exceptions that may be raised
- Provider-specific behavior differences

### 4. Update All Providers

Ensure all providers implement the new method:

- ✅ Local providers (Whisper, NER, Transformers)
- ✅ API providers (OpenAI)
- ✅ Any custom providers

### 5. Add Tests

Add comprehensive tests:

- ✅ Test protocol compliance
- ✅ Test each provider implementation
- ✅ Test error cases
- ✅ Test edge cases

---

## Common Patterns

### Pattern 1: Optional Method with Default

```python

class TranscriptionProvider(Protocol):
    def optional_method(self, param: str) -> dict[str, object]:
        """Optional method with default implementation."""

        # Default: return empty dict

        return {}

```python

        """Override with Whisper-specific implementation."""
        return {"whisper_specific": "value"}

```python

class TranscriptionProvider(Protocol):
    def advanced_method(self, param: str) -> dict[str, object]:
        """Advanced method that requires initialization."""
        ...

```python

        if not self.is_initialized:
            raise RuntimeError("Provider not initialized")

        # Implementation

```text
```python

class TranscriptionProvider(Protocol):
    def get_model_info(self) -> dict[str, object]:
        """Get information about the model being used.

        Returns:
            Dictionary with model information.
            Structure may vary by provider.
        """
        ...

```python

class WhisperTranscriptionProvider:
    def get_model_info(self) -> dict[str, object]:
        return {
            "type": "local",
            "model_name": self.cfg.whisper_model,
            "device": self._device,
        }

```python

        return {
            "type": "api",
            "model": self.model,
            "api_endpoint": "https://api.openai.com/v1/audio/transcriptions",
        }

```text

- [ ] Define method signature in protocol
- [ ] Add method docstring with full documentation
- [ ] Implement method in all existing providers
- [ ] Provide default implementation if possible
- [ ] Update factory functions if needed
- [ ] Add tests for protocol compliance
- [ ] Add tests for each provider implementation
- [ ] Update protocol documentation
- [ ] Update provider guide documentation
- [ ] Update API reference
- [ ] Update migration guide if needed
- [ ] Verify all tests pass

---

## Troubleshooting

### Issue: Method Not Found on Provider

**Error**: `AttributeError: 'Provider' object has no attribute 'new_method'`

**Solution**: Ensure all providers implement the new method. Check that you've updated all provider implementations.

### Issue: Protocol Type Checking Fails

**Error**: Type checker complains about missing method

**Solution**: Ensure the method signature matches exactly in protocol and all implementations. Check return types match.

### Issue: Default Implementation Not Used

**Error**: Provider doesn't use default implementation

**Solution**: Default implementations in protocols are only used if provider doesn't override. If provider needs default, explicitly call it or provide implementation.

---

## Related Documentation

- [Provider Implementation Guide](./PROVIDER_IMPLEMENTATION_GUIDE.md) - Complete guide for implementing new providers (includes OpenAI example, testing, E2E server mocking)

---

## Summary

Extending protocols involves:

1. Adding method signature to protocol
2. Implementing method in all providers
3. Adding tests
4. Updating documentation

Follow best practices for backward compatibility, documentation, and testing to ensure smooth protocol extensions.

````
