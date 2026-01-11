# Protocol Extension Guide

This guide explains how to add new methods to existing protocols or extend protocol functionality.

## Overview

Protocols define the interface that all providers must implement. When adding new functionality, you may need to extend protocols while ensuring all providers stay compliant.

## Protocol Structure

Protocols are defined using Python's `Protocol` type from `typing` in `{capability}/base.py`:

- `TranscriptionProvider` in `transcription/base.py`
- `SpeakerDetector` in `speaker_detectors/base.py`
- `SummarizationProvider` in `summarization/base.py`

## Step-by-Step: Extending a Protocol

### Step 1: Define the Method in the Protocol

Add the method signature to the protocol class. **Best Practice**: Provide a default implementation in the protocol if possible to make the change non-breaking for existing providers.

```python
# transcription/base.py

class TranscriptionProvider(Protocol):
    # ... existing methods ...

    def transcribe_with_segments(self, audio_path: str) -> tuple[dict, float]:
        """Default implementation that can be overridden."""
        text = self.transcribe(audio_path)
        return {"text": text, "segments": []}, 0.0
```

### Step 2: Update Unified Providers

**CRITICAL**: Since `MLProvider` and `OpenAIProvider` each implement **multiple** protocols, you must update the relevant sections of both classes.

1. **`src/podcast_scraper/ml/ml_provider.py`**: Update the local implementation.
2. **`src/podcast_scraper/openai/openai_provider.py`**: Update the API implementation.

### Step 3: Optional Methods

If a method is not applicable to all providers, it can return `None` or a default value. For example, `analyze_patterns` in `SpeakerDetector` is implemented by `MLProvider` but returns `None` in `OpenAIProvider`.

```python
# openai/openai_provider.py

def analyze_patterns(self, episodes, known_hosts):
    """OpenAI provider uses local logic for pattern analysis."""
    return None
```

---

## Best Practices

### 1. Maintain Backward Compatibility

- ✅ Make new parameters optional with defaults.
- ✅ Don't change existing method names or signatures.
- ✅ Use generic types (like `dict[str, object]`) for provider-specific metadata.

### 2. Update Documentation & Tests

- [ ] Add the method to the appropriate Protocol class.
- [ ] Implement/override in `MLProvider` and `OpenAIProvider`.
- [ ] Add unit tests for protocol compliance.
- [ ] Update the API Reference if the method is public.

## Extension Checklist

- [ ] **Protocol**: Signature added with docstring.
- [ ] **Implementations**: Method implemented in all unified providers.
- [ ] **Factories**: Factory functions updated if the new method requires config changes.
- [ ] **Tests**: Verified with `make test-unit`.

## Related Documentation

- [Provider Implementation Guide](./PROVIDER_IMPLEMENTATION_GUIDE.md) - How to implement new providers.
- [ML Provider Reference](./ML_PROVIDER_REFERENCE.md) - Technical details on the local ML implementation.
