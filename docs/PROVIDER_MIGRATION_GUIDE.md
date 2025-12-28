# Provider Migration Guide

This guide explains how to migrate code from direct module calls to the provider pattern. This is useful when updating existing code or integrating new features that need to use providers.

## Overview

The podcast scraper uses a **provider pattern** where capabilities (transcription, speaker detection, summarization) are accessed through protocol-based providers rather than direct module calls. This guide shows how to migrate from the old direct-call pattern to the new provider pattern.

## Why Migrate?

**Benefits of using providers**:

- ✅ **Pluggable implementations**: Switch between providers (e.g., Whisper vs OpenAI) via configuration
- ✅ **Type safety**: Protocols ensure consistent interfaces
- ✅ **Testability**: Easy to mock providers for testing
- ✅ **Consistency**: Single pattern across all capabilities
- ✅ **Future-proof**: Easy to add new providers without changing calling code

## Migration Patterns

### Pattern 1: Transcription

#### Before (Direct Call)

```python
from podcast_scraper import whisper_integration as whisper

# Load model directly
whisper_model = whisper.load_whisper_model(cfg)

# Transcribe directly
result, elapsed = whisper.transcribe_with_whisper(
    whisper_model, audio_path, cfg
)
text = result["text"]
```

#### After (Provider Pattern)

```python
from podcast_scraper.transcription.factory import create_transcription_provider

# Create and initialize provider
transcription_provider = create_transcription_provider(cfg)
transcription_provider.initialize()

# Use provider method
result_dict, elapsed = transcription_provider.transcribe_with_segments(
    audio_path, language=cfg.language
)
text = result_dict["text"]
```

#### Key Changes

- ✅ Replace `whisper.load_whisper_model()` with `create_transcription_provider()`
- ✅ Replace `whisper.transcribe_with_whisper()` with `provider.transcribe_with_segments()`
- ✅ Call `provider.initialize()` before use
- ✅ Call `provider.cleanup()` when done

---

### Pattern 2: Speaker Detection

#### Before (Direct Call)

```python
from podcast_scraper import speaker_detection

# Load model directly
nlp = speaker_detection.get_ner_model(cfg)

# Detect hosts directly
feed_hosts = speaker_detection.detect_hosts_from_feed(
    feed_title, feed_description, feed_authors, nlp=nlp
)

# Detect speakers directly
speaker_names, detected_hosts, success = speaker_detection.detect_speaker_names(
    episode_title, episode_description, cfg, cached_hosts, heuristics
)
```

#### After (Provider Pattern)

```python
from podcast_scraper.speaker_detectors.factory import create_speaker_detector

# Create and initialize detector
speaker_detector = create_speaker_detector(cfg)
speaker_detector.initialize()

# Use provider methods
feed_hosts = speaker_detector.detect_hosts(
    feed_title, feed_description, feed_authors
)

speaker_names, detected_hosts, success = speaker_detector.detect_speakers(
    episode_title, episode_description, known_hosts
)
```

#### Key Changes

- ✅ Replace `speaker_detection.get_ner_model()` with `create_speaker_detector()`
- ✅ Replace `detect_hosts_from_feed()` with `detector.detect_hosts()`
- ✅ Replace `detect_speaker_names()` with `detector.detect_speakers()`
- ✅ Remove manual `nlp` parameter passing (handled internally)
- ✅ Call `detector.initialize()` before use
- ✅ Call `detector.cleanup()` when done

---

### Pattern 3: Summarization

#### Before (Direct Call)

```python
from podcast_scraper import summarizer

# Load models directly
model_name = summarizer.select_summary_model(cfg)
summary_model = summarizer.SummaryModel(
    model_name=model_name,
    device=cfg.summary_device,
    cache_dir=cfg.summary_cache_dir,
)

# Summarize directly
result = summarizer.summarize_long_text(
    transcript_text,
    summary_model=summary_model,
    reduce_model=reduce_model,
    cfg=cfg,
)
summary = result["summary"]
```

#### After (Provider Pattern)

```python
from podcast_scraper.summarization.factory import create_summarization_provider

# Create and initialize provider
summary_provider = create_summarization_provider(cfg)
summary_provider.initialize()

# Use provider method
result = summary_provider.summarize(
    transcript_text,
    episode_title=episode_title,
    episode_description=episode_description,
)
summary = result["summary"]
```

#### Key Changes

- ✅ Replace `summarizer.SummaryModel()` with `create_summarization_provider()`
- ✅ Replace `summarizer.summarize_long_text()` with `provider.summarize()`
- ✅ Remove manual model loading (handled by provider)
- ✅ Call `provider.initialize()` before use
- ✅ Call `provider.cleanup()` when done

---

## Step-by-Step Migration Process

### Step 1: Identify Direct Calls

Look for these patterns in your code:

- `whisper.load_whisper_model()`
- `whisper.transcribe_with_whisper()`
- `speaker_detection.get_ner_model()`
- `speaker_detection.detect_hosts_from_feed()`
- `speaker_detection.detect_speaker_names()`
- `summarizer.SummaryModel()`
- `summarizer.summarize_long_text()`

### Step 2: Replace with Provider Creation

Replace direct module imports and calls with provider factory functions:

```python
# Old
from podcast_scraper import whisper_integration as whisper

# New
from podcast_scraper.transcription.factory import create_transcription_provider
```

### Step 3: Initialize Provider

Add provider initialization before use:

```python
# Create provider
provider = create_transcription_provider(cfg)

# Initialize (loads models, sets up connections)
provider.initialize()
```

### Step 4: Replace Method Calls

Replace direct function calls with provider methods:

```python
# Old
result = whisper.transcribe_with_whisper(model, audio_path, cfg)

# New
result_dict, elapsed = provider.transcribe_with_segments(audio_path)
```

### Step 5: Add Cleanup

Add provider cleanup when done:

```python
try:
    # Use provider
    result = provider.transcribe(audio_path)
finally:
    # Cleanup resources
    provider.cleanup()
```

### Step 6: Remove Direct Model Access

If you were accessing models directly, use provider methods instead:

```python
# Old
whisper_model = whisper.load_whisper_model(cfg)
result = whisper.transcribe_with_whisper(whisper_model, audio_path, cfg)

# New
provider = create_transcription_provider(cfg)
provider.initialize()
result_dict, elapsed = provider.transcribe_with_segments(audio_path)
```

---

## Common Migration Scenarios

### Scenario 1: Function Parameter Migration

#### Before

```python
def process_episode(audio_path: str, whisper_model, cfg: config.Config):
    result, elapsed = whisper.transcribe_with_whisper(whisper_model, audio_path, cfg)
    return result["text"]
```

#### After

```python
def process_episode(
    audio_path: str,
    transcription_provider: TranscriptionProvider,
    cfg: config.Config
):
    result_dict, elapsed = transcription_provider.transcribe_with_segments(audio_path)
    return result_dict["text"]
```

### Scenario 2: Class Attribute Migration

#### Before

```python
class EpisodeProcessor:
    def __init__(self, cfg: config.Config):
        self.whisper_model = whisper.load_whisper_model(cfg)
    
    def transcribe(self, audio_path: str):
        return whisper.transcribe_with_whisper(self.whisper_model, audio_path, self.cfg)
```

#### After

```python
class EpisodeProcessor:
    def __init__(self, cfg: config.Config):
        self.transcription_provider = create_transcription_provider(cfg)
        self.transcription_provider.initialize()
    
    def transcribe(self, audio_path: str):
        result_dict, elapsed = self.transcription_provider.transcribe_with_segments(audio_path)
        return result_dict["text"]
    
    def cleanup(self):
        self.transcription_provider.cleanup()
```

### Scenario 3: Conditional Provider Access

#### Before

```python
if cfg.transcription_provider == "whisper":
    model = whisper.load_whisper_model(cfg)
    result = whisper.transcribe_with_whisper(model, audio_path, cfg)
else:
    # Handle other cases
    pass
```

#### After

```python
# Provider selection is handled by factory
provider = create_transcription_provider(cfg)
provider.initialize()
result_dict, elapsed = provider.transcribe_with_segments(audio_path)
```

---

## Backward Compatibility

### Accessing Provider Internals (If Needed)

If you need to access provider internals for backward compatibility:

```python
# Get model from provider (backward compatibility)
whisper_model = getattr(transcription_provider, "model", None)
if whisper_model:
    # Use model directly if needed
    pass
```

**Note**: This is only for backward compatibility. Prefer using provider methods.

### Fallback Patterns

Some fallback patterns are still supported for backward compatibility:

```python
# In episode_processor.py - backward compatibility fallback
if transcription_provider is not None:
    result_dict, elapsed = transcription_provider.transcribe_with_segments(audio_path)
else:
    # Fallback to direct call (deprecated)
    result, elapsed = whisper.transcribe_with_whisper(whisper_model, audio_path, cfg)
```

**Note**: These fallbacks are deprecated and will be removed in future versions.

---

## Testing Migration

### Update Tests

When migrating, update tests to use providers:

#### Before

```python
def test_transcription():
    model = whisper.load_whisper_model(cfg)
    result = whisper.transcribe_with_whisper(model, "test.mp3", cfg)
    assert result["text"] == "expected text"
```

#### After

```python
def test_transcription():
    provider = create_transcription_provider(cfg)
    provider.initialize()
    result_dict, elapsed = provider.transcribe_with_segments("test.mp3")
    assert result_dict["text"] == "expected text"
    provider.cleanup()
```

### Mock Providers

Use mocks for testing:

```python
from unittest.mock import Mock

def test_with_mock_provider():
    mock_provider = Mock()
    mock_provider.transcribe_with_segments.return_value = (
        {"text": "mocked text"}, 1.0
    )
    
    result_dict, elapsed = mock_provider.transcribe_with_segments("test.mp3")
    assert result_dict["text"] == "mocked text"
```

---

## Migration Checklist

- [ ] Identify all direct module calls
- [ ] Replace imports with provider factory imports
- [ ] Create providers using factory functions
- [ ] Add `initialize()` calls before use
- [ ] Replace direct function calls with provider methods
- [ ] Add `cleanup()` calls when done
- [ ] Remove direct model access (if applicable)
- [ ] Update function signatures to accept providers
- [ ] Update tests to use providers
- [ ] Remove backward compatibility code (if applicable)
- [ ] Verify all functionality works

---

## Examples

### Complete Example: Migrating Episode Processing

#### Before

```python
from podcast_scraper import whisper_integration as whisper
from podcast_scraper import speaker_detection
from podcast_scraper import summarizer

def process_episode(episode, cfg):
    # Load models
    whisper_model = whisper.load_whisper_model(cfg)
    nlp = speaker_detection.get_ner_model(cfg)
    summary_model = summarizer.SummaryModel(...)
    
    # Transcribe
    result, elapsed = whisper.transcribe_with_whisper(whisper_model, episode.audio_path, cfg)
    text = result["text"]
    
    # Detect speakers
    hosts = speaker_detection.detect_hosts_from_feed(...)
    speakers, _, _ = speaker_detection.detect_speaker_names(...)
    
    # Summarize
    summary_result = summarizer.summarize_long_text(text, summary_model, ...)
    
    return text, speakers, summary_result["summary"]
```

#### After

```python
from podcast_scraper.transcription.factory import create_transcription_provider
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider

def process_episode(episode, cfg):
    # Create providers
    transcription_provider = create_transcription_provider(cfg)
    speaker_detector = create_speaker_detector(cfg)
    summary_provider = create_summarization_provider(cfg)
    
    # Initialize
    transcription_provider.initialize()
    speaker_detector.initialize()
    summary_provider.initialize()
    
    try:
        # Transcribe
        result_dict, elapsed = transcription_provider.transcribe_with_segments(
            episode.audio_path
        )
        text = result_dict["text"]
        
        # Detect speakers
        hosts = speaker_detector.detect_hosts(...)
        speakers, _, _ = speaker_detector.detect_speakers(...)
        
        # Summarize
        summary_result = summary_provider.summarize(text, ...)
        
        return text, speakers, summary_result["summary"]
    finally:
        # Cleanup
        transcription_provider.cleanup()
        speaker_detector.cleanup()
        summary_provider.cleanup()
```

---

## Troubleshooting

### Issue: Provider Not Initialized

**Error**: `RuntimeError: Provider not initialized`

**Solution**: Call `provider.initialize()` before using provider methods.

### Issue: Method Not Found

**Error**: `AttributeError: 'Provider' object has no attribute 'method'`

**Solution**: Check that you're using the correct provider method name. See protocol definitions in `{capability}/base.py`.

### Issue: Provider Creation Fails

**Error**: `ValueError: Unsupported provider type`

**Solution**: Check configuration. Ensure `transcription_provider`, `speaker_detector_provider`, or `summary_provider` is set to a valid value.

### Issue: Missing API Key

**Error**: `ValidationError: OpenAI API key required`

**Solution**: Set `OPENAI_API_KEY` environment variable or `openai_api_key` in config when using OpenAI providers.

---

## Fallback Behavior

During migration to the provider pattern, the system includes fallback mechanisms to support backward compatibility and graceful degradation. Understanding these fallbacks is important for debugging and maintenance.

### Fallback Categories

1. **Intentional Fallbacks (Backward Compatibility)**: Documented fallbacks that support backward compatibility during migration
2. **Graceful Degradation Fallbacks**: Allow the system to continue operating when provider initialization fails
3. **Deprecated Fallbacks**: Temporary fallbacks that will be removed in future versions

### Key Fallback Patterns

#### Transcription Provider Fallback

**Location**: `workflow.py:843-847`

**Purpose**: Graceful degradation when transcription provider initialization fails

**Behavior**: Falls back to direct Whisper loading if provider initialization fails

**Status**: ✅ **Intentional** - Documented graceful degradation pattern

**Future**: May be removed when providers are stable, or made configurable (fail-fast vs graceful)

#### Episode Processor Transcription Fallback

**Location**: `episode_processor.py:446-447`

**Purpose**: Backward compatibility when provider is not available

**Behavior**: Uses direct Whisper transcription if `transcription_provider` is `None`

**Status**: ✅ **Intentional** - Backward compatibility pattern

**Future**: Will be removed when all code paths use providers

### Fallback Decision Matrix

| Scenario | Provider Type | Fallback? | Reason |
| --------- | -------------- | ----------- | -------- |
| Transcription init fails | Transcription | ✅ Yes | Graceful degradation |
| Speaker detector init fails | SpeakerDetector | ❌ No | Fail-fast (optional) |
| Summarization init fails | Summarization | ❌ No | Fail-fast (optional) |
| Provider not passed | Any | ✅ Yes | Backward compatibility |
| Method not available | Any | ❌ No | Protocol method required |

### Best Practices

#### ✅ Do

- **Document fallbacks**: Always document why fallbacks exist
- **Log fallback usage**: Log when fallbacks are used for debugging
- **Deprecate gradually**: Mark deprecated fallbacks with warnings
- **Test fallbacks**: Ensure fallback paths are tested

#### ❌ Don't

- **Silent fallbacks**: Don't fail silently - always log
- **Complex fallbacks**: Keep fallback logic simple
- **Permanent fallbacks**: Plan to remove deprecated fallbacks
- **Undocumented fallbacks**: Always document fallback behavior

### Removing Fallbacks

**When to Remove**:

- All code paths use providers
- No backward compatibility needed
- Provider initialization is stable
- Deprecation period has passed

**How to Remove**:

1. Identify all fallback paths
2. Update code to require providers
3. Remove fallback logic
4. Update tests
5. Update documentation

---

## Related Documentation

- [Custom Provider Guide](./CUSTOM_PROVIDER_GUIDE.md) - How to create custom providers
- [Provider Attributes](./PROVIDER_ATTRIBUTES.md) - Provider-specific attributes
- [Protocol Extension Guide](./PROTOCOL_EXTENSION_GUIDE.md) - How to extend protocols

---

## Summary

Migrating to the provider pattern involves:

1. Replacing direct module calls with provider factory functions
2. Using provider methods instead of direct functions
3. Adding initialization and cleanup calls
4. Updating tests and function signatures

The provider pattern provides better flexibility, testability, and consistency across the codebase.
