# Provider-Specific Attributes

This document describes provider-specific attributes that are exposed for backward compatibility and internal access.
These attributes are **not part of the protocol interface** but are available on specific provider implementations.

## Overview

Providers expose certain internal resources (models, NLP objects, etc.) as properties to support:

1. **Backward compatibility** - Code that previously accessed models directly
2. **Internal access** - Workflow code that needs model instances for parallel processing
3. **Debugging and inspection** - Access to provider internals for troubleshooting

**Important**: These attributes are implementation-specific and may not be available on all providers. Always use
`getattr()` with a default value when accessing them.

## TranscriptionProvider Attributes

### `model` (WhisperTranscriptionProvider)

**Type**: `Optional[Any]` (Whisper model object)

**Description**: The loaded Whisper model instance. Available only on `WhisperTranscriptionProvider`.

**Usage**:

````python
provider = create_transcription_provider(cfg)
provider.initialize()
whisper_model = getattr(provider, "model", None)
if whisper_model:

    # Access Whisper model directly (backward compatibility)

    pass
```text

### `nlp` (NERSpeakerDetector)

**Type**: `Optional[Any]` (spaCy Language object)

**Description**: The loaded spaCy NLP model instance. Available only on `NERSpeakerDetector`.

**Usage**:

```python
detector = create_speaker_detector(cfg)
detector.initialize()
nlp = getattr(detector, "nlp", None)
if nlp:

    # Access spaCy model directly (backward compatibility)

    pass
```python

**Type**: `Optional[Dict[str, Any]]`

**Description**: Cached heuristics from pattern analysis. Available only on `NERSpeakerDetector` after `analyze_patterns()` is called.

**Usage**:

```python
detector = create_speaker_detector(cfg)
detector.initialize()
heuristics = getattr(detector, "heuristics", None)
```text

**Type**: `Optional[SummaryModel]`

**Description**: The MAP (chunk summarization) model instance. Available only on `TransformersSummarizationProvider`.

**Usage**:

```python
provider = create_summarization_provider(cfg)
provider.initialize()
map_model = getattr(provider, "map_model", None)
if map_model:

    # Access MAP model directly (backward compatibility)

    pass
```text

**Type**: `Optional[SummaryModel]`

**Description**: The REDUCE (final combine) model instance. Available only on `TransformersSummarizationProvider`. May be the same as `map_model` if a separate reduce model is not used.

**Usage**:

```python
provider = create_summarization_provider(cfg)
provider.initialize()
reduce_model = getattr(provider, "reduce_model", None)
if reduce_model:

    # Access REDUCE model directly (backward compatibility)

    pass
```text

### `is_initialized` (All Providers)

**Type**: `bool`

**Description**: Indicates whether the provider has been initialized. Available on all provider implementations.

**Usage**:

```python
provider = create_provider(cfg)
if not provider.is_initialized:
    provider.initialize()
```text

1. **Use `getattr()` with defaults**: Always use `getattr(provider, "attribute", None)` to safely access provider-specific attributes.

2. **Check for None**: Provider-specific attributes may be `None` even if the provider is initialized (e.g., API-based providers don't expose models).

3. **Prefer protocol methods**: Use protocol methods (`transcribe()`, `summarize()`, etc.) instead of accessing internal models when possible.

4. **Document usage**: If your code accesses provider-specific attributes, document why and which providers support them.

5. **Avoid direct model access**: Prefer using provider methods over direct model access. Direct model access is primarily for backward compatibility.

## Migration Path

As the codebase migrates fully to the provider pattern, direct model access will be deprecated:

1. **Phase 1** (Current): Provider methods + backward compatibility attributes
2. **Phase 2** (Future): Provider methods only, attributes deprecated
3. **Phase 3** (Future): Attributes removed, protocol methods only

## Examples

### Safe Attribute Access

```python

# Good: Safe access with getattr()

provider = create_transcription_provider(cfg)
provider.initialize()
model = getattr(provider, "model", None)
if model:

    # Use model for backward compatibility

    pass

# Bad: Direct access (may raise AttributeError)

model = provider.model  # Don't do this
```text

# Check provider type before accessing attributes

if isinstance(provider, WhisperTranscriptionProvider):
    model = provider.model  # Safe - we know this provider has model
else:

    # API provider - no model attribute

    pass

```text

# Support both provider and direct model access

if transcription_provider:

    # Use provider (preferred)

    result, elapsed = transcription_provider.transcribe_with_segments(audio_path)
else:

    # Fallback to direct model (backward compatibility)

    model = getattr(transcription_provider, "model", None) or load_model_directly()
    result, elapsed = transcribe_with_model(model, audio_path)
```text

- [Provider Protocols](api/) - Protocol definitions
- [Custom Provider Guide](./CUSTOM_PROVIDER_GUIDE.md) - How to create custom providers
- [Provider Factory Functions](https://github.com/chipi/podcast_scraper/blob/main/podcast_scraper/speaker_detectors/factory.py) - Provider creation

````
