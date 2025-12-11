# Transcription Parallelism Architecture: Provider-Specific Control

## Problem Statement

With the introduction of OpenAI transcription providers (external API calls), we need to decide:

1. **Should transcription parallelism be controlled at workflow level or provider level?**
2. **How do we handle different resource constraints?** (Memory/CPU vs Network I/O)
3. **Should parallelism be configurable per provider type?**

## Current State Analysis

### Local Whisper (Current)

| Aspect | Current Implementation | Constraint | Resource Type |
|--------|----------------------|------------|---------------|
| **Processing** | Sequential (one job at a time) | Memory (RAM) + CPU | Each transcription = ~1-4GB RAM, CPU-intensive |
| **Model Loading** | Single model instance shared | Memory (RAM) | ~500MB-2GB per model |
| **Control** | No parallelism config | N/A | Sequential by design |

**Current Code Flow**:

- `_process_transcription_jobs_concurrent()` processes jobs **sequentially** in a background thread
- Each job calls `transcribe_media_to_text()` → `WhisperTranscriptionProvider.transcribe()`
- Whisper model is loaded once and reused for all jobs
- Jobs are processed one at a time (no parallelism)

**Why Sequential?**

- Whisper models are memory-intensive (~1-4GB RAM per model)
- CPU-intensive processing (audio → text conversion)
- Parallelizing would require multiple model instances (memory explosion)
- Current design: Single model instance, sequential processing

### OpenAI API (Future)

| Aspect | Proposed Implementation | Constraint | Resource Type |
|--------|------------------------|------------|---------------|
| **Processing** | Parallel API calls | Rate limits (RPM/TPM) | Network I/O bound |
| **Model Loading** | No local model | N/A | API-based |
| **Control** | `transcription_parallelism` config | Rate limits | I/O-bound, can parallelize |

**Key Differences**:

- ✅ **No local model**: No memory constraints
- ✅ **I/O-bound**: Network calls, not CPU-bound
- ✅ **Can parallelize**: Multiple API calls simultaneously
- ⚠️ **Rate limited**: Need to respect API limits (RPM/TPM)

## Architecture Options

### Option 1: Workflow-Level Control (Provider-Agnostic)

**Approach**: Workflow controls all transcription parallelism, providers are stateless.

```python
# workflow.py
def _process_transcription_jobs_concurrent(...):
    max_workers = cfg.transcription_parallelism  # Workflow controls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process episodes in parallel
        # Each worker calls provider.transcribe()
```

**Pros**:

- ✅ Simple: One place controls parallelism
- ✅ Consistent across providers
- ✅ Easy to reason about

**Cons**:

- ❌ Doesn't account for provider-specific constraints
- ❌ Whisper (local) should be sequential (memory/CPU bound)
- ❌ OpenAI needs rate limiting, not just worker count
- ❌ Can't optimize per-provider

### Option 2: Provider-Specific Control (Recommended)

**Approach**: Each provider handles its own parallelism internally.

```python
# transcription/whisper_provider.py
class WhisperTranscriptionProvider:
    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        # Always sequential - no parallelism
        # Single model instance, CPU/memory intensive
        return self._transcribe_sequential(audio_path, language)

# transcription/openai_provider.py (future)
class OpenAITranscriptionProvider:
    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        # Can parallelize multiple calls
        # Handles rate limiting internally
        return self._transcribe_with_rate_limiting(audio_path, language)
    
    def transcribe_batch(self, audio_paths: List[str], max_concurrent: int = 5):
        # Provider handles parallelism internally
        # Uses rate limiter to respect API limits
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Process with rate limiting
```

**Pros**:

- ✅ Provider-specific optimizations
- ✅ Whisper stays sequential (no memory explosion)
- ✅ OpenAI can parallelize with rate limiting
- ✅ Flexible for different provider types

**Cons**:

- ⚠️ More complex: Each provider implements parallelism
- ⚠️ Need to ensure consistent interface

### Option 3: Hybrid Approach

**Approach**: Workflow controls episode-level parallelism, provider handles internal parallelism.

```python
# workflow.py - Episode-level parallelism
def _process_transcription_jobs_concurrent(...):
    # Workflow controls how many episodes to process in parallel
    max_workers = cfg.transcription_parallelism  # Episode-level
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for job in transcription_jobs:
            executor.submit(provider.transcribe, job.audio_path)

# Provider - Internal parallelism (if needed)
class WhisperTranscriptionProvider:
    def transcribe(self, audio_path: str) -> str:
        # Always sequential - no internal parallelism
        return self._transcribe_sequential(audio_path)

class OpenAITranscriptionProvider:
    def transcribe(self, audio_path: str) -> str:
        # Single API call - no internal parallelism needed
        # But provider can handle rate limiting
        return self._single_api_call(audio_path)
```

**Pros**:

- ✅ Clear separation: Episode = workflow, Internal = provider
- ✅ Provider can optimize based on its constraints
- ✅ Workflow doesn't need to know provider internals

**Cons**:

- ⚠️ For transcription, episode-level IS the only level (no chunking)
- ⚠️ Simpler than summarization (no chunk-level parallelism)

## Recommended Architecture

### Key Insight

**Transcription is simpler than summarization**:

- ✅ **No chunking**: Each episode = one transcription call
- ✅ **Single level**: Only episode-level parallelism (no chunk-level)
- ✅ **Provider-specific**: Whisper sequential, OpenAI parallel

### Configuration Structure

```python
# config.py
class Config:
    # Transcription parallelism (provider-specific)
    transcription_parallelism: int = 1  # Episodes processed in parallel
    
    # OpenAI-specific (future)
    openai_transcription_max_concurrent: int = 5  # Max concurrent API calls
    openai_transcription_requests_per_minute: int = 50  # Rate limit
```

### Implementation Pattern

**Workflow Level** (Episode Parallelism):
```python
# workflow.py
def _process_transcription_jobs_concurrent(...):
    # Get parallelism from config (provider-specific default)
    max_workers = cfg.transcription_parallelism
    
    # Whisper: max_workers = 1 (sequential)
    # OpenAI: max_workers = 5 (parallel with rate limiting)
    
    if max_workers <= 1:
        # Sequential processing
        for job in transcription_jobs:
            provider.transcribe(job.audio_path)
    else:
        # Parallel processing (OpenAI only)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(provider.transcribe, job.audio_path) 
                      for job in transcription_jobs]
            # Wait for completion
```

**Provider Level** (Internal Handling):
```python
# transcription/whisper_provider.py
class WhisperTranscriptionProvider:
    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        # Always sequential - single model instance
        # No parallelism needed or possible (memory/CPU bound)
        return self._transcribe_sequential(audio_path, language)

# transcription/openai_provider.py (future)
class OpenAITranscriptionProvider:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self._rate_limiter = RateLimiter(
            requests_per_minute=cfg.openai_transcription_requests_per_minute,
            tokens_per_minute=cfg.openai_transcription_tokens_per_minute,
        )
    
    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        # Single API call - provider handles rate limiting internally
        with self._rate_limiter:
            return self._single_api_call(audio_path, language)
```

### Default Behavior

**Whisper (Local)**:

- `transcription_parallelism = 1` (default)
- Sequential processing (one job at a time)
- Single model instance shared across all jobs

**OpenAI (Future)**:

- `transcription_parallelism = 5` (default)
- Parallel processing (multiple API calls)
- Rate limiting handled internally by provider

### Migration Path

1. **Stage 1**: Add `transcription_parallelism` config field (default: 1)
2. **Stage 2**: Update workflow to use `transcription_parallelism` for episode-level parallelism
3. **Stage 3**: Whisper provider remains sequential (ignores parallelism > 1)
4. **Stage 4**: OpenAI provider uses parallelism with rate limiting

## Comparison with Summarization

| Aspect | Summarization | Transcription |
|--------|--------------|---------------|
| **Levels** | Episode-level + Chunk-level | Episode-level only |
| **Config Fields** | `summary_batch_size` + `summary_chunk_parallelism` | `transcription_parallelism` |
| **Local Provider** | Can parallelize chunks (CPU-bound) | Sequential (memory/CPU bound) |
| **API Provider** | Can parallelize chunks (I/O-bound) | Can parallelize episodes (I/O-bound) |
| **Complexity** | Higher (two levels) | Lower (one level) |

## Benefits

- ✅ **Provider-specific**: Whisper sequential, OpenAI parallel
- ✅ **Simple**: Single config field (no chunk-level complexity)
- ✅ **Future-proof**: Ready for OpenAI provider
- ✅ **Backward compatible**: Default = 1 (sequential, current behavior)

## Implementation Checklist

- [ ] Add `transcription_parallelism` config field (default: 1)
- [ ] Update workflow to use `transcription_parallelism` for episode-level parallelism
- [ ] Whisper provider: Document that it ignores parallelism > 1 (sequential only)
- [ ] OpenAI provider (future): Implement parallelism with rate limiting
- [ ] Update tests to verify sequential behavior for Whisper
- [ ] Update documentation

## Recommendation

**Use Provider-Specific Control**:

- **Workflow**: Controls episode-level parallelism via `transcription_parallelism`
- **Provider**: Handles internal parallelism (if any)
  - Whisper: Always sequential (ignores parallelism > 1)
  - OpenAI: Parallel with rate limiting (uses parallelism config)

This gives us:

- ✅ Clear separation of concerns
- ✅ Provider-specific optimizations
- ✅ Simple configuration (one field)
- ✅ Future-proof for OpenAI provider
