# Parallelism Architecture: Provider-Agnostic vs Provider-Specific

## Problem Statement

With the introduction of OpenAI providers (external API calls), we need to decide:

1. **Where should parallelism be controlled?** (Workflow level vs Provider level)
2. **Should episode-level and chunk-level parallelism be separate controls?**
3. **How do we handle different resource constraints?** (Memory vs CPU vs Network I/O)

## Current State Analysis

### Local Models (Current)

| Level | Constraint | Current Control | Resource Type |
|-------|-----------|----------------|---------------|
| **Episode-level** | Memory (RAM) | `summary_batch_size` | Each episode = new model instance (~500MB-2GB) |
| **Chunk-level** | CPU cores | `summary_batch_size` | Shared model, multiple threads |

**Problem**: Same config controls both, but they have different constraints.

### OpenAI API (Future)

| Level | Constraint | Proposed Control | Resource Type |
|-------|-----------|------------------|---------------|
| **Episode-level** | Rate limits (RPM/TPM) | `openai_max_concurrent_requests` | Network I/O bound |
| **Chunk-level** | Rate limits (RPM/TPM) | Same rate limiter | Network I/O bound |

**Key Difference**: Both are I/O-bound, but rate-limited by API.

## Architecture Options

### Option 1: Provider-Agnostic Parallelism (Workflow Level)

**Approach**: Workflow controls all parallelism, providers are stateless.

```python
# workflow.py
def _parallel_episode_summarization(...):
    max_workers = cfg.summary_batch_size  # Episode-level
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process episodes in parallel
        # Each worker calls provider.summarize()
```

**Pros**:

- ✅ Simple: One place controls parallelism
- ✅ Consistent across providers
- ✅ Easy to reason about

**Cons**:

- ❌ Doesn't account for provider-specific constraints
- ❌ OpenAI rate limits need different handling
- ❌ Can't optimize per-provider

### Option 2: Provider-Specific Parallelism

**Approach**: Each provider handles its own parallelism internally.

```python
# summarization/local_provider.py
class TransformersSummarizationProvider:
    def summarize_chunks(self, chunks, params):
        max_workers = params.get("chunk_parallelism", 1)
        # Provider handles parallelism internally

# summarization/openai_provider.py  
class OpenAISummarizationProvider:
    def summarize_chunks(self, chunks, params):
        max_concurrent = params.get("max_concurrent_requests", 5)
        # Provider handles rate limiting internally
```

**Pros**:

- ✅ Provider-specific optimizations
- ✅ Can handle rate limits per provider
- ✅ Flexible for different provider types

**Cons**:

- ❌ More complex: Each provider implements parallelism
- ❌ Harder to reason about overall parallelism
- ❌ Potential for conflicts (too many total workers)

### Option 3: Hybrid Approach (Recommended)

**Approach**:

- **Episode-level**: Workflow controls (provider-agnostic)
- **Chunk-level**: Provider controls (provider-specific)

```python
# workflow.py - Episode-level parallelism
def _parallel_episode_summarization(...):
    # Workflow controls how many episodes to process in parallel
    max_workers = cfg.summary_batch_size  # Episode-level
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for episode in episodes:
            executor.submit(provider.summarize, ...)

# Provider - Chunk-level parallelism
class TransformersSummarizationProvider:
    def summarize(self, text, params):
        # Provider controls chunk parallelism internally
        chunk_parallelism = params.get("chunk_parallelism", 1)
        return self._summarize_with_chunks(text, chunk_parallelism)
```

**Pros**:

- ✅ Clear separation: Episode = workflow, Chunks = provider
- ✅ Provider can optimize based on its constraints
- ✅ Workflow doesn't need to know provider internals

**Cons**:

- ⚠️ Two levels of parallelism to configure
- ⚠️ Need to ensure they don't conflict

## Recommended Architecture

### Configuration Structure

```python
# config.py
class Config:
    # Episode-level parallelism (workflow controls)
    summary_batch_size: int = 1  # Episodes processed in parallel
    
    # Chunk-level parallelism (provider-specific, passed via params)
    summary_chunk_parallelism: int = 1  # Chunks processed in parallel (local only)
    
    # OpenAI-specific parallelism (provider handles internally)
    openai_max_concurrent_requests: int = 5  # Max concurrent API calls
    openai_requests_per_minute: int = 50  # Rate limit
    openai_tokens_per_minute: int = 100000  # Rate limit
```

### Implementation Pattern

**Workflow Level** (Episode Parallelism):

```python
# workflow.py
def _parallel_episode_summarization(...):
    # Workflow controls episode-level parallelism
    max_episode_workers = cfg.summary_batch_size
    
    with ThreadPoolExecutor(max_workers=max_episode_workers) as executor:
        for episode in episodes:
            executor.submit(
                provider.summarize,
                text=transcript,
                params={
                    "chunk_parallelism": cfg.summary_chunk_parallelism,  # Pass to provider
                    # Provider uses this for chunk-level parallelism
                }
            )
```

**Provider Level** (Chunk Parallelism):

```python
# summarization/local_provider.py
class TransformersSummarizationProvider:
    def summarize(self, text, params):
        chunk_parallelism = params.get("chunk_parallelism", 1)
        
        if needs_chunking(text):
            # Provider handles chunk parallelism internally
            return self._summarize_chunks_parallel(
                chunks, 
                max_workers=chunk_parallelism  # CPU-bound
            )
        else:
            return self._summarize_direct(text)

# summarization/openai_provider.py
class OpenAISummarizationProvider:
    def summarize(self, text, params):
        # OpenAI: Most transcripts fit in 128k context window
        # No chunking needed for most cases
        if self._fits_in_context(text):
            return self._single_api_call(text)  # One API call
        else:
            # Rare case: Need chunking
            # Provider handles rate limiting internally
            return self._summarize_with_chunks(
                chunks,
                rate_limiter=self._rate_limiter  # Handles RPM/TPM limits
            )
```

## Resource Constraints by Provider Type

### Local Transformers Provider

| Level | Constraint | Control | Typical Limit |
|-------|-----------|---------|---------------|
| Episode | Memory (RAM) | `summary_batch_size` | 2-4 episodes |
| Chunk | CPU cores | `summary_chunk_parallelism` | 4-8 chunks |

### OpenAI Provider

| Level | Constraint | Control | Typical Limit |
|-------|-----------|---------|---------------|
| Episode | Rate limits (RPM) | `openai_max_concurrent_requests` | 5-10 episodes |
| Chunk | Rate limits (RPM/TPM) | Same rate limiter | Usually N/A (no chunking) |

**Key Insight**: OpenAI rarely needs chunking (128k context window), so chunk parallelism is less relevant.

## Decision Matrix

| Scenario | Episode Parallelism | Chunk Parallelism | Control Location |
|----------|-------------------|-------------------|------------------|
| **Local, High RAM, Low CPU** | High (4) | Low (1) | Workflow + Provider |
| **Local, Low RAM, High CPU** | Low (1) | High (4) | Workflow + Provider |
| **OpenAI, Standard** | Medium (5-10) | N/A (no chunking) | Workflow + Provider (rate limiter) |
| **OpenAI, Long transcripts** | Medium (5-10) | Rate-limited | Workflow + Provider (rate limiter) |

## Recommended Implementation

### 1. Separate Controls

```python
# config.py
summary_batch_size: int = 1  # Episode-level parallelism
summary_chunk_parallelism: int = 1  # Chunk-level parallelism (local only)
```

### 2. Provider-Agnostic Episode Parallelism

```python
# workflow.py - Controls episode-level parallelism
max_episode_workers = cfg.summary_batch_size
```

### 3. Provider-Specific Chunk Parallelism

```python
# Provider receives chunk_parallelism via params
# Provider decides how to use it based on its constraints:
# - Local: Uses for CPU parallelism
# - OpenAI: Uses for rate-limited API calls (if chunking needed)
```

### 4. OpenAI Rate Limiting

```python
# Provider handles rate limiting internally
# Shared rate limiter across all provider instances
# Respects openai_max_concurrent_requests, RPM, TPM limits
```

## Benefits of This Approach

1. **Clear Separation**: Episode-level = workflow concern, Chunk-level = provider concern
2. **Provider Flexibility**: Each provider optimizes based on its constraints
3. **Backward Compatible**: Can add `summary_chunk_parallelism` without breaking existing configs
4. **Future-Proof**: Works for local, OpenAI, Anthropic, or any future provider

## Migration Path

1. **Phase 1**: Add `summary_chunk_parallelism` config field (default: 1)
2. **Phase 2**: Update local provider to use `chunk_parallelism` from params
3. **Phase 3**: Keep `summary_batch_size` for episode-level parallelism
4. **Phase 4**: OpenAI provider uses rate limiter (doesn't need chunk parallelism for most cases)

## Conclusion

**Recommendation**: **Hybrid Approach**

- **Episode-level parallelism**: Workflow controls (provider-agnostic)
- **Chunk-level parallelism**: Provider controls (provider-specific, passed via params)
- **OpenAI rate limiting**: Provider handles internally (shared rate limiter)

This gives us:

- ✅ Clear separation of concerns
- ✅ Provider-specific optimizations
- ✅ Flexible for different provider types
- ✅ Backward compatible
