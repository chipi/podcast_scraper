# Performance Guide

This guide covers performance considerations, optimization opportunities, and performance-related troubleshooting for podcast_scraper.

---

## Audio Preprocessing Performance (Issue #392)

### Cache Miss Cost

**Observation:** Audio preprocessing (especially Voice Activity Detection) is computationally expensive. The first time an episode is processed, preprocessing can take 30-60 seconds for a typical 30-60 minute podcast episode. However, subsequent runs with the same audio file benefit from caching.

**Performance Impact:**

- **Cache Hit:** Near-instant (milliseconds) - preprocessed audio is reused
- **Cache Miss:** 30-60 seconds for typical episodes (depends on audio length and preprocessing settings)
- **Net Benefit:** Despite preprocessing overhead, total pipeline time typically decreases by 20-40% due to:
  - Faster transcription (smaller audio files)
  - Reduced API costs (30-60% reduction for API providers)
  - Better API compatibility (files fit within size limits)

**Cache Strategy:**

- Cache key: SHA256 hash of first 1MB of audio + preprocessing configuration
- Cache location: `.cache/preprocessing/`
- Cache invalidation: Automatic when preprocessing settings change

**Optimization Opportunities:**

1. **Pre-warm cache** for frequently processed episodes:

   ```bash
   # Process episodes once to populate cache
   python3 -m podcast_scraper.cli feed.xml --enable-preprocessing
   ```

2. **Adjust preprocessing settings** to balance quality vs. performance:
   - Lower `preprocessing_silence_threshold` = more aggressive silence removal = faster transcription
   - Higher `preprocessing_silence_duration` = fewer processing steps = faster preprocessing

3. **Monitor cache hit rate** in metrics:
   - Check `preprocessing_cache_hit_rate` in pipeline metrics
   - Target: ≥80% cache hit rate in development/testing scenarios

**When Cache Misses Are Expensive:**

- First-time processing of new episodes
- Changing preprocessing configuration (invalidates cache)
- Processing episodes from different sources (different audio = different hash)

**Mitigation Strategies:**

- Use consistent preprocessing settings across runs
- Pre-process episodes in batch to populate cache
- Consider preprocessing as a separate step for large batches

---

## Transcription Performance

### Provider Selection

**Local Whisper:**

- **Pros:** No API costs, no rate limits, complete privacy
- **Cons:** Slower (CPU-bound), requires GPU for reasonable speed
- **Best for:** Small batches, privacy-sensitive content, offline processing

**API Providers (OpenAI, Gemini):**

- **Pros:** Fast (parallel processing), high quality, no local compute
- **Cons:** API costs, rate limits, file size limits (25MB for OpenAI)
- **Best for:** Large batches, production workloads, when speed matters

### Model Selection

**Whisper Models:**

- `tiny`: Fastest, lowest quality
- `base`: Fast, acceptable quality
- `small`: Balanced (recommended default)
- `medium`: Slower, higher quality
- `large`: Slowest, highest quality

**Recommendation:** Use `small` for most use cases. Upgrade to `medium` or `large` only if quality is critical.

---

## Summarization Performance

### Summarization Model Selection

**Local Models (BART, LED):**

- **Pros:** No API costs, complete privacy
- **Cons:** Slower, lower quality, requires GPU for reasonable speed
- **Best for:** Small batches, privacy-sensitive content

**API Providers (OpenAI, Anthropic, Gemini):**

- **Pros:** Fast, high quality, parallel processing
- **Cons:** API costs, rate limits, context window limits
- **Best for:** Large batches, production workloads

### Context Window Management

- **Long transcripts:** May exceed model context windows
- **Solution:** Transcripts are automatically chunked when needed
- **Performance impact:** Chunking adds overhead but enables processing of any length transcript

---

## Memory Usage

### ML Model Loading

**Memory per model:**

- Whisper `small`: ~500MB
- Whisper `medium`: ~1.5GB
- Whisper `large`: ~3GB
- BART/LED summarization: ~1-2GB

**Recommendations:**

- Preload models once: `make preload-ml-models`
- Use smaller models if memory-constrained
- Process episodes sequentially if memory is limited

### Parallel Processing

**Default behavior:**

- Downloads: Parallel (limited by `max_workers`)
- Transcription: Sequential (provider-dependent)
- Summarization: Sequential (provider-dependent)

**Memory-aware worker calculation:**

- Automatically calculates optimal workers based on available RAM
- Reserves 4GB for system operations
- Caps at 8 workers for CPU efficiency

---

## Network Performance

### RSS Feed Fetching

- **Caching:** RSS feeds are cached to avoid redundant fetches
- **Timeout:** Default 60s timeout for feed fetching
- **Retry:** Automatic retry with exponential backoff

### Media Download

- **Parallel downloads:** Multiple episodes downloaded concurrently
- **Resume support:** Partial downloads can be resumed
- **Timeout:** Configurable per-episode download timeout

---

## Disk I/O Performance

### Cache Directories

**Cache locations:**

- `.cache/whisper/` - Whisper models
- `.cache/preprocessing/` - Preprocessed audio
- `.cache/transcripts/` - Cached transcripts
- `.cache/huggingface/` - Transformers models

**Performance tips:**

- Use fast storage (SSD) for cache directories
- Monitor cache directory size (can grow large)
- Periodic cleanup recommended for long-running systems

### Output Directory

- **Write performance:** Multiple episodes writing concurrently
- **Recommendation:** Use fast storage (SSD) for output directory
- **Network storage:** Supported but may be slower

---

## Profiling and Monitoring

### Metrics Collection

Pipeline metrics track:

- Processing times per stage
- Cache hit/miss rates
- API call counts and costs
- Token usage

**Access metrics:**

```bash
# JSON output (end of run)
cat output/run_*.json

# JSONL streaming (during run)
tail -f output/run_*.jsonl
```

### Debug Logging

Enable debug logging to see detailed performance information:

```bash
export LOG_LEVEL=DEBUG
python3 -m podcast_scraper.cli feed.xml
```

**What to look for:**

- Preprocessing times
- Cache hit/miss messages
- API call durations
- Memory usage warnings

---

## Optimization Checklist

- [ ] Use preprocessing caching (enable preprocessing)
- [ ] Preload ML models (`make preload-ml-models`)
- [ ] Use appropriate model sizes for your hardware
- [ ] Monitor cache hit rates
- [ ] Use fast storage (SSD) for cache and output
- [ ] Adjust worker counts based on available resources
- [ ] Use API providers for large batches (if cost is acceptable)
- [ ] Monitor memory usage and adjust batch sizes

---

## Related Documentation

- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions
- [Development Guide](DEVELOPMENT_GUIDE.md) - Environment setup
- [RFC-040: Audio Preprocessing Pipeline](../rfc/RFC-040-audio-preprocessing-pipeline.md) - Preprocessing details
- [RFC-028: ML Model Preloading](../rfc/RFC-028-ml-model-preloading-and-caching.md) - Model caching
