# RFC-028: ML Model Preloading and Caching

**Status:** Implemented
**Created:** 2025-12-30
**Related Issues:** #131

## Summary

This RFC addresses the problem of ML models (Whisper, spaCy, Transformers) downloading from the internet during test execution, which causes slow CI runs, network dependencies, and test failures when network is blocked. We implement model preloading for local development and GitHub Actions caching for CI to eliminate network dependencies and significantly speed up test execution.

## Problem Statement

### Current Issues

1. **Slow CI/CD:**
   - Models are deleted after each CI run
   - E2E tests and `ml_models`-marked integration tests re-download models (~2-5 minutes)
   - Wasted bandwidth and compute time

2. **Network Dependency:**
   - E2E tests and `ml_models` integration tests fail if model servers are down
   - Development requires internet connection when using real models
   - Adds flakiness to CI/CD pipeline

3. **Test Categories:**
   - **Unit tests:** Mock models (don't need cached models)
   - **Integration tests:** Most mock models, only `@pytest.mark.ml_models` tests use real models
   - **E2E tests:** Use real models (need cached models)
   - **Real usage:** Uses real models (needs cached models)

### Models Affected

**Note**: The test suite uses smaller, faster models for speed, while production uses quality models. The preload script preloads both sets to ensure flexibility.

1. **Whisper Models** (Transcription)
   - Test default: `tiny.en` (smallest, fastest)
   - Production default: `base.en` (better quality, matches app config)
   - Cache: `~/.cache/whisper/`
   - Status: ✅ Preloaded in Dockerfile (`base.en`), ✅ Preloaded locally (both)

2. **spaCy Models** (Speaker Detection)
   - Default: `en_core_web_sm` (same for tests and production)
   - Cache: `~/.local/share/spacy/` or site-packages
   - Status: ✅ Preloaded locally

3. **Transformers Models** (Summarization)
   - Test default (MAP): `facebook/bart-base` (small, ~500MB, fast)
   - Production default (MAP): `facebook/bart-large-cnn` (large, ~2GB, quality)
   - REDUCE default: `allenai/led-base-16384` (long-context, ~1GB, used in both)
   - Additional: `sshleifer/distilbart-cnn-12-6` (fast option)
   - Cache: `~/.cache/huggingface/hub/`
   - Status: ✅ Preloaded locally (all 4 models)

## Goals

1. **Eliminate Network Dependency:**
   - Models pre-downloaded and cached locally
   - CI uses cached models across runs
   - Works offline after initial download

2. **Speed Up Tests:**
   - No download time during test execution
   - CI runs 10-30x faster after first cache
   - Faster developer feedback

3. **Improve Reliability:**
   - Tests work even if model servers are down
   - No flakiness from network issues
   - Consistent test execution

## Design

### Local Development

**Makefile Target: `preload-ml-models`**

Preloads all required ML models to local cache:

```bash
make preload-ml-models
```

- Whisper: `tiny.en` (test default), `base.en` (production default)
- spaCy: `en_core_web_sm` (same for tests and production)
- Transformers: `facebook/bart-base` (test default), `facebook/bart-large-cnn` (production default), `sshleifer/distilbart-cnn-12-6` (fast option), `allenai/led-base-16384` (REDUCE default)

**Rationale**: Preloading both test and production defaults ensures:
- Fast test execution (using small models)
- Production quality (using large models)
- Flexibility to switch between models

**Cache Locations:**
- Whisper: `~/.cache/whisper/`
- spaCy: `~/.local/share/spacy/`
- Transformers: `~/.cache/huggingface/hub/`

**Persistence:**
- Models persist across runs indefinitely
- Only deleted if user runs `make clean-cache`
- Works offline after initial download

### CI/CD (GitHub Actions)

**Model Caching Strategy:**

1. **Cache Step:**
   ```yaml


   - name: Cache ML models
     uses: actions/cache@v4

     id: cache-models
     with:
       path: |
         ~/.cache/whisper
         ~/.local/share/spacy
         ~/.cache/huggingface
       key: ml-models-${{ runner.os }}-v1
       restore-keys: |
         ml-models-${{ runner.os }}-
   ```

2. **Preload on Cache Miss:**
   ```yaml


   - name: Preload ML models (if cache miss)
     if: steps.cache-models.outputs.cache-hit != 'true'

     run: make preload-ml-models
   ```

3. **Keep Models in Cleanup:**
   - Removed model deletion from cleanup steps
   - Models persist for next CI run via cache

**Jobs Updated:**
- `test-integration-slow` (includes `ml_models`-marked integration tests)
- `test-e2e-slow` (includes E2E tests with real models)
- `test` (full test suite)

**Cache Key Strategy:**
- Versioned keys (`ml-models-v1`) for cache invalidation
- OS-specific keys for cross-platform compatibility
- Restore keys for fallback to older cache

### Docker

**Current State:**
- ✅ Whisper models preloaded in Dockerfile
- ❌ spaCy models NOT preloaded (future enhancement)
- ❌ Transformers models NOT preloaded (future enhancement)

**Docker Layer Caching:**
- Models baked into image layers
- GitHub Actions caches Docker layers
- Models persist across Docker builds

## Implementation

### Files Changed

1. **`Makefile`:**
   - Added `preload-ml-models` target
   - Preloads Whisper, spaCy, and Transformers models
   - Added to `.PHONY` and help text

2. **`.github/workflows/python-app.yml`:**
   - Added `actions/cache@v4` step to 3 jobs
   - Added `make preload-ml-models` step for cache misses
   - Removed model deletion from cleanup steps

3. **Documentation:**
   - Created analysis documents (now consolidated into this RFC)
   - Updated references to use `preload-ml-models`

### Cache Size

**Model Sizes:**
- Whisper `base.en`: ~150 MB
- Whisper `tiny`: ~75 MB
- spaCy `en_core_web_sm`: ~50 MB
- Transformers `facebook/bart-base`: ~500 MB
- Transformers `facebook/bart-large-cnn`: ~1.6 GB
- Transformers `sshleifer/distilbart-cnn-12-6`: ~300 MB
- **Total: ~2.7 GB**

**GitHub Actions Limits:**
- 10 GB per repository (free tier)
- 10 GB per cache entry
- Our usage: ~2.7 GB (well within limits)

## Benefits

### Performance

- **CI Speed:** 10-30x faster after first cache
  - First run: Downloads models (~2-5 min)
  - Subsequent runs: Uses cache (~10-30 seconds)

- **Local Development:**
  - No download time during test execution
  - Faster feedback loop
  - Works offline

### Reliability

- **Network Independence:**
  - E2E tests and `ml_models` integration tests work even if model servers are down
  - No flakiness from network issues
  - Consistent test execution

- **Test Categories:**
  - **Unit tests:** Continue to mock models (as they should)
  - **Integration tests:** Most mock models, `ml_models`-marked tests use cached models
  - **E2E tests:** Use cached models (real models, but from cache)
  - **Real usage:** Uses cached models (faster startup, works offline)

### Cost Savings

- **Bandwidth:** Reduced by ~90% (only first run downloads)
- **Compute Time:** Faster CI = lower compute costs
- **Reliability:** Fewer failed runs = less wasted compute

## Testing

### Local Testing

1. **Preload models:**
   ```bash

   make preload-ml-models
   ```

2. **Verify cache:**
   ```bash

   ls -la ~/.cache/whisper/
   ls -la ~/.local/share/spacy/
   ls -la ~/.cache/huggingface/hub/
   ```

2. **Test E2E and ml_models integration tests are faster:**
   ```bash

   make test-e2e-slow          # E2E tests with real models (faster with cache)
   make test-integration-slow   # Integration tests marked ml_models (faster with cache)
   ```

3. **Note on test categories:**
   - **Unit tests:** Mock models (don't need cached models)
   - **Integration tests:** Most mock models, only `@pytest.mark.ml_models` tests use real models
   - **E2E tests:** Use real models (need cached models)
   - **Real usage:** Uses real models (needs cached models)

### CI Testing

1. **First run:** Cache miss → Downloads models → Saves to cache
2. **Subsequent runs:** Cache hit → Restores models → No download
3. **Verify:** Check cache hit rate in GitHub Actions

## Monitoring

### Metrics to Track

- **Cache Hit Rate:** Should be >80% after first run
- **CI Run Time:** Should be faster with cache
- **Cache Size:** Should be ~2.7 GB
- **Network Errors:** Should decrease significantly

### Cache Management

- View cache usage in GitHub repository settings
- Manually delete cache if needed (via API or UI)
- Increment cache key version when models change

## Future Enhancements

1. **Dockerfile Updates:**
   - Preload spaCy models in Dockerfile
   - Preload Transformers models in Dockerfile

2. **Additional Models:**
   - Support preloading additional model variants
   - Allow configuration of which models to preload

3. **Cache Invalidation:**
   - Automatic cache invalidation on model version changes
   - Manual cache refresh command

## Migration Notes

### Breaking Changes

- None - this is fully additive

### Backward Compatibility

- Works with existing workflows
- Cache is optional (falls back to download if cache fails)
- Models still download if not cached (backward compatible)

### Developer Impact

**Before:**
- Models download on first use (slow)
- Network required for tests
- CI downloads models every run

**After:**
- Run `make preload-ml-models` once
- Models cached locally
- CI uses cached models (fast)

## Related Documents

- Issue #131: "prelaod ml models outside of test runs"
- `tests/unit/conftest.py`: Network blocking implementation
- `Dockerfile`: Whisper model preloading
- `src/podcast_scraper/speaker_detection.py`: spaCy model loading
- `src/podcast_scraper/summarizer.py`: Transformers model loading
- `src/podcast_scraper/whisper_integration.py`: Whisper model loading

## Conclusion

This RFC successfully addresses the network dependency and performance issues with ML model downloads. By implementing local preloading and CI caching, we've eliminated network dependencies, significantly improved CI performance, and made the development experience more reliable and faster.

The implementation is complete and ready for use. Developers should run `make preload-ml-models` once to cache models locally, and CI will automatically cache models across runs for optimal performance.

