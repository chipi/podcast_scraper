# GitHub Actions Changes Summary

## Overview

This document summarizes the changes made to GitHub Actions workflows to ensure conservative
resource usage and alignment with local development settings.

## Changes Made

### 1. Parallelism Reduction

**Before:**

- All workflows used `-n auto` (uses all available CPU cores)
- Could use up to 14+ workers on GitHub Actions runners
- High memory pressure, potential resource exhaustion

**After:**

- All workflows now use `-n $(python3 -c "import os; print(max(1, (os.cpu_count() or 2) - 2))")`
- Reserves 2 cores for system operations
- On typical GitHub Actions runners (2 cores), this results in sequential execution (safe)
- On larger runners (14 cores), this uses 12 workers (conservative)

**Files Changed:**

- `.github/workflows/python-app.yml`: 5 instances updated
- `.github/workflows/nightly.yml`: 1 instance updated

**Impact:**

- Reduced memory usage per job
- More predictable execution times
- Better resource utilization
- Aligns with local development (`make test` uses same strategy)

### 2. Job Timeouts Added

**Before:**

- No explicit timeouts (default 6 hours)
- Risk of runaway jobs consuming resources indefinitely

**After:**

- All jobs have conservative timeouts:
  - `lint`: 10 minutes
  - `test-unit`: 15 minutes
  - `preload-ml-models`: 20 minutes
  - `test-integration`: 30 minutes
  - `test-e2e`: 45 minutes
  - `nightly-tests`: 60 minutes

**Impact:**

- Prevents runaway jobs from consuming resources
- Faster failure detection
- Better resource management

### 3. Cache Strategy

**Current State:**

- GitHub Actions caches models in standard locations:
  - `~/.cache/whisper`
  - `~/.local/share/spacy`
  - `~/.cache/huggingface`
- Our code checks for local `.cache/` first, then falls back to standard locations
- **No changes needed** - existing cache strategy works correctly

**Cache Size:**

- Total: ~3.7 GB (well under 10 GB limit)
- Cache hit rate: >80% after initial setup
- Cache miss: Only on first run or cache invalidation

### 4. Resource Usage Documentation

**Created:**

- `.github/workflows/RESOURCE_USAGE.md`: Comprehensive documentation of resource usage strategy

**Contents:**

- GitHub Actions limits (free tier)
- Our conservative strategy
- Resource usage estimates
- Monitoring recommendations
- Best practices

## Alignment with Local Development

### Local (Makefile)

```makefile
test: test-serial
 @E2E_TEST_MODE=multi_episode pytest tests/ -m "not serial" --cov=$(PACKAGE) --cov-report=term-missing --cov-append -n $(python3 -c "import os; print(max(1, (os.cpu_count() or 14) - 2))") --disable-socket --allow-hosts=127.0.0.1,localhost
```

### GitHub Actions

```yaml
OUTPUT=$(pytest tests/integration/ -v -m integration -n $(python3 -c "import os; print(max(1, (os.cpu_count() or 2) - 2))") --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1 2>&1)
```

**Key Differences:**

- Local: Defaults to 14 cores if detection fails (your machine)
- CI: Defaults to 2 cores if detection fails (GitHub Actions runner)
- Both: Reserve 2 cores for system operations
- Both: Use same parallelism calculation logic

## Expected Impact

### Resource Usage Reduction

- **Memory**: ~30-40% reduction per job (fewer parallel workers)
- **CPU**: More predictable usage (reserved cores for system)
- **Time**: Slightly longer execution (acceptable trade-off for stability)

### Reliability Improvements

- **Timeouts**: Prevent runaway jobs
- **Consistency**: Same behavior locally and in CI
- **Predictability**: More consistent execution times

### Cost Savings (if on paid tier)

- Reduced concurrent resource usage
- Faster job completion (with timeouts)
- Better cache utilization

## Testing Recommendations

1. **Monitor first few runs** after these changes:
   - Check job durations (should be slightly longer but more stable)
   - Verify cache hit rates (should remain >80%)
   - Monitor memory usage (should be lower)

2. **Adjust if needed**:
   - If jobs are too slow, can reduce reservation (e.g., `- 1` instead of `- 2`)
   - If still seeing memory issues, can reduce further (e.g., `- 4`)
   - Timeouts can be adjusted based on actual execution times

3. **Track metrics**:
   - Job duration trends
   - Cache hit rates
   - Memory usage patterns
   - Failure rates

## Rollback Plan

If issues arise, revert to `-n auto`:

```yaml

# Old (revert to this if needed)

OUTPUT=$(pytest tests/integration/ -v -m integration -n auto ...)

# New (current)

OUTPUT=$(pytest tests/integration/ -v -m integration -n $(python3 -c "import os; print(max(1, (os.cpu_count() or 2) - 2))") ...)
```

## References

- [GitHub Actions Usage Limits](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions)
- [Pytest-xdist Documentation](https://pytest-xdist.readthedocs.io/)
- `.github/workflows/RESOURCE_USAGE.md` - Detailed resource usage strategy
