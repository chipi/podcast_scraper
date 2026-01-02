# GitHub Actions Resource Usage Strategy

## Overview

This document outlines our conservative resource usage strategy for GitHub Actions to avoid hitting
limits and ensure reliable CI/CD.

## GitHub Actions Limits

### Free Tier Limits

- **Minutes per month**: 2,000 minutes
- **Concurrent jobs**: 20 jobs
- **Job timeout**: 6 hours (default, can be set lower)
- **Cache size**: 10 GB per repository

### Paid Tier Limits

- **Minutes per month**: Varies by plan
- **Concurrent jobs**: Varies by plan
- **Job timeout**: 6 hours (default)

## Our Conservative Strategy

### 1. Parallelism Reduction

**Local Development (Makefile):**

- Uses `auto - 2` for pytest-xdist (reserves 2 cores for system)
- Sequential summarization in test environments (1 worker)

**GitHub Actions:**

- **MUST match local strategy** to ensure consistent behavior
- Uses `auto - 2` for pytest-xdist (reserves 2 cores for system)
- Sequential summarization in test environments (1 worker)

**Rationale:**

- Reduces memory pressure
- Prevents resource exhaustion
- More predictable execution times
- Better for shared CI runners

### 2. Cache Strategy

**Model Cache Locations:**

- **Local development**: `.cache/` in repo root (versioned structure, contents ignored)
- **GitHub Actions**: `~/.cache/whisper`, `~/.local/share/spacy`, `~/.cache/huggingface` (standard locations)

**Cache Key Strategy:**

- Versioned keys (`ml-models-${{ runner.os }}-v1`) for cache invalidation
- OS-specific keys for cross-platform compatibility
- Restore keys for fallback to older cache

**Cache Size:**

- Whisper models: ~150 MB
- spaCy models: ~50 MB (installed as packages)
- Transformers models: ~3.5 GB (facebook/bart-base + allenai/led-base-16384)
- **Total**: ~3.7 GB (well under 10 GB limit)

### 3. Job Timeouts

**Recommended Timeouts:**

- **Lint job**: 10 minutes (fast, no ML dependencies)
- **Unit tests**: 15 minutes (fast, no ML dependencies)
- **Integration tests**: 30 minutes (ML models, parallel execution)
- **E2E tests**: 45 minutes (ML models, parallel execution, multiple episodes)
- **Nightly tests**: 60 minutes (comprehensive suite with metrics)

**Implementation:**

```yaml
timeout-minutes: 30  # Add to each job
```

### 4. Job Dependencies

**Optimized Job Graph:**

```text
test-unit (parallel) ──┐
preload-ml-models (parallel) ──┐
docs (parallel) ──┐
build (parallel) ──┐
                   │
                   ├──> test-integration
                   └──> test-e2e
```

**Benefits:**

- Fast jobs (lint, unit tests) run immediately
- ML model preloading runs in parallel (doesn't block)
- Heavy jobs (integration, E2E) wait for model cache

### 5. Conditional Execution

**Path-based filtering:**

- Only run tests when relevant files change
- Skip ML-heavy jobs for documentation-only PRs
- Skip full test suite for markdown-only changes

**Branch-based filtering:**

- Full test suite on `main` branch
- Full test suite on PRs (except docs-only)
- Nightly tests only on schedule/manual trigger

## Resource Usage Estimates

### Per PR (typical)

- **Lint**: ~3 minutes
- **Unit tests**: ~5 minutes
- **Preload models**: ~2 minutes (cache hit) or ~15 minutes (cache miss)
- **Integration tests**: ~20 minutes
- **E2E tests**: ~25 minutes
- **Total**: ~55 minutes per PR (with cache) or ~70 minutes (cache miss)

### Per Nightly Run

- **Comprehensive tests**: ~45 minutes
- **Data quality tests**: ~15 minutes
- **Total**: ~60 minutes per nightly run

### Monthly Estimates

- **PRs**: ~10 PRs/month × 55 minutes = 550 minutes
- **Main branch pushes**: ~5 pushes/month × 55 minutes = 275 minutes
- **Nightly runs**: ~30 runs/month × 60 minutes = 1,800 minutes
- **Total**: ~2,625 minutes/month (exceeds free tier limit)

**Recommendation**: Disable nightly schedule, use manual triggers only for now.

## Monitoring

### Key Metrics to Track

1. **Job duration**: Should decrease with cache hits
2. **Cache hit rate**: Should be >80% after initial setup
3. **Memory usage**: Monitor for leaks or excessive usage
4. **Concurrent jobs**: Should stay well under 20

### Actions to Take if Limits Approached

1. Reduce parallelism further (e.g., `auto - 4`)
2. Disable non-critical jobs
3. Increase cache hit rate
4. Split jobs into smaller chunks
5. Consider GitHub Actions paid tier

## Best Practices

1. **Always use conservative parallelism** (`auto - 2` minimum)
2. **Set job timeouts** to prevent runaway jobs
3. **Monitor cache hit rates** and optimize cache keys
4. **Use path-based filtering** to skip unnecessary runs
5. **Test locally first** before pushing (saves CI minutes)
6. **Use `make ci-fast`** for quick local validation

## References

- [GitHub Actions Usage Limits](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions)
- [GitHub Actions Cache](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [Pytest-xdist Documentation](https://pytest-xdist.readthedocs.io/)
