# E2E Testing Strategy: Code Quality vs Data Quality

## Problem Statement

E2E tests currently process **1 episode** in fast mode for quick CI/CD feedback. We need to:

1. **Expand functional coverage** in fast E2E tests (more scenarios, still 1 episode)
2. **Separate data quality validation** from code quality checks
3. **Run data quality tests nightly** (not blocking PRs/commits)

**Key Insight:**

- **Code Quality** = Does the code work correctly? (1 episode, fast, run on every PR)
- **Data Quality** = Does data processing work well with volume? (3-5 episodes, nightly only)

Processing **all episodes** from full feeds would:

- Take **hours** to complete
- Strain transcription and summarization
- Slow down CI/CD feedback
- Not be practical for regular test runs

## Proposed Solution: Two-Tier E2E Testing Strategy

**Core Principle:** Separate code quality (fast, 1 episode, PR-blocking) from data quality (slower, 3-5 episodes, nightly only)

### Tier 1: Fast E2E Tests (Expand Functional Coverage)

**Purpose:** Fast CI/CD feedback, validate critical path and functional coverage

**Configuration:**

- **1 episode** per test (keep fast)
- **Fast fixtures** (1-minute audio, short transcripts)
- **Mocked Whisper** (for speed)
- **Real ML models** for NER and summarization (if cached)
- **Real OpenAI providers** (via E2E server mocks)

**What to Expand:**

- ✅ **More test scenarios** (different configurations, edge cases)
- ✅ **More functional coverage** (all provider combinations, all paths)
- ✅ **More API coverage** (CLI, Library API, Service API)
- ✅ **More edge cases** (error handling, recovery, concurrent processing)
- ❌ **NOT more episodes** (stay at 1 episode for speed)

**When to Run:**

- ✅ On every PR
- ✅ On every commit to main
- ✅ Local development (fast feedback)
- ✅ Part of code quality checks

**Duration:** ~30-60 seconds per test

**Markers:** `@pytest.mark.e2e` (default, no additional markers)

### Tier 2: Data Quality E2E Tests (New - Nightly Only)

**Purpose:** Validate data quality, consistency, and volume handling across multiple episodes

**Configuration:**

- **3-5 episodes** per test (enough to catch volume issues, not hours of processing)
- **Real fixtures** (full-length audio/transcripts)
- **Real Whisper** (if marked `@pytest.mark.ml_models`)
- **Real ML models** for NER and summarization
- **Real OpenAI providers** (via E2E server mocks)

**What to Test:**

- ✅ **Data consistency**: Same episode processed multiple times produces same results
- ✅ **Multi-episode workflows**: Processing 3-5 episodes in sequence
- ✅ **Resource usage**: Memory, processing time with volume
- ✅ **Edge case detection**: Issues that only appear with multiple episodes
- ✅ **Data quality metrics**: Summary quality, speaker detection accuracy across episodes

**When to Run:**

- ✅ **Nightly builds ONLY** (scheduled, separate from code quality)
- ✅ **On-demand** (manual trigger for specific validation)
- ❌ **NOT on PRs** (not part of code quality checks)
- ❌ **NOT on main branch commits** (nightly only)
- ❌ **NOT part of CI/CD code quality pipeline**

**Duration:** ~5-15 minutes per test suite

**Markers:** `@pytest.mark.e2e` + `@pytest.mark.data_quality` + `@pytest.mark.slow`

**Key Principle:** Data quality tests are **separate from code quality** - they validate data processing quality, not code correctness.

**Example Test:**

```python
@pytest.mark.e2e
@pytest.mark.data_quality
@pytest.mark.slow
def test_data_quality_across_multiple_episodes(e2e_server):
    """Test data quality and consistency across 3 episodes."""
    cfg = config.Config(
        rss_url=e2e_server.urls.feed("podcast1"),
        output_dir=tmpdir,
        max_episodes=3,  # Process 3 episodes
        # ... other config
    )
    # Validate consistency, quality metrics, etc.
```python

### Optional: Comprehensive E2E Tests (Future - If Needed)

**Purpose:** Full feed validation, stress testing, production-like scenarios

**Configuration:**

- **All episodes** from a feed (or subset of feeds)
- **Full production-like** processing
- **Real everything** (Whisper, ML models, OpenAI if configured)

**When to Run:**

- ✅ **Weekly/Monthly** (if needed for specific validation)
- ✅ **Before major releases** (optional)
- ✅ **On-demand** for specific validation
- ❌ **NOT in regular CI/CD** (too slow)

**Duration:** Hours (acceptable for rare runs)

**Markers:** `@pytest.mark.e2e` + `@pytest.mark.comprehensive` + `@pytest.mark.slow`

**Note:** This tier is optional and may not be needed if Tier 2 (data quality) provides sufficient coverage.

## Implementation Plan

### Phase 1: Document Strategy (This Document)

- ✅ Define three-tier strategy
- ✅ Document when each tier runs
- ✅ Define markers and configuration

### Phase 2: Create Data Quality Test Suite

**Tasks:**

1. Create new test file: `tests/e2e/test_data_quality_e2e.py`
2. Add `@pytest.mark.data_quality` marker
3. Implement tests for:
   - Multi-episode processing (3-5 episodes)
   - Data consistency validation
   - Resource usage monitoring
   - Edge case detection
4. Update `pyproject.toml` to define marker
5. Update CI/CD to run data quality tests in nightly builds

**Example Test Structure:**

```python
@pytest.mark.e2e
@pytest.mark.data_quality
@pytest.mark.slow
class TestDataQualityE2E:
    """Data quality validation across multiple episodes."""

    def test_consistency_across_episodes(self, e2e_server):
        """Test that same episodes produce consistent results."""
        # Process same episodes twice, compare outputs

    def test_multi_episode_workflow(self, e2e_server):
        """Test processing 3-5 episodes in sequence."""
        # Process multiple episodes, validate all succeed

    def test_resource_usage_with_volume(self, e2e_server):
        """Test memory and processing time with multiple episodes."""
        # Monitor resource usage, validate within limits
```python

### Phase 3: Update CI/CD Configuration

**GitHub Actions:**

- ✅ **PR/Main branch**: Run Tier 1 (fast E2E tests) only - **code quality checks**
- ✅ **Nightly builds**: Run Tier 2 (data quality tests) only - **separate from code quality**
- ✅ **Manual dispatch**: Option to run Tier 2 or Tier 3 on-demand

**Key Principle:**

- **Code quality** = Tier 1 (fast E2E with 1 episode, expanded functional coverage)
- **Data quality** = Tier 2 (nightly only, 3-5 episodes, separate workflow)

**Makefile:**

```makefile
test-e2e-fast:          # Tier 1 only - code quality (current, expand coverage)
test-e2e-data-quality: # Tier 2 only - data quality (new, nightly only)
test-e2e-all:          # All tiers (for local validation)
```

### Phase 4: Update Documentation

- Update `TESTING_STRATEGY.md` with three-tier strategy
- Update `CRITICAL_PATH_TESTING_GUIDE.md` with data quality section
- Update `CI_CD.md` with new test tiers

## Benefits

1. **Fast CI/CD feedback**: Tier 1 stays fast (1 episode)
2. **Data quality validation**: Tier 2 catches issues with volume (3-5 episodes)
3. **Comprehensive validation**: Tier 3 validates full production scenarios (all episodes)
4. **Flexible execution**: Run appropriate tier based on context
5. **Resource efficiency**: Don't waste CI time on slow tests unless needed

## Recommendations

**For E2E Tests (Tier 1 - Code Quality):**

- ✅ **Keep at 1 episode** - Fast feedback is critical
- ✅ **Expand functional coverage** - More test scenarios, not more episodes
- ✅ **Use fast fixtures** - 1-minute audio, short transcripts
- ✅ **Mock Whisper** - Speed over accuracy in fast tests
- ✅ **Real ML models** - If cached, use real NER/summarization
- ✅ **Run on every PR/commit** - Part of code quality pipeline

**For Data Quality Tests (Tier 2 - Nightly Only):**

- ✅ **3-5 episodes** - Enough to catch volume issues, not hours
- ✅ **Real fixtures** - Full-length audio/transcripts
- ✅ **Real Whisper** - If marked `@pytest.mark.ml_models`
- ✅ **Focus on quality** - Consistency, edge cases, resource usage
- ✅ **Run nightly ONLY** - Separate from code quality, not blocking
- ✅ **Separate workflow** - Don't mix with code quality checks

**Key Distinction:**

- **Code Quality** = Does the code work correctly? (Tier 1, 1 episode, fast)
- **Data Quality** = Does the data processing work well with volume? (Tier 2, 3-5 episodes, nightly)

## Next Steps

1. **Review and approve** this strategy
2. **Create data quality test suite** (Phase 2)
3. **Update CI/CD** to run data quality tests in nightly builds
4. **Monitor and adjust** based on results
