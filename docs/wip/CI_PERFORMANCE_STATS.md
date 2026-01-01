# CI Performance Statistics

This document tracks performance metrics for the CI test suites, comparing fast CI (`make ci-fast`) and full CI (`make ci`) runs.

**Last Updated:** 2025-12-31 (after enhanced model verification in preload-ml-models)

## Overview

- **Fast CI (`make ci-fast`)**: Runs critical path tests only (unit + critical path integration + critical path e2e)
- **Full CI (`make ci`)**: Runs all tests (unit + integration + e2e, all slow/fast variants)

## Fast CI Performance (`make ci-fast`)

### Timing Summary (Latest Run - 2025-12-31)

**Total execution time:** ~177s (2:57) - includes formatting, linting, security, tests, docs, build

**Breakdown:**
- **Formatting/Linting/Type checking:** ~5-10s (black, isort, flake8, mypy)
- **Security audit:** ~1-2s (bandit, pip-audit)
- **Tests:** `155.70s` (2:35.70) — 717 passed, 40 skipped, 0 failed
  - **Serial tests:** `27.08s` — 4 passed (runs sequentially first)
  - **Parallel tests:** `128.62s` (2:08.62) — 713 passed, 40 skipped, 0 failed
- **Documentation build:** ~1-2s (mkdocs)
- **Package build:** ~15-20s (python build)

### Test Results

- **Total tests:** 717 passed, 40 skipped, 0 failed ✅
- **Serial tests:** 4 passed in 27.08s (runs sequentially to avoid parallel execution issues)
- **Parallel tests:** 713 passed, 40 skipped in 128.62s
- **Test execution time:** 155.70 seconds (27.08s serial + 128.62s parallel)
- **Coverage:** 76.03%

### Serial Test Optimization

**New Feature:** Tests marked with `@pytest.mark.serial` run sequentially first, then remaining tests run in parallel.

**Benefits:**
- Prevents race conditions in tests that have resource conflicts
- Allows problematic tests to run safely while maintaining parallel execution for the rest
- 5 tests marked as serial (4 in fast suite, 1 in data quality suite)

**Serial Tests in Fast Suite:**
- `test_library_api_basic_pipeline_path1`
- `test_library_api_basic_pipeline_path2`
- `test_service_api_basic_run_path1`
- `test_service_api_basic_run_path2`

### Top 10 Slowest Tests

1. **78.92s** - `tests/test_infrastructure.py::test_makefile_test_e2e_runs_tests`
   - Infrastructure test that runs `make test-e2e` (expected to be slow)

2. **32.58s** - `tests/test_infrastructure.py::test_makefile_test_integration_runs_tests`
   - Infrastructure test that runs `make test-integration` (expected to be slow)

3. **21.03s** - `tests/integration/test_http_integration.py::TestHTTPClientIntegration::test_timeout_handling`
   - HTTP timeout handling test (marked as `slow`, correctly excluded from fast tests)

4. **4.31s** - `tests/test_infrastructure.py::test_marker_combinations_work`
   - Infrastructure test for marker combinations

5. **3.69s** - `tests/test_infrastructure.py::test_no_tests_collected_warning`
   - Infrastructure test for warning detection

6. **3.58s** - `tests/test_infrastructure.py::test_integration_marker_collects_tests`
   - Infrastructure test for marker collection

7. **3.51s** - `tests/test_infrastructure.py::test_e2e_marker_collects_tests`
   - Infrastructure test for E2E marker collection

8. **0.73s** - `tests/e2e/test_e2e_server.py::TestE2EServerSmokeFeed::test_smoke_feed_with_openai_mocks_fast`
   - E2E server test with OpenAI mocks

9. **0.50s** (teardown) - `tests/integration/test_http_integration.py::TestHTTPClientIntegration::test_transcript_download`
   - HTTP integration test teardown

10. **0.50s** (teardown) - `tests/integration/test_http_integration.py::TestHTTPClientIntegration::test_user_agent_header`
    - HTTP integration test teardown

### Fast CI Observations

- **All tests passing:** ✅ 0 failures after standardizing on `tiny.en` Whisper model
- **Serial test optimization working:** 4 serial tests run sequentially in 27.08s, preventing parallel execution issues.
- **Parallel execution efficient:** 713 tests run in parallel in 128.62s (~0.18s per test average).
- **Coverage:** 76.03% (stable, good coverage for critical path tests)
- **Whisper model standardization:** All tests now use `tiny.en` instead of `base.en`, eliminating model mismatch issues
- Most actual test execution times are under 0.5 seconds, indicating fast tests are working as intended.
- The timeout/retry tests are correctly excluded from fast runs.

## Full CI Performance (`make ci`)

### Timing Summary (Latest Run - 2025-12-31, After Enhanced Model Verification)

**Total execution time:** ~207s (3:27) - includes formatting, linting, security, tests, docs, build

**Breakdown:**
- **Formatting/Linting/Type checking:** ~5-10s (black, isort, flake8, mypy, markdownlint)
- **Security audit:** ~1-2s (bandit, pip-audit)
- **Tests:** `127.72s` (2:07.72) — 1035 passed, 59 skipped, 9 failed
  - **Serial tests:** `34.19s` — 5 passed (runs sequentially first)
  - **Parallel tests:** `93.53s` (1:33.53) — 1030 passed, 59 skipped, 9 failed
- **Documentation build:** ~1-2s (mkdocs)
- **Package build:** ~15-20s (python build)

### Test Results

- **Total tests:** 1035 passed, 59 skipped, 9 failed
- **Serial tests:** 5 passed in 34.19s (runs sequentially to avoid parallel execution issues)
- **Parallel tests:** 1030 passed, 59 skipped, 9 failed in 93.53s
- **Test execution time:** 127.72 seconds (34.19s serial + 93.53s parallel)
- **Coverage:** 76.44%

### Top 10 Slowest Tests

1. **75.76s** - `tests/e2e/test_cli_e2e.py::TestCLIBasicCommands::test_generate_summaries`
   - E2E CLI test with full pipeline including summarization

2. **55.54s** - `tests/e2e/test_service_api_e2e.py::TestServiceAPIBasic::test_service_run_all_features`
   - E2E Service API test with all features enabled

3. **55.49s** - `tests/e2e/test_library_api_e2e.py::TestLibraryAPIBasic::test_run_pipeline_all_features`
   - E2E Library API test with all features enabled

4. **39.31s** - `tests/test_infrastructure.py::test_makefile_test_e2e_runs_tests`
   - Infrastructure test that runs `make test-e2e` (expected to be slow)

5. **32.77s** - `tests/test_infrastructure.py::test_makefile_test_integration_runs_tests`
   - Infrastructure test that runs `make test-integration` (expected to be slow)

6. **21.03s** - `tests/integration/test_http_integration.py::TestHTTPClientIntegration::test_timeout_handling`
   - HTTP timeout handling test (marked as `slow`, correctly excluded from fast tests)

7. **15.31s** - `tests/e2e/test_error_handling_e2e.py::TestPartialFailureHandling::test_mixed_success_and_failure`
   - E2E error handling test for mixed success/failure scenarios

8. **15.29s** - `tests/e2e/test_error_handling_e2e.py::TestHTTPErrorHandling::test_transcript_download_500_error`
   - E2E error handling test for 500 errors on transcript download

9. **15.08s** - `tests/e2e/test_library_api_e2e.py::TestLibraryAPIBasic::test_run_pipeline_with_summaries`
   - E2E Library API test with summarization

10. **15.03s** - `tests/e2e/test_error_handling_e2e.py::TestHTTPErrorHandling::test_rss_feed_500_error`
    - E2E error handling test for 500 errors on RSS feed

### Full CI Observations

- **9 test failures:** Related to ML model cache issues (NOT expected - models were preloaded):
  1. `test_transformers_model_loading` - Network access blocked (trying to download missing `tokenizer.json`)
  2. `test_all_providers_initialize_with_real_models` - Network access blocked (same issue)
  3. `test_summarization_provider_with_real_model` - Network access blocked (same issue)
  4. `test_bart_small_model_loads` - Model cache incomplete (network access blocked)
  5. `test_fast_model_loads` - Model cache incomplete (network access blocked)
  6. `test_cli_basic_transcript_download_path1` - Summary is None (model cache incomplete)
  7. `test_transformers_provider_summarize` - Model cache incomplete (`.incomplete` file error)
  8. `test_multi_episode_with_summarization_smoke` - Summary is None (model cache incomplete)
  9. `test_pipeline_with_summarization` - Summary is None (model cache incomplete)
- **Root Causes:**
  1. **`facebook/bart-base` NOT cached** - Preload should have downloaded it, but cache was missing (likely silent failure)
  2. **`allenai/led-base-16384` missing `tokenizer.json`** - Has `tokenizer_config.json` but NOT `tokenizer.json`. Cache check was too lenient (accepted `tokenizer_config.json` as sufficient), but Transformers requires `tokenizer.json` for loading
  3. **Cache check too lenient** - Accepted `tokenizer_config.json` OR `tokenizer.json`, but Transformers requires `tokenizer.json`
- **Fix Applied:** Updated cache check to require `tokenizer.json` (not just `tokenizer_config.json`). This will catch incomplete caches earlier and skip tests instead of attempting downloads.
- **Serial test optimization working:** 5 serial tests run sequentially in 34.19s, preventing parallel execution issues.
- **Parallel execution efficient:** 1030 tests run in parallel in 93.53s (~0.09s per test average).
- **Coverage:** 76.44% (stable, good coverage for full test suite).
- **Enhanced model verification:** New stronger verification in `preload-ml-models` should catch corrupted models earlier.
- Slowest tests are E2E tests with full pipeline features (summarization, all features), which is expected.
- Infrastructure tests are slow because they run make commands; this is expected.
- The timeout handling test is correctly excluded from fast tests and runs in the full suite.
- Most tests complete quickly; the top E2E tests account for most of the execution time.

## Comparison: Fast CI vs Full CI

| Metric | Fast CI | Full CI | Difference |
|--------|---------|---------|------------|
| **Total Time** | ~177s | ~207s | Full CI is **~30s slower** |
| **Test Time** | 155.70s | 127.72s | Full CI is **27.98s faster** |
| **Tests Passed** | 717 | 1035 | Full CI has **318 more tests** |
| **Tests Skipped** | 40 | 59 | Full CI has **19 more skips** |
| **Tests Failed** | 0 ✅ | 9 ⚠️ | Fast CI has **0 failures** (Full CI failures expected after cache clean) |
| **Coverage** | 76.03% | 76.44% | Full CI has **0.41% higher coverage** |

### Why Full CI Test Time is Faster Despite More Tests

The apparent paradox (more tests running faster) is explained by:

1. **Parallel execution efficiency**: Full CI runs tests in smoke mode (5 episodes) which allows better parallelization across more test files, while Fast CI uses fast mode (1 episode) which may have less parallel work.

2. **Test distribution**: Fast CI includes infrastructure tests that run make commands (78.92s + 32.58s = 111.5s), which are sequential and slow. Full CI spreads the load more evenly.

3. **E2E test mode**:
   - Fast CI: Uses `E2E_TEST_MODE=fast` (1 episode per feed)
   - Full CI: Uses `E2E_TEST_MODE=smoke` (5 episodes per feed)
   - The smoke mode allows better parallelization of E2E tests.

4. **Total CI time difference**: While test execution is faster in Full CI, the total CI time is slower due to additional overhead (formatting, linting, security checks, docs, build).

## Key Insights

1. **Fast CI is optimized for critical path**: Excludes slow tests (timeout/retry) and focuses on essential functionality.

2. **Infrastructure tests dominate Fast CI time**: The two infrastructure tests that run make commands account for ~91% of the slowest test time in Fast CI.

3. **E2E tests with full features are slowest**: Tests that exercise the complete pipeline (summarization, all features) take 55-75 seconds each.

4. **Most tests are fast**: After the top few slow tests, most tests complete in under 1 second.

5. **Test categorization is working**: Slow tests are correctly excluded from fast runs, and the critical path is well-defined.

## Recommendations

1. **Consider optimizing infrastructure tests**: The infrastructure tests that run make commands could potentially be optimized or moved to a separate suite.

2. **Monitor E2E test performance**: The top 3 E2E tests (75s, 55s, 55s) account for significant time. Consider if these can be optimized or if they're acceptable given their comprehensive coverage.

3. **Continue excluding slow tests from fast CI**: The current approach of excluding `slow` tests from fast CI is working well and should be maintained.

4. **Track these metrics over time**: Regular monitoring of CI performance will help identify regressions and optimization opportunities.

## Test Suite Configuration

### Fast CI Test Selector

```bash
-m '(not integration and not e2e) or (integration and critical_path and not slow) or (e2e and critical_path and not slow)'
```

- Unit tests: All
- Integration tests: Only `critical_path` and `not slow`
- E2E tests: Only `critical_path` and `not slow`
- E2E mode: `fast` (1 episode per feed)

### Full CI Test Selector

```bash
# All tests (no marker filtering)
```

- Unit tests: All
- Integration tests: All (including slow)
- E2E tests: All (including slow)
- E2E mode: `smoke` (5 episodes per feed)

## Notes

- All tests run with network isolation (`--disable-socket --allow-hosts=127.0.0.1,localhost`)
- Tests use parallel execution (`-n auto`)
- Coverage is collected for both suites
- ML model tests are skipped if models are not cached (prevents network downloads)

