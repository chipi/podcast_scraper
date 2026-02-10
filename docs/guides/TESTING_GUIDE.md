# Testing Guide

> **Document Structure:**
>
> - **[Testing Strategy](../TESTING_STRATEGY.md)** - High-level philosophy, test pyramid, decision criteria
> - **This document** - Quick reference, test execution commands
> - **[Unit Testing Guide](UNIT_TESTING_GUIDE.md)** - Unit test mocking patterns and isolation
> - **[Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md)** - Integration test mocking guidelines
> - **[E2E Testing Guide](E2E_TESTING_GUIDE.md)** - E2E server, real ML models, OpenAI mocking
> - **[Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md)** - What to test and prioritization

## Quick Reference

| Layer | Speed | Scope | Mocking |
| ------- | ------- | ------- | --------- |
| **Unit** | < 100ms | Single function | All dependencies mocked |
| **Integration** | < 5s | Component interactions | External services mocked |
| **E2E** | < 60s | Complete workflow | No mocking (real everything) |

**Decision Tree:**

1. Testing a complete user workflow? → **E2E Test**
2. Testing component interactions? → **Integration Test**
3. Testing a single function? → **Unit Test**

## Running Tests

### Default Commands

```bash

# Unit tests (parallel, network isolated)

make test-unit

# Integration tests (parallel, with reruns)

make test-integration

# E2E tests (serial first, then parallel)

make test-e2e

# All tests

make test
```

## Fast Variants (Critical Path Only)

```bash
make test-fast              # Unit + critical path integration + e2e
make test-integration-fast  # Critical path integration tests
make test-e2e-fast          # Critical path E2E tests
```

### Sequential (For Debugging)

For debugging test failures, run tests sequentially using pytest directly:

```bash

# Run all tests sequentially

pytest tests/ -n 0

# Run unit tests sequentially

pytest tests/unit/ -n 0

# Run integration tests sequentially

pytest tests/integration/ -n 0

# Run E2E tests sequentially

pytest tests/e2e/ -n 0
```

## Specific Tests

```bash

# Run specific test file

pytest tests/unit/podcast_scraper/test_config.py -v

# Run with marker

pytest tests/integration/ -m "integration and critical_path" -v

# Run with coverage

pytest tests/unit/ --cov=podcast_scraper --cov-report=term-missing
```yaml

## Test Markers

| Marker | Purpose |
| -------- | --------- |
| `@pytest.mark.unit` | Unit tests |
| `@pytest.mark.integration` | Integration tests |
| `@pytest.mark.e2e` | End-to-end tests |
| `@pytest.mark.critical_path` | Critical path tests (run in fast suite) |
| `@pytest.mark.nightly` | Nightly-only tests (excluded from regular CI) |
| `@pytest.mark.flaky` | May fail intermittently (gets reruns) |
| `@pytest.mark.serial` | Must run sequentially (rarely needed) |
| `@pytest.mark.ml_models` | Requires ML dependencies |
| `@pytest.mark.slow` | Slow-running tests |
| `@pytest.mark.network` | Hits external network |
| `@pytest.mark.llm` | Uses LLM APIs (excluded from nightly to avoid costs) |
| `@pytest.mark.openai` | Uses OpenAI specifically (subset of `llm`) |

## Network Isolation

All tests use network isolation:

```bash
--disable-socket --allow-hosts=127.0.0.1,localhost
```

- **Unit tests:** Network calls blocked by pytest plugin
- **Integration/E2E:** Network calls blocked by pytest-socket

## Parallel Execution

Tests run in parallel by default using `pytest-xdist`:

1. Tests marked `@pytest.mark.serial` run sequentially first (if any)
2. Remaining tests run in parallel with memory-aware worker calculation

The Makefile automatically calculates the optimal number of workers based on:

- Available system memory
- CPU core count
- Test type (unit/integration/e2e have different memory requirements)
- Platform (more conservative on macOS)

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#memory-issues-with-ml-models) for details on memory-aware worker calculation.

> **Note:** The `@pytest.mark.serial` marker is rarely needed now. Global state cleanup
> fixtures in `conftest.py` reset shared state between tests, allowing most tests to run
> in parallel safely. Only use `serial` for tests with genuine resource conflicts.

### Memory Cleanup Best Practices

**Automatic Cleanup:**

The test suite includes an automatic cleanup fixture (`cleanup_ml_resources_after_test`) that:

- Limits PyTorch thread pools to prevent excessive thread spawning
- Cleans up the global preloaded ML provider
- **Finds and cleans up ALL SummaryModel and provider instances** created during tests (Issue #351)
- Forces garbage collection after integration/E2E tests

**Explicit Cleanup (Recommended):**

While automatic cleanup handles most cases, explicit cleanup is recommended for clarity and immediate memory release:

```python
from tests.conftest import cleanup_model, cleanup_provider

def test_something():
    # Create model directly
    model = summarizer.SummaryModel(...)
    try:
        # test code
    finally:
        cleanup_model(model)  # Explicit cleanup

def test_with_provider():
    # Create provider directly
    provider = create_summarization_provider(cfg)
    try:
        # test code
    finally:
        cleanup_provider(provider)  # Explicit cleanup
```

**Why Explicit Cleanup?**

1. **Immediate memory release** - Models are unloaded as soon as the test completes
2. **Clarity** - Makes it obvious that cleanup is happening
3. **Defensive** - Works even if automatic cleanup has issues
4. **Best practice** - Matches the pattern of resource management (try/finally)

**Helper Functions:**

- `cleanup_model(model)` - Unloads a SummaryModel instance
- `cleanup_provider(provider)` - Cleans up a provider instance (MLProvider, etc.)

Both functions are idempotent (safe to call multiple times) and handle None gracefully.

### ⚠️ Warning: `-s` Flag and Parallel Execution

**Do not use `-s` (no capture) with parallel tests** — it causes hangs due to tqdm
progress bars competing for terminal access.

```bash

# DON'T DO THIS (hangs)

pytest tests/e2e/ -s -n auto

# DO THIS INSTEAD

pytest tests/e2e/ -v -n auto     # Use -v for verbose output
pytest tests/e2e/ -s -n 0        # Or disable parallelism
make test-e2e-sequential         # Or use sequential target
```

See [Issue #176](https://github.com/chipi/podcast_scraper/issues/176) for details.

## Flaky Test Reruns

Integration and E2E tests use reruns:

```bash
pytest --reruns 3 --reruns-delay 1
```yaml

## Flaky Test Markers

Some tests are marked with `@pytest.mark.flaky` to indicate they may fail intermittently
due to inherent non-determinism. These tests get automatic reruns.

### Why Some Tests Are Flaky

| Category | Tests | Root Cause |
| -------- | ----- | ---------- |
| **Whisper Transcription** | 4 | ML inference variability - audio transcription has natural variation |
| **Full Pipeline + Whisper** | 2 | Whisper timing + audio file I/O |
| **OpenAI Mock Integration** | 2 | Mock response parsing timing |
| **Full Pipeline + OpenAI** | 7 | Complex multi-component timing |

### Current Flaky Test Count: 15

| File | Count | Category |
| ---- | ----- | -------- |
| `test_basic_e2e.py` | 7 | Full pipeline with OpenAI mocks |
| `test_whisper_e2e.py` | 4 | Whisper inference variability |
| `test_full_pipeline_e2e.py` | 2 | Whisper transcription |
| `test_openai_provider_integration_e2e.py` | 2 | OpenAI mock responses |

### Tests That Are NOT Flaky

The following categories are now **stable** and don't need flaky markers:

- **Transformers/spaCy model loading** - Uses offline mode (`HF_HUB_OFFLINE=1`)
- **ML model tests** - Explicit `summary_reduce_model` prevents cache misses
- **HTTP integration tests** - Explicit server waits prevent timing issues
- **Parallel execution** - Global state cleanup prevents race conditions

### Reducing Flakiness

If you encounter a flaky test, check these common causes:

1. **Network access** - Should be blocked via pytest-socket
2. **Model cache** - Run `make preload-ml-models` first
3. **Global state** - Ensure cleanup fixtures reset shared state
4. **Progress bars** - `TQDM_DISABLE=1` is set automatically in tests

See [Issue #177](https://github.com/chipi/podcast_scraper/issues/177) for investigation details.

## E2E Test Modes

Set via `E2E_TEST_MODE` environment variable:

| Mode | Episodes | Use Case |
| ------ | ---------- | ---------- |
| `fast` | 1 per test | Quick feedback |
| `multi_episode` | 5 per test | Full validation |
| `data_quality` | Multiple | Nightly only |

## ML Model Preloading

Tests require models to be pre-cached:

```bash
make preload-ml-models
```

See [E2E Testing Guide](E2E_TESTING_GUIDE.md) for model defaults.

## E2E Acceptance Tests

Acceptance tests allow you to run multiple configuration files sequentially, collect structured data (logs, outputs, timing, resource usage), and compare results against baselines. This is useful for:

- Running the same configs across different code versions to detect regressions
- Testing multiple configs with different RSS feeds or settings
- Validating system acceptance of different provider/model configurations
- Comparing performance metrics across runs

### Running Acceptance Tests

```bash
# Run a single config file
make test-acceptance CONFIGS="examples/config.example.yaml"

# Run multiple configs (using glob patterns)
make test-acceptance CONFIGS="examples/config.my.*.yaml"

# Save current runs as a baseline for future comparison
make test-acceptance CONFIGS="examples/config.example.yaml" SAVE_AS_BASELINE=baseline_v1

# Compare against an existing baseline
make test-acceptance CONFIGS="examples/config.example.yaml" COMPARE_BASELINE=baseline_v1

# Use fixture feeds (mock data) instead of real RSS feeds
make test-acceptance CONFIGS="examples/config.example.yaml" USE_FIXTURES=1

# Disable real-time log streaming (only save to files)
make test-acceptance CONFIGS="examples/config.example.yaml" NO_SHOW_LOGS=1

# Disable automatic analysis and benchmark reports
make test-acceptance CONFIGS="examples/config.example.yaml" NO_AUTO_ANALYZE=1 NO_AUTO_BENCHMARK=1
```

### Understanding Sessions vs Runs

**Session** = One execution of the acceptance test tool

- Triggered by a single command invocation
- Can process multiple config files sequentially
- Has a unique `session_id` (timestamp-based)
- Contains a summary of all runs in that session

**Run** = One execution of a single config file within a session

- Each config file you pass creates one run
- Has its own `run_id`, timing, exit code, logs, and outputs
- Runs execute sequentially within the session

**Example:**

If you run:

```bash
make test-acceptance CONFIGS="config1.yaml config2.yaml config3.yaml"
```

You get:

- **1 Session** (with `session_id = 20260208_101601`)
  - **3 Runs** (one for each config file)
    - `run_20260208_101601_123` (config1.yaml)
    - `run_20260208_101601_456` (config2.yaml)
    - `run_20260208_101601_789` (config3.yaml)

### Output Structure

Results are saved to `.test_outputs/acceptance/` by default:

```text
.test_outputs/acceptance/
├── sessions/
│   └── session_20260208_101601/          ← ONE SESSION
│       ├── session.json                   ← Summary of all runs
│       └── runs/
│           ├── run_20260208_101601_123/  ← RUN #1 (config1)
│           │   ├── config.original.yaml  ← Original config for this run
│           │   ├── config.yaml           ← Modified config used for execution
│           │   ├── run_data.json
│           │   ├── stdout.log
│           │   ├── stderr.log
│           │   └── ... (service outputs)
│           ├── run_20260208_101601_456/  ← RUN #2 (config2)
│           │   ├── config.original.yaml  ← Original config for this run
│           │   ├── config.yaml
│           │   └── ...
│           └── run_20260208_101601_789/  ← RUN #3 (config3)
│               ├── config.original.yaml  ← Original config for this run
│               ├── config.yaml
│               └── ...
└── baselines/
    └── baseline_v1/                       ← Saved baselines
        ├── baseline.json
        └── run_20260208_101601_123/      ← Copied run data
```

### Analyzing Results

Use the analysis script to generate reports:

```bash
# Basic analysis
make analyze-acceptance SESSION_ID=20260208_101601

# Comprehensive analysis with baseline comparison
make analyze-acceptance SESSION_ID=20260208_101601 MODE=comprehensive COMPARE_BASELINE=baseline_v1

# Or use the script directly
python scripts/acceptance/analyze_bulk_runs.py \
    --session-id 20260208_101601 \
    --output-dir .test_outputs/acceptance \
    --mode comprehensive \
    --compare-baseline baseline_v1
```

### Performance Benchmarking

Generate performance benchmarking reports that group runs by provider/model configuration:

```bash
# Generate benchmark report
make benchmark-acceptance SESSION_ID=20260208_101601

# Generate benchmark report with baseline comparison
make benchmark-acceptance SESSION_ID=20260208_101601 COMPARE_BASELINE=baseline_v1

# Or use the script directly
python scripts/acceptance/generate_performance_benchmark.py \
    --session-id 20260208_101601 \
    --output-dir .test_outputs/acceptance \
    --compare-baseline baseline_v1
```

The benchmark report includes:

- **Summary table** comparing all provider/model configurations
- **Performance metrics** per configuration (time per episode, throughput, memory)
- **Detailed analysis** for each configuration
- **Performance comparison** (fastest vs. slowest, memory usage)
- **Baseline comparison** (if `--compare-baseline` is provided):
  - Performance changes vs. baseline (time, throughput, memory)
  - Regression detection (20% slower, 100MB more memory)
  - Improvement detection (10% faster, 50MB less memory)
  - Detailed per-configuration comparison

**Baseline Comparison Features:**

- Compares provider/model configurations between current run and baseline
- Detects regressions (performance degradation)
- Detects improvements (performance gains)
- Shows percentage changes for all metrics
- Groups comparisons by provider/model (not just config name)

Reports are generated in both Markdown and JSON formats for easy review and programmatic analysis.

## Test Organization

```text
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── conftest.py          # Network/filesystem isolation
│   └── podcast_scraper/     # Per-module tests
├── integration/             # Integration tests
│   ├── conftest.py          # Shared fixtures
│   └── test_*.py            # Component interaction tests
├── e2e/                     # E2E tests
│   ├── fixtures/            # E2E server, HTTP server
│   └── test_*.py            # Complete workflow tests
└── conftest.py              # Shared fixtures, ML cleanup
```yaml

## Coverage Thresholds

Per-tier thresholds enforced in CI (prevents regression):

| Tier | Threshold | Current |
| ---- | --------- | ------- |
| **Unit** | 70% | ~74% |
| **Integration** | 40% | ~42% |
| **E2E** | 40% | ~50% |
| **Combined** | 80% | ~82% |

**Note:** Local make targets now run with coverage:

```bash
make test-unit          # includes --cov
make test-integration   # includes --cov
make test-e2e           # includes --cov
```

## Test Count Targets

- **Unit tests:** 200+
- **Integration tests:** 50+
- **E2E tests:** 100+

## Layer-Specific Guides

For detailed implementation patterns:

- **[Unit Testing Guide](UNIT_TESTING_GUIDE.md)** - What to mock, isolation enforcement, test structure
- **[Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md)** - Mock vs real decisions, ML model usage
- **[E2E Testing Guide](E2E_TESTING_GUIDE.md)** - E2E server, OpenAI mocking, network guard

## What to Test

- **[Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md)** - What to test based on the critical path,
  test prioritization, fast vs slow tests, `@pytest.mark.critical_path` marker

## Domain-Specific Testing

- **[Provider Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md#testing-your-provider)** - Provider testing
  across all tiers (unit, integration, E2E), E2E server mock endpoints, testing checklist

## References

- [Testing Strategy](../TESTING_STRATEGY.md) - Overall testing philosophy
- [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md) - Prioritization
- [CI/CD Documentation](../ci/index.md) - GitHub Actions workflows
- [Architecture](../ARCHITECTURE.md) - Testing Notes section
