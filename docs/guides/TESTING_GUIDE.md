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

### Specific Tests

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
2. Remaining tests run in parallel with `-n auto`

> **Note:** The `@pytest.mark.serial` marker is rarely needed now. Global state cleanup
> fixtures in `conftest.py` reset shared state between tests, allowing most tests to run
> in parallel safely. Only use `serial` for tests with genuine resource conflicts.

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
```

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
```

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
- [CI/CD Documentation](../CI_CD.md) - GitHub Actions workflows
- [Architecture](../ARCHITECTURE.md) - Testing Notes section
