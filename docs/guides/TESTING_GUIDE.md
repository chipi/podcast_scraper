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

```bash
make test-sequential
make test-unit-sequential
make test-integration-sequential
make test-e2e-sequential
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
| `@pytest.mark.serial` | Must run sequentially |
| `@pytest.mark.ml_models` | Requires ML dependencies |
| `@pytest.mark.slow` | Slow-running tests |
| `@pytest.mark.network` | Hits external network |
| `@pytest.mark.llm` | Uses LLM APIs (may incur costs) |
| `@pytest.mark.openai` | Uses OpenAI specifically |

## Network Isolation

All tests use network isolation:

```bash
--disable-socket --allow-hosts=127.0.0.1,localhost
```

- **Unit tests:** Network calls blocked by pytest plugin
- **Integration/E2E:** Network calls blocked by pytest-socket

## Parallel Execution

Tests run in parallel by default. Serial tests run first:

1. Tests marked `@pytest.mark.serial` run sequentially
2. Remaining tests run in parallel with `-n auto`

## Flaky Test Reruns

Integration and E2E tests use reruns:

```bash
pytest --reruns 2 --reruns-delay 1
```yaml

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

## Coverage Targets

- **Overall:** >80%
- **Critical modules:** >90%
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
