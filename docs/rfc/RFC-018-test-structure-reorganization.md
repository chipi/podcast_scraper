# RFC-018: Test Structure Reorganization

- **Status**: Completed
- **Authors**:
- **Stakeholders**: Maintainers, developers writing tests, CI/CD pipeline maintainers
- **Related PRDs**: N/A
- **Related RFCs**:
  - `docs/rfc/RFC-001-workflow-orchestration.md` (workflow tests)
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (provider tests)
  - `docs/rfc/RFC-019-e2e-test-improvements.md` (E2E test improvements - follow-up work)
  - `docs/rfc/RFC-020-integration-test-improvements.md` (Integration test improvements - follow-up work)
- **Related Issues**: Issue #98

## Abstract

Reorganize the test suite structure to improve maintainability, test discoverability, and enable better test execution strategies. The new structure organizes tests by type (unit, integration, workflow_e2e) and mirrors the source code structure for unit tests, making it easier to locate tests and understand test coverage. This reorganization also enables parallel test execution, flaky test reruns, and stricter CI/CD enforcement of test isolation.

**Key Changes:**

- Organize tests into `unit/`, `integration/`, and `workflow_e2e/` directories
- Mirror `src/podcast_scraper/` structure in `tests/unit/podcast_scraper/`
- Add pytest markers for test categorization and network usage
- Enforce test isolation in CI/CD (unit tests must not hit network or perform filesystem I/O)
- Enable parallel test execution and flaky test reruns

## Problem Statement

**Current Issues:**

1. **Flat Structure**: All 32 test files are in a single `tests/` directory, making it hard to:
   - Understand test types at a glance
   - Run specific test suites (unit vs integration vs e2e)
   - Navigate between source code and corresponding tests

2. **Unclear Test Boundaries**: Tests mix unit, integration, and e2e without clear organization:
   - `test_integration.py` contains full workflow tests (more like e2e)
   - Some unit tests may hit network (violating isolation)
   - No clear distinction between component integration and full workflow tests

3. **Limited Test Execution Control**: Current pytest config doesn't support:
   - Running only fast unit tests by default
   - Excluding network-dependent tests
   - Parallel execution
   - Automatic reruns for flaky tests

4. **CI/CD Limitations**: Cannot easily:
   - Run different test suites at different stages
   - Enforce test isolation (fail if unit test hits network or performs filesystem I/O)
   - Optimize test execution time

**Impact:**

- Slower development feedback (running all tests when only unit tests needed)
- Harder to maintain and extend test suite
- Risk of unit tests hitting network (non-deterministic, slow)
- Difficult to scale test suite as project grows

## Goals

1. **Clear Test Organization**: Tests organized by type (unit/integration/workflow_e2e) and structure
2. **Easy Navigation**: Unit tests mirror source structure for easy code-to-test navigation
3. **Test Execution Control**: Ability to run specific test types, exclude network tests, run in parallel
4. **CI/CD Enforcement**: Automated checks to ensure test isolation (unit tests don't hit network or perform filesystem I/O)
5. **Scalability**: Structure supports growth without becoming unwieldy
6. **Backward Compatibility**: All existing tests continue to work (just reorganized)

## Constraints & Assumptions

**Constraints:**

- Must preserve all existing test functionality
- Must maintain absolute imports (already using `from podcast_scraper import ...`)
- Must work with existing CI/CD pipelines (with updates)
- Must not break test discovery

**Assumptions:**

- All tests use absolute imports from installed package (already true)
- pytest is the primary test framework (some unittest.TestCase usage is acceptable)
- Tests can be categorized into unit/integration/workflow_e2e with reasonable accuracy
- Network tests can be identified and marked appropriately

## Design & Implementation

### 1. New Test Directory Structure

```text
tests/
├── conftest.py                    # Shared fixtures (keep at root)
├── unit/                          # Fast, isolated, fully mocked
│   ├── podcast_scraper/
│   │   ├── __init__.py
│   │   ├── test_config.py         # From test_config_validation.py
│   │   ├── test_downloader.py     # Mocked HTTP, no network
│   │   ├── test_rss_parser.py
│   │   ├── test_filesystem.py
│   │   ├── test_metadata.py
│   │   ├── test_models.py
│   │   ├── test_preprocessing.py
│   │   ├── test_progress.py
│   │   ├── test_prompt_store.py
│   │   ├── test_speaker_detection.py
│   │   ├── test_workflow.py       # Unit tests for workflow (mocked)
│   │   ├── speaker_detectors/
│   │   │   ├── __init__.py
│   │   │   ├── test_base.py
│   │   │   ├── test_factory.py
│   │   │   ├── test_ner_detector.py
│   │   │   └── test_openai_detector.py
│   │   ├── summarization/
│   │   │   ├── __init__.py
│   │   │   ├── test_base.py
│   │   │   ├── test_factory.py
│   │   │   ├── test_local_provider.py
│   │   │   └── test_openai_provider.py
│   │   └── transcription/
│   │       ├── __init__.py
│   │       ├── test_base.py
│   │       ├── test_factory.py
│   │       ├── test_openai_provider.py
│   │       └── test_whisper_provider.py
│   ├── test_api_versioning.py     # Top-level API tests
│   ├── test_package_imports.py    # Package structure tests
│   └── test_utilities.py          # Utility function tests
│
├── integration/                   # Component interactions
│   ├── __init__.py
│   ├── test_provider_integration.py      # Provider system integration
│   ├── test_protocol_compliance.py       # Protocol implementation
│   ├── test_protocol_compliance_extended.py
│   ├── test_fallback_behavior.py        # Fallback chains
│   ├── test_parallel_summarization.py    # Parallel processing
│   ├── test_summarization_provider.py    # Provider integration
│   ├── test_speaker_detector_provider.py
│   ├── test_transcription_provider.py
│   ├── test_provider_error_handling_extended.py
│   └── test_stage0_foundation.py         # Foundation stage tests
│
└── workflow_e2e/                  # Full workflow tests (renamed from integration)
    ├── __init__.py
    ├── test_cli.py                # CLI end-to-end
    ├── test_service.py            # Service mode end-to-end
    ├── test_workflow_e2e.py       # Full workflow (renamed from test_integration.py)
    ├── test_podcast_scraper.py    # Main pipeline E2E
    ├── test_summarizer.py         # Full summarization workflow
    ├── test_summarizer_edge_cases.py
    ├── test_summarizer_security.py
    ├── test_eval_scripts.py      # Evaluation script tests
    └── test_env_variables.py      # Environment variable E2E
```

**Key Decisions:**

- **Unit tests mirror src structure**: Makes navigation easy, clear test-to-code mapping
- **Integration tests by scenario**: Groups related component interactions
- **Workflow_e2e for full workflows**: Renamed from "integration" to be more descriptive
- **Flat structure for top-level tests**: `test_api_versioning.py` doesn't need deep nesting

### 2. Test Type Definitions

**Unit Tests** (`tests/unit/`):

- Test a single module/function/class in isolation
- All external dependencies are mocked (HTTP, filesystem, etc.)
- Fast execution (< 1 second per test typically)
- No network access, no filesystem I/O (use tempfile operations or mocks)
- Deterministic (same input = same output)

**Integration Tests** (`tests/integration/`):

- Test multiple components working together
- May use real implementations (not mocked) for some components
- Test component interactions and interfaces
- May be slower but still isolated from external services
- Can use real filesystem (temp directories)
- Should not hit network unless explicitly marked

**Workflow E2E Tests** (`tests/workflow_e2e/`):

- Test complete workflows from entry point to output
- Test CLI commands, service mode, full pipelines
- May use real implementations and real data (mocked where appropriate)
- Slowest tests (may take seconds to minutes)
- May hit network if explicitly marked
- Test user-facing behavior

### 3. Pytest Configuration

**Updated `pyproject.toml`:**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
# Default: run only unit tests (fast feedback)
addopts = "-q -ra -m 'not integration and not workflow_e2e and not network'"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: integration tests (slower, test component interactions)",
    "workflow_e2e: end-to-end workflow tests (slowest, test full workflows)",
    "network: hits the network (off by default, use -m network to enable)",
]
```

**Test Execution Examples:**

```bash
# Default: unit tests only (fast)
pytest

# All tests except network
pytest -m "not network"

# Only integration tests
pytest -m integration

# Only workflow e2e tests
pytest -m workflow_e2e

# All tests including network
pytest -m network

# Parallel execution (Stage 6)
pytest -n auto

# With reruns for flaky tests (Stage 6)
pytest --reruns 2 --reruns-delay 1
```

### 4. Test Markers and Enforcement

**Marker Usage:**

- All integration tests: `@pytest.mark.integration`
- All workflow_e2e tests: `@pytest.mark.workflow_e2e`
- Tests that hit network: `@pytest.mark.network`
- Slow tests: `@pytest.mark.slow` (existing)

**CI/CD Enforcement:**

- Unit tests must not have `@pytest.mark.network`
- CI will fail if unit test hits network (detected via pytest plugin)
- CI will fail if unit test performs filesystem I/O (detected via pytest plugin)
- Integration and workflow_e2e tests can use network and filesystem I/O if marked

**Implementation:**

- Create pytest plugin to detect network calls in unit tests
- Create pytest plugin to detect filesystem I/O in unit tests
- Fail test run if unit test makes network request or performs filesystem I/O
- Add to CI pipeline as separate check

### 5. Implementation Stages

**Stage 1: Setup Structure**

- Create `tests/unit/`, `tests/integration/`, `tests/workflow_e2e/` directories
- Create `tests/unit/podcast_scraper/` and subdirectories matching src structure
- Add `__init__.py` files where needed
- Update `pyproject.toml` pytest config

**Stage 2: Categorize Tests**

- Review each of 32 test files to determine type
- Create mapping: `test_file.py` → `unit/integration/workflow_e2e`
- Document categorization decisions
- Identify tests that hit network

**Stage 3: Move Tests**

- Move unit tests to `tests/unit/podcast_scraper/` (mirroring src structure)
- Move integration tests to `tests/integration/`
- Move workflow_e2e tests to `tests/workflow_e2e/`
- Rename `test_integration.py` → `test_workflow_e2e.py`
- Verify all imports still work (should - already using absolute imports)

**Stage 4: Update Markers**

- Add `@pytest.mark.integration` to all integration tests
- Add `@pytest.mark.workflow_e2e` to all workflow_e2e tests
- Add `@pytest.mark.network` to tests that hit network
- Update existing `@pytest.mark.slow` markers if needed

**Stage 5: CI/CD Enforcement**

- Create pytest plugin to detect network calls in unit tests
- Create pytest plugin to detect filesystem I/O in unit tests
- Add CI check that fails if unit test hits network or performs filesystem I/O
- Update GitHub Actions workflows to run appropriate test suites
- Add separate jobs for unit/integration/workflow_e2e if beneficial

**Stage 6: Parallel Execution & Reruns**

- Add `pytest-xdist` dependency for parallel execution
- Add `pytest-rerunfailures` dependency for flaky test reruns
- Update pytest config to support `-n auto` for parallel execution
- Update CI to use parallel execution for faster feedback
- Configure reruns (2 retries with 1 second delay) for flaky tests

**Stage 7: Documentation Updates**

- Update `docs/TESTING_STRATEGY.md` with new test structure and organization
- Update `CONTRIBUTING.md` with new test running examples and structure
- Update `docs/DEVELOPMENT_GUIDE.md` if it references test structure
- Update `README.md` if it has test-related sections
- Document test type definitions and decision tree
- Add examples of running different test suites

**Stage 8: Makefile & CI/CD Updates**

- Update `Makefile` test target to work with new structure
- Update GitHub Actions workflows (`.github/workflows/python-app.yml`) to:
  - Use new test paths (`tests/unit/`, `tests/integration/`, `tests/workflow_e2e/`)
  - Run unit tests by default (fast feedback)
  - Run integration and workflow_e2e tests in separate jobs if beneficial
  - Add network and filesystem I/O detection enforcement for unit tests
  - Enable parallel execution with `pytest-xdist`
  - Configure reruns for flaky tests
- Update workflow paths triggers if needed
- Verify all CI checks pass with new structure

### 6. Test Categorization Guide

**Unit Tests** (move to `tests/unit/podcast_scraper/`):

- `test_config_validation.py` → `test_config.py`
- `test_downloader.py` (with mocked HTTP)
- `test_rss_parser.py`
- `test_filesystem.py`
- `test_metadata.py`
- `test_prompt_store.py`
- `test_speaker_detection.py`
- `test_api_versioning.py` → `test_api_versioning.py` (top-level)
- `test_package_imports.py` → `test_package_imports.py` (top-level)
- `test_utilities.py` → `test_utilities.py` (top-level)
- Provider tests (factory, base, individual providers) → subdirectories

**Integration Tests** (move to `tests/integration/`):

- `test_provider_integration.py`
- `test_protocol_compliance.py`
- `test_protocol_compliance_extended.py`
- `test_fallback_behavior.py`
- `test_parallel_summarization.py`
- `test_summarization_provider.py`
- `test_speaker_detector_provider.py`
- `test_transcription_provider.py`
- `test_provider_error_handling_extended.py`
- `test_stage0_foundation.py`

**Workflow E2E Tests** (move to `tests/workflow_e2e/`):

- `test_cli.py`
- `test_service.py`
- `test_integration.py` → `test_workflow_e2e.py` (rename)
- `test_podcast_scraper.py`
- `test_summarizer.py`
- `test_summarizer_edge_cases.py`
- `test_summarizer_security.py`
- `test_eval_scripts.py`
- `test_env_variables.py`

### 7. Network and Filesystem I/O Detection Plugin

**Purpose:** Enforce that unit tests never hit the network or perform filesystem I/O.

**Network Detection:**

- Block `requests.get()`, `requests.post()`, `requests.Session()` methods
- Block `urllib.request.urlopen()`
- Block `urllib3.PoolManager()`
- Block `socket.create_connection()`
- Raise `NetworkCallDetectedError` if network call detected

**Filesystem I/O Detection:**

- Block `open()` for file operations (outside temp directories)
- Block `os.makedirs()`, `os.remove()`, `os.unlink()`, `os.rmdir()`, etc.
- Block `shutil.copy()`, `shutil.move()`, `shutil.rmtree()`, etc.
- Block `Path.write_text()`, `Path.write_bytes()`, `Path.mkdir()`, `Path.unlink()`, etc.
- Raise `FilesystemIODetectedError` if filesystem I/O detected

**Exceptions (allowed in unit tests):**

- `tempfile.mkdtemp()`, `tempfile.NamedTemporaryFile()` (designed for testing)
- Operations within temp directories (detected automatically)
- Cache directories (`~/.cache/`, `~/.local/share/`) for model loading
- Site-packages (read-only access to installed packages)
- Python cache files (`.pyc`, `__pycache__/`) created during imports
- `test_filesystem.py` tests (they need to test filesystem operations)

**Implementation:**

```python
# tests/unit/conftest.py
import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def block_network_and_filesystem_io(request):
    """Automatically block network calls and filesystem I/O in unit tests."""
    # Patch network libraries
    # Patch filesystem operations
    # Allow exceptions (tempfile, cache dirs, etc.)
    yield
    # Clean up patches
```

### 8. Documentation Updates

**Files to Update:**

- `docs/TESTING_STRATEGY.md`: Update with new test structure, organization, and test type definitions
- `CONTRIBUTING.md`: Update test running examples, add section on new test structure
- `docs/DEVELOPMENT_GUIDE.md`: Update any test-related references
- `README.md`: Update test running examples if present

**New Documentation Sections:**

- Test type decision tree (when to use unit vs integration vs workflow_e2e)
- Examples of each test type with code samples
- How to add new tests in correct location
- Test execution patterns (running specific test suites)
- Network test guidelines (when and how to mark tests)
- Filesystem I/O isolation guidelines (when to use tempfile vs move to integration)

## Testing

**Verification Steps:**

1. **Test Discovery**: Verify pytest finds all tests after reorganization

   ```bash
   pytest --collect-only
   ```

2. **Test Execution**: Run each test suite independently

   ```bash
   pytest tests/unit/
   pytest tests/integration/
   pytest tests/workflow_e2e/
   ```

3. **Marker Verification**: Verify markers work correctly

   ```bash
   pytest -m unit
   pytest -m integration
   pytest -m workflow_e2e
   pytest -m network
   ```

4. **Network and Filesystem I/O Enforcement**: Verify unit tests fail if they hit network or perform filesystem I/O

   ```bash
   # Should fail if unit test makes network call or performs filesystem I/O
   pytest tests/unit/
   # Network calls will raise NetworkCallDetectedError
   # Filesystem I/O will raise FilesystemIODetectedError
   ```

5. **Parallel Execution**: Verify parallel execution works (Stage 6)

   ```bash
   pytest -n auto
   ```

6. **Reruns**: Verify flaky test reruns work (Stage 6)

   ```bash
   pytest --reruns 2 --reruns-delay 1
   ```

7. **CI/CD**: Verify all CI checks pass with new structure

## Migration Strategy

**Approach:** Direct migration (no POC needed per requirements)

1. **Create structure** (Stage 1)
2. **Categorize and move tests** (Stages 2-3)
3. **Add markers** (Stage 4)
4. **Update CI/CD** (Stage 5)
5. **Add parallel/reruns** (Stage 6)
6. **Update documentation** (Stage 7)
7. **Update Makefile & CI/CD** (Stage 8)

**Risk Mitigation:**

- Use `git mv` to preserve file history
- Verify imports after each move (should work - already absolute)
- Run full test suite after each stage
- Keep old structure in git history for reference
- Test CI/CD changes in a branch before merging

## Dependencies

**New Dependencies:**

- `pytest-xdist` (Stage 6): Parallel test execution
- `pytest-rerunfailures` (Stage 6): Flaky test reruns

**Optional:**

- Custom pytest plugin for network and filesystem I/O detection (Stage 5)

## Makefile Updates

**Current `test` target:**

```makefile
test:
    pytest --cov=$(PACKAGE) --cov-report=term-missing
```

**Updated `test` target (after reorganization):**

```makefile
test:
    pytest --cov=$(PACKAGE) --cov-report=term-missing

test-unit:
    pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing

test-integration:
    pytest tests/integration/ -m integration

test-workflow-e2e:
    pytest tests/workflow_e2e/ -m workflow_e2e

test-all:
    pytest tests/ -m "not network" --cov=$(PACKAGE) --cov-report=term-missing

test-parallel:
    pytest -n auto --cov=$(PACKAGE) --cov-report=term-missing
```

**Note:** Default `test` target remains unchanged for backward compatibility. New targets added for specific test suites.

## GitHub Actions Workflow Updates

**Current workflow** (`.github/workflows/python-app.yml`):

- Runs `make test` which executes all tests
- Single test job

**Updated workflow** (after reorganization):

```yaml
jobs:
  # ... existing lint job ...

  # Unit tests - fast, run on every PR
  test-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
          cache-dependency-path: pyproject.toml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,ml]
      - name: Run unit tests
        run: |
          export PYTHONPATH="${PYTHONPATH}:$(pwd)"
          pytest tests/unit/ -n auto --cov=podcast_scraper --cov-report=term-missing
      - name: Verify no network calls or filesystem I/O in unit tests
        run: |
          # Plugin automatically detects network calls and filesystem I/O
          pytest tests/unit/ || exit 1

  # Integration tests - run on main branch and PRs
  test-integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
          cache-dependency-path: pyproject.toml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,ml]
      - name: Run integration tests
        run: |
          export PYTHONPATH="${PYTHONPATH}:$(pwd)"
          pytest tests/integration/ -m integration -n auto --reruns 2 --reruns-delay 1

  # Workflow E2E tests - run on main branch
  test-workflow-e2e:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
          cache-dependency-path: pyproject.toml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,ml]
      - name: Run workflow E2E tests
        run: |
          export PYTHONPATH="${PYTHONPATH}:$(pwd)"
          pytest tests/workflow_e2e/ -m workflow_e2e --reruns 2 --reruns-delay 1

  # Full test suite (backward compatibility)
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
          cache-dependency-path: pyproject.toml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,ml]
      - name: Run all tests (except network)
        run: |
          export PYTHONPATH="${PYTHONPATH}:$(pwd)"
          pytest tests/ -m "not network" -n auto --cov=podcast_scraper --cov-report=term-missing --reruns 2 --reruns-delay 1
```

**Workflow Path Triggers:**

Update path triggers to include new test directories:

```yaml
paths:
  - '**.py'
  - 'tests/**'  # Already covers all test subdirectories
  - 'pyproject.toml'
  - 'Makefile'
```

**Note:** The existing `tests/**` pattern already covers all subdirectories, so no changes needed to path triggers.

## Alternatives Considered

1. **Keep flat structure**: Rejected - doesn't solve navigation or execution control
2. **Only organize by type (no mirroring)**: Rejected - loses code-to-test navigation benefit
3. **Gradual migration with POC**: Rejected per requirements - direct migration preferred
4. **Different naming**: Considered "e2e" vs "workflow_e2e" - chose "workflow_e2e" for clarity

## Success Criteria

1. ✅ All tests organized into unit/integration/workflow_e2e
2. ✅ Unit tests mirror src structure
3. ✅ Pytest markers correctly applied
4. ✅ Default test run executes only unit tests (fast)
5. ✅ CI enforces unit tests don't hit network or perform filesystem I/O
6. ✅ Parallel execution works (Stage 6)
7. ✅ Flaky test reruns work (Stage 6)
8. ✅ All existing tests pass after reorganization
9. ✅ Documentation updated

## Open Questions

1. Should we add `__init__.py` files to all test subdirectories? (Yes - for proper Python packages)
2. Should we enforce test naming conventions? (e.g., `test_<module>.py` in unit tests)
3. Should we add a script to help categorize tests? (Nice to have, not required)

## Relationship to Other Test RFCs

This RFC (RFC-018) establishes the **foundation** for the test suite structure. It was followed by two comprehensive improvement efforts:

1. **RFC-020: Integration Test Improvements** - Built upon RFC-018's structure to add comprehensive integration test coverage (182 tests across 15 files) with real component workflows, real ML models, real HTTP client testing, error handling, concurrent execution, and OpenAI provider integration.

2. **RFC-019: E2E Test Improvements** - Plans comprehensive E2E test infrastructure improvements, including local HTTP server, real data files, and complete coverage of all major user-facing entry points.

Together, these three RFCs form a complete testing strategy:

- **RFC-018**: Establishes test structure and boundaries ✅ **Completed**
- **RFC-020**: Comprehensive integration test improvements ✅ **Completed**
- **RFC-019**: Comprehensive E2E test improvements 📋 **Planned**

## References

- Issue #98: Test structure reorganization
- Issue #94: Move Python files to src/ (completed - enables this RFC)
- [pytest-xdist documentation](https://pytest-xdist.readthedocs.io/)
- [pytest-rerunfailures documentation](https://pytest-rerunfailures.readthedocs.io/)
- Follow-up work: `docs/rfc/RFC-019-e2e-test-improvements.md`, `docs/rfc/RFC-020-integration-test-improvements.md`
