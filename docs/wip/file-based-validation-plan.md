# File-Based Validation & Impacted Test Discovery - Implementation Plan

**Status**: Planning
**Created**: 2026-01-30
**Goal**: Speed up development by running only impacted tests when validating changes to specific files

## Problem Statement

When fixing a few files to stabilize a failing PR, running the entire `make ci` suite is time-consuming (~6-10 minutes for `make ci-fast`, even longer for full suite). Developers need a way to:

1. **Run linting/formatting** on only changed files
2. **Discover and run only impacted tests** (unit/integration/e2e) that test those specific files
3. **Keep test subset tiny** - main goal is speed, not comprehensive coverage

**Classic scenario**: Fixing a few files to stabilize a failing PR - need fast validation without running entire test suite.

## Solution Approach

### Core Strategy: Module-Based Test Markers

Use **pytest markers** to tag tests with the modules they test. This allows:
- Fast discovery: `pytest -m "config"` runs all tests for config module
- Precise targeting: Only run tests that directly test changed modules
- Works across test types: Unit, integration, and E2E tests can all be tagged

### Key Constraints

1. **Must run tiny subset** - Primary goal is speed, not comprehensive coverage
2. **Use test markers** - Leverage pytest's marker system for discovery
3. **Fast execution** - Target: < 1 minute for typical changes (vs 6-10 min for ci-fast)

## Module Mapping

### Source Modules → Markers

Based on `src/podcast_scraper/` structure, define these module markers:

| Source Module | Marker | Description |
|--------------|--------|-------------|
| `config.py`, `config_constants.py` | `module_config` | Configuration management |
| `cli.py` | `module_cli` | CLI interface |
| `service.py` | `module_service` | Service API |
| `workflow/*.py` | `module_workflow` | Workflow orchestration |
| `rss/downloader.py` | `module_downloader` | HTTP downloading |
| `rss/parser.py` | `module_rss_parser` | RSS parsing |
| `providers/ml/*.py` | `module_ml_providers` | ML providers (Whisper, BART, etc.) |
| `providers/openai/*.py` | `module_openai_providers` | OpenAI providers |
| `summarization/*.py` | `module_summarization` | Summarization logic |
| `transcription/*.py` | `module_transcription` | Transcription logic |
| `speaker_detectors/*.py` | `module_speaker_detection` | Speaker detection |
| `preprocessing/*.py` | `module_preprocessing` | Text/audio preprocessing |
| `cache/*.py` | `module_cache` | Cache management |
| `evaluation/*.py` | `module_evaluation` | Evaluation/scoring |
| `utils/*.py` | `module_utils` | Utility functions |
| `models.py` | `module_models` | Data models |
| `exceptions.py` | `module_exceptions` | Exception classes |

### Marker Naming Convention

- Format: `module_<name>` (e.g., `module_config`, `module_workflow`)
- Lowercase, underscores
- Matches module name from source structure

## Implementation Plan

### Phase 1: Add Module Markers to pyproject.toml

**File**: `pyproject.toml`

Add new markers to `[tool.pytest.ini_options.markers]`:

```toml
markers = [
    # ... existing markers ...
    "module_config: tests for config.py and config_constants.py",
    "module_cli: tests for cli.py",
    "module_service: tests for service.py",
    "module_workflow: tests for workflow/*.py",
    "module_downloader: tests for rss/downloader.py",
    "module_rss_parser: tests for rss/parser.py",
    "module_ml_providers: tests for providers/ml/*.py",
    "module_openai_providers: tests for providers/openai/*.py",
    "module_summarization: tests for summarization/*.py",
    "module_transcription: tests for transcription/*.py",
    "module_speaker_detection: tests for speaker_detectors/*.py",
    "module_preprocessing: tests for preprocessing/*.py",
    "module_cache: tests for cache/*.py",
    "module_evaluation: tests for evaluation/*.py",
    "module_utils: tests for utils/*.py",
    "module_models: tests for models.py",
    "module_exceptions: tests for exceptions.py",
]
```

**Effort**: 15 minutes

### Phase 2: Tag Existing Tests with Module Markers

**Strategy**: Add module markers to test files based on:
1. **Unit tests**: Direct mapping - `test_config.py` → `@pytest.mark.module_config`
2. **Integration tests**: Based on what they test (may have multiple markers)
3. **E2E tests**: Based on workflow components they exercise

**Approach**:
- Start with unit tests (clear 1:1 mapping)
- Then integration tests (review each file)
- Finally E2E tests (tag based on components)

**Example**:

```python
# tests/unit/podcast_scraper/test_config.py
import pytest

@pytest.mark.unit
@pytest.mark.module_config  # NEW
def test_config_validation():
    ...

# tests/integration/test_workflow_integration.py
@pytest.mark.integration
@pytest.mark.module_workflow  # NEW
@pytest.mark.module_downloader  # NEW (if it tests downloader too)
def test_workflow_with_download():
    ...

# tests/e2e/test_full_pipeline_e2e.py
@pytest.mark.e2e
@pytest.mark.module_workflow  # NEW
@pytest.mark.critical_path
def test_full_pipeline():
    ...
```

**Effort**: 2-3 hours (review and tag ~100 test files)

**Automation**: Could create a script to auto-tag unit tests based on naming convention, but manual review recommended for accuracy.

### Phase 3: Create Test Discovery Script

**File**: `scripts/tools/find_impacted_tests.py`

**Purpose**: Given a list of source files, discover which tests to run.

**Algorithm**:

1. **Parse source files** → Extract module names
   - `src/podcast_scraper/config.py` → `module_config`
   - `src/podcast_scraper/workflow/orchestration.py` → `module_workflow`
   - `src/podcast_scraper/rss/downloader.py` → `module_downloader`

2. **Build marker expression** for pytest
   - Single module: `-m "module_config"`
   - Multiple modules: `-m "module_config or module_workflow"`

3. **Filter by test type** (optional)
   - Only unit tests: `-m "module_config and unit"`
   - Only critical path: `-m "module_config and critical_path"`

4. **Output**: pytest marker expression + list of test files

**Usage**:

```bash
python scripts/tools/find_impacted_tests.py \
    --files src/podcast_scraper/config.py src/podcast_scraper/workflow/orchestration.py \
    --test-type unit  # optional: unit, integration, e2e, or all
```

**Output**:

```
Discovered markers: module_config, module_workflow
Pytest expression: -m "module_config or module_workflow"
Test files:
  - tests/unit/podcast_scraper/test_config.py
  - tests/unit/podcast_scraper/test_workflow_*.py
  - tests/integration/test_workflow_integration.py
```

**Effort**: 3-4 hours

### Phase 4: Create Makefile Target

**File**: `Makefile`

**Target**: `validate-files`

**Behavior**:

1. **Lint/format** changed files (black, isort, flake8, mypy)
2. **Discover impacted tests** using script
3. **Run only those tests** with pytest

**Implementation**:

```makefile
validate-files:
	@# Usage: make validate-files FILES="src/podcast_scraper/config.py src/podcast_scraper/workflow/orchestration.py"
	@# Optional: TEST_TYPE=unit (default: all), FAST_ONLY=1 (only critical_path tests)
	@if [ -z "$(FILES)" ]; then \
		echo "❌ Error: FILES required"; \
		echo "Usage: make validate-files FILES='file1.py file2.py' [TEST_TYPE=unit] [FAST_ONLY=1]"; \
		exit 1; \
	fi
	@echo "🔍 Step 1: Linting and formatting changed files..."
	@for file in $(FILES); do \
		if [ ! -f "$$file" ]; then \
			echo "⚠️  Warning: File not found: $$file"; \
			continue; \
		fi; \
		echo "  Formatting: $$file"; \
		$(PYTHON) -m black $$file; \
		$(PYTHON) -m isort $$file; \
		echo "  Linting: $$file"; \
		$(PYTHON) -m flake8 --config .flake8 $$file || true; \
		$(PYTHON) -m mypy $$file || true; \
	done
	@echo ""
	@echo "🔍 Step 2: Discovering impacted tests..."
	@TEST_TYPE=$${TEST_TYPE:-all}; \
	FAST_ONLY=$${FAST_ONLY:-0}; \
	$(PYTHON) scripts/tools/find_impacted_tests.py \
		--files $(FILES) \
		--test-type $$TEST_TYPE \
		$(if $(filter 1,$(FAST_ONLY)),--fast-only,) \
		--output-format makefile > /tmp/impacted_tests.mk || exit 1
	@echo ""
	@echo "🧪 Step 3: Running impacted tests..."
	@# Source the generated makefile to get TEST_MARKERS variable
	@. /tmp/impacted_tests.mk && \
	if [ -z "$$TEST_MARKERS" ]; then \
		echo "⚠️  No tests found for changed files"; \
		exit 0; \
	fi; \
	echo "  Running tests with markers: $$TEST_MARKERS"; \
	$(PYTHON) -m pytest -m "$$TEST_MARKERS" \
		-n $(PYTEST_WORKERS) \
		--disable-socket --allow-hosts=127.0.0.1,localhost \
		-v \
		--tb=short
	@echo ""
	@echo "✅ Validation complete!"

validate-files-fast:
	@# Fast mode: only critical_path tests
	@$(MAKE) validate-files FILES="$(FILES)" TEST_TYPE=all FAST_ONLY=1

validate-files-unit:
	@# Unit tests only (fastest)
	@$(MAKE) validate-files FILES="$(FILES)" TEST_TYPE=unit
```

**Effort**: 1-2 hours

### Phase 5: Optimize for Speed (Keep Subset Tiny)

**Strategies to minimize test count**:

1. **Default to unit tests only** (fastest)
   - `validate-files` → unit tests by default
   - `validate-files-all` → all test types

2. **Fast-only mode** (`FAST_ONLY=1`)
   - Only run tests marked `critical_path`
   - Skips slow integration/E2E tests

3. **Limit test types** (`TEST_TYPE=unit`)
   - Unit tests are fastest (< 1 second each)
   - Integration/E2E are slower (seconds to minutes)

4. **Early exit on lint failures**
   - Don't run tests if linting fails
   - Saves time on syntax errors

5. **Parallel execution**
   - Use existing `PYTEST_WORKERS` calculation
   - Already configured in Makefile

**Target Performance**:
- **Unit tests only**: < 30 seconds for typical changes
- **With integration**: < 2 minutes
- **Full suite**: < 5 minutes (still faster than `make ci-fast`)

### Phase 6: Documentation & Examples

**File**: `docs/guides/DEVELOPMENT_GUIDE.md` (update)

Add section on file-based validation:

```markdown
## Fast Validation for Changed Files

When fixing a few files, use `make validate-files` to run only impacted tests:

```bash
# Validate specific files (runs unit tests only)
make validate-files FILES="src/podcast_scraper/config.py src/podcast_scraper/workflow/orchestration.py"

# Include integration/E2E tests
make validate-files FILES="..." TEST_TYPE=all

# Fast mode (critical_path only)
make validate-files-fast FILES="..."

# Unit tests only (fastest)
make validate-files-unit FILES="..."
```

This runs:
1. Linting/formatting on changed files
2. Discovery of impacted tests (via module markers)
3. Execution of only those tests

**Performance**: Typically < 1 minute for unit tests, < 5 minutes for full suite.
```

**Effort**: 30 minutes

## Implementation Checklist

### Phase 1: Foundation
- [ ] Add module markers to `pyproject.toml`
- [ ] Verify markers are registered: `pytest --markers | grep module_`

### Phase 2: Tag Tests
- [ ] Tag all unit tests with module markers (automated script + manual review)
- [ ] Tag integration tests with module markers (manual review)
- [ ] Tag E2E tests with module markers (manual review)
- [ ] Verify: `pytest -m "module_config"` runs expected tests

### Phase 3: Discovery Script
- [ ] Create `scripts/tools/find_impacted_tests.py`
- [ ] Implement file → module mapping
- [ ] Implement marker expression generation
- [ ] Add `--test-type` filtering
- [ ] Add `--fast-only` filtering
- [ ] Test with various file combinations

### Phase 4: Makefile Integration
- [ ] Create `validate-files` target
- [ ] Create `validate-files-fast` target
- [ ] Create `validate-files-unit` target
- [ ] Test with real file changes
- [ ] Verify performance targets

### Phase 5: Optimization
- [ ] Measure performance (unit tests only)
- [ ] Measure performance (with integration)
- [ ] Optimize if needed
- [ ] Add early exit on lint failures

### Phase 6: Documentation
- [ ] Update `docs/guides/DEVELOPMENT_GUIDE.md`
- [ ] Add examples to `README.md` (optional)
- [ ] Update `.cursor/rules/testing-strategy.mdc` (optional)

## Testing the Solution

### Test Cases

1. **Single file change**:
   ```bash
   make validate-files FILES="src/podcast_scraper/config.py"
   ```
   - Should run: `test_config.py` (unit)
   - Should NOT run: unrelated tests

2. **Multiple files, same module**:
   ```bash
   make validate-files FILES="src/podcast_scraper/config.py src/podcast_scraper/config_constants.py"
   ```
   - Should run: `test_config.py` (unit)
   - Should deduplicate markers

3. **Multiple files, different modules**:
   ```bash
   make validate-files FILES="src/podcast_scraper/config.py src/podcast_scraper/workflow/orchestration.py"
   ```
   - Should run: `test_config.py` + `test_workflow_*.py` (unit)
   - Should use OR expression: `-m "module_config or module_workflow"`

4. **Integration/E2E tests**:
   ```bash
   make validate-files FILES="src/podcast_scraper/workflow/orchestration.py" TEST_TYPE=all
   ```
   - Should run: unit + integration + e2e tests for workflow

5. **Fast mode**:
   ```bash
   make validate-files-fast FILES="src/podcast_scraper/config.py"
   ```
   - Should run: only `critical_path` tests

## Edge Cases & Considerations

### 1. Tests with Multiple Module Markers

Some tests (especially integration/E2E) may test multiple modules. Solution:
- Tag with all relevant markers: `@pytest.mark.module_config @pytest.mark.module_workflow`
- Use OR expression: `-m "module_config or module_workflow"` will match both

### 2. Missing Markers

If a test doesn't have a module marker, it won't be discovered. Solution:
- Phase 2 ensures all tests are tagged
- Script can warn about untagged tests (optional)

### 3. Indirect Dependencies

Tests may test module A but depend on module B. Solution:
- **Conservative approach**: Tag with all modules it imports/uses
- **Fast approach**: Only tag with primary module being tested
- **Recommendation**: Start conservative, optimize later

### 4. False Positives/Negatives

- **False positives**: Test tagged with module but doesn't actually test it
  - **Impact**: Runs extra tests (slower, but safe)
  - **Mitigation**: Careful tagging in Phase 2

- **False negatives**: Test should run but isn't tagged
  - **Impact**: Misses test (risky)
  - **Mitigation**: Comprehensive tagging in Phase 2

### 5. Performance

If discovered test set is still too large:
- Default to `TEST_TYPE=unit` (fastest)
- Use `FAST_ONLY=1` (critical_path only)
- Consider limiting to specific test files (future enhancement)

## Future Enhancements

1. **Git integration**: Auto-detect changed files from git diff
   ```bash
   make validate-changed  # Uses git diff to find changed files
   ```

2. **Coverage-based discovery**: Use coverage data to find tests (more accurate but slower)

3. **Test file mapping**: Maintain explicit mapping file for edge cases

4. **CI integration**: Use in CI to run only impacted tests on PRs

5. **Caching**: Cache test discovery results for unchanged files

## Success Criteria

✅ **Speed**: Unit tests complete in < 1 minute for typical changes
✅ **Accuracy**: All tests that should run are discovered
✅ **Usability**: Simple command: `make validate-files FILES="..."`
✅ **Safety**: No false negatives (all relevant tests run)
✅ **Flexibility**: Supports unit/integration/e2e filtering

## Estimated Total Effort

- **Phase 1**: 15 minutes
- **Phase 2**: 2-3 hours (can be done incrementally)
- **Phase 3**: 3-4 hours
- **Phase 4**: 1-2 hours
- **Phase 5**: 1 hour (testing/optimization)
- **Phase 6**: 30 minutes

**Total**: ~8-11 hours (can be done incrementally over multiple sessions)

## Notes

- This solution prioritizes **speed** over comprehensive coverage
- For full validation before PR, still use `make ci-fast` or `make ci`
- This is a **development tool**, not a replacement for CI
- Markers can be added incrementally - start with unit tests, expand later
