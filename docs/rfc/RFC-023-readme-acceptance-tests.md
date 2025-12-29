# RFC-023: README Acceptance Tests

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, developers, CI/CD pipeline maintainers, first-time users
- **Related PRDs**:
  - `docs/prd/PRD-001-transcript-pipeline.md` (core pipeline)
  - `docs/prd/PRD-002-whisper-fallback.md` (Whisper transcription)
  - `docs/prd/PRD-003-user-interface-config.md` (CLI and config)
  - `docs/prd/PRD-004-metadata-generation.md` (metadata)
  - `docs/prd/PRD-005-episode-summarization.md` (summarization)
- **Related RFCs**:
  - `rfc/RFC-019-e2e-test-improvements.md` (E2E test infrastructure - foundation)
  - `rfc/RFC-018-test-structure-reorganization.md` (test structure - foundation)
  - `rfc/RFC-020-integration-test-improvements.md` (integration test improvements)
  - `rfc/RFC-007-cli-interface.md` (CLI interface)
- **Related Documents**:
  - [README.md](https://github.com/chipi/podcast_scraper/blob/main/README.md) - Main project README (source of truth for examples)
  - [TESTING_STRATEGY.md](../TESTING_STRATEGY.md) - Overall testing strategy and test categories

## Abstract

This RFC defines a new category of **Acceptance Tests** that verify all examples, commands, and workflows documented in the project README actually work as described. These tests serve as a final validation gate before releases, ensuring that what users read in the README is accurate and functional. Acceptance tests are distinct from E2E tests: they test **documentation accuracy** rather than **feature completeness**, and they run as the **final CI step** after all other tests pass.

**Key Characteristics:**

- **Documentation-Driven**: Tests are derived directly from README examples and claims
- **Comprehensive Coverage**: Every CLI example, installation option, and key feature claim is tested
- **Final Validation Gate**: Runs as last CI step, only after all other tests pass
- **Slow but Thorough**: May take 10-20 minutes, but provides confidence that README is accurate
- **User Journey Focus**: Tests the exact commands a new user would run following the README
- **Separate Category**: New test category `acceptance` (not `e2e` or `integration`)

## Problem Statement

**Current Issues:**

1. **README Examples May Break**
   - CLI examples in README are not automatically tested
   - When code changes, README examples may become outdated or broken
   - No automated way to detect when README examples stop working
   - Risk of first-time users encountering broken examples

2. **Installation Instructions Not Verified**
   - Installation commands (`pip install -e ".[ml]"`, `make init`) are not tested
   - No verification that installation instructions actually work
   - Risk of installation failures for new users

3. **Key Features Claims Not Validated**
   - README lists key features, but no tests verify these claims
   - Features may be broken without README being updated
   - Risk of misleading users about project capabilities

4. **No Documentation Regression Testing**
   - Changes to code may break documented workflows
   - No automated detection of documentation drift
   - Manual verification is error-prone and time-consuming

5. **Unclear Test Boundaries**
   - E2E tests focus on feature completeness, not documentation accuracy
   - No clear distinction between "does the feature work?" vs "does the README example work?"
   - Risk of test duplication or gaps

**Impact:**

- First-time users may encounter broken examples in README
- Installation instructions may fail silently
- Key features may be claimed but not actually work
- Documentation may drift from actual behavior
- Reduced confidence in project reliability

## Goals

1. **Documentation Accuracy**: Every README example is automatically tested
2. **Installation Verification**: All installation commands are verified to work
3. **Feature Claims Validation**: All key features mentioned in README are tested
4. **User Journey Testing**: Tests follow the exact path a new user would take
5. **Final Validation Gate**: Acceptance tests run as last CI step, only after all other tests pass
6. **Clear Test Category**: New `acceptance` test category, distinct from `e2e`
7. **Comprehensive Coverage**: All README sections with executable examples are tested
8. **CI/CD Integration**: Acceptance tests run in CI with proper markers and timeouts

## Constraints & Assumptions

**Constraints:**

- Acceptance tests must **not** hit external networks (use E2E server fixture)
- Acceptance tests must use **real implementations** (no mocking of core functionality)
- Acceptance tests may be **slow** (10-20 minutes acceptable for final validation)
- Acceptance tests must test **exact README examples** (copy-pasteable commands)
- Test fixtures must be **realistic** (use existing E2E test fixtures)

**Assumptions:**

- README is the **source of truth** for user-facing examples
- E2E server fixture is sufficient for acceptance testing (no external network needed)
- Slow execution is acceptable for final validation gate
- Installation testing can use isolated virtual environments
- All acceptance tests can use existing E2E test infrastructure

## Design & Implementation

### Test Category: `acceptance`

**New Test Category:**

- **Location**: `tests/acceptance/`
- **Marker**: `@pytest.mark.acceptance`
- **Purpose**: Verify README examples and documentation accuracy
- **Distinction from E2E**: E2E tests verify feature completeness; acceptance tests verify documentation accuracy

**Test Structure:**

````text
├── __init__.py
├── conftest.py              # Shared fixtures (reuse E2E server)
├── test_readme_installation.py    # Installation examples
├── test_readme_basic_usage.py     # Basic usage examples
├── test_readme_key_features.py     # Key features validation
└── README.md                # Acceptance test documentation
```text

**Tests to Implement:**

1. **`test_install_core_only`**
   - Command: `pip install -e .`
   - Verify: Package imports successfully
   - Verify: Core functionality works (no ML deps)

2. **`test_install_with_ml`**
   - Command: `pip install -e ".[ml]"`
   - Verify: Package imports successfully
   - Verify: ML dependencies are available (Whisper, spaCy, transformers)
   - Verify: ML functionality works

3. **`test_install_with_dev_ml`**
   - Command: `pip install -e ".[dev,ml]"`
   - Verify: Package imports successfully
   - Verify: Dev tools are available (pytest, black, mypy)
   - Verify: ML dependencies are available

4. **`test_make_init`** (if make is available)
   - Command: `make init`
   - Verify: Package installs correctly
   - Verify: Dev and ML dependencies are available

**Implementation Notes:**

- Use isolated virtual environments for each test
- Clean up virtual environments after tests
- Skip tests if system dependencies are missing (e.g., `make`)
- Use `subprocess` to run installation commands
- Verify imports and basic functionality after installation

**Deliverables:**

- `tests/acceptance/test_readme_installation.py`
- Installation test fixtures in `conftest.py`
- Documentation of installation test approach

### Stage 2: Basic Usage Examples Testing

**Goal**: Verify all CLI examples from README work correctly.

**README Section**: Lines 60-75 (Basic Usage)

**Tests to Implement:**

1. **`test_basic_transcript_download`**
   - Command: `python3 -m podcast_scraper.cli <rss_url>`
   - Verify: Command exits with code 0
   - Verify: Transcript file is created
   - Verify: Transcript content is valid

2. **`test_whisper_fallback`**
   - Command: `python3 -m podcast_scraper.cli <rss_url> --transcribe-missing --whisper-model base`
   - Verify: Command exits with code 0
   - Verify: Transcript is generated using Whisper
   - Verify: Transcript content is valid

3. **`test_metadata_and_summaries`**
   - Command: `python3 -m podcast_scraper.cli <rss_url> --generate-metadata --generate-summaries`
   - Verify: Command exits with code 0
   - Verify: Metadata file is created
   - Verify: Summary is generated
   - Verify: Output files are valid

**Implementation Notes:**

- Use E2E server fixture for RSS feeds
- Use existing E2E test fixtures (RSS, transcripts, audio)
- Test exact commands from README (copy-pasteable)
- Verify output files and content
- Use temporary directories for output

**Deliverables:**

- `tests/acceptance/test_readme_basic_usage.py`
- Integration with E2E server fixture
- Documentation of basic usage test approach

### Stage 3: Key Features Validation

**Goal**: Verify all key features mentioned in README actually work.

**README Section**: Lines 10-21 (Key Features)

**Features to Test:**

1. **Transcript Downloads**
   - Test: Automatic detection and download of transcripts
   - Verify: Transcripts are downloaded from RSS feeds
   - Verify: Transcript files are created correctly

2. **Whisper Fallback**
   - Test: Generate transcripts using Whisper when none exist
   - Verify: Whisper transcription works
   - Verify: Transcripts are generated correctly

3. **Speaker Detection**
   - Test: Automatic speaker name detection using NER
   - Verify: Speaker detection works
   - Verify: Speaker names are detected correctly

4. **Screenplay Formatting**
   - Test: Format Whisper transcripts as dialogue
   - Verify: Screenplay format is applied
   - Verify: Speaker labels are correct

5. **Episode Summarization**
   - Test: Generate summaries using local transformer models
   - Verify: Summarization works
   - Verify: Summaries are generated correctly

6. **Metadata Generation**
   - Test: Create database-friendly JSON/YAML metadata
   - Verify: Metadata files are created
   - Verify: Metadata structure is correct

7. **Multi-threaded Downloads**
   - Test: Concurrent processing with worker pools
   - Verify: Multiple episodes are processed concurrently
   - Verify: Worker pool configuration works

8. **Resumable Operations**
   - Test: Skip existing files, handle interruptions
   - Verify: Existing files are skipped
   - Verify: Interrupted operations can be resumed

9. **Configuration Files**
   - Test: JSON/YAML config support
   - Verify: Config files are parsed correctly
   - Verify: Config options are applied

10. **Service Mode**
    - Test: Non-interactive daemon mode
    - Verify: Service mode works
    - Verify: Service can be run as daemon

**Implementation Notes:**

- Each feature should have at least one test
- Tests should be realistic but focused
- Use E2E server fixture for HTTP requests
- Verify feature claims are accurate
- Some features may already be tested in E2E tests (that's OK)

**Deliverables:**

- `tests/acceptance/test_readme_key_features.py`
- Tests for all 10 key features
- Documentation of key features test approach

### Stage 4: CI/CD Integration

**Goal**: Integrate acceptance tests into CI/CD pipeline as final validation gate.

**CI/CD Strategy:**

1. **Test Execution Order**:
   - Unit tests → Integration tests → E2E tests → **Acceptance tests** (last)
   - Acceptance tests only run if all previous tests pass

2. **CI Job Configuration**:
   - **Job Name**: `test-acceptance`
   - **Triggers**: Only on main branch, or manual trigger
   - **Dependencies**: All other test jobs must pass first
   - **Timeout**: 30 minutes (acceptance tests may be slow)
   - **Parallel Execution**: Disabled (acceptance tests should run sequentially)

3. **Test Execution**:

   ```bash
   pytest tests/acceptance/ -v -m acceptance --disable-socket --allow-hosts=127.0.0.1,localhost
````

4. **Failure Handling**:
   - If acceptance tests fail, CI should fail
   - Clear error messages indicating which README example failed
   - Link to README section that failed

5. **Makefile Target**:

   ```makefile
   test-acceptance:
       pytest tests/acceptance/ -v -m acceptance
   ```

**GitHub Actions Workflow:**

````yaml
test-acceptance:
  runs-on: ubuntu-latest
  needs: [test-unit, test-integration, test-e2e]
  if: github.ref == 'refs/heads/main' || github.event_name == 'workflow_dispatch'
  steps:

    - uses: actions/checkout@v4
    - name: Set up Python 3.11
      uses: actions/setup-python@v5

      with:
        python-version: "3.11"

    - name: Install full dependencies (including ML)
      run: |

        pip install -e ".[dev,ml]"

    - name: Run acceptance tests (final validation gate)
      timeout-minutes: 30

      run: |
        pytest tests/acceptance/ -v -m acceptance \
          --disable-socket --allow-hosts=127.0.0.1,localhost
```text

- GitHub Actions workflow update
- Makefile target for acceptance tests
- CI/CD documentation updates
- Test execution strategy documentation

## Test Coverage Matrix

**README Sections Covered:**

| Section | Lines | Tests | Status |
| --------- | ------- | ------- | -------- |
| Installation | 35-58 | `test_readme_installation.py` | ⏳ Planned |
| Basic Usage | 60-75 | `test_readme_basic_usage.py` | ⏳ Planned |
| Key Features | 10-21 | `test_readme_key_features.py` | ⏳ Planned |
| Requirements | 25-33 | Installation tests | ⏳ Planned |
| Documentation | 77-98 | Not tested (links only) | ❌ Out of scope |

**Total Test Count Estimate:**

- Installation tests: ~4 tests
- Basic usage tests: ~3 tests
- Key features tests: ~10 tests
- **Total: ~17 acceptance tests**

## Success Criteria

1. ✅ All README installation examples are tested
2. ✅ All README CLI usage examples are tested
3. ✅ All key features mentioned in README are validated
4. ✅ Acceptance tests run as final CI step
5. ✅ Acceptance tests use `@pytest.mark.acceptance` marker
6. ✅ Acceptance tests are in `tests/acceptance/` directory
7. ✅ CI/CD pipeline includes acceptance test job
8. ✅ Acceptance tests fail fast with clear error messages
9. ✅ Documentation explains acceptance test category
10. ✅ All acceptance tests pass before releases

## Risks & Mitigations

**Risk 1: Slow Test Execution**

- **Mitigation**: Acceptance tests run only on main branch, after all other tests pass
- **Mitigation**: Clear timeout configuration (30 minutes)
- **Mitigation**: Tests are marked as `slow` for local development filtering

**Risk 2: Installation Test Flakiness**

- **Mitigation**: Use isolated virtual environments
- **Mitigation**: Skip tests if system dependencies are missing
- **Mitigation**: Clear error messages for installation failures

**Risk 3: Test Duplication with E2E Tests**

- **Mitigation**: Clear distinction: E2E tests verify features, acceptance tests verify README
- **Mitigation**: Acceptance tests use exact README examples
- **Mitigation**: Some overlap is acceptable (different purposes)

**Risk 4: README Changes Breaking Tests**

- **Mitigation**: Tests are documentation-driven (README is source of truth)
- **Mitigation**: When README changes, tests must be updated
- **Mitigation**: This is a feature, not a bug (catches documentation drift)

## Future Enhancements

1. **Documentation Link Testing**: Test that documentation links are valid (separate tool)
2. **Code Example Testing**: Test code examples in documentation (if any)
3. **Tutorial Testing**: Test step-by-step tutorials (if added to README)
4. **Multi-Platform Testing**: Test installation on different platforms (Linux, macOS, Windows)
5. **Version Compatibility Testing**: Test installation with different Python versions

## Open Questions

1. **Should acceptance tests run on every PR or only on main?**
   - **Proposal**: Only on main branch (slow, final validation gate)
   - **Alternative**: Run on PRs but allow failures (warning only)

2. **Should installation tests use real pip or mocked pip?**
   - **Proposal**: Real pip in isolated virtual environments
   - **Alternative**: Mocked pip (faster, but less realistic)

3. **Should acceptance tests be part of release process?**
   - **Proposal**: Yes, all acceptance tests must pass before release
   - **Alternative**: Acceptance tests are informational only

4. **How to handle README examples that require external services?**
   - **Proposal**: Use E2E server fixture (no external services)
   - **Alternative**: Skip tests that require external services

## References

- [RFC-019: E2E Test Infrastructure](RFC-019-e2e-test-improvements.md)
- [RFC-018: Test Structure Reorganization](RFC-018-test-structure-reorganization.md)
- [Testing Strategy](../TESTING_STRATEGY.md)
- [README.md](https://github.com/chipi/podcast_scraper/blob/main/README.md)
````
