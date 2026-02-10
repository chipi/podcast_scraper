# Testing Strategy

> **Document Structure:**
>
> - **This document** - High-level strategy, test pyramid, decision criteria
> - **[Testing Guide](guides/TESTING_GUIDE.md)** - Quick reference, test execution commands
> - **[Unit Testing Guide](guides/UNIT_TESTING_GUIDE.md)** - Unit test mocking patterns and isolation
> - **[Integration Testing Guide](guides/INTEGRATION_TESTING_GUIDE.md)** - Integration test mocking guidelines
> - **[E2E Testing Guide](guides/E2E_TESTING_GUIDE.md)** - E2E server, real ML models, OpenAI mocking
> - **[Critical Path Testing Guide](guides/CRITICAL_PATH_TESTING_GUIDE.md)** - What to test and prioritization

## Overview

This document defines the testing strategy for the
podcast scraper codebase. It establishes the test
pyramid approach, decision criteria for choosing test
types, and high-level testing patterns.

The system supports **8 providers** (ML, OpenAI,
Gemini, Anthropic, Mistral, DeepSeek, Grok, Ollama)
with per-provider unit, integration, and E2E tests.
All LLM providers use versioned Jinja2 prompt
templates managed by `PromptStore` (RFC-017).

**Planned**: The Grounded Insight Layer (GIL,
PRD-017) will add testing for insight/quote extraction,
grounding contract validation, and `kg.json` schema
compliance. See
[GIL Testing (Planned)](#gil-testing-planned).

For detailed implementation guides per test layer,
see the layer-specific guides linked above.

## Problem Statement

Testing requirements and strategies were previously scattered across individual RFCs, making it difficult to:

- Understand the overall testing approach
- Ensure consistent testing patterns across modules
- Plan new test infrastructure
- Onboard new contributors to testing practices
- Track testing coverage and requirements

This unified testing strategy document provides a single source of truth for all testing decisions and requirements.

## Test Pyramid

The testing strategy follows a three-tier pyramid:

```text
        /\
       /E2E\          ← Few, realistic end-to-end tests
      /------\
     /Integration\    ← Moderate, focused integration tests
    /------------\
   /    Unit      \   ← Many, fast unit tests
  /----------------\
```yaml

| Layer | Scope | Entry Point | HTTP | Fixtures | ML Models |
| ----- | ----- | ----------- | ---- | -------- | --------- |
| **Unit** | Individual functions/modules | Function/class level | Mocked | Mocked | Mocked |
| **Integration** | Component interactions | Component level | Local test server (or mocked) | Test fixtures | Real (optional) |
| **E2E** | Complete user workflows | User level (CLI/API) | Real HTTP client (local server) | Real data files | Real (in workflow) |

### Decision Questions

1. **Am I testing a complete user workflow?** (CLI command, library API call, service API call)
   - **YES** → E2E Test
   - **NO** → Continue to question 2

2. **Am I testing how multiple components work together?** (RSS parser → Episode → Provider → File)
   - **YES** → Integration Test
   - **NO** → Continue to question 3

3. **Am I testing a single function/module in isolation?**
   - **YES** → Unit Test
   - **NO** → Review test scope and purpose

### Common Patterns

- **Component workflow** (RSS → Episode → Provider) → Integration Test
- **Complete CLI command** (`podcast-scraper <url>`) → E2E Test
- **Library API call** (`run_pipeline(config)`) → E2E Test
- **Error handling in pipeline** → Integration Test (if focused) or E2E Test (if complete workflow)
- **HTTP client behavior** → Integration Test (if isolated) or E2E Test (if in workflow)

### Unit Tests

- **Purpose**: Test individual functions/modules in isolation
- **Speed**: Fast (< 100ms each)
- **Scope**: Single module, mocked dependencies
- **Coverage**: High (target: ≥70% code coverage)
- **Examples**: Config validation, filename sanitization, URL normalization

### Integration Tests

- **Purpose**: Test interactions between multiple modules/components (component interactions, data flow)
- **Speed**: Moderate (< 5s each for fast tests)
- **Scope**: Multiple modules working together, real internal implementations
- **Entry Point**: Component-level (functions, classes, not user-facing APIs)
- **I/O Policy**:
  - ✅ **Allowed**: Real filesystem I/O (temp directories), real component interactions
  - ❌ **Mocked**: External services (HTTP APIs, external APIs) - mocked for speed/reliability
  - ⚠️ **ML models**: Mocked by default for speed. Use real models with `@pytest.mark.ml_models`
    for ML workflow integration tests (excluded from fast suite, see
    [Integration Testing Guide](guides/INTEGRATION_TESTING_GUIDE.md))
  - ✅ **Optional**: Local HTTP server for HTTP client testing in isolation
- **Coverage**: Critical paths and edge cases, component interactions
- **Examples**: Provider factory → provider implementation, RSS parser → Episode → Provider → File output, HTTP client with local test server
- **Key Distinction**: Tests how components work together, not complete user workflows. Mock ML
  models for fast tests; use real models with `@pytest.mark.ml_models` for ML workflow tests.

### End-to-End Tests

- **Purpose**: Test complete user workflows from entry
  point to final output (CLI commands, library API
  calls, service API calls)
- **Speed**: Slow (< 60s each, may be minutes for
  full workflows)
- **Scope**: Full pipeline from entry point to output,
  real HTTP client, real data files, real ML models
- **Entry Point**: User-level (CLI commands,
  `run_pipeline()`, `service.run()`)
- **I/O Policy**:
  - ✅ **Allowed**: Real HTTP client with local HTTP
    server (no external network), real filesystem I/O,
    real data files
  - ✅ **Real implementations**: Use actual HTTP
    clients (no mocking), real file operations, real
    model loading
  - ✅ **Real ML models**: Use real Whisper, spaCy,
    and Transformers models (NO mocks)
  - ✅ **Real data files**: RSS feeds, transcripts,
    audio files from `tests/fixtures/e2e_server/`
  - ❌ **No external network**: All HTTP calls go to
    local server (network guard prevents external
    calls)
  - ❌ **No mocks**: E2E tests use real
    implementations throughout (no Whisper mocks,
    no ML model mocks)
- **Provider E2E tests**: Dedicated E2E test files
  exist for each of the 8 providers:
  `test_ml_models_e2e.py`, `test_openai_provider_e2e.py`,
  `test_gemini_provider_e2e.py`,
  `test_anthropic_provider_e2e.py`,
  `test_mistral_provider_e2e.py`,
  `test_deepseek_provider_e2e.py`,
  `test_grok_provider_e2e.py`,
  `test_ollama_provider_e2e.py`. LLM providers use
  mock API endpoints served by the E2E HTTP server.
- **Coverage**: Complete user workflows,
  production-like scenarios
- **Examples**: CLI command
  (`podcast-scraper <rss_url>`) → Full pipeline →
  Output files, Library API
  (`run_pipeline(config)`) → Full pipeline →
  Output files
- **Key Distinction**: Tests complete user workflows
  with real implementations. NO mocks allowed — tests
  the system as users would use it.

## Decision Criteria

The decision questions above provide a quick way to determine test type. For critical path prioritization, see [Critical Path Testing Guide](guides/CRITICAL_PATH_TESTING_GUIDE.md). For detailed implementation guidelines, see [Testing Guide](guides/TESTING_GUIDE.md).

**Quick Reference:**

- **Unit Test**: Single function/module in isolation, all dependencies mocked
- **Integration Test**: Multiple components working together, real internal implementations, mocked external services (including ML models for speed)
- **E2E Test**: Complete user workflow from entry point to output, real HTTP client, real data files, real ML models (NO mocks)

**Critical Path Priority**: If your test covers the critical path (RSS → Parse → Download/Transcribe → NER → Summarization → Metadata → Files), prioritize it. See [Critical Path Testing Guide](guides/CRITICAL_PATH_TESTING_GUIDE.md) for details.

## Test Categories

### 1. Unit Tests

#### Configuration & Validation (`config.py`)

- **RFC-008**: Validate coercion logic, error messages, alias handling
- **RFC-007**: Test argument parsing edge cases (invalid speaker counts, unknown config keys)
- **Test Cases**:
  - Type coercion (string → int, validation failures)
  - Config file loading (JSON/YAML, invalid formats)
  - Default value application
  - Alias resolution (`rss` vs `rss_url`)

#### Filesystem Operations (`filesystem.py`)

- **RFC-004**: Sanitization edge cases, output derivation logic
- **Test Cases**:
  - Filename sanitization (special characters, reserved names)
  - Output directory derivation and validation
  - Run suffix generation
  - Path normalization across platforms

#### RSS Parsing (`rss_parser.py`)

- **RFC-002**: Varied RSS shapes, namespace differences, missing attributes
- **Test Cases**:
  - Namespace handling (Podcasting 2.0, standard RSS)
  - Relative URL resolution
  - Missing optional fields
  - Malformed XML handling
  - Edge cases (uppercase tags, mixed namespaces)

#### Transcript Downloads (`downloader.py`, `episode_processor.py`)

- **RFC-003**: Extension derivation edge cases
- **Test Cases**:
  - URL normalization (encoding, special characters)
  - Extension inference (from URL, Content-Type, declared type)
  - HTTP retry logic (unit test with mocked responses)
  - Transcript type preference ordering

#### Whisper Integration (`whisper_integration.py`)

- **RFC-005**: Mock Whisper library, loading paths, error handling
- **RFC-006**: Screenplay formatting with synthetic segments
- **Test Cases**:
  - Model loading (success, missing dependency, invalid model)
  - Screenplay formatting (gap handling, speaker rotation, aggregation)
  - Language parameter propagation
  - Model selection logic (`.en` variants for English)

#### Progress Reporting (`progress.py`)

- **RFC-009**: Noop factory, `set_progress_factory` behavior
- **Test Cases**:
  - Factory registration and replacement
  - Progress update calls
  - Context manager behavior

#### Speaker Detection (`speaker_detection.py`) - RFC-010

- **RFC-010**: NER extraction scenarios, host/guest distinction
- **Test Cases**:
  - Title-only detection (`"Alice interviews Bob"`)
  - Description-rich detection (multiple guest names)
  - Feed-level host inference
  - CLI override precedence
  - spaCy missing/disabled scenarios
  - Name capping when too many detected

#### Summarization (`summarizer.py`) - RFC-012

- **RFC-012**: Local transformer model integration, summary generation with map-reduce strategy
- **Test Cases**:
  - Model selection (explicit, auto-detection for MPS/CUDA/CPU)
  - Model loading and initialization on different devices
  - **Model integration tests** (marked as `@pytest.mark.slow` and `@pytest.mark.integration`):
    - Verify all models in `DEFAULT_SUMMARY_MODELS` can be loaded when configured
    - Test each model individually: `bart-large`, `bart-small`, `long`, `long-fast`
    - Catch dependency issues (e.g., missing protobuf for PEGASUS models)
    - Verify model and tokenizer are properly initialized
    - Test model unloading after loading
  - **Map-reduce strategy**:

```text
    - Map phase: chunking (word-based and token-based), chunk summarization
    - Reduce phase decision logic: single abstractive (≤800 tokens), mini map-reduce (800-4000 tokens), extractive (>4000 tokens)
    - Mini map-reduce: re-chunking combined summaries into 3-5 sections (650 words each), second map phase (summarize each section), final abstractive reduce
    - Extractive fallback behavior for extremely large combined summaries
```

- Summary generation with various text lengths
- Key takeaways extraction
- Text chunking for long transcripts
- Safe summarization error handling (OOM, missing dependencies)
- Memory optimization (CUDA/MPS)
- Model unloading and cleanup
- Integration with metadata generation pipeline

### Provider System (RFC-013, RFC-029)

- **RFC-013/RFC-029**: Protocol-based provider
  architecture for transcription, speaker detection,
  and summarization with 8 unified providers.
- **Provider Coverage**: Each provider has a
  consistent test structure:

| Provider | Unit Tests | Integration | E2E |
| --- | --- | --- | --- |
| MLProvider | `test_ml_provider.py`, `_lifecycle.py` | `test_provider_real_models.py` | `test_ml_models_e2e.py` |
| OpenAIProvider | `test_openai_provider.py`, `_factory.py`, `_lifecycle.py` | `test_openai_providers.py` | `test_openai_provider_e2e.py` |
| GeminiProvider | `test_gemini_provider.py`, `_factory.py`, `_lifecycle.py` | `test_gemini_providers.py` | `test_gemini_provider_e2e.py` |
| AnthropicProvider | `test_anthropic_provider.py`, `_factory.py`, `_lifecycle.py` | `test_anthropic_providers.py` | `test_anthropic_provider_e2e.py` |
| MistralProvider | `test_mistral_provider.py`, `_factory.py`, `_lifecycle.py` | `test_mistral_providers.py` | `test_mistral_provider_e2e.py` |
| DeepSeekProvider | `test_deepseek_provider.py`, `_factory.py`, `_lifecycle.py` | `test_deepseek_providers.py` | `test_deepseek_provider_e2e.py` |
| GrokProvider | `test_grok_provider.py`, `_factory.py`, `_lifecycle.py` | `test_grok_providers.py` | `test_grok_provider_e2e.py` |
| OllamaProvider | `test_ollama_provider.py`, `_factory.py`, `_lifecycle.py` | `test_ollama_providers.py` | `test_ollama_provider_e2e.py` |

- **Provider Capabilities**: `test_capabilities.py`
  (unit) and `test_capabilities_e2e.py` (E2E) test
  `ProviderCapabilities` — which providers support
  which features (JSON mode, tool calls, streaming).

- **Test Cases**:
  - **Unit Tests (Standalone Provider)**: Each
    provider tested directly with all dependencies
    mocked. Tests: creation, initialization, protocol
    method implementation, error handling, cleanup,
    configuration validation.
  - **Unit Tests (Factory)**: Factory tests verify
    correct provider instantiation per config,
    protocol compliance, error handling.
  - **Unit Tests (Lifecycle)**: Provider lifecycle
    tests verify initialization, cleanup,
    resource management.
  - **Integration Tests**: Real provider
    implementations with mocked external services.
    Tests: provider factory, protocol compliance,
    component interactions, provider switching, error
    handling in workflow context.
  - **E2E Tests**: Real providers in full pipeline.
    LLM providers use E2E server mock endpoints.
    MLProvider uses real ML models (Whisper, spaCy,
    Transformers). Tests: complete workflows,
    error scenarios (API failures, rate limits).

### Prompt Management (RFC-017)

- **Prompt Store**: `test_prompt_store.py` (unit)
  tests versioned Jinja2 prompt template management.
- **Test Cases**:
  - Template loading per provider/task/version
  - Jinja2 variable rendering
  - Missing template error handling
  - Template version selection
  - All 8 providers have prompt directories
    (`prompts/<provider>/ner/`, `summarization/`)
  - Template validation (syntax, required variables)

#### Service API (`service.py`)

- **Public API**: Service interface for daemon/non-interactive use
- **Test Cases**:
  - `ServiceResult` dataclass (success/failure states, attributes)
  - `service.run()` with valid Config (success path)
  - `service.run()` with logging configuration
  - `service.run()` exception handling (returns failed ServiceResult)
  - `service.run_from_config_file()` with JSON config
  - `service.run_from_config_file()` with YAML config
  - `service.run_from_config_file()` with missing file (returns failed ServiceResult)
  - `service.run_from_config_file()` with invalid config (returns failed ServiceResult)
  - `service.run_from_config_file()` with Path objects
  - `service.main()` CLI entry point (success/failure exit codes)
  - `service.main()` version flag handling
  - `service.main()` missing config argument handling
  - Service API importability via `__getattr__`
  - ServiceResult equality and string representation
  - Integration with public API (`Config`, `load_config_file`, `run_pipeline`)

#### Reproducibility & Operational Hardening (Issue #379)

- **Determinism**: Seed-based reproducibility for `torch`, `numpy`, and `transformers`. Test that seeds are set correctly and outputs are consistent across runs.
- **Run Tracking**: Test run manifest creation (system state capture), run summary generation (manifest + metrics), and episode index generation (episode status tracking).
- **Failure Handling**: Test `--fail-fast` and `--max-failures` flags, episode failure tracking in metrics, and exit code behavior.
- **Retry Policies**: Test exponential backoff retry for transient errors (network failures, model loading errors), retry counts and delays, and fallback to cache clearing.
- **Timeout Enforcement**: Test transcription and summarization timeout enforcement, timeout error handling, and timeout configuration.
- **Security**: Test path validation (directory traversal prevention), model allowlist validation, safetensors format preference, and `trust_remote_code=False` enforcement.
- **Structured Logging**: Test JSON log formatting, log aggregation compatibility, and structured log fields.
- **Diagnostics**: Test `doctor` command checks (Python version, ffmpeg, write permissions, model cache, network connectivity).

**For detailed unit test execution commands, test file descriptions, fixtures, requirements, and coverage, see [Unit Testing Guide](guides/UNIT_TESTING_GUIDE.md).**

### 2. Integration Tests

#### CLI Integration (`cli.py` + `workflow.py`)

- **RFC-001**: Success, dry-run, concurrency edge cases, error handling
- **RFC-007**: CLI happy path, invalid args, config file precedence
- **Test Cases**:
  - CLI argument parsing → Config validation → pipeline execution
  - Config file loading and precedence (CLI > config file)
  - Dry-run mode (no disk writes)
  - Error handling and exit codes
  - Dependency injection hooks (`apply_log_level_fn`, `run_pipeline_fn`)

#### Workflow Orchestration (`workflow.py`)

- **RFC-001**: End-to-end coordination, concurrency, cleanup
- **Test Cases**:
  - RSS fetch → episode parsing → transcript download → file writing
  - Concurrent transcript downloads (ThreadPoolExecutor)
  - Whisper job queuing and sequential processing
  - Temp directory cleanup
  - Skip-existing and clean-output flags

#### Episode Processing (`episode_processor.py`)

- **RFC-003**: HTTP response simulation with various headers
- **RFC-004**: Directory management interactions
- **Test Cases**:
  - Transcript download with various formats (VTT, SRT, JSON)
  - Media download for Whisper transcription
  - File naming and storage
  - Error handling (network failures, missing files)

#### Config + CLI + Workflow

- **RFC-008**: CLI + config files → Config instantiation
- **Test Cases**:
  - Config file loading → validation → pipeline execution
  - Config override precedence
  - Invalid config error handling

#### Whisper + Screenplay Formatting

- **RFC-006**: Screenplay flags → formatting → file output
- **RFC-010**: Detected speaker names → screenplay formatting
- **Test Cases**:
  - Whisper transcription → screenplay formatting with detected names
  - Speaker name override precedence
  - Language-aware model selection

**For detailed integration test execution commands, test file descriptions, fixtures, requirements,
and coverage, see [Integration Testing Guide](guides/INTEGRATION_TESTING_GUIDE.md).**

### 3. End-to-End Tests

#### E2E Test Coverage Goals

**Critical Path Priority**: The critical path must have E2E tests for all three entry points
(CLI, Library API, Service API). See
[Critical Path Testing Guide](guides/CRITICAL_PATH_TESTING_GUIDE.md) for details.

**Every major user-facing entry point should have at least one E2E test:**

1. **CLI Commands** - Each main CLI command should have E2E tests
2. **Library API Endpoints** - Each public API function should have E2E tests
3. **Critical User Scenarios** - Important workflows should have E2E tests

**What Doesn't Need E2E Tests:**

- Not every CLI flag combination needs an E2E test
- Every possible configuration value (tested in integration/unit tests)
- Edge cases in specific components (tested in integration tests)

**Rule of Thumb**: E2E tests should cover "as a user, I want to..." scenarios, not every possible configuration combination.

**For detailed E2E test execution commands and implementation, see [E2E Testing Guide](guides/E2E_TESTING_GUIDE.md).**

### E2E Test Tiers (Code Quality vs Data Quality)

E2E tests are organized into three tiers to balance fast CI feedback with comprehensive validation:

| Tier | Purpose | Episodes | Models | When | Makefile Target |
| ---- | ------- | -------- | ------ | ---- | --------------- |
| **Tier 1: Fast** | Code quality, critical path | 1 | Test (tiny/base) | Every PR | `test-e2e-fast` |
| **Tier 2: Data Quality** | Volume validation | 3 | Test (tiny/base) | Nightly | `test-e2e-data-quality` |
| **Tier 3: Nightly Full** | Production validation | 15 (5×3) | Production (base, BART-large, LED-large) | Nightly | `test-nightly` |

**Tier 1: Fast E2E Tests** (`@pytest.mark.e2e` + `@pytest.mark.critical_path`)

- 1 episode per test for fast feedback
- Test models (Whisper tiny.en, BART-base, LED-base)
- Run on every PR and push to main
- Focus: Does the code work correctly?

**Tier 2: Data Quality Tests** (`@pytest.mark.e2e` + `@pytest.mark.data_quality`)

- 3 episodes per test for volume validation
- Test models (same as Tier 1)
- Run in nightly builds only
- Focus: Does data processing work with volume?

**Tier 3: Nightly Full Suite** (`@pytest.mark.nightly`)

- 15 episodes across 5 podcasts (p01-p05)
- Production models (Whisper base, BART-large-cnn, LED-large-16384)
- Run in nightly builds only
- Focus: Production-quality validation with real models

**Key Principle:** Code quality tests (Tier 1) run on every PR. Data quality and nightly tests
(Tiers 2-3) run only in nightly builds to avoid slowing down CI/CD feedback.

**LLM/OpenAI Exclusion:** Tests marked `@pytest.mark.llm` are excluded from nightly builds
(`-m "not llm"`) to avoid API costs. See issue #183.

## Test Infrastructure

### Test Framework

- **Primary**: pytest (with unittest compatibility)
- **Fixtures**: pytest fixtures for test setup and teardown
- **Markers**: pytest markers for test categorization

### Mocking Strategy

- **Unit Tests**: Mock all external dependencies (HTTP, ML models, file system, API clients)
- **Integration Tests**: Mock external services (HTTP APIs, external APIs) and ML models (Whisper,
  spaCy, Transformers), use real internal implementations (real providers, real Config, real

  workflow logic)

- **E2E Tests**: Use real implementations (HTTP client, ML models, file system) with local test
  server. For API providers, use E2E server mock endpoints instead of direct API calls. ML models

  (Whisper, spaCy, Transformers) are REAL - no mocks allowed.

**Provider Testing Strategy (8 providers):**

- **Unit Tests (Standalone Provider)**: Each of the
  8 providers tested directly with all dependencies
  mocked (API clients, ML models). Dedicated test
  files: `test_<provider>_provider.py`,
  `_factory.py`, `_lifecycle.py`.
- **Unit Tests (Factory)**: Test factories create
  correct unified providers for each
  `<provider>_provider` config value. Verify protocol
  compliance, test factory error handling.
- **Integration Tests**: Per-provider integration
  files (`test_<provider>_providers.py`) use real
  provider implementations with mocked external
  services. ML models mocked for speed.
- **E2E Tests**: Per-provider E2E files
  (`test_<provider>_provider_e2e.py`) use E2E server
  mock endpoints for LLM providers, real ML models
  for MLProvider.
- **Key Principle**: Always verify protocol
  compliance, not class names. All providers
  implement the same protocol interfaces.
- **Prompt Templates**: LLM providers load prompts
  via `PromptStore`. Unit tests mock `PromptStore`;
  integration/E2E tests use real templates.
- **Test Organization**: See
  `docs/wip/PROVIDER_TEST_STRATEGY.md` for detailed
  test organization and separation.

### Test Organization

The test suite is organized into three main categories:

- **`tests/unit/`** - Unit tests per module (fast, isolated, fully mocked)
- **`tests/integration/`** - Integration tests (component interactions, real internal implementations, mocked external services)
- **`tests/e2e/`** - E2E tests (complete workflows, real HTTP client, real ML models)

### Test Markers

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (component interactions)
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.critical_path` - Critical path tests (run in fast suite)
- `@pytest.mark.serial` - Tests that must run sequentially (resource conflicts)
- `@pytest.mark.ml_models` - Requires ML dependencies (whisper, spacy, transformers) or uses real
  ML models

- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.network` - Tests that hit external network
- `@pytest.mark.multi_episode` - Multi-episode tests (nightly)
- `@pytest.mark.data_quality` - Data quality validation (nightly only, 3 episodes)
- `@pytest.mark.nightly` - Comprehensive nightly tests (15 episodes, production models)
- `@pytest.mark.llm` - Tests using LLM APIs (excluded
  from nightly to avoid costs)
- `@pytest.mark.openai` - Tests using OpenAI API
  specifically (subset of llm)
- `@pytest.mark.gil` - (planned) GIL extraction tests
- `@pytest.mark.grounding` - (planned) Grounding
  contract validation tests

**Execution Pattern**: Tests marked `serial` run first sequentially, then remaining tests run in
parallel with `-n auto`. All tests use network isolation via
`--disable-socket --allow-hosts=127.0.0.1,localhost`.

**For detailed test infrastructure information, see [Testing Guide](guides/TESTING_GUIDE.md).**

## ML Quality Evaluation

Beyond functional testing, the project uses objective
metrics to evaluate the **quality** of ML-generated
outputs. These evaluations are performed using
dedicated scripts and a "golden" dataset.

### Evaluation Layers

| Layer | Focus | Metrics | Tools |
| :--- | :--- | :--- | :--- |
| **Cleaning** | Effective removal of ads/outro | Removal %, Brand detection | Automatic (via experiment runner) |
| **Summarization** | Accuracy and synthesis quality | ROUGE-1/2/L, Compression ratio | Automatic (via experiment runner) |
| **GIL** (planned) | Insight/quote extraction quality | Grounding rate, Quote verbatim %, Insight coverage | Per-provider comparison |

### Golden Datasets

Evaluation is performed against human-verified ground
truth data stored in `data/eval/`. This dataset is
versioned and frozen to provide a stable baseline for
comparison. See
[ADR-026](adr/ADR-026-explicit-golden-dataset-versioning.md)
for details.

**Planned (GIL)**: A golden dataset for GIL
evaluation will include human-annotated insights,
quotes, and grounding links for a representative
set of episodes. This enables comparison across
extraction tiers (ML-only, Hybrid, Cloud LLM).

### Continuous Improvement

Quality evaluation is integrated into the
**[AI Quality & Experimentation Platform](prd/PRD-007-ai-quality-experiment-platform.md)**
(PRD-007), which uses these metrics to gate new model
deployments and configuration changes. The complete
evaluation loop (runner, scorer, comparator) is
documented in the
**[Experiment Guide](guides/EXPERIMENT_GUIDE.md)**
(Step 4: Evaluate Results).

## CI/CD Integration

### Continuous Integration Strategy

The CI/CD pipeline (GitHub Actions) implements a multi-layered validation strategy to balance fast feedback with comprehensive quality control.

#### 1. Every Pull Request (Per-Commit Feedback)

- **lint**: Fast formatting, code linting, markdown
  linting, and type checking (~2 min)
- **test-unit**: All unit tests with coverage,
  verified network isolation (~3-5 min)
- **test-integration-fast**: Critical path integration
  tests using test ML models (~5-8 min)
- **test-e2e-fast**: Critical path E2E tests (Tier 1)
  using test ML models (~8-12 min)
- **build**: Package build validation (~2 min)
- **docs**: Documentation build validation (~3 min)

#### 2. Main Branch & Release Branches (Full Validation)

- **Unified Coverage**: Combines unit, integration,
  and E2E coverage reports into a single artifact.
- **test-integration**: Complete integration test
  suite.
- **test-e2e**: Complete E2E test suite (Tier 1).
- **Automated Metrics**: Collection of test health,
  code quality, and pipeline performance metrics.
- **Unified Dashboard**: Deployment of the
  [Metrics Dashboard](ci/METRICS.md) to GitHub Pages.

#### 3. Nightly Comprehensive (Deep Analysis)

- **nightly-only-tests**: Full validation (Tier 3) using production-quality ML models (Whisper base, BART-large, LED-large).
- **Data Quality (Tier 2)**: Volume validation with multiple episodes.
- **Module Dependency Analysis**: Automated graph generation and coupling analysis.
- **Trend Tracking**: Historical metrics accumulation for long-term health monitoring.

#### Test Execution Pattern

- **Parallel Execution**: Most jobs run in parallel with reserved cores for system stability (`auto - 2`).
- **Flaky Test Resilience**: Integration and E2E tests use automatic reruns (`--reruns 2`).
- **Network Isolation**: Enforced via `pytest-socket` across all test tiers.
- **LLM Exclusion**: API-based tests (OpenAI) are excluded from nightly runs to avoid costs.

**For detailed test execution commands, parallel execution, and coverage configuration, see [Testing Guide](guides/TESTING_GUIDE.md).**

## Testing Patterns

### Dependency Injection

- **CLI Testing**: Use `cli.main()` override callables (`apply_log_level_fn`, `run_pipeline_fn`, `logger`)
- **Workflow Testing**: Mock `run_pipeline` for CLI-focused tests
- **Benefit**: Test CLI behavior without executing full pipeline

### Mocking External Dependencies

- **HTTP**: Mock `requests.Session` and responses (unit/integration tests), use E2E server for E2E tests
- **Whisper**: Mock `whisper.load_model()` and `whisper.transcribe()` (unit tests), use real models (integration/E2E tests)
- **ML Dependencies (spacy, torch, transformers)**:
  - **Unit Tests**: Mock in `sys.modules` before importing dependent modules
  - **Integration Tests**: Real ML dependencies required
  - **Verification**: CI runs `scripts/tools/check_unit_test_imports.py` to ensure modules can import without ML deps
- **File System**: Use `tempfile` for isolated test environments
- **API Providers** (OpenAI, Gemini, Anthropic,
  Mistral, DeepSeek, Grok, Ollama):
  - **Unit Tests**: Mock API clients (e.g., `OpenAI`,
    `genai`, `anthropic.Anthropic` classes)
  - **Integration Tests**: Mock API clients or use
    E2E server mock endpoints
  - **E2E Tests**: Use E2E server mock endpoints
    (real HTTP client, mock API responses). Dedicated
    mock fixtures: `openai_mock.py`,
    `gemini_mock_client.py`

### Test Isolation

- Each test uses `tempfile.TemporaryDirectory` for output
- Tests clean up after themselves
- No shared state between tests
- Mock external services (HTTP, file system)
- No network calls in unit tests (enforced by pytest plugin)
- No filesystem I/O in unit tests (enforced by pytest plugin, except tempfile operations)

### Error Testing

- Test validation errors (invalid config, malformed RSS)
- Test network failures (timeouts, connection errors)
- Test missing dependencies (Whisper, spaCy unavailable)
- Test edge cases (empty feeds, missing transcripts, invalid URLs)

## Test Requirements by Module

### `cli.py`

- [ ] Argument parsing (valid, invalid, edge cases)
- [ ] Config file loading and precedence
- [ ] Error handling and exit codes
- [ ] Dependency injection hooks
- [ ] Version flag behavior

### `config.py`

- [ ] Type coercion and validation
- [ ] Default value application
- [ ] Config file loading (JSON/YAML)
- [ ] Alias resolution
- [ ] Error messages

### `workflow.py`

- [ ] Pipeline orchestration
- [ ] Concurrent downloads
- [ ] Whisper job queuing
- [ ] Cleanup operations
- [ ] Dry-run mode

### `rss_parser.py`

- [ ] RSS parsing (various formats)
- [ ] Namespace handling
- [ ] URL resolution
- [ ] Episode creation
- [ ] Error handling (malformed XML)

### `downloader.py`

- [ ] HTTP session configuration
- [ ] Retry logic
- [ ] URL normalization
- [ ] Streaming downloads
- [ ] Error handling

### `episode_processor.py`

- [ ] Transcript download
- [ ] Media download
- [ ] File naming
- [ ] Whisper job creation
- [ ] Error handling

### `filesystem.py`

- [ ] Filename sanitization
- [ ] Output directory derivation
- [ ] Run suffix generation
- [ ] Path validation

### `whisper_integration.py`

- [ ] Model loading
- [ ] Transcription invocation
- [ ] Screenplay formatting
- [ ] Language handling
- [ ] Error handling (missing dependency)

### `speaker_detection.py` (RFC-010)

- [ ] NER extraction
- [ ] Host/guest distinction
- [ ] CLI override precedence
- [ ] Fallback behavior
- [ ] Caching logic

### `progress.py`

- [ ] Factory registration
- [ ] Progress updates
- [ ] Context manager behavior

### `summarizer.py` (RFC-012)

- [x] Model selection logic (explicit, auto-detection for MPS/CUDA/CPU)
- [x] Model loading and initialization
- [x] **Model integration tests** (all models in `DEFAULT_SUMMARY_MODELS` can be loaded)
- [ ] Summary generation
- [ ] Key takeaways generation
- [x] Text chunking for long transcripts
- [ ] Safe summarization with error handling
- [ ] Memory optimization (CUDA/MPS)
- [x] Model unloading
- [ ] Integration with metadata generation

### `service.py` (Public API)

- [x] `ServiceResult` dataclass
- [x] `service.run()` with valid Config
- [x] `service.run()` with logging configuration
- [x] `service.run()` exception handling
- [x] `service.run_from_config_file()` JSON/YAML
- [x] `service.run_from_config_file()` error cases
- [x] `service.main()` CLI entry point
- [x] `service.main()` version/missing config
- [x] Service API importability via `__getattr__`
- [x] ServiceResult equality and string repr
- [x] Integration with public API

### `providers/` (All 8 Providers)

- [x] MLProvider: unit, lifecycle, integration, E2E
- [x] OpenAIProvider: unit, factory, lifecycle,
  integration, E2E
- [x] GeminiProvider: unit, factory, lifecycle,
  integration, E2E
- [x] AnthropicProvider: unit, factory, lifecycle,
  integration, E2E
- [x] MistralProvider: unit, factory, lifecycle,
  integration, E2E
- [x] DeepSeekProvider: unit, factory, lifecycle,
  integration, E2E
- [x] GrokProvider: unit, factory, lifecycle,
  integration, E2E
- [x] OllamaProvider: unit, factory, lifecycle,
  integration, E2E
- [x] Provider capabilities (`test_capabilities.py`)
- [x] Provider params (`test_provider_params.py`)

### `prompts/store.py` (Prompt Management)

- [x] Template loading per provider/task/version
- [x] Jinja2 variable rendering
- [x] Missing template error handling
- [x] Template version selection

### `schemas/summary_schema.py`

- [x] Summary schema validation
  (`test_summary_schema.py`)

### GIL Modules (Planned — RFC-049)

- [ ] `workflow/gil_extraction.py` — GIL orchestration
- [ ] `kg/schema.py` — `kg.json` validation
- [ ] `kg/writer.py` — kg.json serialization
- [ ] Grounding contract enforcement
- [ ] Three-tier extraction (ML-only, Hybrid, Cloud)
- [ ] Quote verbatim validation
- [ ] Insight grounding status tracking

## Test Pyramid Status

> **Note**: Test distribution numbers should be verified periodically by running test collection
> and counting tests by layer. Historical progress tracking was documented in
> `docs/wip/TEST_PYRAMID_ANALYSIS.md` and `docs/wip/TEST_PYRAMID_PLAN.md` (now consolidated here).

### Current State vs. Ideal Distribution

The test pyramid shows our current distribution compared to the ideal:

**Current Distribution (verify with `pytest --collect-only`):**

- **Unit Tests**: Target 70-80% ⚠️ Need to verify current percentage
- **Integration Tests**: Target 15-20% ⚠️ Need to verify current percentage
- **E2E Tests**: Target 5-10% ⚠️ Need to verify current percentage

**Visual Representation:**

```text
        ╱╲
       ╱  ╲      E2E: 31% (should be 5-10%)
      ╱    ╲
     ╱      ╲    Integration: 27% (should be 15-20%)
    ╱        ╲
   ╱          ╲  Unit: 41% (should be 70-80%)
  ╱____________╲

Ideal Pyramid:
        ╱╲
       ╱  ╲      E2E: 5-10%
      ╱    ╲
     ╱      ╲    Integration: 15-20%
    ╱        ╲
   ╱          ╲  Unit: 70-80%
  ╱____________╲
```text

1. **Too Few Unit Tests**: Core business logic is being tested at E2E level instead of unit level
   - **Critical modules with zero unit tests**: `workflow.py`, `cli.py`, `service.py`, `episode_processor.py`, `preprocessing.py`, `progress.py`, `metrics.py`, `filesystem.py`
   - **67 summarizer tests misclassified**: Currently at E2E level but test individual functions with mocked dependencies → should be unit tests
   - **Missing unit test coverage**: Many core functions in `summarizer.py`, `workflow.py`, and `speaker_detection.py` have no unit tests

2. **Too Many E2E Tests**: Many tests are misclassified
   - Tests that use function-level entry points with mocked dependencies should be unit tests
   - Tests that use component-level entry points should be integration tests
   - **Root cause**: Tests were written at E2E level for convenience, violating testing strategy definitions

3. **Integration Layer Underutilized**: Component interactions are often tested at E2E level
   - **Missing integration test coverage**: RSS Parser + Downloader, Downloader + Episode Processor, Progress Reporting + Workflow, Metrics + Workflow, Filesystem + Workflow
   - Some E2E tests should be integration tests (component-level entry points with mocked HTTP)

### Goals and Targets

**Target Distribution:**

- **Unit Tests**: 70-80% (~550-650 tests)
- **Integration Tests**: 15-20% (~120-150 tests)
- **E2E Tests**: 5-10% (~50-80 tests)

**Success Metrics:**

- Unit test execution time: < 30 seconds
- Integration test execution time: < 5 minutes
- E2E test execution time: < 20 minutes
- Test coverage: Maintain ≥70%

### Improvement Strategy

**Phase 1: Reclassify Misplaced Tests** (High Priority)

- Move 67 summarizer tests from E2E to unit (they test individual functions with mocked dependencies)
- Review and reclassify E2E tests that violate testing strategy definitions
- **Expected Result**: Unit: ~50-51%, Integration: ~27-28%, E2E: ~22-23%

**Phase 2: Add Missing Unit Tests** (High Priority)

- Add unit tests for core functions currently untested:
  - `workflow.py` helper functions (8-10 tests)
  - `episode_processor.py` functions (5-8 tests)
  - `summarizer.py` core functions (45-65 tests)
  - `speaker_detection.py` functions (19-31 tests)
  - `preprocessing.py`, `progress.py`, `metrics.py`, `filesystem.py` (17-28 tests)
  - `cli.py` and `service.py` (8-13 tests)
- **Target**: +150-200 new unit tests
- **Expected Result**: Unit: ~69-83%, Integration: ~27-28%, E2E: ~22-23%

**Phase 3: Optimize Integration Layer** (Medium Priority)

- Move component interaction tests from E2E to integration (~20-40 tests)
- Add missing integration tests for component-to-component interactions (~19-31 tests)
- **Expected Result**: Unit: ~69-83%, Integration: ~30-32%, E2E: ~16-19%

**Phase 4: Reduce E2E to True E2E** (Low Priority)

- Keep only true end-to-end user workflow tests
- Focus on complete user journeys (CLI commands, library API calls, service API calls)
- **Target**: ~50-80 true E2E tests (5-10% of total)
- **Expected Final Result**: Unit: ~70-80%, Integration: ~15-20%, E2E: ~5-10% ✅

### Priority Areas for Unit Test Coverage

**High Priority Modules:**

- `summarizer.py`: Text cleaning, chunking, validation functions (45-65 tests needed)
- `workflow.py`: Pipeline orchestration helpers (8-10 tests needed)
- `episode_processor.py`: Episode processing logic (5-8 tests needed)

**Medium Priority Modules:**

- `speaker_detection.py`: Detection and scoring logic (19-31 tests needed)
- `cli.py` and `service.py`: Argument parsing and service logic (8-13 tests needed)
- `preprocessing.py`, `progress.py`, `metrics.py`, `filesystem.py`: Utility functions (17-28 tests needed)

## Future Testing Enhancements

### E2E Test Infrastructure Improvements (Issue #14)

- [x] Local HTTP test server
- [x] Test audio file fixtures
- [x] Real Whisper integration tests
- [x] Test markers and CI integration

### Library API Tests (Issue #16)

- [x] `run_pipeline()` E2E tests
- [x] `load_config_file()` tests
- [x] Error handling tests
- [x] Return value validation

### GIL Testing (Planned)

Testing for the Grounded Insight Layer (PRD-017,
RFC-049/050/051) will follow the established test
pyramid:

**Unit Tests:**

- `kg/schema.py` — JSON schema validation against
  `docs/kg/kg.schema.json`
- `kg/writer.py` — Serialization of insights, quotes,
  edges to `kg.json` format
- `workflow/gil_extraction.py` — Extraction
  orchestration with mocked providers
- Grounding contract validation (every quote verbatim,
  every insight declares grounding status)
- Three extraction tiers (ML-only, Hybrid, Cloud LLM)
  with mocked models

**Integration Tests:**

- GIL extraction → provider interaction (real
  providers, mocked external services)
- `kg.json` → Postgres export pipeline (RFC-051)
- GIL extraction within workflow pipeline stages
- Cross-tier consistency (same transcript, different
  tiers should produce compatible outputs)

**E2E Tests:**

- Full pipeline with GIL extraction enabled
  (`--enable-gil`)
- CLI commands: `kg inspect`, `kg show-insight`,
  `kg explore` (RFC-050)
- `kg.json` output validation against schema
- Per-provider GIL extraction (ML, OpenAI, Ollama)

**Quality Evaluation:**

- Quote verbatim accuracy (% exact match with
  transcript spans)
- Grounding rate (% of insights with ≥1 supporting
  quote)
- Insight coverage (% of key topics covered)
- Cross-provider comparison using golden dataset

### Model Registry Testing (RFC-044, Planned)

- `ModelRegistry` initialization and model lookup
- `ModelCapabilities` validation (token limits,
  device defaults)
- Registry used by summarizer and GIL extractor
- Invalid model name error handling

### Hybrid ML Platform Testing (RFC-042, Planned)

- Hybrid MAP-REDUCE pipeline (LED map → FLAN-T5
  reduce)
- Extractive QA for quote extraction
- NLI for grounding validation
- Sentence embeddings for topic deduplication
- `StructuredExtractor` protocol compliance

### Performance Testing

- [ ] Benchmark large feed processing (1000+ episodes)
- [ ] Measure Whisper transcription performance
- [ ] Profile memory usage
- [ ] Test concurrent download limits
- [ ] GIL extraction latency per tier (planned)

### Property-Based Testing

- [ ] Generate random RSS feeds
- [ ] Test filename sanitization with fuzzing
- [ ] Test URL normalization with edge cases
- [ ] `kg.json` schema fuzzing (planned)

## References

- **[Critical Path Testing Guide](guides/CRITICAL_PATH_TESTING_GUIDE.md)**
  — What to test, prioritization
- **[Testing Guide](guides/TESTING_GUIDE.md)**
  — Test execution, fixtures, coverage
- **[Unit Testing Guide](guides/UNIT_TESTING_GUIDE.md)**
  — Mocking patterns and isolation
- **[Integration Testing Guide](guides/INTEGRATION_TESTING_GUIDE.md)**
  — Integration test mocking guidelines
- **[E2E Testing Guide](guides/E2E_TESTING_GUIDE.md)**
  — E2E server, real ML models, API mocking
- Test structure reorganization:
  `docs/rfc/RFC-018-test-structure-reorganization.md`
- CI workflow: `.github/workflows/python-app.yml`
- Related RFCs: RFC-001–RFC-018 (testing strategies),
  RFC-029 (unified providers), RFC-017 (prompt store)
- Planned RFCs: RFC-042 (Hybrid ML), RFC-044 (Model
  Registry), RFC-049/050/051 (GIL)
- Related Issues: #14 (E2E testing), #16 (Library API
  E2E tests), #94 (src/ layout), #98 (Test structure
  reorganization)
- Architecture: `docs/ARCHITECTURE.md`
- Contributing guide: `CONTRIBUTING.md`

````
