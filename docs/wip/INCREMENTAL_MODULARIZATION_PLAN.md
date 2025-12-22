# Incremental Modularization Plan

**Status**: Draft  
**Related Documents**:

- `docs/wip/MODULARIZATION_REFACTORING_PLAN.md` - Overall refactoring strategy (baseline)
- `docs/prd/PRD-006-openai-provider-integration.md` - OpenAI provider product requirements
- `docs/prd/PRD-007-ai-experiment-pipeline.md` - AI experiment pipeline product requirements
- `docs/rfc/RFC-013-openai-provider-implementation.md` - OpenAI provider technical design
- `docs/rfc/RFC-015-ai-experiment-pipeline.md` - AI experiment pipeline technical design
- `docs/rfc/RFC-016-modularization-for-ai-experiments.md` - Provider system architecture
- `docs/rfc/RFC-017-prompt-management.md` - Prompt management implementation

## Overview

This document provides a **holistic, risk-balanced, incremental implementation plan** for:

1. **Modularizing the podcast scraper architecture** to support provider abstraction (OpenAI, local HF, etc.)
2. **Building the AI experiment pipeline** that enables rapid iteration on models, prompts, and parameters
3. **Implementing prompt management** for versioned, parameterized prompts

**Key Concept**: Think of the AI experiment pipeline exactly like your unit/integration test pipeline – just for models instead of code. The experiment pipeline wraps existing pieces (gold data, HF baseline, eval scripts) in a repeatable pipeline that sits next to your normal build/CI.

Each stage is **complete, tested, and fully working** before moving to the next. The plan integrates:

- **Provider modularization** (Stages 0-6) - Enables pluggable backends
- **Prompt management** (Stage 7) - Versioned, parameterized prompts
- **AI experiment pipeline** (Stages 8-10) - Configuration-driven experimentation

**Core Principles**:

1. ✅ Each stage delivers working functionality
2. ✅ Each stage is fully tested before proceeding
3. ✅ Backward compatibility maintained at every step
4. ✅ Incremental risk reduction (start with lowest risk)
5. ✅ Build on previous stages (no rework)

---

## Stage 0: Foundation & Preparation

**Goal**: Set up infrastructure with zero risk to existing functionality

**Duration**: 1-2 days  
**Risk Level**: ⚪ Very Low (no code changes, only additions)

### Deliverables

1. **Create package structure** (empty packages, no imports yet):

   ```text
   podcast_scraper/
   ├── preprocessing.py         # NEW (empty for now)
   ├── speaker_detectors/
   │   ├── __init__.py
   │   ├── base.py              # NEW (Protocol definitions only)
   │   └── factory.py           # NEW (empty factory)
   ├── transcription/
   │   ├── __init__.py
   │   ├── base.py              # NEW (Protocol definitions only)
   │   └── factory.py           # NEW (empty factory)
   └── summarization/
       ├── __init__.py
       ├── base.py              # NEW (Protocol definitions only)
       └── factory.py           # NEW (empty factory)
   ```

2. **Add config fields** (backward compatible defaults):

   ```python
   # config.py - Add new fields with defaults matching current behavior
   speaker_detector_type: Literal["ner", "openai"] = Field(default="ner")
   transcription_provider: Literal["whisper", "openai"] = Field(default="whisper")
   summary_provider: Literal["transformers", "openai"] = Field(default="transformers")
   openai_api_key: Optional[str] = Field(default=None)
   ```

3. **Create Protocol definitions** (no implementations yet):
   - `SpeakerDetector` protocol in `speaker_detectors/base.py`
   - `TranscriptionProvider` protocol in `transcription/base.py`
   - `SummarizationProvider` protocol in `summarization/base.py`

### Tests

- ✅ All existing tests pass (no regressions)
- ✅ Config validation tests for new fields
- ✅ Protocol type checking tests (verify protocols are valid)
- ✅ Import tests (verify new packages can be imported)

### Risk Mitigation

- **Risk**: New packages might cause import issues
  - **Mitigation**: Empty `__init__.py` files, no imports in workflow yet
- **Risk**: Config changes might break existing configs
  - **Mitigation**: All new fields have defaults matching current behavior

### Success Criteria

- ✅ New packages exist and can be imported
- ✅ Protocols are defined and type-checkable
- ✅ Config accepts new fields with defaults
- ✅ All existing tests pass
- ✅ No changes to existing functionality

---

## Stage 1: Extract Preprocessing Module

**Goal**: Extract provider-agnostic preprocessing to shared module

**Duration**: 1-2 days  
**Risk Level**: 🟢 Low (isolated refactoring, easy to test)

### Stage 1 Deliverables

1. **Create `preprocessing.py` module**:
   - Move `clean_transcript()` from `summarizer.py`
   - Move `remove_sponsor_blocks()` from `summarizer.py`
   - Move `clean_for_summarization()` from `summarizer.py`
   - Keep function signatures identical (backward compatible)

2. **Update imports**:
   - Update `metadata.py` to import from `preprocessing.py`
   - Update `summarizer.py` to import from `preprocessing.py` (for backward compatibility)
   - Keep `summarizer.py` functions as wrappers initially (deprecation path)

3. **Add deprecation warnings** (optional, for future cleanup):
   - Add deprecation warnings in `summarizer.py` wrapper functions

### Stage 1 Tests

- ✅ Unit tests for each preprocessing function (moved from summarizer tests)
- ✅ Integration tests: `metadata.py` produces identical results
- ✅ Backward compatibility tests: `summarizer.py` functions still work
- ✅ Test with various transcript formats (timestamps, speakers, sponsors)

### Stage 1 Risk Mitigation

- **Risk**: Function behavior might change during move
  - **Mitigation**: Copy-paste exact code, add tests before refactoring
- **Risk**: Import paths might break
  - **Mitigation**: Keep wrapper functions in `summarizer.py` initially

### Stage 1 Success Criteria

- ✅ Preprocessing functions work identically in new location
- ✅ All existing tests pass
- ✅ `metadata.py` uses new preprocessing module
- ✅ No changes to output or behavior

---

## Stage 2: Transcription Provider Abstraction

**Goal**: Refactor transcription to use provider pattern (lowest coupling, easiest first)

**Duration**: 2-3 days  
**Risk Level**: 🟡 Medium (refactoring core functionality)

### Stage 2 Deliverables

1. **Create `WhisperTranscriptionProvider`**:
   - Move `whisper_integration.py` logic to `transcription/whisper_provider.py`
   - Implement `TranscriptionProvider` protocol
   - Wrap existing functions as methods:
     - `initialize()` - Load Whisper model
     - `transcribe()` - Call `transcribe_with_whisper()`
     - `cleanup()` - Unload model

2. **Create factory**:
   - `TranscriptionProviderFactory.create()` returns `WhisperTranscriptionProvider` for `"whisper"`
   - Factory reads `transcription_provider` config field

3. **Update `workflow.py`**:
   - Replace direct `whisper_integration` imports with factory
   - Use provider pattern for transcription
   - Keep `_TranscriptionResources` but update to use provider

4. **Update `episode_processor.py`** (if exists):
   - Use provider instead of direct Whisper calls

### Stage 2 Tests

- ✅ Unit tests for `WhisperTranscriptionProvider` (mock Whisper)
- ✅ Protocol compliance tests (verify implements `TranscriptionProvider`)
- ✅ Integration tests: Transcription produces identical results
- ✅ Factory tests: Factory returns correct provider
- ✅ End-to-end tests: Full workflow with transcription works

### Stage 2 Risk Mitigation

- **Risk**: Transcription might break during refactoring
  - **Mitigation**: Copy-paste exact logic, test thoroughly before switching
- **Risk**: Resource management might leak
  - **Mitigation**: Test cleanup() is called, verify no memory leaks

### Stage 2 Success Criteria

- ✅ Transcription works identically via provider
- ✅ All existing transcription tests pass
- ✅ Factory pattern works correctly
- ✅ No memory leaks or resource issues
- ✅ Backward compatible (default behavior unchanged)

---

## Stage 3: Speaker Detection Provider Abstraction

**Goal**: Refactor speaker detection to use provider pattern

**Duration**: 2-3 days  
**Risk Level**: 🟡 Medium (moderate coupling, well-isolated)

### Stage 3 Deliverables

1. **Refactor `speaker_detection.py` → `speaker_detectors/ner_detector.py`**:
   - Extract helper functions from large functions:
     - `_calculate_heuristic_score()` from `detect_speaker_names()`
     - `_build_guest_candidates()` from `detect_speaker_names()`
     - `_select_best_guest()` from `detect_speaker_names()`
     - `_extract_entities_from_text()` from `extract_person_entities()`
     - `_extract_entities_from_segments()` from `extract_person_entities()`
     - `_pattern_based_fallback()` from `extract_person_entities()`
   - Implement `SpeakerDetector` protocol
   - Wrap existing functions as methods:
     - `detect_hosts()` - Call `detect_hosts_from_feed()`
     - `detect_speakers()` - Call `detect_speaker_names()`
     - `analyze_patterns()` - Call pattern analysis functions

2. **Create factory**:
   - `SpeakerDetectorFactory.create()` returns `NERSpeakerDetector` for `"ner"`
   - Factory reads `speaker_detector_provider` config field (renamed from speaker_detector_type)

3. **Update `workflow.py`**:
   - Replace direct `speaker_detection` imports with factory
   - Use provider pattern for speaker detection

### Stage 3 Tests

- ✅ Unit tests for `NERSpeakerDetector` (mock spaCy)
- ✅ Protocol compliance tests (verify implements `SpeakerDetector`)
- ✅ Integration tests: Speaker detection produces identical results
- ✅ Factory tests: Factory returns correct provider
- ✅ End-to-end tests: Full workflow with speaker detection works
- ✅ Test extracted helper functions independently

### Stage 3 Risk Mitigation

- **Risk**: Speaker detection logic might break during extraction
  - **Mitigation**: Extract functions incrementally, test after each extraction
- **Risk**: Large functions might be hard to refactor
  - **Mitigation**: Extract helper functions first, then wrap in protocol

### Stage 3 Success Criteria

- ✅ Speaker detection works identically via provider
- ✅ All existing speaker detection tests pass
- ✅ Helper functions are testable independently
- ✅ Factory pattern works correctly
- ✅ Code is more maintainable (smaller functions)

---

## Stage 4: Summarization Provider Abstraction

**Goal**: Refactor summarization to use provider pattern (most complex, done last)

**Duration**: 3-4 days  
**Risk Level**: 🟠 Medium-High (most coupling, complex logic)

### Stage 4 Deliverables

1. **Refactor `summarizer.py` → `summarization/local_provider.py`**:
   - Move `SummaryModel` class to `TransformersSummarizationProvider`
   - Implement `SummarizationProvider` protocol
   - Wrap existing methods:
     - `initialize()` - Load model (current `__init__` logic)
     - `summarize()` - Call `generate_summary()` for single text
     - `summarize_chunks()` - Call `generate_summary()` for chunks (MAP phase)
     - `combine_summaries()` - Call `generate_summary()` for final combine (REDUCE phase)
     - `cleanup()` - Unload model (current cleanup logic)

2. **Create factory**:
   - `SummarizationProviderFactory.create()` returns `TransformersSummarizationProvider` for `"transformers"`
   - Factory reads `summary_provider` config field

3. **Update `workflow.py`**:
   - Replace direct `summarizer` imports with factory
   - Use provider pattern for summarization
   - Pass provider to `metadata.py` functions

4. **Update `metadata.py`**:
   - Refactor `_generate_episode_summary()` to use provider
   - Use provider's `summarize_chunks()` and `combine_summaries()` methods
   - Remove direct `summarizer` imports (use provider instead)

### Stage 4 Tests

- ✅ Unit tests for `LocalSummarizationProvider` (mock transformers)
- ✅ Protocol compliance tests (verify implements `SummarizationProvider`)
- ✅ Integration tests: Summarization produces identical results
- ✅ Factory tests: Factory returns correct provider
- ✅ End-to-end tests: Full workflow with summarization works
- ✅ Test MAP/REDUCE phases independently
- ✅ Test model loading/unloading (memory management)

### Stage 4 Risk Mitigation

- **Risk**: Model loading/unloading might break
  - **Mitigation**: Test memory management thoroughly, verify cleanup
- **Risk**: MAP/REDUCE logic might break during refactoring
  - **Mitigation**: Test each phase independently, verify end-to-end
- **Risk**: `metadata.py` refactoring might be complex
  - **Mitigation**: Refactor incrementally, test after each change

### Stage 4 Success Criteria

- ✅ Summarization works identically via provider
- ✅ All existing summarization tests pass
- ✅ MAP/REDUCE phases work correctly
- ✅ Factory pattern works correctly
- ✅ Memory management works (no leaks)
- ✅ `metadata.py` is cleaner (uses provider)

### Stage 4 Documentation Tasks

**Create:** `docs/wip/TESTING_STRATEGY_MODULARIZATION.md`

- Protocol compliance test examples
- Mock provider patterns
- Integration test requirements
- Performance benchmark baselines
- Example test cases for each provider type

**Purpose:** Support testing efforts in Stage 5 and Stage 6

---

## Stage 5: Provider Integration & Testing

**Goal**: Ensure all providers work together, comprehensive testing

**Duration**: 2-3 days  
**Risk Level**: 🟡 Medium (integration testing)

### Stage 5 Deliverables

1. **Integration tests**:
   - Test workflow with all providers (transcription, speaker detection, summarization)
   - Test provider switching (change config, verify behavior)
   - Test error handling (provider fails, verify graceful handling)

2. **Protocol compliance tests**:
   - Verify all providers implement protocols correctly
   - Test type checking (mypy compliance)

3. **Backward compatibility tests**:
   - Test default behavior (all local providers)
   - Test existing configs still work
   - Test existing CLI commands still work

4. **Performance tests**:
   - Compare performance (provider vs direct calls)
   - Verify no performance regression

5. **Documentation**:
   - Update docstrings for providers
   - Document provider interfaces
   - Add examples for each provider
   - **Create:** `docs/CUSTOM_PROVIDER_GUIDE.md` (for external contributors)
   - **Create:** `docs/ENVIRONMENT_VARIABLES.md` (before Stage 6)

### Stage 5 Tests

- ✅ Full pipeline integration tests
- ✅ Provider switching tests
- ✅ Error handling tests
- ✅ Backward compatibility tests
- ✅ Performance benchmarks
- ✅ Protocol compliance tests

### Stage 5 Risk Mitigation

- **Risk**: Integration issues might surface
  - **Mitigation**: Comprehensive integration tests, fix issues before proceeding
- **Risk**: Performance might degrade
  - **Mitigation**: Benchmark before/after, optimize if needed

### Stage 5 Success Criteria

- ✅ All providers work together correctly
- ✅ All integration tests pass
- ✅ No performance regression
- ✅ Backward compatibility maintained
- ✅ Documentation complete

### Stage 5 Documentation Tasks

**Create during Stage 5:**

1. **`docs/CUSTOM_PROVIDER_GUIDE.md`**
   - Step-by-step provider creation guide
   - Protocol interface documentation
   - Factory registration pattern
   - Testing requirements
   - Three example implementations (minimal, full-featured, custom config)
   - Pull request process

2. **`docs/ENVIRONMENT_VARIABLES.md`**
   - Complete list of supported environment variables
   - Usage examples (macOS, Linux, Windows, Docker)
   - Security best practices
   - Troubleshooting guide
   - Prepare for Stage 6 (OpenAI API keys)

**Purpose:** Enable external contributions and prepare for Stage 6 (OpenAI)

---

## Stage 6: OpenAI Provider Implementation (Optional - After Core Refactoring)

**Goal**: Add OpenAI providers for each capability

**Duration**: 3-5 days per provider (can be done incrementally)  
**Risk Level**: 🟡 Medium (new functionality, well-isolated)

### Prerequisites

- ✅ Stages 0-5 completed
- ✅ Provider pattern fully implemented
- ✅ All tests passing

### Implementation Order

1. **OpenAI Transcription Provider** (easiest, most isolated)
2. **OpenAI Speaker Detection Provider** (moderate complexity)
3. **OpenAI Summarization Provider** (most complex, leverages large context window)

### Deliverables (Per Provider)

1. **Create provider implementation**:
   - `transcription/openai_provider.py`
   - `speaker_detectors/openai_detector.py`
   - `summarization/openai_provider.py`

2. **Update factories**:
   - Add OpenAI provider to factory selection logic

3. **Add config validation**:
   - Validate API key when OpenAI provider selected (using `python-dotenv` for `.env` files)
   - Add per-provider model configuration
   - Use `prompt_store` (RFC-017) for prompt loading

4. **API Key Management**:
   - Use `python-dotenv` to load `.env` files automatically
   - Support `OPENAI_API_KEY` environment variable
   - Create `examples/.env.example` template
   - Add `.env` to `.gitignore`

5. **Tests**:
   - Unit tests with mocked OpenAI API
   - Integration tests with real API (optional, requires key)
   - Error handling tests (API failures, rate limits)

### Success Criteria (Per Provider)

- ✅ Provider implements protocol correctly
- ✅ All protocol tests pass
- ✅ Integration tests pass (with mocked API)
- ✅ Error handling works correctly
- ✅ API key management works (`.env` file support)
- ✅ Uses `prompt_store` for prompts (RFC-017)
- ✅ Documentation complete

**Reference**: See `docs/rfc/RFC-013-openai-provider-implementation.md` for detailed implementation design.

---

## Stage 7: Prompt Management System

**Goal**: Implement versioned, parameterized prompt management system

**Duration**: 2-3 days  
**Risk Level**: 🟢 Low (new functionality, well-isolated)

### Stage 7 Prerequisites

- ✅ Stage 6 completed (or can be done in parallel)
- ✅ OpenAI providers use prompts

### Stage 7 Deliverables

1. **Create `prompt_store.py` module**:
   - Load prompts from `prompts/` directory
   - Jinja2 templating support
   - Caching with `lru_cache`
   - SHA256 hashing for tracking
   - Functions: `render_prompt()`, `get_prompt_metadata()`, `get_prompt_source()`

2. **Create prompt directory structure**:

   ```text
   prompts/
   ├── summarization/
   │   ├── system_v1.j2
   │   ├── long_v1.j2
   │   └── short_v1.j2
   └── ner/
       ├── system_ner_v1.j2
       └── guest_host_v1.j2
   ```

3. **Update OpenAI providers**:
   - Use `prompt_store.render_prompt()` instead of hardcoded prompts
   - Track prompt metadata in results

4. **Create `experiment_config.py`**:
   - Pydantic models for experiment configs
   - YAML loader
   - Data discovery helpers
   - Prompt config models

5. **Tests**:
   - Unit tests for prompt loading and rendering
   - Template parameterization tests
   - Metadata tracking tests

### Stage 7 Success Criteria

- ✅ Prompts stored as versioned files
- ✅ Prompts support Jinja2 templating
- ✅ Prompt metadata tracked in results
- ✅ OpenAI providers use `prompt_store`
- ✅ Experiment config models ready

**Reference**: See `docs/rfc/RFC-017-prompt-management.md` for detailed implementation design.

---

## Stage 8: AI Experiment Pipeline - Foundation

**Goal**: Normalize existing structure and build generic runner

**Duration**: 3-4 days  
**Risk Level**: 🟡 Medium (new functionality, wraps existing pieces)

### Stage 8 Prerequisites

- ✅ Stage 7 completed (prompt management)
- ✅ Stages 0-6 completed (provider system)

### Stage 8 Deliverables

1. **Normalize data structure**:
   - Move gold data under `data/eval/episodes/*`
   - Ensure consistent episode structure
   - Document golden dataset format

2. **Establish baseline**:
   - Keep existing baseline as `results/summarization_bart_led_v1/metrics.json`
   - Create baseline experiment config file
   - Document baseline experiment

3. **Create generic `run_experiment.py`**:
   - Takes config path as input
   - Loads episodes listed in config
   - Calls appropriate backend (local HF or OpenAI API)
   - Writes predictions + metrics separately
   - Support episode filtering (e.g., `--episodes ep01`)

4. **Create `eval_experiment.py` wrapper**:
   - Bridges experiment output to existing eval scripts
   - Reuses `eval_summaries.py` logic
   - No changes to existing eval scripts required

5. **Tests**:
   - Test generic runner with baseline config
   - Test episode filtering
   - Test prediction generation
   - Test metrics computation

### Stage 8 Success Criteria

- ✅ Gold data normalized under `data/eval/episodes/*`
- ✅ Baseline experiment documented and config file created
- ✅ Generic runner works with baseline config
- ✅ Generates predictions.jsonl
- ✅ Can evaluate predictions using existing eval logic

**Reference**: See `docs/prd/PRD-007-ai-experiment-pipeline.md` and `docs/rfc/RFC-015-ai-experiment-pipeline.md` for detailed design.

---

## Stage 9: AI Experiment Pipeline - CI/CD Integration

**Goal**: Add two-layer CI/CD integration (smoke tests + full pipeline)

**Duration**: 2-3 days  
**Risk Level**: 🟡 Medium (CI/CD integration)

### Stage 9 Prerequisites

- ✅ Stage 8 completed (generic runner)

### Stage 9 Deliverables

1. **Layer A: CI Smoke Tests**:
   - GitHub Actions workflow for smoke tests
   - Runs on every push/PR
   - Uses tiny subset (e.g., `ep01` only)
   - Uses single baseline config
   - Asserts quality thresholds (e.g., `rougeL_f >= threshold`)
   - Asserts no errors, no NaNs, no missing fields

2. **Layer B: Full Eval Pipeline**:
   - Script that loops over all YAMLs in `experiments/*.yaml`
   - Runs them, writes results
   - Prints summary table
   - GitHub Actions workflow for full evaluation
   - Runs nightly or on-demand
   - Generates summary report

3. **Comparison tooling**:
   - Script to compare experiment results
   - Generate comparison reports (markdown, JSON)
   - Detect regressions

### Stage 9 Success Criteria

- ✅ Smoke tests run on every push/PR
- ✅ Asserts quality thresholds
- ✅ Catches breakages quickly
- ✅ Full eval script runs all experiments
- ✅ Generates summary report with metrics table
- ✅ Can compare experiments and detect regressions

**Reference**: See `docs/rfc/RFC-015-ai-experiment-pipeline.md` Section 7 for detailed CI/CD design.

---

## Stage 10: AI Experiment Pipeline - Comparison Tooling

**Goal**: Build comparison tool that creates Excel with all experiments and key metrics

**Duration**: 2-3 days  
**Risk Level**: 🟢 Low (new tooling, well-isolated)

### Stage 10 Prerequisites

- ✅ Stage 9 completed (CI/CD integration)

### Stage 10 Deliverables

1. **Build comparison tool**:
   - Reads all experiment results
   - Creates Excel workbook with all experiments
   - One tab per task type
   - Key metrics columns (ROUGE, precision, F1, etc.)

2. **Enable data-driven decisions**:
   - Answer "which model + prompt is best?" becomes a data question
   - Visual comparison tables
   - Trend analysis

3. **Integration**:
   - Integrate with full eval pipeline
   - Generate Excel workbook automatically
   - Update workbook on each run

### Stage 10 Success Criteria

- ✅ Comparison tool creates Excel workbook
- ✅ All experiments and key metrics visible
- ✅ Easy to compare and make data-driven decisions
- ✅ Integrated with full eval pipeline

**Reference**: See `docs/prd/PRD-007-ai-experiment-pipeline.md` FR5 for requirements.

---

## Risk Assessment Summary

| Stage | Risk Level | Mitigation Strategy |
| ----- | ---------- | -------------------- |
| **Provider Modularization** | | |
| Stage 0: Foundation | ⚪ Very Low | Empty packages, defaults match current behavior |
| Stage 1: Preprocessing | 🟢 Low | Isolated refactoring, easy to test |
| Stage 2: Transcription | 🟡 Medium | Well-isolated, copy-paste logic |
| Stage 3: Speaker Detection | 🟡 Medium | Extract incrementally, test frequently |
| Stage 4: Summarization | 🟠 Medium-High | Most complex, done last with experience |
| Stage 5: Integration | 🟡 Medium | Comprehensive testing |
| Stage 6: OpenAI | 🟡 Medium | Well-isolated, optional, can be incremental |
| **Prompt Management** | | |
| Stage 7: Prompt Management | 🟢 Low | New functionality, well-isolated |
| **AI Experiment Pipeline** | | |
| Stage 8: Foundation | 🟡 Medium | Wraps existing pieces, new functionality |
| Stage 9: CI/CD Integration | 🟡 Medium | CI/CD integration, well-tested patterns |
| Stage 10: Comparison Tooling | 🟢 Low | New tooling, well-isolated |

---

## Testing Strategy

### Unit Tests (Each Stage)

- Test individual functions/methods
- Mock external dependencies (spaCy, Whisper, transformers, OpenAI)
- Test error handling
- Test edge cases

### Integration Tests (Each Stage)

- Test provider with real dependencies (where feasible)
- Test workflow integration
- Test config handling
- Test resource management

### Protocol Compliance Tests (Stages 2-4)

- Verify providers implement protocols correctly
- Test type checking (mypy)
- Test interface contracts

### Backward Compatibility Tests (All Stages)

- Test default behavior unchanged
- Test existing configs still work
- Test existing CLI commands still work
- Test existing output format unchanged

### Performance Tests (Stage 5)

- Benchmark before/after refactoring
- Verify no performance regression
- Test memory usage

### Experiment Pipeline Tests (Stages 8-10)

- Test generic runner with various configs
- Test episode filtering
- Test prediction generation
- Test metrics computation
- Test CI/CD workflows (smoke tests and full pipeline)
- Test comparison tooling

---

## Success Metrics

### Code Quality

- ✅ All existing tests pass (no regressions)
- ✅ New tests added for each stage
- ✅ Code coverage maintained or improved
- ✅ Type checking passes (mypy)
- ✅ Linting passes (flake8, black)

### Functionality

- ✅ All existing functionality works identically
- ✅ Provider pattern works correctly
- ✅ Factory pattern works correctly
- ✅ Backward compatibility maintained

### Maintainability

- ✅ Code is more modular (smaller functions)
- ✅ Clear separation of concerns
- ✅ Protocols are well-defined
- ✅ Documentation is complete

---

## Timeline Estimate

| Stage | Duration | Cumulative |
| ----- | -------- | ---------- |
| **Provider Modularization** | | |
| Stage 0: Foundation | 1-2 days | 1-2 days |
| Stage 1: Preprocessing | 1-2 days | 2-4 days |
| Stage 2: Transcription | 2-3 days | 4-7 days |
| Stage 3: Speaker Detection | 2-3 days | 6-10 days |
| Stage 4: Summarization | 3-4 days | 9-14 days |
| Stage 5: Integration | 2-3 days | 11-17 days |
| Stage 6: OpenAI (optional) | 3-5 days each | 14-32 days |
| **Prompt Management** | | |
| Stage 7: Prompt Management | 2-3 days | 16-35 days |
| **AI Experiment Pipeline** | | |
| Stage 8: Foundation | 3-4 days | 19-39 days |
| Stage 9: CI/CD Integration | 2-3 days | 21-42 days |
| Stage 10: Comparison Tooling | 2-3 days | 23-45 days |

**Total Core Refactoring**: ~11-17 days (2-3 weeks)  
**With OpenAI Providers**: ~14-32 days (3-6 weeks)  
**With Prompt Management**: ~16-35 days (3-7 weeks)  
**With Full AI Experiment Pipeline**: ~23-45 days (5-9 weeks)

**Note**: Stages can be done incrementally. The experiment pipeline (Stages 8-10) can be started once provider system (Stages 0-6) and prompt management (Stage 7) are in place.

---

## Dependencies Between Stages

```text
Provider Modularization:
  Stage 0 (Foundation)
    ↓
  Stage 1 (Preprocessing) - Independent
    ↓
  Stage 2 (Transcription) - Independent
    ↓
  Stage 3 (Speaker Detection) - Independent
    ↓
  Stage 4 (Summarization) - Uses preprocessing from Stage 1
    ↓
  Stage 5 (Integration) - Requires all Stages 1-4
    ↓
  Stage 6 (OpenAI) - Requires Stage 5 (optional)

Prompt Management:
  Stage 7 (Prompt Management) - Can be done in parallel with Stage 6

AI Experiment Pipeline:
  Stage 8 (Foundation) - Requires Stages 0-6 (provider system) + Stage 7 (prompts)
    ↓
  Stage 9 (CI/CD Integration) - Requires Stage 8
    ↓
  Stage 10 (Comparison Tooling) - Requires Stage 9
```

**Note**:

- Stages 1-4 can be done in parallel after Stage 0, but sequential is recommended for risk management
- Stage 7 (Prompt Management) can be done in parallel with Stage 6 (OpenAI)
- Stages 8-10 (AI Experiment Pipeline) require provider system (Stages 0-6) and prompt management (Stage 7)

---

## Rollback Plan

If any stage fails or introduces issues:

1. **Immediate**: Revert to previous stage (git revert)
2. **Investigation**: Identify root cause
3. **Fix**: Address issue in isolation
4. **Re-test**: Verify fix works
5. **Continue**: Proceed to next stage

Each stage is designed to be independently revertible without affecting previous stages.

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Set up development branch**: `feature/modularization-refactoring`
3. **Start with Stage 0**: Foundation & Preparation
4. **Iterate**: Complete each stage fully before proceeding
5. **Document**: Update this plan with lessons learned

---

## Notes

- This plan prioritizes **risk reduction** and **incremental value delivery**
- Each stage can be **reviewed and approved** independently
- **OpenAI providers** can be added later (Stage 6) or in parallel by different developers
- **Prompt management** (Stage 7) can be done in parallel with OpenAI providers (Stage 6)
- **AI experiment pipeline** (Stages 8-10) wraps existing pieces - think of it like your test pipeline for models
- **Testing is critical** at each stage - don't proceed without passing tests
- **Backward compatibility** is maintained at every step
- The experiment pipeline enables rapid iteration on models/prompts without code changes - just config files

## Key Concepts

### Test Pipeline Analogy

The AI experiment pipeline is exactly like your unit/integration test pipeline – just for models instead of code:

- **Unit Tests** → **Smoke Tests** (Layer A): Fast sanity checks on every push/PR
- **Integration Tests** → **Full Eval Pipeline** (Layer B): Comprehensive evaluation nightly/on-demand
- **Test Files** → **Experiment Configs**: Configuration-driven, version-controlled
- **Test Results** → **Experiment Metrics**: Tracked, comparable, reproducible

### Wrapping Existing Pieces

The experiment pipeline doesn't rebuild - it wraps:

- Existing gold data → Normalized under `data/eval/episodes/*`
- Existing HF baseline → Baseline experiment config
- Existing eval scripts → Wrapped by `eval_experiment.py`
- Existing providers → Used by generic runner

### Configuration-Driven

Treat model + prompt + params as configuration:

- No hardcoded experiments in Python
- Config files define experiments (like GitHub Actions workflows)
- "Trying a different model or prompt" = adding another config file
