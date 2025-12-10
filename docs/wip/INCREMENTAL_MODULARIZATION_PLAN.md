# Incremental Modularization Plan

**Status**: Draft  
**Related Documents**:

- `docs/wip/MODULARIZATION_REFACTORING_PLAN.md` - Overall refactoring strategy
- `docs/prd/PRD-006-openai-provider-integration.md` - OpenAI provider requirements
- `docs/rfc/RFC-013-openai-provider-implementation.md` - OpenAI implementation design
- `docs/rfc/RFC-016-modularization-for-ai-experiments.md` - Code structure refactoring

## Overview

This document provides a **risk-balanced, incremental implementation plan** for modularizing the podcast scraper architecture to support OpenAI provider integration and the AI experiment pipeline. Each stage is **complete, tested, and fully working** before moving to the next.

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
   summary_provider: Literal["local", "openai"] = Field(default="local")
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
   - Factory reads `speaker_detector_type` config field

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
   - Move `SummaryModel` class to `LocalSummarizationProvider`
   - Implement `SummarizationProvider` protocol
   - Wrap existing methods:
     - `initialize()` - Load model (current `__init__` logic)
     - `summarize()` - Call `generate_summary()` for single text
     - `summarize_chunks()` - Call `generate_summary()` for chunks (MAP phase)
     - `combine_summaries()` - Call `generate_summary()` for final combine (REDUCE phase)
     - `cleanup()` - Unload model (current cleanup logic)

2. **Create factory**:
   - `SummarizationProviderFactory.create()` returns `LocalSummarizationProvider` for `"local"`
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
   - Validate API key when OpenAI provider selected
   - Add per-provider model configuration

4. **Tests**:
   - Unit tests with mocked OpenAI API
   - Integration tests with real API (optional, requires key)
   - Error handling tests (API failures, rate limits)

### Success Criteria (Per Provider)

- ✅ Provider implements protocol correctly
- ✅ All protocol tests pass
- ✅ Integration tests pass (with mocked API)
- ✅ Error handling works correctly
- ✅ Documentation complete

---

## Risk Assessment Summary

| Stage | Risk Level | Mitigation Strategy |
| ----- | ---------- | -------------------- |
| Stage 0: Foundation | ⚪ Very Low | Empty packages, defaults match current behavior |
| Stage 1: Preprocessing | 🟢 Low | Isolated refactoring, easy to test |
| Stage 2: Transcription | 🟡 Medium | Well-isolated, copy-paste logic |
| Stage 3: Speaker Detection | 🟡 Medium | Extract incrementally, test frequently |
| Stage 4: Summarization | 🟠 Medium-High | Most complex, done last with experience |
| Stage 5: Integration | 🟡 Medium | Comprehensive testing |
| Stage 6: OpenAI | 🟡 Medium | Well-isolated, optional, can be incremental |

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
| Stage 0: Foundation | 1-2 days | 1-2 days |
| Stage 1: Preprocessing | 1-2 days | 2-4 days |
| Stage 2: Transcription | 2-3 days | 4-7 days |
| Stage 3: Speaker Detection | 2-3 days | 6-10 days |
| Stage 4: Summarization | 3-4 days | 9-14 days |
| Stage 5: Integration | 2-3 days | 11-17 days |
| Stage 6: OpenAI (optional) | 3-5 days each | 14-32 days |

**Total Core Refactoring**: ~11-17 days (2-3 weeks)  
**With OpenAI Providers**: ~14-32 days (3-6 weeks)

---

## Dependencies Between Stages

```text
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
Stage 6 (OpenAI) - Requires Stage 5
```

**Note**: Stages 1-4 can be done in parallel after Stage 0, but sequential is recommended for risk management.

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
- **Testing is critical** at each stage - don't proceed without passing tests
- **Backward compatibility** is maintained at every step
