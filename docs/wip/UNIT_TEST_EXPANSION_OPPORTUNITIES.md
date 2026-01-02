# Unit Test Expansion Opportunities

**Generated:** 2026-01-01
**Current Coverage:** 49.96%
**Target Coverage:** >80% (per Testing Strategy)

## Executive Summary

The codebase has **significant opportunities** to expand unit test coverage. Current
coverage is **49.96%**, with several critical modules below 50% coverage. This analysis
identifies specific functions, classes, and modules that need unit tests, prioritized by
impact and testability.

## Coverage by Module

### Critical Gaps (0-40% Coverage)

| Module | Lines | Covered | Coverage | Priority |
| -------- | ------- | --------- | ---------- | ---------- |
| `experiment_config.py` | 68 | 0 | **0.00%** | Medium |
| `workflow.py` | 841 | 222 | **22.93%** | **High** |
| `whisper_integration.py` | 175 | 41 | **21.81%** | **High** |
| `episode_processor.py` | 303 | 79 | **33.72%** | **High** |
| `transcription/openai_provider.py` | 75 | 37 | **44.44%** | Medium |
| `speaker_detectors/openai_detector.py` | 129 | 64 | **45.86%** | Medium |
| `summarizer.py` | 732 | 366 | **46.99%** | **High** |
| `metadata.py` | 259 | 121 | **47.95%** | Medium |

### Moderate Gaps (50-70% Coverage)

| Module | Lines | Covered | Coverage | Priority |
| -------- | ------- | --------- | ---------- | ---------- |
| `rss_parser.py` | 299 | 98 | **62.13%** | Medium |
| `speaker_detection.py` | 422 | 135 | **63.53%** | Medium |

### Good Coverage (70%+)

| Module | Lines | Covered | Coverage | Status |
| -------- | ------- | --------- | ---------- | -------- |
| `filesystem.py` | 99 | 15 | **81.40%** | ✅ Good |
| `preprocessing.py` | 45 | 0 | **98.46%** | ✅ Excellent |
| `prompt_store.py` | 59 | 4 | **89.33%** | ✅ Good |
| `service.py` | 51 | 1 | **94.74%** | ✅ Excellent |
| `transcription/whisper_provider.py` | 52 | 9 | **80.30%** | ✅ Good |
| `summarization/local_provider.py` | 76 | 11 | **81.52%** | ✅ Good |

## High-Priority Expansion Opportunities

### 1. `experiment_config.py` (0% Coverage) - **NEW MODULE**

**Status:** No unit tests exist for this module.

**Functions/Classes to Test:**

- `PromptConfig` - Pydantic model validation
- `HFBackendConfig` - Pydantic model validation
- `OpenAIBackendConfig` - Pydantic model validation
- `DataConfig` - Pydantic model validation
- `ExperimentParams` - Field validation, `collect_extra` validator
- `ExperimentConfig` - Full config validation, `ensure_non_empty_id` validator
- `load_experiment_config()` - YAML loading, file validation, error handling
- `discover_input_files()` - Glob pattern matching, file filtering
- `episode_id_from_path()` - Path-to-ID conversion logic

**Estimated Tests:** 15-20 tests
**Estimated Effort:** 2-3 hours
**Testability:** High (pure functions, Pydantic models)

**Example Test Cases:**

```python
def test_prompt_config_validation():
    """Test PromptConfig field validation."""
    # Valid config
    cfg = PromptConfig(user="summarization/system_v1")
    assert cfg.user == "summarization/system_v1"
    assert cfg.system is None
    assert cfg.params == {}

    # With params
    cfg = PromptConfig(
        user="summarization/long_v1",
        params={"paragraphs_min": 3}
    )
    assert cfg.params["paragraphs_min"] == 3

def test_load_experiment_config_success():
    """Test loading valid experiment config."""
    # Create temp YAML file
    # Load and validate

```text

    # Check all fields are populated correctly

```python
def test_load_experiment_config_missing_file():

```text

    """Test error handling for missing file."""
    with pytest.raises(FileNotFoundError):
        load_experiment_config("nonexistent.yaml")

```python
def test_discover_input_files():

```text

    """Test file discovery with glob patterns."""
    # Create temp directory with test files
    # Test glob matching
    # Test file filtering (only files, not dirs)

```python

def test_episode_id_from_path_stem():

```python

    """Test episode ID extraction from file stem."""
    path = Path("data/episodes/ep01/transcript.txt")
    cfg = DataConfig(episodes_glob="*.txt", id_from="stem")
    assert episode_id_from_path(path, cfg) == "transcript"

```python

def test_episode_id_from_path_parent_dir():

```python

    """Test episode ID extraction from parent directory."""
    path = Path("data/episodes/ep01/transcript.txt")
    cfg = DataConfig(episodes_glob="*.txt", id_from="parent_dir")
    assert episode_id_from_path(path, cfg) == "ep01"

```
### 2. `workflow.py` (22.93% Coverage) - **CRITICAL**

**Status:** Only helper functions are tested (`test_workflow_helpers.py`). Main
pipeline and many internal functions lack coverage.

**Functions NOT Tested (High Priority):**
- `_setup_pipeline_environment()` - Partially tested, needs more edge cases
- `_detect_feed_hosts_and_patterns()` - **0% coverage** - Complex logic, high impact
- `_setup_transcription_resources()` - **0% coverage** - Resource initialization
- `_setup_processing_resources()` - **0% coverage** - Queue setup
- `_prepare_episode_download_args()` - **0% coverage** - Argument preparation
- `_process_episodes()` - **0% coverage** - Core episode processing logic
- `_process_transcription_jobs()` - **0% coverage** - Sequential transcription
- `_process_transcription_jobs_concurrent()` - **0% coverage** - Concurrent transcription
- `_process_processing_jobs_concurrent()` - **0% coverage** - Concurrent metadata/summarization
- `_cleanup_pipeline()` - **0% coverage** - Cleanup logic
- `_generate_episode_metadata()` - **0% coverage** - Metadata generation call
- `_parallel_episode_summarization()` - **0% coverage** - Parallel summarization
- `_summarize_single_episode()` - **0% coverage** - Single episode summarization

**Functions Partially Tested:**
- `_fetch_and_parse_feed()` - Basic tests exist, needs error cases
- `_extract_feed_metadata_for_generation()` - Basic tests exist, needs edge cases
- `_prepare_episodes_from_feed()` - Basic tests exist, needs more scenarios
- `_generate_pipeline_summary()` - Basic tests exist, needs more metrics scenarios

**Estimated Tests:** 40-60 tests
**Estimated Effort:** 12-16 hours
**Testability:** Medium-High (requires mocking providers, HTTP, file I/O)

**Priority Test Cases:**

1. **`_detect_feed_hosts_and_patterns()`** (High Impact):

   ```python
   def test_detect_hosts_from_rss_authors():

```python

       """Test host detection from RSS author tags."""

```python
   def test_detect_hosts_fallback_to_ner():

```text

       """Test fallback to NER when no author tags."""

```python
   def test_detect_hosts_with_auto_speakers_disabled():

```text

       """Test behavior when auto_speakers is False."""

```python
   def test_analyze_patterns_with_multiple_episodes():

```text

       """Test pattern analysis across episodes."""

```python

2. **`_setup_transcription_resources()`**:

   ```python
   def test_setup_transcription_resources_with_whisper():

```text

       """Test transcription resource setup with Whisper enabled."""

```python
   def test_setup_transcription_resources_without_whisper():

```text

       """Test setup when transcription is disabled."""

```python
   def test_setup_transcription_resources_with_openai():

```text

       """Test setup with OpenAI transcription provider."""

```python

3. **`_process_episodes()`**:

   ```python
   def test_process_episodes_download_transcripts():

```text

       """Test episode processing with transcript downloads."""

```python
   def test_process_episodes_queue_for_transcription():

```text

       """Test episode processing queuing for transcription."""

```python
   def test_process_episodes_with_skip_existing():

```text

       """Test skip_existing flag behavior."""

```

### 3. `episode_processor.py` (33.72% Coverage) - **CRITICAL**

**Status:** Many helper functions are not tested.

**Functions NOT Tested:**
- `derive_media_extension()` - **0% coverage** - Extension derivation logic
- `derive_transcript_extension()` - **0% coverage** - Extension derivation logic
- `download_media_for_transcription()` - **0% coverage** - Media download
- `_format_transcript_if_needed()` - **0% coverage** - Transcript formatting
- `_save_transcript_file()` - **0% coverage** - File saving logic
- `_cleanup_temp_media()` - **0% coverage** - Cleanup logic
- `transcribe_media_to_text()` - **0% coverage** - Transcription orchestration
- `_determine_output_path()` - **0% coverage** - Path determination
- `_check_existing_transcript()` - **0% coverage** - File existence checks
- `_fetch_transcript_content()` - **0% coverage** - HTTP fetching
- `_write_transcript_file()` - **0% coverage** - File writing
- `process_transcript_download()` - **0% coverage** - Download orchestration
- `process_episode_download()` - **0% coverage** - Main entry point

**Estimated Tests:** 30-40 tests
**Estimated Effort:** 8-12 hours
**Testability:** Medium (requires mocking HTTP, file I/O, providers)

**Priority Test Cases:**

1. **Extension Derivation** (Pure Functions - Easy):

   ```python
   def test_derive_media_extension_from_type():

```python

       """Test extension derivation from media type."""
       assert derive_media_extension("audio/mpeg", "") == ".mp3"
       assert derive_media_extension("audio/mp4", "") == ".m4a"

```python
   def test_derive_media_extension_from_url():

```python

       """Test extension derivation from URL."""
       assert derive_media_extension(None, "https://example.com/audio.mp3") == ".mp3"
       assert derive_media_extension(None, "https://example.com/audio.m4a") == ".m4a"

```python
   def test_derive_transcript_extension():

```text

       """Test transcript extension derivation."""
       assert derive_transcript_extension("text/vtt", "") == ".vtt"
       assert derive_transcript_extension("text/srt", "") == ".srt"

```
```python

2. **Path Determination** (Pure Functions - Easy):
   ```python

   def test_determine_output_path_basic():
       """Test basic output path determination."""

   def test_determine_output_path_with_run_suffix():
       """Test output path with run suffix."""

   def test_determine_output_path_with_whisper_suffix():
       """Test output path with Whisper model suffix."""

```python

3. **File Operations** (Requires Mocking):
   ```python

   def test_check_existing_transcript_exists():
       """Test checking for existing transcript file."""

   def test_check_existing_transcript_not_exists():
       """Test when transcript doesn't exist."""

   def test_write_transcript_file():
       """Test writing transcript to file."""

```

### 4. `whisper_integration.py` (21.81% Coverage) - **HIGH PRIORITY**

**Status:** Core Whisper integration functions are not tested.

**Functions NOT Tested:**
- `load_whisper_model()` - **0% coverage** - Model loading
- `transcribe_with_whisper()` - **0% coverage** - Transcription
- `format_screenplay_from_segments()` - **0% coverage** - Screenplay formatting
- `_select_whisper_model()` - **0% coverage** - Model selection logic
- `_normalize_language_code()` - **0% coverage** - Language normalization

**Estimated Tests:** 20-25 tests
**Estimated Effort:** 6-8 hours
**Testability:** Medium (requires mocking Whisper library)

**Priority Test Cases:**

```python

def test_load_whisper_model_success():
    """Test successful model loading."""
    # Mock whisper.load_model()
    # Verify model is loaded correctly

def test_load_whisper_model_missing_dependency():
    """Test error handling when Whisper is not installed."""
    # Mock ImportError
    # Verify graceful error handling

def test_transcribe_with_whisper():
    """Test Whisper transcription."""
    # Mock whisper model and transcribe()
    # Verify transcription result

def test_format_screenplay_from_segments():
    """Test screenplay formatting."""
    # Test with speaker names

```text

    # Test without speaker names

```
```text

    # Test gap handling
    # Test speaker rotation

```python

def test_select_whisper_model_prefers_en_variant():
    """Test model selection prefers .en variants for English."""
    assert _select_whisper_model("en", "base") == "base.en"
    assert _select_whisper_model("en", "small") == "small.en"

def test_normalize_language_code():
    """Test language code normalization."""
    assert _normalize_language_code("en-US") == "en"
    assert _normalize_language_code("EN") == "en"

```

### 5. `summarizer.py` (46.99% Coverage) - **HIGH PRIORITY**

**Status:** Many core functions are tested, but complex logic paths are not.

**Functions/Areas NOT Tested:**
- `select_summary_model()` - Edge cases, model selection logic
- `select_reduce_model()` - Edge cases, model selection logic
- `summarize_long_text()` - Complex chunking/reduction logic
- `_chunk_text_for_summarization()` - Token-based chunking edge cases
- `_combine_summaries()` - Combination logic, edge cases
- `_decide_reduce_strategy()` - Decision logic for reduce phase
- `_mini_map_reduce()` - Mini map-reduce implementation
- `_extractive_fallback()` - Extractive summarization fallback
- Model loading/unloading edge cases
- Memory optimization edge cases

**Estimated Tests:** 30-40 tests
**Estimated Effort:** 10-14 hours
**Testability:** Medium (requires mocking transformers, but many pure functions)

**Note:** 18 tests are currently skipped due to model loading issues. These should be moved to integration tests or properly mocked.

### 6. `metadata.py` (47.95% Coverage) - **MEDIUM PRIORITY**

**Status:** Basic metadata generation is tested, but complex scenarios are not.

**Functions/Areas NOT Tested:**
- `generate_episode_metadata()` - Complex metadata generation scenarios
- Summary integration in metadata
- Multiple transcript sources
- Error handling in metadata generation
- YAML output formatting
- JSON output formatting

**Estimated Tests:** 15-20 tests
**Estimated Effort:** 4-6 hours
**Testability:** Medium (requires mocking providers, file I/O)

### 7. Provider Tests - **MEDIUM PRIORITY**

**OpenAI Providers:**
- `transcription/openai_provider.py` (44.44% coverage)
- `speaker_detectors/openai_detector.py` (45.86% coverage)
- `summarization/openai_provider.py` (72.94% coverage)

**Missing Test Cases:**
- Error handling (API failures, rate limits, timeouts)
- Custom base URL handling (E2E testing scenarios)
- Request/response validation
- Edge cases in API response parsing

**Estimated Tests:** 20-30 tests
**Estimated Effort:** 6-8 hours
**Testability:** High (can mock OpenAI client)

## Medium-Priority Expansion Opportunities

### 8. `rss_parser.py` (62.13% Coverage)

**Missing Coverage:**
- Complex namespace handling scenarios
- Relative URL resolution edge cases
- Missing field handling
- Malformed XML error handling
- Episode metadata extraction edge cases

**Estimated Tests:** 15-20 tests
**Estimated Effort:** 4-6 hours

### 9. `speaker_detection.py` (63.53% Coverage)

**Missing Coverage:**
- Pattern analysis logic
- Heuristics application
- Cache management
- Fallback behavior
- Multi-language support

**Note:** 5 tests are skipped due to spaCy mocking issues. These should be fixed.

**Estimated Tests:** 15-20 tests
**Estimated Effort:** 4-6 hours

## Low-Priority / Maintenance

### 10. Fix Skipped Tests (22 tests)

**Categories:**
- **Model Loading Issues (18 tests)**: `test_summarizer.py` - Tests that create `SummaryModel` instances
- **spaCy Mocking Issues (5 tests)**: `test_speaker_detection.py`, `test_utilities.py`
- **Whisper Mocking Issues (4 tests)**: `test_utilities.py`

**Recommendation:**
- Move model-dependent tests to integration tests
- Fix spaCy mocking setup
- Fix Whisper mocking setup

**Estimated Effort:** 4-6 hours

## Test Organization Recommendations

### New Test Files to Create

1. **`tests/unit/podcast_scraper/test_experiment_config.py`** (NEW)
   - Test all `experiment_config.py` functions and classes

2. **`tests/unit/podcast_scraper/test_workflow_core.py`** (NEW)
   - Test core workflow functions not covered by `test_workflow_helpers.py`
   - Focus on `_detect_feed_hosts_and_patterns`, `_setup_*_resources`, `_process_*` functions

3. **`tests/unit/podcast_scraper/test_episode_processor_helpers.py`** (NEW)
   - Test pure helper functions (extension derivation, path determination)
   - Separate from integration-style tests

4. **`tests/unit/podcast_scraper/test_whisper_integration.py`** (EXPAND)
   - Add tests for missing functions
   - Fix mocking approach

5. **`tests/unit/podcast_scraper/test_metrics.py`** (EXPAND)
   - Add tests for `Metrics` class methods
   - Test `finish()`, `log_metrics()`, timing methods

## Implementation Strategy

### Phase 1: Quick Wins (High Testability, High Impact)
1. **`experiment_config.py`** - Pure functions, Pydantic models (2-3 hours)
2. **Extension derivation functions** in `episode_processor.py` (1-2 hours)
3. **Path determination functions** in `episode_processor.py` (1-2 hours)
4. **`Metrics` class methods** (1-2 hours)

**Total:** 5-9 hours, ~30-40 new tests

### Phase 2: Core Logic (Medium Testability, High Impact)
1. **`workflow.py` helper functions** - `_detect_feed_hosts_and_patterns`, `_setup_*_resources` (6-8 hours)
2. **`episode_processor.py` core functions** - File operations, download logic (6-8 hours)
3. **`whisper_integration.py`** - Model loading, transcription, formatting (6-8 hours)

**Total:** 18-24 hours, ~60-80 new tests

### Phase 3: Complex Logic (Lower Testability, High Impact)
1. **`workflow.py` orchestration** - `_process_episodes`, `_process_transcription_jobs`, etc. (8-10 hours)
2. **`summarizer.py` complex paths** - Chunking, reduction, fallback (10-14 hours)
3. **Provider error handling** - OpenAI providers, edge cases (6-8 hours)

**Total:** 24-32 hours, ~50-70 new tests

### Phase 4: Fixes and Polish
1. **Fix skipped tests** - Move to integration or fix mocking (4-6 hours)
2. **Edge cases** - `rss_parser.py`, `speaker_detection.py`, `metadata.py` (8-12 hours)

**Total:** 12-18 hours, ~30-40 new tests

## Expected Outcomes

### Coverage Improvements

| Phase | New Tests | Coverage Increase | Total Coverage |
| ------- | ----------- | ------------------- | ---------------- |
| Phase 1 | 30-40 | +5-7% | ~55-57% |
| Phase 2 | 60-80 | +10-15% | ~65-72% |
| Phase 3 | 50-70 | +8-12% | ~73-84% |
| Phase 4 | 30-40 | +5-7% | **~78-91%** |

### Benefits

1. **Higher Confidence**: Better test coverage for critical paths
2. **Faster Debugging**: Unit tests catch issues early
3. **Better Refactoring Safety**: Tests ensure refactoring doesn't break functionality
4. **Documentation**: Tests serve as executable documentation
5. **CI Speed**: Unit tests run fast, providing quick feedback

## Recommendations

### Immediate Actions (This Sprint)

1. ✅ **Create `test_experiment_config.py`** - New module, high testability
2. ✅ **Add extension derivation tests** - Pure functions, easy to test
3. ✅ **Add path determination tests** - Pure functions, easy to test
4. ✅ **Expand `Metrics` tests** - Simple class, high value

### Short-Term (Next Sprint)

1. **Expand `workflow.py` tests** - Focus on helper functions first
2. **Expand `episode_processor.py` tests** - Focus on pure functions first
3. **Fix skipped tests** - Move model-dependent tests to integration

### Long-Term (Next Quarter)

1. **Complete `workflow.py` coverage** - All orchestration functions
2. **Complete `summarizer.py` coverage** - All complex logic paths
3. **Complete provider coverage** - All error scenarios

## Related Documentation

- [Testing Strategy](../TESTING_STRATEGY.md) - Overall testing approach
- [Testing Guide](../guides/TESTING_GUIDE.md) - Detailed test implementation
- [Provider Implementation Guide](../guides/PROVIDER_IMPLEMENTATION_GUIDE.md) - Provider testing patterns
