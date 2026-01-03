# Unit Test Coverage Expansion Plan

**Generated:** 2026-01-27
**Current Coverage:** 51.95%
**Target Coverage:** >80% (per Testing Strategy)
**Total Lines:** 5,989 | **Covered:** 3,340 | **Missing:** 2,649

## Executive Summary

This plan identifies **high-impact, high-testability** opportunities to expand unit test coverage from 51.95% to >80%. The plan is organized by priority, with specific functions and estimated effort for each module.

## Priority Tiers

### 🔴 **Tier 1: Critical Gaps (0-30% Coverage)**

High impact, often testable with mocks. These should be addressed first.

### 🟡 **Tier 2: Low Coverage (30-50% Coverage)**

Medium-high impact, may require more complex mocking.

### 🟢 **Tier 3: Moderate Coverage (50-70% Coverage)**

Fill remaining gaps, edge cases, error paths.

---

## Tier 1: Critical Gaps (0-30% Coverage)

### 1. `experiment_config.py` - **0% Coverage** ⭐ **QUICK WIN**

**Status:** No unit tests exist. Pure functions and Pydantic models - highly testable.

**Functions/Classes to Test:**
- `PromptConfig` - Field validation, optional params
- `HFBackendConfig` - Model name validation
- `OpenAIBackendConfig` - API key validation
- `DataConfig` - Glob patterns, ID extraction methods
- `ExperimentParams` - Field validation, `collect_extra` validator
- `ExperimentConfig` - Full config validation, `ensure_non_empty_id` validator
- `load_experiment_config()` - YAML loading, file validation, error handling
- `discover_input_files()` - Glob pattern matching, file filtering
- `episode_id_from_path()` - Path-to-ID conversion (stem, parent_dir, custom)

**Test File:** `tests/unit/podcast_scraper/test_experiment_config.py` (NEW)

**Estimated Tests:** 20-25 tests
**Estimated Effort:** 3-4 hours
**Testability:** ⭐⭐⭐⭐⭐ (Pure functions, Pydantic models)

**Example Test Cases:**

```python
def test_prompt_config_validation():
    """Test PromptConfig field validation."""
    cfg = PromptConfig(user="summarization/system_v1")
    assert cfg.user == "summarization/system_v1"
    assert cfg.system is None
    assert cfg.params == {}

def test_load_experiment_config_success():
    """Test loading valid experiment config from YAML."""
    # Create temp YAML file, load, validate all fields

def test_episode_id_from_path_stem():
    """Test episode ID extraction from file stem."""
    path = Path("data/episodes/ep01/transcript.txt")
    cfg = DataConfig(episodes_glob="*.txt", id_from="stem")
    assert episode_id_from_path(path, cfg) == "transcript"

def test_discover_input_files_glob_pattern():
    """Test file discovery with glob patterns."""
    # Test various glob patterns, filtering
```yaml

---

### 2. `workflow.py` - **22.35% Coverage** ⭐ **HIGH PRIORITY**

**Status:** Only helper functions tested. Core orchestration functions have 0% coverage.

**Functions NOT Tested (High Priority):**
- `_initialize_ml_environment()` - Environment variable setup
- `_update_metric_safely()` - Safe metric updates (error handling)
- `_detect_feed_hosts_and_patterns()` - **0% coverage** - Complex host detection logic
- `_setup_transcription_resources()` - **0% coverage** - Provider initialization
- `_setup_processing_resources()` - **0% coverage** - Queue setup
- `_prepare_episode_download_args()` - **0% coverage** - Argument preparation
- `_process_episodes()` - **0% coverage** - Core episode processing
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

**Test File:** `tests/unit/podcast_scraper/test_workflow_core.py` (NEW)

**Estimated Tests:** 40-50 tests
**Estimated Effort:** 12-16 hours
**Testability:** ⭐⭐⭐ (Requires extensive mocking of providers, filesystem, threading)

**Strategy:**
1. Start with pure helper functions (`_initialize_ml_environment`, `_update_metric_safely`)
2. Test resource setup functions with mocked providers
3. Test processing functions with mocked dependencies
4. Use `unittest.mock` extensively for providers, filesystem, threading

**Example Test Cases:**

```python
def test_initialize_ml_environment_sets_variables():
    """Test ML environment variable initialization."""
    with patch.dict(os.environ, {}, clear=True):
        _initialize_ml_environment()
        assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"

def test_detect_feed_hosts_and_patterns():
    """Test host detection from feed metadata."""
    feed = create_test_feed()
    feed_metadata = {"author": "John Doe"}
    result = _detect_feed_hosts_and_patterns(feed, feed_metadata, cfg, provider)
    assert result.hosts == ["John Doe"]

def test_setup_transcription_resources():
    """Test transcription provider initialization."""
    with patch("podcast_scraper.workflow.create_transcription_provider") as mock_create:
        provider = _setup_transcription_resources(cfg)
        mock_create.assert_called_once()
```yaml

---

### 3. `downloader.py` - **31.33% Coverage** ⭐ **HIGH PRIORITY**

**Status:** Core download functions have low coverage. Critical for reliability.

**Functions NOT Tested:**
- `normalize_url()` - URL normalization logic
- `_configure_http_session()` - Session configuration
- `_get_thread_request_session()` - Thread-local session management
- `_close_all_sessions()` - Session cleanup
- `_open_http_request()` - Low-level request handling
- `http_get()` - HTTP GET with retries, error handling
- `http_download_to_file()` - File download with progress

**Test File:** `tests/unit/podcast_scraper/test_downloader.py` (EXPAND)

**Estimated Tests:** 25-30 tests
**Estimated Effort:** 6-8 hours
**Testability:** ⭐⭐⭐⭐ (Mock `requests`, filesystem)

**Example Test Cases:**

```python
def test_normalize_url_removes_fragments():
    """Test URL normalization removes fragments."""
    assert normalize_url("https://example.com/path#fragment") == "https://example.com/path"

def test_http_get_success():
    """Test successful HTTP GET request."""
    with patch("podcast_scraper.downloader.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"test content"
        content, content_type = http_get("https://example.com", "test-agent", 10)
        assert content == b"test content"

def test_http_get_retries_on_failure():
    """Test HTTP GET retries on transient failures."""
    # Test retry logic with mocked requests

def test_http_download_to_file_progress():
    """Test file download with progress reporting."""
    # Mock requests, filesystem, progress reporter
```yaml

---

### 4. `episode_processor.py` - **33.33% Coverage** ⭐ **HIGH PRIORITY**

**Status:** Core episode processing logic has low coverage.

**Functions NOT Tested:**
- Extension derivation functions (`.mp3`, `.m4a`, etc.)
- Path determination functions
- `process_episode_download()` - Download logic, error handling
- `transcribe_media_to_text()` - Transcription orchestration
- File operation helpers

**Test File:** `tests/unit/podcast_scraper/test_episode_processor.py` (EXPAND)

**Estimated Tests:** 30-35 tests
**Estimated Effort:** 8-10 hours
**Testability:** ⭐⭐⭐ (Mock filesystem, transcription provider)

**Example Test Cases:**

```python
def test_derive_extension_from_url():
    """Test extension derivation from URL."""
    assert derive_extension("https://example.com/episode.mp3") == ".mp3"

def test_process_episode_download_success():
    """Test successful episode download."""
    # Mock downloader, filesystem, verify file creation

def test_transcribe_media_to_text():
    """Test media transcription."""
    # Mock transcription provider, verify transcript creation
```yaml

---

### 5. `whisper_integration.py` - **40.74% Coverage** ⭐ **MEDIUM-HIGH PRIORITY**

**Status:** Whisper model loading and transcription have low coverage.

**Functions NOT Tested:**
- `_import_third_party_whisper()` - Import handling
- `_intercept_whisper_progress()` - Progress reporting
- Model loading with cache
- Transcription with various models
- Error handling (model not found, transcription failures)

**Test File:** `tests/unit/podcast_scraper/test_whisper_integration.py` (EXPAND)

**Estimated Tests:** 20-25 tests
**Estimated Effort:** 6-8 hours
**Testability:** ⭐⭐⭐ (Mock Whisper library, filesystem)

---

## Tier 2: Low Coverage (30-50%)

### 6. `cache_utils.py` - **47.17% Coverage** ⭐ **QUICK WIN**

**Status:** Cache directory utilities have partial coverage.

**Functions NOT Tested:**
- `get_project_root()` - Project root detection (edge cases)
- `get_whisper_cache_dir()` - Cache path resolution (local vs home)
- `get_transformers_cache_dir()` - Cache path resolution
- `get_spacy_cache_dir()` - Cache path resolution

**Test File:** `tests/unit/podcast_scraper/test_cache_utils.py` (NEW)

**Estimated Tests:** 12-15 tests
**Estimated Effort:** 2-3 hours
**Testability:** ⭐⭐⭐⭐⭐ (Pure functions, mock filesystem)

**Example Test Cases:**

```python
def test_get_project_root_finds_pyproject_toml():
    """Test project root detection via pyproject.toml."""
    # Mock filesystem, test root detection

def test_get_whisper_cache_dir_prefers_local():
    """Test Whisper cache prefers local .cache/ directory."""
    # Mock local cache exists, verify it's returned

def test_get_whisper_cache_dir_falls_back_to_home():
    """Test Whisper cache falls back to ~/.cache/whisper."""
    # Mock local cache doesn't exist, verify home fallback
```yaml

---

### 7. `speaker_detectors/ner_detector.py` - **0% Coverage** ⭐ **MEDIUM PRIORITY**

**Status:** NER-based speaker detector has no unit tests.

**Class to Test:**
- `NERSpeakerDetector` - Initialization, detection, error handling

**Test File:** `tests/unit/podcast_scraper/speaker_detectors/test_ner_detector.py` (NEW)

**Estimated Tests:** 15-20 tests
**Estimated Effort:** 4-6 hours
**Testability:** ⭐⭐⭐ (Mock spaCy model)

---

### 8. `transcription/whisper_provider.py` - **0% Coverage** ⭐ **MEDIUM PRIORITY**

**Status:** Whisper transcription provider has no unit tests.

**Functions/Classes to Test:**
- `_import_third_party_whisper()` - Import handling
- `_intercept_whisper_progress()` - Progress reporting
- `WhisperTranscriptionProvider` - Initialization, transcription, error handling

**Test File:** `tests/unit/podcast_scraper/transcription/test_whisper_provider.py` (NEW)

**Estimated Tests:** 20-25 tests
**Estimated Effort:** 6-8 hours
**Testability:** ⭐⭐⭐ (Mock Whisper library)

---

### 9. `summarization/local_provider.py` - **0% Coverage** ⭐ **MEDIUM PRIORITY**

**Status:** Local Transformers summarization provider has no unit tests.

**Class to Test:**
- `TransformersSummarizationProvider` - Initialization, summarization, error handling

**Test File:** `tests/unit/podcast_scraper/summarization/test_local_provider.py` (NEW)

**Estimated Tests:** 15-20 tests
**Estimated Effort:** 4-6 hours
**Testability:** ⭐⭐⭐ (Mock Transformers models)

---

### 10. `metadata.py` - **47.95% Coverage** ⭐ **MEDIUM PRIORITY**

**Status:** Metadata generation has partial coverage. Missing edge cases and error paths.

**Functions NOT Tested:**
- Complex metadata generation scenarios
- Error handling paths
- Edge cases in ID generation
- File writing error handling

**Test File:** `tests/unit/podcast_scraper/test_metadata.py` (EXPAND)

**Estimated Tests:** 20-25 tests
**Estimated Effort:** 6-8 hours
**Testability:** ⭐⭐⭐⭐ (Mock filesystem, providers)

---

## Tier 3: Moderate Coverage (50-70%)

### 11. `cli.py` - **50.14% Coverage** ⭐ **MEDIUM PRIORITY**

**Status:** CLI argument parsing has partial coverage. Missing error cases and edge paths.

**Functions NOT Tested:**
- Complex argument combinations
- Error handling (invalid files, missing options)
- Config file loading edge cases
- Service mode handling

**Test File:** `tests/unit/podcast_scraper/test_cli.py` (EXPAND)

**Estimated Tests:** 15-20 tests
**Estimated Effort:** 4-6 hours
**Testability:** ⭐⭐⭐⭐ (Mock filesystem, sys.argv)

---

### 12. `config.py` - **61.81% Coverage** ⭐ **MEDIUM PRIORITY**

**Status:** Config validation has good coverage but missing edge cases.

**Functions NOT Tested:**
- Complex validation scenarios
- Field validator edge cases
- Environment variable handling
- Config file loading error paths

**Test File:** `tests/unit/podcast_scraper/test_config.py` (EXPAND)

**Estimated Tests:** 20-25 tests
**Estimated Effort:** 6-8 hours
**Testability:** ⭐⭐⭐⭐ (Pydantic models, mock environment)

---

### 13. `rss_parser.py` - **62.13% Coverage** ⭐ **LOW-MEDIUM PRIORITY**

**Status:** RSS parsing has decent coverage. Missing edge cases.

**Functions NOT Tested:**
- Malformed RSS feeds
- Edge cases in date parsing
- Missing optional fields
- Encoding issues

**Test File:** `tests/unit/podcast_scraper/test_rss_parser.py` (EXPAND)

**Estimated Tests:** 15-20 tests
**Estimated Effort:** 4-6 hours
**Testability:** ⭐⭐⭐⭐ (Mock XML content)

---

### 14. `speaker_detection.py` - **69.31% Coverage** ⭐ **LOW PRIORITY**

**Status:** Good coverage. Focus on remaining edge cases.

**Estimated Tests:** 10-15 tests
**Estimated Effort:** 3-4 hours
**Testability:** ⭐⭐⭐⭐

---

## Implementation Phases

### Phase 1: Quick Wins (High Testability, High Impact)

**Target:** +5-7% coverage | **Effort:** 15-20 hours

1. ✅ `experiment_config.py` (0% → 90%+) - 3-4 hours
2. ✅ `cache_utils.py` (47% → 90%+) - 2-3 hours
3. ✅ `downloader.py` helper functions - 3-4 hours
4. ✅ `workflow.py` helper functions (`_initialize_ml_environment`, `_update_metric_safely`) - 2-3 hours
5. ✅ `episode_processor.py` helper functions (extension, path) - 2-3 hours
6. ✅ `metadata.py` edge cases - 3-4 hours

**Expected Result:** Coverage ~57-59%

---

### Phase 2: Core Logic (Medium Testability, High Impact)

**Target:** +8-10% coverage | **Effort:** 30-40 hours

1. ✅ `workflow.py` core functions (`_detect_feed_hosts_and_patterns`, `_setup_*_resources`) - 8-10 hours
2. ✅ `downloader.py` core functions (`http_get`, `http_download_to_file`) - 4-6 hours
3. ✅ `episode_processor.py` core functions - 6-8 hours
4. ✅ `whisper_integration.py` - 6-8 hours
5. ✅ `speaker_detectors/ner_detector.py` - 4-6 hours
6. ✅ `transcription/whisper_provider.py` - 6-8 hours

**Expected Result:** Coverage ~65-69%

---

### Phase 3: Provider Tests (Medium Testability, Medium Impact)

**Target:** +5-7% coverage | **Effort:** 20-25 hours

1. ✅ `summarization/local_provider.py` - 4-6 hours
2. ✅ `transcription/openai_provider.py` edge cases - 4-6 hours
3. ✅ `speaker_detectors/openai_detector.py` edge cases - 4-6 hours
4. ✅ `workflow.py` processing functions (with extensive mocking) - 8-10 hours

**Expected Result:** Coverage ~70-76%

---

### Phase 4: Polish and Edge Cases (Lower Priority)

**Target:** +4-6% coverage | **Effort:** 15-20 hours

1. ✅ `cli.py` edge cases - 4-6 hours
2. ✅ `config.py` edge cases - 6-8 hours
3. ✅ `rss_parser.py` edge cases - 4-6 hours
4. ✅ `speaker_detection.py` remaining cases - 3-4 hours

**Expected Result:** Coverage ~74-82% ✅ **TARGET ACHIEVED**

---

## Test Organization

### New Test Files to Create

1. **`tests/unit/podcast_scraper/test_experiment_config.py`** (NEW)
   - All `experiment_config.py` functions and classes

2. **`tests/unit/podcast_scraper/test_cache_utils.py`** (NEW)
   - All `cache_utils.py` functions

3. **`tests/unit/podcast_scraper/test_workflow_core.py`** (NEW)
   - Core workflow functions not covered by `test_workflow_helpers.py`

4. **`tests/unit/podcast_scraper/speaker_detectors/test_ner_detector.py`** (NEW)
   - `NERSpeakerDetector` class

5. **`tests/unit/podcast_scraper/transcription/test_whisper_provider.py`** (NEW)
   - `WhisperTranscriptionProvider` class

6. **`tests/unit/podcast_scraper/summarization/test_local_provider.py`** (NEW)
   - `TransformersSummarizationProvider` class

### Test Files to Expand

1. **`tests/unit/podcast_scraper/test_downloader.py`** (EXPAND)
   - Add tests for missing functions

2. **`tests/unit/podcast_scraper/test_episode_processor.py`** (EXPAND)
   - Add tests for helper functions and core logic

3. **`tests/unit/podcast_scraper/test_whisper_integration.py`** (EXPAND)
   - Add tests for missing functions

4. **`tests/unit/podcast_scraper/test_metadata.py`** (EXPAND)
   - Add edge cases and error paths

5. **`tests/unit/podcast_scraper/test_cli.py`** (EXPAND)
   - Add error cases and edge paths

6. **`tests/unit/podcast_scraper/test_config.py`** (EXPAND)
   - Add validation edge cases

7. **`tests/unit/podcast_scraper/test_rss_parser.py`** (EXPAND)
   - Add malformed feed tests

---

## Testing Best Practices

### Mocking Strategy

1. **ML Models:** Always mock ML model loading in unit tests
   - Use `@patch` decorators for `spacy.load()`, `whisper.load_model()`, `transformers.AutoModel`
   - Mock model methods (`.predict()`, `.transcribe()`, etc.)

2. **Network Calls:** Always mock HTTP requests
   - Use `@patch("podcast_scraper.downloader.requests.get")`
   - Mock responses with appropriate status codes and content

3. **Filesystem:** Mock file operations when testing logic
   - Use `unittest.mock.mock_open()` for file reads/writes
   - Use `tempfile` for actual file operations when needed

4. **Threading:** Mock threading when testing concurrent logic
   - Use `@patch("threading.Thread")` or `ThreadPoolExecutor` mocks

### Test Structure

```python
class TestModuleName(unittest.TestCase):
    """Test module_name module."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # ... setup ...

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("module.dependency")
    def test_function_success(self, mock_dependency):
        """Test successful function execution."""
        # Arrange
        mock_dependency.return_value = expected_value

        # Act

```text

        result = function_under_test(input)

```

        # Assert
        self.assertEqual(result, expected_result)
        mock_dependency.assert_called_once_with(...)

```python

    def test_function_error_handling(self):
        """Test function error handling."""
        # Test error cases

```yaml

---

## Success Metrics

- **Coverage Target:** >80% overall coverage
- **Phase 1 Target:** 57-59% (+5-7%)
- **Phase 2 Target:** 65-69% (+8-10%)
- **Phase 3 Target:** 70-76% (+5-7%)
- **Phase 4 Target:** 74-82% (+4-6%)

---

## Notes

- All unit tests should run without ML dependencies (use mocks)
- All unit tests should run without network access (mock HTTP)
- All unit tests should be fast (<1 second each)
- Focus on testing logic, not implementation details
- Test error paths and edge cases, not just happy paths
- Use descriptive test names that explain what is being tested
