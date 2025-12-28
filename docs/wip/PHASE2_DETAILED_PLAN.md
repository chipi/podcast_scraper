# Phase 2: Add Missing Unit Tests - Detailed Plan

## Overview

**Goal:** Add ~150-200 new unit tests for untested core functions to reach 70-80% unit test coverage.

**Current State:** 356 unit tests (50.2%)  
**Target State:** 500-600 unit tests (70-80%)  
**Gap:** ~144-244 tests needed

## Strategy: Divide and Conquer

This plan breaks Phase 2 into **8 sub-phases**, each focusing on a specific module or component. Each sub-phase can be completed independently, making it easy to track progress and work incrementally.

---

## Sub-Phase 2.1: Preprocessing Functions (Quick Win) ⭐

**Priority:** High (Easy, High Impact)  
**Estimated Time:** 2-3 hours  
**Target Tests:** 15-20 tests  
**File:** `tests/unit/podcast_scraper/test_preprocessing.py`

### Functions to Test

1. **`clean_transcript()`** - 5-6 tests
   - Test timestamp removal (`[00:12:34]`, `[1:23:45]`, `[12:34]`)
   - Test generic speaker tag removal (preserves real names)
   - Test blank line collapsing
   - Test filler word removal (when enabled)
   - Test edge cases (empty string, no timestamps, no speakers)

2. **`remove_sponsor_blocks()`** - 3-4 tests
   - Test removal of "this episode is brought to you by"
   - Test removal of "today's episode is sponsored by"
   - Test multiple sponsor blocks
   - Test no sponsor blocks (no change)

3. **`remove_outro_blocks()`** - 3-4 tests
   - Test removal of "thank you so much for listening"
   - Test removal of "please rate/review/subscribe"
   - Test multiple outro patterns
   - Test no outro blocks (no change)

4. **`clean_for_summarization()`** - 3-4 tests
   - Test full pipeline (timestamps → sponsors → outros)
   - Test with various input formats
   - Test edge cases (empty, already clean)

### Implementation Steps

1. Create `tests/unit/podcast_scraper/test_preprocessing.py`
2. Import preprocessing module (mock ML deps if needed)
3. Write test cases for each function
4. Verify tests pass and are fast (< 100ms each)

**Success Criteria:**
- ✅ All 4 functions have unit tests
- ✅ Tests run in < 1 second total
- ✅ No network/filesystem I/O (unit test isolation)

---

## Sub-Phase 2.2: Filesystem Utilities (Quick Win) ⭐

**Priority:** High (Easy, High Impact)  
**Estimated Time:** 2-3 hours  
**Target Tests:** 12-18 tests  
**File:** `tests/unit/podcast_scraper/test_filesystem.py` (extend existing)

### Functions to Test

1. **`sanitize_filename()`** - 3-4 tests
   - Test special character replacement
   - Test newline/tab removal
   - Test empty string handling
   - Test edge cases (all special chars, unicode)

2. **`validate_and_normalize_output_dir()`** - 3-4 tests
   - Test valid paths (home, cwd, platformdirs)
   - Test invalid paths (empty, relative, outside safe roots)
   - Test path normalization (expanduser, resolve)
   - Test warning for paths outside safe roots

3. **`derive_output_dir()`** - 2-3 tests
   - Test override path
   - Test default derivation from RSS URL
   - Test URL hash generation

4. **`setup_output_directory()`** - 2-3 tests
   - Test with run_id
   - Test with auto run_id
   - Test with whisper model suffix
   - Test without run_id

5. **`truncate_whisper_title()`** - 2-3 tests
   - Test truncation at max_len
   - Test for_log vs for_log=False
   - Test short titles (no truncation)

6. **`build_whisper_output_name()`** - 2-3 tests
   - Test with run_suffix
   - Test without run_suffix
   - Test episode number formatting

7. **`build_whisper_output_path()`** - 1-2 tests
   - Test path construction
   - Test with/without run_suffix

### Implementation Steps

1. Review existing `test_filesystem.py` to see what's already covered
2. Add missing test cases for untested functions
3. Ensure all tests use mocks for filesystem operations (or tempfile)

**Success Criteria:**
- ✅ All 7 functions have unit tests
- ✅ Tests use tempfile or mocks (no real filesystem I/O)
- ✅ Tests run in < 1 second total

---

## Sub-Phase 2.3: Progress Reporting (Quick Win) ⭐

**Priority:** High (Easy, Isolated)  
**Estimated Time:** 1-2 hours  
**Target Tests:** 8-12 tests  
**File:** `tests/unit/podcast_scraper/test_progress.py` (new)

### Functions to Test

1. **`set_progress_factory()`** - 2-3 tests
   - Test setting custom factory
   - Test setting None (resets to noop)
   - Test factory replacement

2. **`progress_context()`** - 3-4 tests
   - Test with custom factory
   - Test with noop factory (default)
   - Test progress reporter update() calls
   - Test context manager behavior

3. **`_noop_progress()`** - 2-3 tests
   - Test noop reporter update() (does nothing)
   - Test context manager behavior

4. **Backwards compatibility** - 1-2 tests
   - Test `progress()` alias works

### Implementation Steps

1. Create `tests/unit/podcast_scraper/test_progress.py`
2. Write tests with mocked factories
3. Verify no external dependencies

**Success Criteria:**
- ✅ All progress functions have unit tests
- ✅ Tests are fast and isolated
- ✅ No external dependencies

---

## Sub-Phase 2.4: Metrics Collection (Quick Win) ⭐

**Priority:** High (Easy, Isolated)  
**Estimated Time:** 2-3 hours  
**Target Tests:** 15-20 tests  
**File:** `tests/unit/podcast_scraper/test_metrics.py` (new)

### Functions to Test

1. **`Metrics` class initialization** - 2-3 tests
   - Test default values
   - Test all fields initialized correctly

2. **`record_stage()`** - 4-5 tests
   - Test each stage ("scraping", "parsing", "normalizing", "writing_storage")
   - Test multiple calls accumulate
   - Test invalid stage name (no error, just ignored)

3. **`record_download_media_time()`** - 2-3 tests
   - Test time recording
   - Test multiple recordings
   - Test empty list handling

4. **`record_transcribe_time()`** - 2-3 tests
   - Test time recording
   - Test multiple recordings

5. **`record_extract_names_time()`** - 2-3 tests
   - Test time recording
   - Test multiple recordings

6. **`record_summarize_time()`** - 2-3 tests
   - Test time recording
   - Test multiple recordings

7. **`finish()`** - 3-4 tests
   - Test run_duration calculation
   - Test average calculations (with/without data)
   - Test return format (dict with all metrics)
   - Test rounding

8. **`log_metrics()`** - 1-2 tests
   - Test log format
   - Test all metrics included

### Implementation Steps

1. Create `tests/unit/podcast_scraper/test_metrics.py`
2. Mock time.time() for deterministic tests
3. Test each method in isolation

**Success Criteria:**
- ✅ All Metrics methods have unit tests
- ✅ Tests are deterministic (mocked time)
- ✅ Tests run in < 1 second total

---

## Sub-Phase 2.5: Episode Processor Utilities (Medium Complexity)

**Priority:** Medium (Moderate Complexity)  
**Estimated Time:** 3-4 hours  
**Target Tests:** 10-15 tests  
**File:** `tests/unit/podcast_scraper/test_episode_processor.py` (new)

### Functions to Test

1. **`derive_media_extension()`** - 3-4 tests
   - Test MIME type mapping (audio/mpeg → .mp3)
   - Test URL extension fallback
   - Test default extension when unknown
   - Test edge cases (None, empty, invalid)

2. **`derive_transcript_extension()`** - 4-5 tests
   - Test transcript_type preference
   - Test content_type fallback
   - Test URL extension fallback
   - Test priority ordering
   - Test edge cases

3. **Helper functions** (if testable in isolation) - 3-6 tests
   - Test any pure functions that can be unit tested
   - Mock dependencies for functions that need I/O

### Implementation Steps

1. Create `tests/unit/podcast_scraper/test_episode_processor.py`
2. Mock downloader, filesystem, whisper for functions that need them
3. Focus on pure functions first (extension derivation)

**Success Criteria:**
- ✅ Extension derivation functions have unit tests
- ✅ Tests use mocks for I/O operations
- ✅ Tests run in < 2 seconds total

---

## Sub-Phase 2.6: CLI Argument Parsing (Medium Complexity)

**Priority:** Medium (Moderate Complexity)  
**Estimated Time:** 4-5 hours  
**Target Tests:** 20-30 tests  
**File:** `tests/unit/podcast_scraper/test_cli.py` (new)

### Functions to Test

1. **`_validate_rss_url()`** - 3-4 tests
   - Test valid URLs (http, https)
   - Test invalid URLs (missing scheme, missing hostname)
   - Test empty URL
   - Test error list population

2. **`_validate_whisper_config()`** - 3-4 tests
   - Test valid models
   - Test invalid models
   - Test when transcribe_missing=False (skip validation)

3. **`_validate_speaker_config()`** - 4-5 tests
   - Test screenplay with num_speakers < MIN
   - Test speaker_names parsing
   - Test speaker_names with < 2 names
   - Test valid configurations

4. **`_validate_workers_config()`** - 2-3 tests
   - Test workers < 1
   - Test valid workers

5. **`validate_args()`** - 4-5 tests
   - Test multiple validation errors
   - Test valid args (no errors)
   - Test ValueError raised with error list

6. **`_load_and_merge_config()`** - 3-4 tests
   - Test JSON config loading
   - Test YAML config loading
   - Test CLI args override config file
   - Test missing config file

7. **`_build_config()`** - 3-4 tests
   - Test Config creation from args
   - Test default values
   - Test type coercion

8. **Argument parser helpers** (optional) - 2-3 tests
   - Test argument group creation
   - Test argument defaults

### Implementation Steps

1. Create `tests/unit/podcast_scraper/test_cli.py`
2. Mock filesystem for config file operations
3. Test validation functions in isolation
4. Test config building with mocked dependencies

**Success Criteria:**
- ✅ All validation functions have unit tests
- ✅ Config loading/building has unit tests
- ✅ Tests use mocks for filesystem operations
- ✅ Tests run in < 2 seconds total

---

## Sub-Phase 2.7: Service API (Medium Complexity)

**Priority:** Medium (Moderate Complexity)  
**Estimated Time:** 2-3 hours  
**Target Tests:** 8-12 tests  
**File:** `tests/unit/podcast_scraper/test_service.py` (extend existing)

### Functions to Test

1. **`run()`** - 4-5 tests
   - Test successful execution (mock workflow.run_pipeline)
   - Test exception handling (returns failed ServiceResult)
   - Test logging configuration application
   - Test ServiceResult structure

2. **`run_from_config_file()`** - 3-4 tests
   - Test JSON config file
   - Test YAML config file
   - Test missing file (returns failed ServiceResult)
   - Test invalid config (returns failed ServiceResult)

3. **`main()`** - 2-3 tests
   - Test successful execution (exit code 0)
   - Test failure (exit code 1)
   - Test version flag

### Implementation Steps

1. Review existing `test_service.py` to see what's covered
2. Add missing test cases
3. Mock workflow.run_pipeline and config.load_config_file

**Success Criteria:**
- ✅ All service functions have unit tests
- ✅ Tests mock workflow and config dependencies
- ✅ Tests run in < 1 second total

---

## Sub-Phase 2.8: Workflow Helper Functions (High Complexity) 🔥

**Priority:** High (Complex, Core Functionality)  
**Estimated Time:** 8-12 hours  
**Target Tests:** 30-50 tests  
**File:** `tests/unit/podcast_scraper/test_workflow.py` (new)

### Functions to Test

1. **`_update_metric_safely()`** - 3-4 tests
   - Test metric increment (no lock)
   - Test metric increment (with lock)
   - Test thread safety (mock lock)
   - Test invalid metric name (no error)

2. **`_call_generate_metadata()`** - 3-4 tests
   - Test metadata generation call
   - Test parameter passing
   - Test with None values
   - Mock metadata._generate_episode_metadata

3. **`apply_log_level()`** - 4-5 tests
   - Test log level setting (DEBUG, INFO, WARNING, ERROR)
   - Test invalid log level
   - Test log file configuration
   - Test file handler creation

4. **`_setup_pipeline_environment()`** - 5-6 tests
   - Test output directory creation
   - Test run_suffix generation
   - Test clean_output=True (mock shutil.rmtree)
   - Test clean_output with dry_run
   - Test dry_run mode (no directory creation)
   - Test error handling (cleanup failure)

5. **`_fetch_and_parse_feed()`** - 5-6 tests
   - Test successful fetch and parse (mock downloader, rss_parser)
   - Test fetch failure (None response)
   - Test parse failure (invalid XML)
   - Test missing RSS URL (ValueError)
   - Test response URL handling
   - Test response cleanup

6. **`_extract_feed_metadata_for_generation()`** - 3-4 tests
   - Test metadata extraction (mock metadata functions)
   - Test when metadata generation disabled
   - Test with missing feed data
   - Test RSS bytes reuse

7. **`_prepare_episodes_from_feed()`** - 5-6 tests
   - Test episode creation from feed items
   - Test max_episodes limit
   - Test episode filtering
   - Test episode ordering
   - Test edge cases (empty feed, no items)

8. **`_detect_feed_hosts_and_patterns()`** - 4-5 tests
   - Test host detection (mock speaker_detection)
   - Test pattern analysis
   - Test when auto_speakers disabled
   - Test caching behavior
   - Test with no episodes

9. **`_setup_transcription_resources()`** - 3-4 tests
   - Test Whisper model loading (mock whisper)
   - Test temp directory creation (mock tempfile)
   - Test when transcribe_missing disabled
   - Test error handling

10. **`_setup_processing_resources()`** - 2-3 tests
    - Test processing queue creation
    - Test with/without metadata/summarization
    - Test resource initialization

### Implementation Steps

1. Create `tests/unit/podcast_scraper/test_workflow.py`
2. Mock all external dependencies:
   - `downloader.fetch_url`
   - `rss_parser.parse_rss_items`
   - `metadata._generate_episode_metadata`
   - `speaker_detection.detect_hosts_from_feed`
   - `whisper.load_whisper_model`
   - Filesystem operations (os.makedirs, shutil.rmtree)
3. Test each helper function in isolation
4. Use tempfile for any needed temporary directories

**Success Criteria:**
- ✅ All 10 helper functions have unit tests
- ✅ All external dependencies are mocked
- ✅ Tests use tempfile (no real filesystem I/O)
- ✅ Tests run in < 3 seconds total

---

## Sub-Phase 2.9: Summarizer Core Functions (High Complexity) 🔥

**Priority:** High (Complex, Many Functions)  
**Estimated Time:** 10-15 hours  
**Target Tests:** 45-65 tests  
**File:** `tests/unit/podcast_scraper/test_summarizer_functions.py` (new, or extend existing)

### Functions to Test

1. **`clean_transcript()`** (if not in preprocessing) - 5-8 tests
   - Already covered if moved to preprocessing

2. **`remove_sponsor_blocks()`** (if not in preprocessing) - 3-5 tests
   - Already covered if moved to preprocessing

3. **`remove_outro_blocks()`** (if not in preprocessing) - 3-5 tests
   - Already covered if moved to preprocessing

4. **`clean_for_summarization()`** (if not in preprocessing) - 3-5 tests
   - Already covered if moved to preprocessing

5. **`chunk_text_for_summarization()`** - 5-8 tests
   - Test token-based chunking
   - Test word-based chunking
   - Test overlap handling
   - Test edge cases (short text, empty text)
   - Mock tokenizer

6. **`_validate_and_fix_repetitive_summary()`** - 3-5 tests
   - Test repetitive pattern detection
   - Test fix application
   - Test no repetition (no change)

7. **`_strip_instruction_leak()`** - 3-5 tests
   - Test instruction leak detection
   - Test removal
   - Test no leak (no change)

8. **`_summarize_chunks_map()`** - 3-5 tests
   - Test chunk summarization
   - Test multiple chunks
   - Mock SummaryModel

9. **`_summarize_chunks_reduce()`** - 3-5 tests
   - Test reduce phase logic
   - Test different reduce strategies
   - Mock SummaryModel

10. **`_combine_summaries_*()`** - 10-15 tests
    - Test abstractive combination
    - Test extractive combination
    - Test mini map-reduce combination
    - Test strategy selection logic
    - Mock SummaryModel

11. **`safe_summarize()`** - 3-5 tests
    - Test successful summarization
    - Test OOM error handling
    - Test other error handling
    - Mock SummaryModel

### Implementation Steps

1. Review existing `test_summarizer.py` to see what's already covered
2. Create `test_summarizer_functions.py` for pure function tests
3. Mock SummaryModel, tokenizer, pipeline for all tests
4. Focus on logic testing, not model integration

**Success Criteria:**
- ✅ All summarizer core functions have unit tests
- ✅ All ML dependencies are mocked
- ✅ Tests run in < 5 seconds total
- ✅ Tests focus on logic, not model behavior

---

## Sub-Phase 2.10: Speaker Detection Functions (Medium Complexity)

**Priority:** Medium (Some ML Dependencies)  
**Estimated Time:** 4-6 hours  
**Target Tests:** 19-31 tests  
**File:** `tests/unit/podcast_scraper/test_speaker_detection.py` (extend existing)

### Functions to Test

1. **`detect_speaker_names()`** - 5-8 tests
   - Test name extraction from title
   - Test name extraction from description
   - Test pattern-based detection
   - Test scoring logic
   - Mock spacy model

2. **`detect_hosts_from_feed()`** - 3-5 tests
   - Test feed-level host detection
   - Test caching
   - Test with no hosts
   - Mock spacy model

3. **`analyze_episode_patterns()`** - 5-8 tests
   - Test pattern analysis
   - Test prefix/suffix detection
   - Test position analysis
   - Test consistency scoring

4. **Helper functions** - 6-10 tests
   - `_validate_model_name()` - 2-3 tests
   - `_load_spacy_model()` - 2-3 tests (mock spacy)
   - `_extract_names_from_text()` - 3-4 tests (mock spacy)
   - `_score_speaker_candidates()` - 3-4 tests

### Implementation Steps

1. Review existing `test_speaker_detection.py`
2. Add missing test cases
3. Mock spacy model for all tests
4. Test pure logic functions without ML deps

**Success Criteria:**
- ✅ All speaker detection functions have unit tests
- ✅ spacy is mocked (no real model loading)
- ✅ Tests run in < 3 seconds total

---

## Implementation Strategy

### Recommended Order

1. **Start with Quick Wins (2.1-2.4):** 6-8 hours, ~50-70 tests
   - Preprocessing, Filesystem, Progress, Metrics
   - Easy wins, high impact, builds momentum

2. **Medium Complexity (2.5-2.7):** 9-12 hours, ~38-57 tests
   - Episode Processor, CLI, Service
   - Moderate effort, good coverage

3. **High Complexity (2.8-2.10):** 22-33 hours, ~94-146 tests
   - Workflow, Summarizer, Speaker Detection
   - Most complex, but highest value

### Tracking Progress

For each sub-phase:
- ✅ Create test file
- ✅ Write test cases
- ✅ Verify tests pass
- ✅ Check test count increase
- ✅ Update progress tracker

### Success Metrics

- **Unit Tests:** 356 → 500-600 (+144-244 tests)
- **Distribution:** 50.2% → 70-80%
- **Test Execution:** All tests run in < 30 seconds total
- **Coverage:** > 80% code coverage for tested modules

---

## Estimated Total Effort

| Sub-Phase | Tests | Time | Priority |
|-----------|-------|------|----------|
| 2.1: Preprocessing | 15-20 | 2-3h | ⭐ High |
| 2.2: Filesystem | 12-18 | 2-3h | ⭐ High |
| 2.3: Progress | 8-12 | 1-2h | ⭐ High |
| 2.4: Metrics | 15-20 | 2-3h | ⭐ High |
| 2.5: Episode Processor | 10-15 | 3-4h | Medium |
| 2.6: CLI | 20-30 | 4-5h | Medium |
| 2.7: Service | 8-12 | 2-3h | Medium |
| 2.8: Workflow | 30-50 | 8-12h | 🔥 High |
| 2.9: Summarizer | 45-65 | 10-15h | 🔥 High |
| 2.10: Speaker Detection | 19-31 | 4-6h | Medium |
| **Total** | **182-273** | **38-56h** | |

**Note:** Time estimates assume working incrementally, not all at once. Can be spread over 2-3 weeks.

---

## Next Steps

1. **Choose starting point:** Recommend starting with Sub-Phase 2.1 (Preprocessing) - easiest win
2. **Create test file:** Set up the test file structure
3. **Write first test:** Start with simplest function
4. **Iterate:** Add tests incrementally
5. **Verify:** Run tests after each addition
6. **Track:** Update progress as you complete each sub-phase

---

## Related Documentation

- [Testing Strategy](../TESTING_STRATEGY.md) - Unit test requirements
- [Phase 1 Plan](TEST_PYRAMID_PLAN.md) - Completed phase
- [Full Analysis](TEST_PYRAMID_ANALYSIS.md) - Complete analysis

