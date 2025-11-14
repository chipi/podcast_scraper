# Refactoring Opportunities

This document identifies large functions and files that could benefit from refactoring to improve maintainability and enable code reuse.

## Summary

**Files with refactoring opportunities:**

1. `speaker_detection.py` (806 lines) - **HIGH PRIORITY**
2. `metadata.py` (422 lines) - **MEDIUM PRIORITY**
3. `episode_processor.py` (447 lines) - **MEDIUM PRIORITY**
4. `rss_parser.py` (548 lines) - **LOW PRIORITY**
5. `cli.py` (386 lines) - **LOW PRIORITY**

## Detailed Analysis

### 1. `speaker_detection.py` - **HIGH PRIORITY**

**File Size**: 806 lines

#### `detect_speaker_names()` - ~250 lines (lines 548-797)

**Current Issues**:

- Very large function with multiple responsibilities
- Complex nested logic for guest selection
- Inline helper function `calculate_heuristic_score()` could be extracted
- Guest candidate processing logic is complex
- Speaker name capping and fallback logic mixed together

**Refactoring Opportunities**:

```python
# Extract helper functions:
- _calculate_heuristic_score() - Extract inline function
- _build_guest_candidates() - Process title/description guests
- _select_best_guest() - Select guest with highest score
- _build_speaker_names_list() - Combine hosts + guests with capping
- _log_guest_detection() - Centralize logging logic
```

**Benefits**:

- Each function has single responsibility
- Easier to test individual components
- Reusable scoring logic
- Clearer flow in main function

#### `extract_person_entities()` - ~170 lines (lines 198-369)

**Current Issues**:

- Large function with multiple fallback strategies
- Duplicate code for entity extraction (full text vs segments)
- Pattern-based fallback logic embedded in main function

**Refactoring Opportunities**:

```python
# Extract helper functions:
- _extract_entities_from_text() - Core NER extraction logic
- _extract_entities_from_segments() - Segment-based fallback
- _pattern_based_fallback() - Pattern matching fallback
- _validate_person_entity() - Entity validation logic
```

**Benefits**:

- Eliminate code duplication
- Test each extraction strategy independently
- Reusable entity validation
- Clearer fallback chain

#### `analyze_episode_patterns()` - ~115 lines (lines 432-545)

**Current Issues**:

- Medium-large function
- Position analysis and prefix/suffix extraction could be separated
- Counter logic repeated for prefixes/suffixes

**Refactoring Opportunities**:

```python
# Extract helper functions:
- _analyze_title_positions() - Position pattern analysis
- _extract_prefixes_suffixes() - Context extraction
- _find_common_patterns() - Pattern frequency analysis
```

**Benefits**:

- Clearer separation of concerns
- Reusable pattern analysis
- Easier to test individual components

### 2. `metadata.py` - **MEDIUM PRIORITY**

**File Size**: 422 lines

#### `generate_episode_metadata()` - ~200 lines (lines 222-421)

**Current Issues**:

- Large function building multiple Pydantic models
- File I/O and serialization mixed with model building
- Path construction logic embedded

**Refactoring Opportunities**:

```python
# Extract helper functions:
- _build_feed_metadata() - Construct FeedMetadata
- _build_episode_metadata() - Construct EpisodeMetadata
- _build_content_metadata() - Construct ContentMetadata
- _build_processing_metadata() - Construct ProcessingMetadata
- _determine_metadata_path() - Path construction logic
- _serialize_metadata() - JSON/YAML serialization
```

**Benefits**:

- Each model builder is independently testable
- Reusable serialization logic
- Clearer separation: model building vs I/O
- Easier to add new metadata fields

### 3. `episode_processor.py` - **MEDIUM PRIORITY**

**File Size**: 447 lines

#### `transcribe_media_to_text()` - ~80 lines (lines 210-289)

**Current Issues**:

- Has `# noqa: C901` complexity comment
- Mixes transcription, formatting, and file I/O
- Error handling mixed with business logic

**Refactoring Opportunities**:

```python
# Extract helper functions:
- _format_transcript_if_needed() - Screenplay formatting logic
- _save_transcript_file() - File writing logic
- _cleanup_temp_media() - Cleanup logic
```

**Benefits**:

- Reduce complexity
- Test formatting separately
- Reusable file operations

#### `process_transcript_download()` - ~85 lines (lines 292-376)

**Current Issues**:

- Medium-sized function
- URL fetching, file writing, and error handling mixed

**Refactoring Opportunities**:

```python
# Extract helper functions:
- _fetch_transcript_content() - HTTP download logic
- _determine_output_path() - Path construction
- _write_transcript_file() - File writing with skip-existing
```

**Benefits**:

- Clearer separation of concerns
- Reusable download logic
- Easier to test

### 4. `rss_parser.py` - **LOW PRIORITY**

**File Size**: 548 lines

#### `extract_episode_metadata()` - ~90 lines (lines 333-421)

**Current Issues**:

- Medium-sized function
- Multiple metadata extractions in sequence
- Could benefit from extraction helpers

**Refactoring Opportunities**:

```python
# Extract helper functions:
- _extract_duration_seconds() - Duration parsing logic
- _extract_episode_number() - Episode number extraction
- _extract_image_url() - Image URL extraction
```

**Benefits**:

- Reusable extraction logic
- Easier to test individual fields
- Clearer function flow

### 5. `cli.py` - **LOW PRIORITY**

**File Size**: 386 lines

#### `parse_args()` - ~165 lines (lines 142-306)

**Current Issues**:

- Large function but mostly argument definitions
- Config file loading logic embedded
- Argument validation mixed with parsing

**Refactoring Opportunities**:

```python
# Extract helper functions:
- _add_common_arguments() - Common argument definitions
- _add_transcription_arguments() - Transcription-related args
- _add_metadata_arguments() - Metadata-related args
- _load_and_merge_config() - Config file loading logic
```

**Benefits**:

- Group related arguments together
- Reusable config loading
- Easier to add new argument groups

#### `validate_args()` - ~60 lines (lines 79-139)

**Current Issues**:

- Has `# noqa: C901` complexity comment
- Multiple validation checks in sequence
- Could be split by concern

**Refactoring Opportunities**:

```python
# Extract helper functions:
- _validate_rss_url() - RSS URL validation
- _validate_whisper_config() - Whisper model validation
- _validate_speaker_config() - Speaker-related validation
- _validate_workers_config() - Workers validation
```

**Benefits**:

- Reduce complexity
- Test validations independently
- Reusable validation logic

## Recommended Refactoring Order

### Phase 1: High Impact (Maintainability)

1. **`speaker_detection.py`** - `detect_speaker_names()` and `extract_person_entities()`
   - Highest complexity
   - Most reusable logic
   - Biggest maintainability win

### Phase 2: Medium Impact (Code Organization)

1. **`metadata.py`** - `generate_episode_metadata()`
   - Clear separation opportunities
   - Model building logic reusable
   - Easier testing

2. **`episode_processor.py`** - `transcribe_media_to_text()` and `process_transcript_download()`
   - Reduce complexity
   - Better error handling separation

### Phase 3: Low Impact (Code Clarity)

1. **`rss_parser.py`** - `extract_episode_metadata()`
   - Smaller gains
   - Still improves clarity

2. **`cli.py`** - `parse_args()` and `validate_args()`
   - Mostly organizational
   - Less critical for maintainability

## Refactoring Principles

1. **Single Responsibility**: Each function should do one thing
2. **Reusability**: Extract logic that could be reused elsewhere
3. **Testability**: Make functions easier to unit test
4. **Readability**: Improve code flow and clarity
5. **Maintainability**: Reduce complexity and coupling

## Notes

- All refactoring should maintain existing functionality
- Add tests for extracted functions
- Update docstrings to reflect new structure
- Consider backward compatibility for public APIs
