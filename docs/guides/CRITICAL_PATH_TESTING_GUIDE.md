# Critical Path Testing Guide

> **See also:**
>
> - [Testing Strategy](../TESTING_STRATEGY.md) for overall testing strategy and test pyramid concepts
> - [Testing Guide](TESTING_GUIDE.md) for detailed implementation instructions and test execution
>
> This guide bridges testing strategy and implementation by focusing on **what to test** based on the critical path.
> It helps you understand which tests matter most and how to prioritize your testing efforts.

## What is the Critical Path?

**Critical Path = Full Workflow with All Core Features**

The critical path represents the **essence of this project** - the complete workflow that delivers the main value:

```text
RSS → Parse → Download/Transcribe → Speaker Detection → Summarization → Metadata → Files
```

### Path 1: Transcript Download (when transcript URL exists)

**ML Provider Flow:**

- RSS → Parse → Download Transcript → **NER Speaker Detection** → **Local Summarization** → Metadata → Files

**OpenAI Provider Flow:**

- RSS → Parse → Download Transcript → **OpenAI Speaker Detection** → **OpenAI Summarization** → Metadata → Files

### Path 2: Transcription (when transcript URL missing)

**ML Provider Flow:**

- RSS → Parse → Download Audio/Video → **Whisper Transcription** → **NER Speaker Detection** →
  **Local Summarization** → Metadata → Files

**OpenAI Provider Flow:**

- RSS → Parse → Download Audio/Video → **OpenAI Transcription** → **OpenAI Speaker Detection** →
  **OpenAI Summarization** → Metadata → Files

**Why All Paths Matter:**

- All variants are valid configurations users can choose
- All are essential for the tool to work correctly with different provider choices
- All must be tested to ensure complete coverage regardless of provider selection

### Why Speaker Detection and Summarization are Critical

- **Speaker Detection**: Core feature that identifies hosts and guests from metadata
  - **ML Provider**: Uses spaCy NER models (local)
  - **OpenAI Provider**: Uses OpenAI API (cloud)
- **Summarization**: Core feature that generates episode summaries from transcripts
  - **ML Provider**: Uses Transformers models (local)
  - **OpenAI Provider**: Uses OpenAI API (cloud)
- **Both are essential**: They represent the main value proposition of the project
- **Both provider types are critical**: Users can choose ML (local, free) or OpenAI (cloud, paid) based on their needs

## Critical Path and Test Pyramid

The critical path should be tested at **all three levels** of the test pyramid:

```text
        /\
       /E2E\          ← Complete critical path workflows (user-facing)
      /------\
     /Integration\    ← Critical path component interactions
    /------------\
   /    Unit      \   ← Critical path individual functions
  /----------------\
```python

- Individual functions in the critical path
- Each step in isolation: RSS parsing, transcript download, transcription, NER, summarization, metadata generation

**Examples:**

- `rss_parser.py`: Parse RSS feeds correctly
- `downloader.py`: Download transcripts/audio correctly
- `whisper_integration.py`: Transcribe audio correctly
- `speaker_detection.py`: Detect speakers from text
- `summarizer.py`: Generate summaries from transcripts
- `metadata.py`: Generate metadata files

**Priority**: **HIGH** - These are the building blocks of the critical path

### Integration Tests (Middle Layer)

**What to Test:**

- Component interactions along the critical path
- How components work together: RSS → Episode → Provider → File

**Critical Path Integration Tests:**

1. **`test_full_workflow_with_ner_and_summarization`** ⭐ **ESSENTIAL (ML Providers)**
   - Validates: RSS → Parse → Download/Transcribe → **NER** → **Local Summarization** → Metadata → Files
   - Uses: ML providers (Whisper, spaCy NER, Transformers)
   - This is the **complete critical path** with ML providers

2. **`test_full_workflow_with_openai_providers`** ⭐ **ESSENTIAL (OpenAI Providers)**
   - Validates: RSS → Parse → Download/Transcribe → **OpenAI Speaker Detection** →
     **OpenAI Summarization** → Metadata → Files

   - Uses: OpenAI providers (mocked API calls)
   - This is the **complete critical path** with OpenAI providers

3. **`test_critical_path_with_real_models`** (ML Providers with Real Models)
   - Validates: Critical path with real cached ML models
   - Uses: Real Whisper, spaCy, Transformers models (cached)

4. **`test_critical_path_with_openai_providers`** (OpenAI Providers)
   - Validates: Critical path with OpenAI providers (mocked API)
   - Uses: OpenAI providers for all three services

5. **`test_rss_to_metadata_generation`**
   - Validates: RSS → Parse → Download Transcript → Metadata → Files
   - Covers Path 1 (transcript download)

6. **`test_rss_to_transcription_workflow`**
   - Validates: RSS → Parse → Download Audio → Transcription → Metadata → Files
   - Covers Path 2 (transcription)

7. **`test_episode_processor_audio_download_and_transcription`**
   - Validates: Episode processor functions for audio download and transcription

8. **`test_speaker_detection_in_transcription_workflow`**
   - Validates: RSS → Parse → Download Audio → Transcribe → **Speaker Detection** → Metadata → Files

**Priority**: **CRITICAL** - These validate the critical path works end-to-end

### E2E Tests (Top Layer)

**What to Test:**

- Complete user workflows for the critical path
- All three entry points: CLI, Library API, Service API

**Critical Path E2E Tests:**

1. **`test_cli_basic_transcript_download`** (CLI)
   - Validates: CLI transcription end-to-end
   - Uses: `--transcribe-missing`

2. **`test_library_api_basic_pipeline`** (Library API)
   - Validates: `run_pipeline(config)` transcription end-to-end
   - Uses: `transcribe_missing=True`

3. **`test_service_api_basic_run`** (Service API)
   - Validates: `service.run(config)` transcription end-to-end
   - Uses: `transcribe_missing=True`

**Priority**: **CRITICAL** - These validate users can actually use the tool

## Decision Framework: Is This Critical Path?

When deciding what to test, ask:

1. **Is this part of the critical path?**

   - ✅ RSS parsing → **Test it**
   - ✅ Transcript download/transcription → **Test it**
   - ✅ NER speaker detection → **Test it**
   - ✅ Summarization → **Test it**
   - ✅ Metadata generation → **Test it**
   - ❌ Extended features → Lower priority
   - ❌ Configuration edge cases → Lower priority

2. **What test level should I use?**

   - **Unit**: Testing individual functions → Unit test
   - **Integration**: Testing component interactions → Integration test
   - **E2E**: Testing complete user workflow → E2E test

3. **Should this be fast or slow?**

   - **Fast**: Critical path tests that use mocked components → Fast
   - **Fast**: Critical path tests with OpenAI providers (mocked API) → Fast
   - **Slow**: Critical path tests that use real ML models → Slow (but still critical)
   - **Slow**: Non-critical path tests → Slow

## Test Prioritization

### Priority 1: Critical Path (Must Have)

**These tests MUST exist and MUST pass:**

- ✅ Full workflow integration test with NER and summarization
- ✅ Integration tests for both paths (transcript download + transcription)
- ✅ E2E tests for all three entry points (CLI, Library API, Service API)
- ✅ Unit tests for all critical path functions

**Execution**: Run in fast test suite (with mocks) for quick feedback

### Priority 2: Critical Path with Real Models (Should Have)

**These tests validate critical path with real implementations:**

- ✅ E2E tests with real Whisper transcription
- ✅ E2E tests with real NER models
- ✅ E2E tests with real summarization models

**Execution**: Run in slow test suite (with real models) for comprehensive validation

### Priority 3: Extended Features (Nice to Have)

**These tests validate features beyond the critical path:**

- ⚠️ Extended metadata features
- ⚠️ Configuration edge cases
- ⚠️ Error handling edge cases
- ⚠️ HTTP behavior tests

**Execution**: Run in slow test suite

## Critical Path Test Coverage Matrix

|Scenario|Unit Tests|Integration Tests|E2E Tests|
| -------- | ---------- | ----------------- | --------- |
|RSS parsing|✅ Parse RSS feeds|✅ RSS → Episode|✅ Full workflow|
|Transcript download|✅ Download logic|✅ Download → Metadata|✅ Full workflow|
|Transcription|✅ Whisper integration|✅ Audio → Transcript|✅ Full workflow|
|NER (Speaker Detection)|✅ NER extraction|✅ Transcript → Speakers|✅ Full workflow|
|Summarization|✅ Summary generation|✅ Transcript → Summary|✅ Full workflow|
|Metadata generation|✅ Metadata creation|✅ All data → Metadata|✅ Full workflow|

## Current Critical Path Coverage

### Integration-Fast Tests ✅

**6 tests covering critical path:**

1. **`test_full_workflow_with_ner_and_summarization`** ⭐ **ESSENTIAL (ML Providers)**
   - File: `tests/integration/test_component_workflows.py::TestRSSToMetadataWorkflow`
   - Validates: RSS → Parse → Download/Transcribe → **NER** → **Local Summarization** → Metadata → Files
   - Uses: RSS feed with audio URL, mocked Whisper, mocked NER, mocked local summarization
   - **This is the complete critical path with ML providers**

2. **`test_full_workflow_with_openai_providers`** ⭐ **ESSENTIAL (OpenAI Providers)**
   - File: `tests/integration/test_component_workflows.py::TestRSSToMetadataWorkflow`
   - Validates: RSS → Parse → Download/Transcribe → **OpenAI Speaker Detection** →
     **OpenAI Summarization** → Metadata → Files

   - Uses: RSS feed with audio URL, mocked OpenAI transcription, speaker detection, and summarization
   - **This is the complete critical path with OpenAI providers**

   - File: `tests/integration/test_component_workflows.py::TestRSSToMetadataWorkflow`
   - Validates: RSS → Parse → Download Transcript → Metadata → Files
   - Uses: RSS feed with transcript URL

3. **`test_rss_to_transcription_workflow`** (Path 2: Transcription)
   - File: `tests/integration/test_component_workflows.py::TestRSSToMetadataWorkflow`
   - Validates: RSS → Parse → Download Audio → Whisper Transcription → Metadata → Files
   - Uses: RSS feed without transcript URL (has audio URL)

4. **`test_episode_processor_audio_download_and_transcription`**
   - File: `tests/integration/test_component_workflows.py::TestRSSToMetadataWorkflow`
   - Validates: Episode processor functions for audio download and transcription
   - Uses: Mocked Whisper transcription

5. **`test_speaker_detection_in_transcription_workflow`**
   - File: `tests/integration/test_component_workflows.py::TestRSSToMetadataWorkflow`
   - Validates: RSS → Parse → Download Audio → Transcribe → **NER** → Metadata → Files
   - Uses: Mocked Whisper and NER
   - **Note**: Covers NER but not summarization (use test_full_workflow_with_ner_and_summarization for complete path)

**Execution Time**: ~8-9 seconds

### E2E-Fast Tests ✅

**3 tests covering critical path:**

All E2E tests use `transcribe_missing=True`, so they validate Path 2 (transcription):

1. **`test_cli_basic_transcript_download`** (CLI)
   - File: `tests/e2e/test_basic_e2e.py::TestBasicCLIE2E`
   - Validates: CLI transcription end-to-end
   - Uses: `--transcribe-missing`

2. **`test_library_api_basic_pipeline`** (Library API)
   - File: `tests/e2e/test_basic_e2e.py::TestBasicLibraryAPIE2E`
   - Validates: `run_pipeline(config)` transcription end-to-end
   - Uses: `transcribe_missing=True`

3. **`test_service_api_basic_run`** (Service API)
   - File: `tests/e2e/test_basic_e2e.py::TestBasicServiceAPIE2E`
   - Validates: `service.run(config)` transcription end-to-end
   - Uses: `transcribe_missing=True`

**Execution Time**: ~30-45 seconds (with minimal audio file)

### Test Coverage Matrix

| Scenario | Integration-Fast | E2E-Fast |
| ---------- | ------------------ | ---------- |
| Transcript URL exists → Download | ✅ `test_rss_to_metadata_generation` | ❌ (not needed, covered in integration) |
| No transcript URL → Transcribe | ✅ `test_rss_to_transcription_workflow` | ✅ All 3 E2E tests |
| Episode processor transcription | ✅ `test_episode_processor_audio_download_and_transcription` | ❌ (component-level, integration is sufficient) |

## Using This Guide

### When Writing New Tests

1. **Check if it's critical path**: Is this part of the critical path?
   (RSS → Parse → Download/Transcribe → Speaker Detection → Summarization → Metadata → Files)

2. **Choose test level**: Unit (function), Integration (components), or E2E (workflow)?

3. **Prioritize**: Critical path tests should be fast and run frequently

### When Reviewing Test Coverage

1. **Verify critical path coverage**: Do we have tests for all critical path steps?

2. **Check all three levels**: Unit, Integration, and E2E tests for critical path?

3. **Ensure fast feedback**: Critical path tests should run quickly (with mocks)

### When Debugging Failures

1. **Check critical path first**: If critical path tests fail, fix those immediately
2. **Verify all entry points**: CLI, Library API, and Service API should all work
3. **Test both paths**: Transcript download and transcription should both work

## Summary

**Key Principles:**

1. **Critical path = Core value**: RSS → Parse → Download/Transcribe → NER → Summarization → Metadata → Files

2. **Test at all levels**: Unit, Integration, and E2E tests for critical path

3. **Prioritize fast feedback**: Critical path tests should run quickly (with mocks)

4. **Cover all entry points**: CLI, Library API, and Service API must all work

5. **Test both paths**: Transcript download and transcription are both critical

**Remember**: If the critical path doesn't work, nothing else matters. Focus your testing efforts on ensuring the
critical path is solid, then expand to extended features.

## References

- [Testing Strategy](../TESTING_STRATEGY.md) - Overall testing strategy and test pyramid
- [Testing Guide](TESTING_GUIDE.md) - Detailed implementation instructions
- Critical path tests: `tests/integration/test_component_workflows.py`
- Critical path E2E tests: `tests/e2e/test_basic_e2e.py`
