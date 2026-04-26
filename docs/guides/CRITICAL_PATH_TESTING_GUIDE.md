# Critical Path Testing Guide

> **See also:**
>
> - [Testing Strategy](../architecture/TESTING_STRATEGY.md) - Overall testing philosophy and test pyramid
> - [Testing Guide](TESTING_GUIDE.md) - Quick reference and test execution commands
> - [Unit Testing Guide](UNIT_TESTING_GUIDE.md) - Unit test implementation
> - [Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md) - Integration test guidelines
> - [E2E Testing Guide](E2E_TESTING_GUIDE.md) - E2E testing with real ML models

This guide focuses on **what to test** and **how to prioritize**. Use the `@pytest.mark.critical_path`
marker for tests that validate the critical path - these run in the fast test suite.

## What is the Critical Path?

**Critical Path = Anything that, if broken, prevents a user from completing a
primary product workflow.**

This project has **two** primary product workflows. A test belongs on the
critical path if its failure would block users from completing either.

### Workflow 1 — Pipeline ingestion (the original definition)

The complete data-production workflow that delivers the main value:

```text
RSS → Parse → Download/Transcribe → Speaker Detection → Summarization → Metadata → Files
```

The two variants below cover the disk paths users hit in production. Both are
in scope for the marker; both must be tested to ensure complete coverage
regardless of provider selection.

#### Path 1A: Transcript Download (when transcript URL exists)

- **ML Provider Flow** — RSS → Parse → Download Transcript →
  **NER Speaker Detection** → **Local Summarization** → Metadata → Files
- **OpenAI Provider Flow** — RSS → Parse → Download Transcript →
  **OpenAI Speaker Detection** → **OpenAI Summarization** → Metadata → Files

#### Path 1B: Transcription (when transcript URL missing)

- **ML Provider Flow** — RSS → Parse → Download Audio/Video →
  **Whisper Transcription** → **NER Speaker Detection** →
  **Local Summarization** → Metadata → Files
- **OpenAI Provider Flow** — RSS → Parse → Download Audio/Video →
  **OpenAI Transcription** → **OpenAI Speaker Detection** →
  **OpenAI Summarization** → Metadata → Files

#### Why Speaker Detection and Summarization are Critical

- **Speaker Detection** identifies hosts and guests from metadata
  (spaCy NER locally, OpenAI in cloud mode).
- **Summarization** generates episode summaries from transcripts
  (Transformers locally, OpenAI in cloud mode).
- Both are core value; both provider modes are valid user-selectable
  configurations.

### Workflow 2 — Viewer UI usage

The viewer (`web/gi-kg-viewer/`, served by the FastAPI app) is the second
primary product surface. A user opens the app, points it at a corpus, and
expects to:

```text
Health → Set corpus → Browse Library / Digest → Search / Explore →
Configure Feeds + Profile → Run pipeline jobs → See results in Graph + Dashboard
```

API surfaces whose failure breaks this loop are critical-path:

- **App + health surface** — `/api/health`, app startup, route mounting.
  No app, no anything else.
- **Corpus listing** — `/api/corpus/episodes`, `/api/corpus/feeds`.
  Library tab is empty without these.
- **Artifacts loading** — `/api/artifacts`, `/api/artifacts/<rel>`.
  Graph / Search / Explore data flows through here.
- **Operator config + feeds** — `/api/operator-config` (GET / PUT),
  `/api/feeds` (GET / PUT). Without these, the user can't set the corpus
  path, choose a profile, or add a feed.
- **Pipeline jobs** — `/api/jobs` (POST + status + log endpoints).
  This is the entry point that *triggers* Workflow 1; it's the bridge
  between the UI and pipeline-critical-path.
- **Search + Explore** — `/api/search`, `/api/explore`. These are the
  panels users open to actually look at the corpus.
- **Digest** — `/api/corpus/digest`, `/api/corpus/cil-digest-topics`,
  `/api/corpus/topic-clusters`. The Digest tab fronts the corpus value
  proposition.

API surfaces that **degrade** the experience but don't block primary
workflows are NOT critical-path (they belong in the full integration suite,
not `ci-fast`):

- Dashboard auxiliary panels: `/api/index/stats` (Index stats),
  `/api/corpus/metrics`, `/api/corpus/persons`, `/api/corpus/coverage`.
  Failure here means a dashboard panel is empty / red — user can still
  ingest, search, and view.
- Index rebuild orchestration. Eventual-consistency surface; user can
  retrigger.

### Decision rule (use this when adding a new test)

A test is **`critical_path`** when answering "yes" to *both*:

1. **Could this regression block a user from completing a primary workflow?**
   (Pipeline ingest *or* Viewer UI usage.)
2. **Is the test itself fast and deterministic** (no real ML model load,
   no real subprocess work)? Critical-path tests run on every PR via
   `ci-fast`; they must finish in seconds, not minutes.

If either answer is "no", drop the marker. Slow but important regressions
go in the full `make ci` / nightly suites; fast but cosmetic regressions
get caught by ordinary integration tests on the full body.

## Critical Path and Test Pyramid

Both workflows should be tested at **all three levels** of the test pyramid:

```text
        /\
       /E2E\          ← Complete critical path workflows (user-facing)
      /------\
     /Integration\    ← Critical path component interactions
    /------------\
   /    Unit      \   ← Critical path individual functions
  /----------------\
```

### Unit Tests (Bottom Layer)

**What to Test:**

- Individual functions in the critical path
- Each step in isolation: RSS parsing, transcript download, transcription, NER, summarization, metadata generation

**Examples:**

- `rss_parser.py`: Parse RSS feeds correctly
- `rss/downloader.py`: Download transcripts/audio correctly
- `providers/ml/whisper_utils.py`: Transcribe audio correctly
- `speaker_detection.py`: Detect speakers from text
- `providers/ml/summarizer.py`: Generate summaries from transcripts
- `metadata.py`: Generate metadata files

**Priority**: **HIGH** - These are the building blocks of the critical path

### Integration Tests (Middle Layer)

**What to Test:**

- Component interactions along the critical path
- How components work together: RSS → Episode → Provider → File

**Critical Path Integration Tests:**

1. **`test_full_workflow_with_ner_and_summarization`**  **ESSENTIAL (ML Providers)**
   - Validates: RSS → Parse → Download/Transcribe → **NER** → **Local Summarization** → Metadata → Files
   - Uses: ML providers (Whisper, spaCy NER, Transformers)
   - This is the **complete critical path** with ML providers

2. **`test_full_workflow_with_openai_providers`**  **ESSENTIAL (OpenAI Providers)**
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

   - RSS parsing → **Test it**
   - Transcript download/transcription → **Test it**
   - NER speaker detection → **Test it**
   - Summarization → **Test it**
   - Metadata generation → **Test it**
   - Extended features → Lower priority
   - Configuration edge cases → Lower priority

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

- Full workflow integration test with NER and summarization
- Integration tests for both paths (transcript download + transcription)
- E2E tests for all three entry points (CLI, Library API, Service API)
- Unit tests for all critical path functions

**Execution**: Run in fast test suite (with mocks) for quick feedback

### Priority 2: Critical Path with Real Models (Should Have)

**These tests validate critical path with real implementations:**

- E2E tests with real Whisper transcription
- E2E tests with real NER models
- E2E tests with real summarization models

**Execution**: Run in slow test suite (with real models) for comprehensive validation

### Priority 3: Extended Features (Nice to Have)

**These tests validate features beyond the critical path:**

- Extended metadata features
- Configuration edge cases
- Error handling edge cases
- HTTP behavior tests

**Execution**: Run in slow test suite

## Critical Path Test Coverage Matrix

|Scenario|Unit Tests|Integration Tests|E2E Tests|
|--------|----------|-----------------|---------|
|RSS parsing|Parse RSS feeds|RSS → Episode|Full workflow|
|Transcript download|Download logic|Download → Metadata|Full workflow|
|Transcription|Whisper integration|Audio → Transcript|Full workflow|
|NER (Speaker Detection)|NER extraction|Transcript → Speakers|Full workflow|
|Summarization|Summary generation|Transcript → Summary|Full workflow|
|Metadata generation|Metadata creation|All data → Metadata|Full workflow|

## Current Critical Path Coverage

### Integration-Fast Tests

**6 tests covering critical path:**

1. **`test_full_workflow_with_ner_and_summarization`**  **ESSENTIAL (ML Providers)**
   - File: `tests/integration/test_component_workflows.py::TestRSSToMetadataWorkflow`
   - Validates: RSS → Parse → Download/Transcribe → **NER** → **Local Summarization** → Metadata → Files
   - Uses: RSS feed with audio URL, mocked Whisper, mocked NER, mocked local summarization
   - **This is the complete critical path with ML providers**

2. **`test_full_workflow_with_openai_providers`**  **ESSENTIAL (OpenAI Providers)**
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

### E2E-Fast Tests

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
| Transcript URL exists → Download | `test_rss_to_metadata_generation` | (not needed, covered in integration) |
| No transcript URL → Transcribe | `test_rss_to_transcription_workflow` | All 3 E2E tests |
| Episode processor transcription | `test_episode_processor_audio_download_and_transcription` | (component-level, integration is sufficient) |

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

- [Testing Strategy](../architecture/TESTING_STRATEGY.md) - Overall testing philosophy
- [Testing Guide](TESTING_GUIDE.md) - Quick reference and test execution
- [Unit Testing Guide](UNIT_TESTING_GUIDE.md) - Unit test implementation
- [Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md) - Integration test guidelines
- [E2E Testing Guide](E2E_TESTING_GUIDE.md) - E2E testing with real ML
- Critical path tests: `tests/integration/test_component_workflows.py`
- Critical path E2E tests: `tests/e2e/test_basic_e2e.py`
