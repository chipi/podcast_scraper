# Work Plan for Issue #387: Production Hardening

## Overview

This plan breaks down issue #387 into logical segments ordered by dependencies, risk, and complexity. Each segment is designed to be independently testable and deliverable.

---

## Phase 1: Foundation & Quick Wins (Week 1)

**Goal**: Establish foundation, fix critical data quality issues, and get quick wins.

### Segment 1.1: Remove "Guest" Placeholder (2-3 hours)
**Priority**: Critical - Data quality blocker
**Complexity**: Low
**Dependencies**: None

**Why first**: Simplest fix, removes immediate data contamination, enables testing of other fixes.

**Tasks**:
- [ ] Remove "Guest" from `DEFAULT_SPEAKER_NAMES` in `speaker_detection.py`
- [ ] Update speaker detection to use `null` or `unknown_guest_N` instead
- [ ] Update metadata generation to handle `null` speakers
- [ ] Update tests to verify no "Guest" in metadata
- [ ] Run tests: `make test-unit` (speaker detection tests)

**Files**:
- `src/podcast_scraper/providers/ml/speaker_detection.py`
- `src/podcast_scraper/workflow/metadata_generation.py`
- `tests/unit/podcast_scraper/providers/ml/test_speaker_detection.py`

**Validation**: Run pipeline on test episode, verify metadata has no "Guest" entries.

---

### Segment 1.2: Pin LED Revision (2-3 hours)
**Priority**: Critical - Reproducibility blocker
**Complexity**: Low
**Dependencies**: None

**Why second**: Quick fix, prevents future reproducibility issues, low risk.

**Tasks**:
- [ ] Find current LED revision SHA (check HuggingFace model card)
- [ ] Add `LED_BASE_16384_REVISION` constant to `config_constants.py`
- [ ] Update model loader to use pinned revision
- [ ] Add validation: warn if revision is not SHA
- [ ] Add fail-fast: error if production run and unpinned
- [ ] Store revision in run manifest
- [ ] Store revision in episode metadata
- [ ] Update tests

**Files**:
- `src/podcast_scraper/config_constants.py`
- `src/podcast_scraper/providers/ml/model_loader.py`
- `src/podcast_scraper/workflow/run_manifest.py`
- `src/podcast_scraper/workflow/metadata_generation.py`

**Validation**: Run pipeline, check run manifest and metadata for revision SHA.

---

### Segment 1.3: Add Short-Chunk Guard (3-4 hours)
**Priority**: Medium - Efficiency improvement
**Complexity**: Medium
**Dependencies**: None

**Why third**: Prevents HF warnings, improves efficiency, relatively isolated change.

**Tasks**:
- [ ] Add chunk size threshold constant (80-120 tokens)
- [ ] Detect tiny tail chunks during chunking
- [ ] Implement merge logic (merge into previous chunk)
- [ ] Add guard before map-stage summarization
- [ ] Add logging when chunks are merged
- [ ] Update tests

**Files**:
- `src/podcast_scraper/providers/ml/summarizer.py`
- `tests/unit/podcast_scraper/providers/ml/test_summarizer.py`

**Validation**: Run summarization, verify no HF warnings about tiny inputs.

---

## Phase 2: Speaker Detection Improvements (Week 1-2)

**Goal**: Fix false positives in guest detection with multi-layer validation.

### Segment 2.1: Add Guest-Intent Cue Detection (4-5 hours)
**Priority**: Critical - Data quality blocker
**Complexity**: Medium
**Dependencies**: Segment 1.1 (removed "Guest")

**Why first in Phase 2**: Core validation logic, needed for other speaker detection improvements.

**Tasks**:
- [ ] Expand `INTERVIEW_INDICATOR_PATTERNS` with all cue types:
  - "with [name]", "interview with [name]", "we're joined by [name]"
  - "talks to [name]", "speaks with [name]", "our guest [name]"
  - "joining us [name]"
- [ ] Create `_has_guest_intent_cue(name, text)` function
- [ ] Update guest detection to require guest-intent cue
- [ ] Add tests with real-world examples
- [ ] Test with "Who Is the New Fed Chair?" episode

**Files**:
- `src/podcast_scraper/providers/ml/speaker_detection.py`
- `tests/unit/podcast_scraper/providers/ml/test_speaker_detection.py`

**Validation**: Run on problematic episodes, verify "Trump" not detected as guest.

---

### Segment 2.2: Implement Two-Signal Validation (5-6 hours)
**Priority**: Critical - Data quality blocker
**Complexity**: Medium
**Dependencies**: Segment 2.1

**Why second**: Fallback validation when cues aren't present, improves accuracy.

**Tasks**:
- [ ] Implement signal A: Check if name appears in description/title
- [ ] Implement signal B options:
  - [ ] Check transcript patterns ("GUEST:", "—", "I'm X" intro)
  - [ ] Calculate NER frequency in first N minutes/chars
  - [ ] Check screenplay-style speaker labels
- [ ] Combine signals: require both A and B
- [ ] Add tests for two-signal validation
- [ ] Update guest detection logic

**Files**:
- `src/podcast_scraper/providers/ml/speaker_detection.py`
- `tests/unit/podcast_scraper/providers/ml/test_speaker_detection.py`

**Validation**: Test with episodes that have sparse metadata.

---

### Segment 2.3: Two-Pass Extraction & RSS Classification (4-5 hours)
**Priority**: High - Data quality improvement
**Complexity**: Medium
**Dependencies**: Segment 2.1, 2.2

**Why third**: Uses transcript as ground truth, improves accuracy.

**Tasks**:
- [ ] Implement Pass 1: Extract speakers from transcript (screenplay format)
- [ ] Implement Pass 2: Use RSS/title/description as hints only
- [ ] Prefer transcript-derived speakers over RSS metadata
- [ ] Add organization detection for RSS author tags
- [ ] Classify RSS authors as `publisher`/`network` when appropriate
- [ ] Update metadata generation to use transcript-derived speakers
- [ ] Add tests

**Files**:
- `src/podcast_scraper/providers/ml/speaker_detection.py`
- `src/podcast_scraper/rss/parser.py`
- `src/podcast_scraper/workflow/metadata_generation.py`

**Validation**: Test with episodes where RSS metadata is incorrect.

---

### Segment 2.4: Make Relaxed Filter Opt-In (2-3 hours)
**Priority**: Medium - Configuration improvement
**Complexity**: Low
**Dependencies**: Segment 2.1, 2.2

**Why last in Phase 2**: Configuration change, depends on strict validation being in place.

**Tasks**:
- [ ] Add `relaxed_guest_detection` config flag (default: `False`)
- [ ] Update guest detection to check config flag
- [ ] Only use relaxed filter when flag is enabled
- [ ] Update documentation
- [ ] Update tests

**Files**:
- `src/podcast_scraper/config.py`
- `src/podcast_scraper/providers/ml/speaker_detection.py`
- `docs/api/CONFIGURATION.md`

**Validation**: Test with flag enabled/disabled.

---

## Phase 3: Entity Normalization & Reconciliation (Week 2)

**Goal**: Fix entity mismatches with normalization and fuzzy matching.

### Segment 3.1: Entity Normalization (4-5 hours)
**Priority**: High - Quality improvement
**Complexity**: Medium
**Dependencies**: None (can work in parallel with Phase 2)

**Why first**: Foundation for fuzzy matching and reconciliation.

**Tasks**:
- [ ] Create `_normalize_entity(name)` function:
  - Lowercase, strip punctuation
  - Extract full name and last name
  - Create aliases (full name, last name)
- [ ] Create `EntityAlias` model (canonical form + aliases)
- [ ] Update entity extraction to normalize entities
- [ ] Store normalized form and aliases in metadata
- [ ] Add tests for normalization

**Files**:
- `src/podcast_scraper/workflow/metadata_generation.py`
- `src/podcast_scraper/models.py` (EntityAlias model)
- `tests/unit/podcast_scraper/test_metadata_generation.py`

**Validation**: Test normalization with various name formats.

---

### Segment 3.2: Fuzzy Matching with Constraints (5-6 hours)
**Priority**: High - Quality improvement
**Complexity**: Medium
**Dependencies**: Segment 3.1

**Why second**: Uses normalization, implements matching logic.

**Tasks**:
- [ ] Enhance `_calculate_levenshtein_distance` (already exists)
- [ ] Implement last-name matching logic:
  - Check if last name is rare (not common words)
  - Check if paired with first name elsewhere
- [ ] Add common word rejection list
- [ ] Implement fuzzy matching with constraints
- [ ] Update `_reconcile_entities` to use fuzzy matching
- [ ] Populate `corrected_entities` field
- [ ] Add tests for fuzzy matching

**Files**:
- `src/podcast_scraper/workflow/metadata_generation.py`
- `tests/unit/podcast_scraper/test_metadata_generation.py`

**Validation**: Test with "Warsh" vs "Walsh" scenario.

---

### Segment 3.3: Entity Extraction from Summary (4-5 hours)
**Priority**: High - Quality improvement
**Complexity**: Medium
**Dependencies**: Segment 3.1, 3.2

**Why third**: Uses normalization and matching, completes entity set.

**Tasks**:
- [ ] Run NER on summary text (in addition to transcript)
- [ ] Union entities from transcript + summary
- [ ] Track provenance (transcript vs summary)
- [ ] Update metadata schema to include provenance
- [ ] Add tests

**Files**:
- `src/podcast_scraper/workflow/metadata_generation.py`
- `src/podcast_scraper/providers/ml/ner_extraction.py`
- `src/podcast_scraper/models.py`

**Validation**: Test with episodes where summary has additional entities.

---

### Segment 3.4: QA Backfill & Mismatch Severity (3-4 hours)
**Priority**: High - Quality improvement
**Complexity**: Medium
**Dependencies**: Segment 3.3

**Why last**: Uses entity extraction, adds backfill logic.

**Tasks**:
- [ ] Implement QA backfill rule:
  - Detect missing entities in summary
  - Backfill with `source=summary_backfill` tag
- [ ] Update mismatch severity logic:
  - Only flag zero-evidence entities as high severity
  - Normalization issues stay as debug/info
- [ ] Update `_check_entity_consistency` to use new logic
- [ ] Update QA flags
- [ ] Add tests

**Files**:
- `src/podcast_scraper/workflow/metadata_generation.py`
- `tests/unit/podcast_scraper/test_metadata_generation.py`

**Validation**: Test with episodes that have entity mismatches.

---

## Phase 4: Summary Quality & Faithfulness (Week 2-3)

**Goal**: Add faithfulness checks to detect hallucinations.

### Segment 4.1: Summary Faithfulness Check (5-6 hours)
**Priority**: High - Quality improvement
**Complexity**: Medium
**Dependencies**: Segment 3.3 (entity extraction)

**Why first**: Uses entity extraction, adds quality guardrail.

**Tasks**:
- [ ] Extract top N entities from transcript (by frequency)
- [ ] Extract entities from episode description
- [ ] Create source entity set (union)
- [ ] Extract entities from summary
- [ ] Compare summary entities with source entities
- [ ] Flag unverifiable entities in `qa_flags`
- [ ] Add `summary_entity_out_of_source` to QA flags
- [ ] Log which entities are out of source
- [ ] Add tests

**Files**:
- `src/podcast_scraper/workflow/metadata_generation.py`
- `src/podcast_scraper/models.py` (QA flags)
- `tests/unit/podcast_scraper/test_metadata_generation.py`

**Validation**: Test with summaries that contain hallucinations.

---

### Segment 4.2: Optional 2nd-Pass Distill Prompt (4-5 hours)
**Priority**: Low - Optional enhancement
**Complexity**: Medium
**Dependencies**: Segment 4.1

**Why second**: Optional enhancement, can be deferred if needed.

**Tasks**:
- [ ] Create distill prompt template
- [ ] Implement 2nd-pass summarization (if flag enabled)
- [ ] Add config flag for 2nd-pass
- [ ] Update tests
- [ ] Document when to use

**Files**:
- `src/podcast_scraper/workflow/metadata_generation.py`
- `src/podcast_scraper/config.py`
- `docs/api/CONFIGURATION.md`

**Validation**: Test with hallucination-prone summaries.

---

## Phase 5: Metrics & Observability (Week 3)

**Goal**: Improve metrics to make performance tuning data-driven.

### Segment 5.1: Preprocessing Metrics (3-4 hours)
**Priority**: Medium - Observability
**Complexity**: Low
**Dependencies**: None

**Why first**: Quick win, improves observability.

**Tasks**:
- [ ] Add `preprocessing_wall_ms` metric (even for cache hits)
- [ ] Add `preprocessing_cache_hit_ms` and `preprocessing_cache_miss_ms`
- [ ] Add `preprocessing_cache_hit` boolean
- [ ] Add `preprocessing_saved_bytes`
- [ ] Track audio metadata (bitrate, sample rate, codec, channels)
- [ ] Update metrics schema
- [ ] Update tests

**Files**:
- `src/podcast_scraper/workflow/metrics.py`
- `src/podcast_scraper/preprocessing/core.py`
- `src/podcast_scraper/workflow/metadata_generation.py`

**Validation**: Run pipeline, check metrics show preprocessing data.

---

### Segment 5.2: Split `io_and_waiting` into Sub-buckets (5-6 hours)
**Priority**: Medium - Observability
**Complexity**: Medium
**Dependencies**: None

**Why second**: More complex but valuable for performance tuning.

**Tasks**:
- [ ] Add sub-bucket metrics to `Metrics` class:
  - `time_download_wait_seconds`
  - `time_transcription_wait_seconds`
  - `time_summarization_wait_seconds`
  - `time_thread_sync_seconds`
  - `time_queue_wait_seconds`
- [ ] Instrument each wait time location
- [ ] Keep `time_io_and_waiting` as sum (backward compatibility)
- [ ] Update metrics schema
- [ ] Update logging
- [ ] Add tests

**Files**:
- `src/podcast_scraper/workflow/metrics.py`
- `src/podcast_scraper/workflow/orchestration.py`
- `src/podcast_scraper/workflow/stages/transcription.py`
- `src/podcast_scraper/workflow/stages/processing.py`

**Validation**: Run pipeline, check metrics show sub-bucket breakdown.

---

### Segment 5.3: Metrics Hygiene (2-3 hours)
**Priority**: Low - Data consistency
**Complexity**: Low
**Dependencies**: Segment 5.1, 5.2

**Why last**: Cleanup task, ensures metrics are consistent.

**Tasks**:
- [ ] Audit metrics writing locations
- [ ] Ensure single write point with complete metrics
- [ ] Add validation before writing
- [ ] Fix any partial writes
- [ ] Add tests

**Files**:
- `src/podcast_scraper/workflow/metrics.py`
- `src/podcast_scraper/workflow/orchestration.py`

**Validation**: Run pipeline multiple times, verify metrics consistency.

---

## Phase 6: Performance Optimizations (Week 3-4)

**Goal**: Improve throughput and efficiency.

### Segment 6.1: spaCy Model Reuse (3-4 hours)
**Priority**: Medium - Efficiency improvement
**Complexity**: Low
**Dependencies**: None

**Why first**: Quick win, reduces overhead.

**Tasks**:
- [ ] Audit all spaCy model loading locations
- [ ] Ensure preloaded instance is passed through to all consumers
- [ ] Remove redundant model loading calls
- [ ] Add logging to track model reuse
- [ ] Update tests

**Files**:
- `src/podcast_scraper/providers/ml/speaker_detection.py`
- `src/podcast_scraper/providers/ml/ner_extraction.py`
- `src/podcast_scraper/workflow/orchestration.py`
- `src/podcast_scraper/workflow/stages/setup.py`

**Validation**: Run pipeline, verify single spaCy load in logs.

---

### Segment 6.2: Per-Stage Device Configuration (5-6 hours)
**Priority**: Medium - Throughput improvement
**Complexity**: Medium
**Dependencies**: None

**Why second**: More complex, allows CPU/GPU overlap.

**Tasks**:
- [ ] Add `transcription_device` and `summarization_device` config options
- [ ] Update device selection logic
- [ ] Allow CPU/GPU mix to regain overlap
- [ ] Update MPS exclusive logic to handle mixed devices
- [ ] Document performance trade-offs
- [ ] Add metrics to track device usage per stage
- [ ] Update tests

**Files**:
- `src/podcast_scraper/config.py`
- `src/podcast_scraper/workflow/orchestration.py`
- `docs/api/CONFIGURATION.md`

**Validation**: Test with CPU/GPU mix, verify overlap.

---

## Testing Strategy

### Unit Tests
- Run after each segment: `make test-unit`
- Focus on changed modules
- Add new tests for new functionality

### Integration Tests
- Run after Phase 1: `make test-integration`
- Run after Phase 2: Test speaker detection with real episodes
- Run after Phase 3: Test entity extraction and reconciliation

### E2E Tests
- Run after Phase 1: `make test-e2e-fast`
- Run after Phase 2: Test with problematic episodes
- Run after Phase 3: Test entity quality
- Run after Phase 4: Test faithfulness detection
- Run after Phase 5: Test metrics output
- Run after Phase 6: Test performance improvements

### Manual Testing
- Test with "Who Is the New Fed Chair?" episode after Phase 2
- Test with episodes that have entity mismatches after Phase 3
- Test with hallucination-prone summaries after Phase 4
- Verify metrics after Phase 5
- Measure performance improvements after Phase 6

---

## Risk Mitigation

### High-Risk Segments
- **Segment 2.1-2.3**: Speaker detection changes could break existing functionality
  - **Mitigation**: Extensive testing with real episodes, backward compatibility checks
- **Segment 3.2**: Fuzzy matching could introduce false positives
  - **Mitigation**: Conservative thresholds, extensive testing, logging

### Medium-Risk Segments
- **Segment 4.1**: Faithfulness check could flag too many false positives
  - **Mitigation**: Tune thresholds, add logging, make it informative not blocking
- **Segment 6.2**: Device configuration changes could affect stability
  - **Mitigation**: Default to safe behavior, extensive testing

---

## Estimated Timeline

- **Week 1**: Phase 1 (Foundation) + Phase 2 (Speaker Detection) - Segments 1.1-1.3, 2.1-2.2
- **Week 2**: Phase 2 (cont.) + Phase 3 (Entity Normalization) - Segments 2.3-2.4, 3.1-3.4
- **Week 3**: Phase 4 (Summary Quality) + Phase 5 (Metrics) - Segments 4.1-4.2, 5.1-5.3
- **Week 4**: Phase 6 (Performance) + Final Testing - Segments 6.1-6.2

**Total Estimated Time**: 3-4 weeks (assuming full-time focus)

---

## Success Criteria

### Phase 1 Complete
- ✅ No "Guest" in metadata
- ✅ LED revision pinned
- ✅ No HF warnings about tiny chunks

### Phase 2 Complete
- ✅ "Trump" not detected as guest in "Who Is the New Fed Chair?"
- ✅ Guest detection requires guest-intent cues or two signals
- ✅ RSS orgs classified correctly

### Phase 3 Complete
- ✅ Entity normalization working
- ✅ "Warsh" vs "Walsh" reconciled correctly
- ✅ Entities extracted from transcript + summary
- ✅ `corrected_entities` populated

### Phase 4 Complete
- ✅ Faithfulness check flags hallucinations
- ✅ `summary_entity_out_of_source` in QA flags

### Phase 5 Complete
- ✅ Preprocessing metrics show cache hit/miss data
- ✅ `io_and_waiting` split into sub-buckets
- ✅ Metrics consistent across runs

### Phase 6 Complete
- ✅ spaCy model loaded once
- ✅ Per-stage device configuration working
- ✅ Performance improvements measurable

---

## Notes

- Each segment should be independently testable
- Commit after each segment (or logical group)
- Run `make ci-fast` before committing
- Update documentation as you go
- Keep tests passing throughout
