# RFC-046: Materialization Architecture

- **Status**: Draft
- **Authors**: Marko Dragoljevic
- **Stakeholders**: AI/ML team, Evaluation pipeline users
- **Related PRDs**:
  - `docs/prd/PRD-007-ai-experiment-pipeline.md`
- **Related RFCs**:
  - `docs/rfc/RFC-015-ai-experiment-pipeline.md`
  - `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md`
  - `docs/rfc/RFC-045-ml-model-optimization-guide.md`

## Abstract

This RFC proposes shifting preprocessing from a run-time parameter to a dataset materialization parameter. Instead of runs deciding how to preprocess input text, preprocessing becomes part of the dataset definition through explicit materialization configs. This ensures honest, reproducible comparisons between experiment runs by making the input contract explicit and frozen.

**Architecture Alignment:** This extends RFC-015 (AI Experiment Pipeline) and RFC-041 (Benchmarking Framework) by formalizing the relationship between datasets, preprocessing, and experiment runs.

## Problem Statement

Currently, preprocessing profiles are specified in run configurations alongside model parameters. This creates several problems:

1. **Ambiguous comparisons**: Two runs with different `preprocessing_profile` values reference the same `dataset_id`, making it easy to accidentally compare "apples to oranges"
2. **Hidden input differences**: The actual input to models depends on runtime parameters, not the dataset definition
3. **Reproducibility risk**: Changing a preprocessing profile retroactively changes what "the dataset" means

**Evidence from experiments:**

| Experiment  | Preprocessing | Speaker Leak Rate | Notes                              |
| ----------- | ------------- | ----------------- | ---------------------------------- |
| v1 baseline | cleaning_v3   | 80%               | Original                           |
| v7          | cleaning_v4   | 0%                | Same dataset_id, different profile |

The v7 experiment showed an 80 percentage point improvement by changing only the preprocessing profile. This massive impact should not be a casual run parameter—it fundamentally changes what the input data is.

**Use Cases:**

1. **Fair model comparison**: Compare BART vs LED using identical preprocessed inputs
2. **Preprocessing A/B testing**: Compare cleaning_v3 vs cleaning_v4 as explicit dataset variants
3. **Provider-specific optimization**: Allow ML providers to use aggressive preprocessing while LLMs use minimal cleanup

## Goals

1. **Explicit input contracts**: Materialization ID makes it unambiguous what inputs were used
2. **Reproducible comparisons**: Materialized inputs are frozen and versioned
3. **Honest evaluation**: Cannot accidentally compare runs with different preprocessing
4. **Provider flexibility**: Support provider-specific adapters while keeping canonical cleanup shared
5. **Backward compatibility**: Support existing configs during migration period

## Constraints & Assumptions

**Constraints:**

- Must not break existing experiment configs (deprecation, not removal)
- Materialized datasets must be storage-efficient (text only, no duplication)
- Must integrate with existing fingerprinting system

**Assumptions:**

- Preprocessing has significant impact on model output quality (validated by v7 experiment)
- Different providers may need different preprocessing strategies
- Chunking is model-dependent and should remain a run parameter

## Design & Implementation

### 1. Core Principle

> **Materialization is part of the dataset definition. Runs should not decide what the dataset "looks like."**

This means:

- Runs reference a `materialization_id`
- Materialization is produced by: `(dataset_id + canonical_profile + adapter)`
- Experiments comparing preprocessing become comparisons of different materialized datasets

### 2. Two-Layer Preprocessing Model

#### Layer A: Canonical Cleanup (Shared)

Minimum cleanup that is always safe, even for strong LLMs:

| Cleanup                 | Description                       | Safe for LLMs?        |
| ----------------------- | --------------------------------- | --------------------- |
| Remove junk lines       | `////`, `=-`, `___`, etc.         | ✅ Yes                |
| Normalize whitespace    | Collapse blank lines              | ✅ Yes                |
| Remove stage directions | `[music]`, `(pause)`              | ✅ Yes (configurable) |
| Strip headers           | Episode titles, `Host:`, `Guest:` | ✅ Yes                |

This becomes the **canonical materialization**: "what summarization input means in our system."

#### Layer B: Provider Adapter (Optional)

More aggressive transforms applied per-provider:

| Adapter                  | Description                    | Use Case               |
| ------------------------ | ------------------------------ | ---------------------- |
| `adapter_none`           | No additional transforms       | LLMs, fair comparison  |
| `adapter_ml_dialogue_v1` | Speaker anonymization (A:, B:) | ML models (BART, LED)  |
| `adapter_narrative_v1`   | Convert dialogue to narrative  | Future                 |

**Key insight from experiments:** Speaker anonymization eliminated 80% speaker leak in ML models but may not be needed for LLMs.

### 3. What Goes Where

| In Materialization Config | In Run Config          |
| ------------------------- | ---------------------- |
| Source dataset_id         | Materialization ID     |
| Canonical cleanup rules   | Model selection        |
| Adapter selection         | Generation parameters  |
| Output formatting         | **Chunking strategy**  |
|                           | **Chunk size/overlap** |

**Important:** Chunking stays in run config because:

- Chunking is model-dependent (BART: 1024 tokens, LED: 4096 tokens)
- Different models may need different chunk sizes from the same clean text
- Materializing chunks would lock to one size

### 4. Materialization Config Schema

```yaml
# data/eval/materializations/summarization_canonical_v1.yaml
id: "summarization_canonical_v1"
version: "1.0.0"
task: "summarization"

source:
  dataset_id: "curated_5feeds_smoke_v1"

canonical:
  remove_junk_lines: true        # ////, =-, etc.
  remove_stage_directions: true  # [music], (pause)
  normalize_whitespace: true
  strip_headers: true            # Episode titles, Host:/Guest:

adapter:
  id: "none"  # or "ml_dialogue_v1"
```

With adapter:

```yaml
# data/eval/materializations/summarization_canonical_v1__ml_dialogue.yaml
id: "summarization_canonical_v1__ml_dialogue"
version: "1.0.0"
task: "summarization"

source:
  dataset_id: "curated_5feeds_smoke_v1"

canonical:
  remove_junk_lines: true
  remove_stage_directions: true
  normalize_whitespace: true
  strip_headers: true

adapter:
  id: "ml_dialogue_v1"
  config:
    anonymize_speakers: true  # Maya: → A:
    remove_speaker_roles: true  # (host), (guest)
```

### 5. Run Config Changes

**Before (current):**

```yaml
id: "baseline_bart_v7"

data:
  dataset_id: "curated_5feeds_smoke_v1"

preprocessing_profile: "cleaning_v4"  # Hidden input change!

backend:
  type: "hf_local"
  map_model: "bart-small"
```

**After (proposed):**

```yaml
id: "baseline_bart_v7"

data:
  materialization_id: "summarization_canonical_v1__ml_dialogue"

backend:
  type: "hf_local"
  map_model: "bart-small"
  reduce_model: "long-fast"

# Chunking is HERE, not in materialization
chunking:
  strategy: "word_chunking"
  word_chunk_size: 900
  word_overlap: 150

map_params:
  max_new_tokens: 200
  # ...
```

### 6. Folder Structure

**Materialization configs:**

```text
data/eval/materializations/
  summarization_canonical_v1.yaml
  summarization_canonical_v1__ml_dialogue.yaml
  summarization_canonical_v2.yaml  # future version
```

**Materialized outputs:**

```text
data/eval/materialized/
  summarization_canonical_v1/
    curated_5feeds_smoke_v1/
      p01_e01.txt
      p01_e02.txt
      index.json
  summarization_canonical_v1__ml_dialogue/
    curated_5feeds_smoke_v1/
      p01_e01.txt
      p01_e02.txt
      index.json
```

**Index file:**

```json
{
  "materialization_id": "summarization_canonical_v1__ml_dialogue",
  "source_dataset_id": "curated_5feeds_smoke_v1",
  "created_at": "2026-02-01T12:00:00Z",
  "canonical_version": "1.0.0",
  "adapter": "ml_dialogue_v1",
  "episode_count": 5,
  "episodes": ["p01_e01", "p01_e02", "..."]
}
```

### 7. Evaluation Honesty Framework

#### A) Comparable Comparisons (Default)

Compare providers using:

- Same materialization_id
- Same canonical cleanup
- Adapter = `none`

This answers: "Which model performs better on identical inputs?"

#### B) Best-Effort Per Provider

Compare:

- ML with `adapter_ml_dialogue_v1`
- LLM with `adapter_none`

This answers: "What output do we ship per provider?"

**Must be labeled as not apples-to-apples.**

## Key Decisions

1. **Chunking remains in run config**
   - **Decision**: Chunking is not part of materialization
   - **Rationale**: Chunking is model-dependent; BART needs 1024 tokens, LED can handle 4096. Materializing chunks would lock to one size.

2. **Two-layer preprocessing model**
   - **Decision**: Separate canonical (shared) from adapter (provider-specific)
   - **Rationale**: Allows fair comparison (canonical only) while enabling provider optimization (with adapters)

3. **Adapter applied at materialization time**
   - **Decision**: Adapters are applied during materialization, not at runtime
   - **Rationale**: Ensures reproducibility; materialized text is frozen

4. **Version in materialization ID**
   - **Decision**: Use semantic versioning in materialization configs
   - **Rationale**: Allows evolution while preserving historical materializations

## Alternatives Considered

1. **Keep preprocessing as run parameter**
   - **Description**: Status quo, preprocessing_profile in run config
   - **Pros**: Simple, no migration needed
   - **Cons**: Ambiguous comparisons, hidden input differences
   - **Why Rejected**: v7 experiment showed 80% impact from preprocessing alone

2. **Include chunking in materialization**
   - **Description**: Materialize pre-chunked text
   - **Pros**: Fully frozen inputs
   - **Cons**: Model-dependent, would need separate materialization per chunk size
   - **Why Rejected**: Chunking is model-specific, not dataset-specific

3. **Apply adapters at runtime**
   - **Description**: Load canonical materialization, apply adapter at run time
   - **Pros**: One materialization, multiple adapters
   - **Cons**: Less reproducible, adapter is still a "hidden" run parameter
   - **Why Rejected**: Defeats purpose of explicit input contracts

## Testing Strategy

**Test Coverage:**

- **Unit tests**: Materialization config loading and validation
- **Integration tests**: Materialization generation from source datasets
- **E2E tests**: Full pipeline with materialization_id in run config

**Test Organization:**

- `tests/unit/evaluation/test_materialization_config.py`
- `tests/integration/evaluation/test_materialization_generation.py`

**Test Execution:**

- Unit tests run in CI-fast
- Integration tests run in CI-full

## Rollout & Monitoring

**Rollout Plan:**

1. **Phase 1**: Add `materialization_id` field to ExperimentConfig (optional, backward compatible)
2. **Phase 2**: Create canonical_v1 and ml_dialogue_v1 materialization configs
3. **Phase 3**: Generate materialized datasets
4. **Phase 4**: Migrate experiment configs to use materialization_id
5. **Phase 5**: Deprecate preprocessing_profile in run configs
6. **Phase 6**: Re-run baselines on new materializations

**Monitoring:**

- Track which configs use old vs new format
- Log warnings for deprecated preprocessing_profile usage

**Success Criteria:**

1. ✅ All new experiment configs use materialization_id
2. ✅ Baseline runs are reproducible via materialization
3. ✅ No accidental apples-to-oranges comparisons

## Relationship to Other RFCs

This RFC (RFC-046) extends the evaluation infrastructure:

1. **RFC-015: AI Experiment Pipeline** - Defines experiment config structure; this RFC adds materialization_id
2. **RFC-041: Benchmarking Framework** - Defines metrics and comparison; this RFC ensures honest comparisons
3. **RFC-045: ML Model Optimization** - Documents preprocessing impact; this RFC formalizes it

**Key Distinction:**

- **RFC-045**: Discovered that preprocessing matters (80% impact)
- **RFC-046**: Formalizes preprocessing as dataset definition, not run parameter

## Benefits

1. **Explicit comparisons**: Materialization ID makes it clear what inputs were used
2. **Reproducibility**: Materialized inputs are frozen and versioned
3. **Honest evaluation**: Cannot accidentally compare runs with different preprocessing
4. **Provider flexibility**: Adapters allow provider-specific optimizations
5. **Clear contract**: Runs pick a prepared input contract

## Migration Path

1. **Phase 1: Backward compatible addition**
   - Add `materialization_id` field to ExperimentConfig
   - Support both old (`dataset_id` + `preprocessing_profile`) and new (`materialization_id`) formats
   - Log deprecation warning for old format

2. **Phase 2: Create materializations**
   - Extract canonical cleanup from `cleaning_v4` → `canonical_v1`
   - Extract speaker anonymization → `adapter_ml_dialogue_v1`
   - Generate materialized datasets

3. **Phase 3: Migrate configs**
   - Update experiment configs to use `materialization_id`
   - Re-run baselines for clean comparison history

4. **Phase 4: Deprecation**
   - Remove `preprocessing_profile` from run configs
   - Keep backward compatibility layer for 1 release cycle

## Open Questions

1. **Adapter inheritance**: Can adapters extend other adapters?
   - Proposed: No, keep flat for simplicity

2. **Materialization caching**: How to handle large materialized datasets?
   - Proposed: Store as text files, compress if needed

3. **Cross-task materializations**: Same canonical for summarization and transcription?
   - Proposed: Task-specific canonicals (`summarization_canonical_v1`, `transcription_canonical_v1`)

## References

- **Related PRD**: `docs/prd/PRD-007-ai-experiment-pipeline.md`
- **Related RFC**: `docs/rfc/RFC-015-ai-experiment-pipeline.md`
- **Related RFC**: `docs/rfc/RFC-045-ml-model-optimization-guide.md`
- **Source Code**: `podcast_scraper/evaluation/config.py`
- **Experiment Evidence**: `data/eval/runs/baseline_bart_v7_cleaning_v4/`
