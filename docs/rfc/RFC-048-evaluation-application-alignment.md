# RFC-048: Evaluation ↔ Application Tightening & Alignment

- **Status**: Draft
- **Authors**: Marko Dragoljevic
- **Stakeholders**: ML evaluation team, Core developers
- **Related RFCs**:
  - `docs/rfc/RFC-015-ai-experiment-pipeline.md` (AI Experiment Pipeline)
  - `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md` (Benchmarking Framework)
  - `docs/rfc/RFC-045-ml-model-optimization-guide.md` (ML Model Optimization)
  - `docs/rfc/RFC-046-materialization-architecture.md` (Materialization Architecture)

## Abstract

Over multiple iterations, we identified that evaluation runs and application (prod/dev) runs can diverge in behavior despite using the same underlying models. This RFC documents the concrete alignment rules, decisions made, and remaining optional improvements, so the system can ship cleanly now while leaving a clear path for future evaluation hardening.

This RFC explicitly separates:

- **What is required to ship now**
- **What is optional and deferred (eval-only enhancements)**

**Architecture Alignment:** This RFC ensures that evaluation results are representative of application behavior by enforcing a single code path, explicit parameter configuration, and comprehensive fingerprinting. It aligns with RFC-015 (AI Experiment Pipeline) and RFC-041 (Benchmarking Framework) by establishing clear contracts between evaluation and production execution.

## Problem Statement

Evaluation runs and application runs can diverge in behavior despite using the same underlying models. This divergence creates several problems:

1. **Hidden parameter drift**: Implicit defaults in eval configs differ from app configs
2. **Unrepresentative results**: Evaluation metrics don't reflect actual production behavior
3. **Debugging difficulty**: When production behavior differs from eval, root cause is unclear
4. **Silent behavior changes**: Defaults and implicit logic can change without being tracked

**Key Findings:**

1. **Eval vs App Differences Are Normal — But Must Be Controlled**

   Differences observed were caused by:
   - Implicit defaults vs explicit config values
   - Eval-only scoring logic (gates, caps, scope filters)
   - Different preprocessing profiles applied implicitly
   - Reduce-stage parameter overrides triggered dynamically

   **Conclusion:**
   Evaluation and application must share the same execution path. Any difference must be:
   - Explicit in config
   - Logged clearly
   - Optional (never implicit)

**Use Cases:**

1. **Production deployment**: Ensure that what was evaluated is what gets deployed
2. **Regression detection**: Identify when eval and prod behavior diverge
3. **Reproducibility**: Recreate production behavior from eval artifacts
4. **Debugging**: Trace production issues back to eval runs

## Goals

1. **Ensure evaluation results are representative of application behavior**
2. **Avoid hidden parameter drift between eval configs and app configs**
3. **Make all summarization / NER behavior explainable from logs + config**
4. **Ship a stable production version without introducing a new eval framework chapter**

## Non-goals

- Building dashboards
- Introducing new eval modes or presets
- Changing baseline model choices

## Design & Implementation

### Alignment Rules (Authoritative)

#### Rule 1: One Code Path

**Requirement:**
Eval runs must call the same providers and pipeline stages as the app.

**Implementation:**
- No duplicated logic ("eval summarizer", "prod summarizer")
- Shared provider initialization
- Shared preprocessing pipeline
- Shared generation logic

**Allowed differences:**
- Scorers (eval-only, read-only observers)
- Reference loading (eval-only)
- Gating thresholds (eval-only validation)

#### Rule 2: Explicit Parameters Only

**Requirement:**
All ML behavior must be driven by config.

**Required config sections:**
- `map_params`
- `reduce_params`
- `tokenization`
- `chunking`
- `preprocessing_profile`

**No silent defaults for:**
- `early_stopping`
- `min_new_tokens`
- `max_new_tokens`
- `no_repeat_ngram_size`

**Implementation:**
- All parameters must appear in config and fingerprint
- If a value matters, it must be explicit
- Defaults are only allowed for non-behavioral settings (e.g., logging level)

#### Rule 3: Preprocessing Is Part of the Model

**Requirement:**
Preprocessing materially affects output quality and must be treated as part of the model contract.

**Decisions:**
- Preprocessing profile (e.g. `cleaning_v4`) is treated as part of the model contract
- Profile must be:
  - Explicit in config
  - Logged during execution
  - Stored in fingerprints

**Implementation:**
- Eval and app must use the same preprocessing profile unless intentionally testing differences
- Profile version is included in fingerprint
- Profile changes are tracked as model changes

#### Rule 4: Dynamic Safeguards Are Allowed (and Required)

**Requirement:**
Dynamic controls that protect model behavior are not eval-only hacks. They are core runtime safety.

**Examples:**
- Capping `max_new_tokens` based on input size
- Forcing `min_new_tokens=0` to prevent expansion
- Switching reduce strategy based on combined input size

**Implementation:**
- These safeguards:
  - Run in both eval and app
  - Are logged with clear reasoning
  - Appear in validation logs
- Safeguard logic is part of the provider, not the scorer

#### Rule 5: Scoring Never Mutates Behavior

**Requirement:**
Scorers are read-only observers.

**Implementation:**
- Scorers must:
  - Never change generation parameters
  - Never filter or rewrite predictions
  - Never affect chunking or reduce decisions
- All filtering (e.g. NER scope filtering) happens inside the scorer only
- Scorers operate on predictions after they are generated

### Summarization Alignment (Final)

#### What Is Locked In

- MAP/REDUCE pipeline
- Dynamic reduce capping logic
- Expansion prevention via runtime caps
- Cleaning profile included in fingerprint

#### What Is Explicitly Deferred

- Eval-only "no expansion ever" policies
- Special eval presets
- Length-normalized eval scoring

**Note:** These may be added later without changing prod behavior.

### NER Alignment (Final)

#### Key Decisions

- **Gold entities define scope, not positions**: Position mismatch = FP by design
- **Scope-aware filtering**: Only ignores out-of-scope predictions
- **Entity identity matters more than exact span offsets**: Matches real-world KG usage

#### Alignment Rule

NER in eval and app:
- Uses identical spaCy pipeline
- Uses identical preprocessing input
- Differs only in scorer and reference loading

### Artifacts Per Run (Required)

**Requirement:**
Every run (eval or app) must produce:

1. `predictions.jsonl`
2. `metrics.json`
3. `metrics_report.md`
4. `fingerprint.json`
5. `run.log`

**Implementation:**
- If any are missing, the run is incomplete
- All artifacts must be generated by the same code path
- Artifacts must be validatable against schemas

### Fingerprinting Contract

**Requirement:**
Fingerprints must include:

- Model IDs (raw HF IDs)
- Transformers version
- Preprocessing profile
- Effective generation parameters (after caps)
- Chunking strategy

**Implementation:**
- Fingerprints are the source of truth for explaining behavior
- Fingerprints must be deterministic and reproducible
- Fingerprint mismatches indicate behavior differences

## Key Decisions

1. **Single Code Path**
   - **Decision**: Eval and app use identical execution path
   - **Rationale**: Ensures evaluation results are representative of production behavior

2. **Explicit Parameters**
   - **Decision**: All behavioral parameters must be explicit in config
   - **Rationale**: Prevents hidden drift and makes behavior explainable

3. **Preprocessing as Model Contract**
   - **Decision**: Preprocessing profile is part of the model fingerprint
   - **Rationale**: Preprocessing materially affects output quality

4. **Dynamic Safeguards in Provider**
   - **Decision**: Runtime safety logic runs in provider, not scorer
   - **Rationale**: Safeguards are part of production behavior, not eval-only

5. **Scorers Are Read-Only**
   - **Decision**: Scorers never mutate behavior
   - **Rationale**: Ensures scoring doesn't affect production-equivalent behavior

6. **NER Scope-Based Evaluation**
   - **Decision**: Gold entities define scope, position mismatches are FPs
   - **Rationale**: Matches real-world knowledge graph usage patterns

## Shipping Readiness Checklist

You are ready to ship when:

- ✅ Eval and app use identical code path
- ✅ All parameters are explicit in config
- ✅ Preprocessing profile is in fingerprint
- ✅ Dynamic safeguards are logged
- ✅ Scorers are read-only
- ✅ All required artifacts are generated
- ✅ Fingerprints are complete and deterministic

## Deferred Work (Next Chapter)

Explicitly out of scope for this release:

- Eval presets (eval-fast, eval-strict)
- Dashboarding
- Benchmark dataset integration for NER
- LLM-based NER fallback

**Note:** These can be layered later without breaking shipped behavior.

## Benefits

1. **Representative Evaluation**: Eval results accurately reflect production behavior
2. **No Hidden Drift**: All differences between eval and app are explicit and tracked
3. **Explainable Behavior**: All behavior can be explained from logs and config
4. **Stable Production**: Production version ships without eval framework churn
5. **Clear Path Forward**: Deferred work is clearly documented and can be added later

## Final Note

This alignment work ensures that:

- **What you evaluate is what you ship.**
- **No hidden behavior. No silent drift. No eval illusions.**

## References

- **Related RFC**: `docs/rfc/RFC-015-ai-experiment-pipeline.md`
- **Related RFC**: `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md`
- **Related RFC**: `docs/rfc/RFC-045-ml-model-optimization-guide.md`
- **Related RFC**: `docs/rfc/RFC-046-materialization-architecture.md`
- **Source Code**: `src/podcast_scraper/evaluation/`, `src/podcast_scraper/providers/ml/`
