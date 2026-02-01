# RFC: ML Models – Next Steps for Podcast Summarization

## Status
Draft

## Owner
Podcast ML Pipeline

## Context

The current summarization pipeline uses BART and LED models with:
- Map → Reduce summarization
- Complex chunking logic
- Two-pass aggregation for final output

This approach works, but podcasts vary significantly:
- Duration: 10 minutes to multiple hours
- Structure: monologue vs dialogue vs panel
- Content: technical vs abstract vs narrative

A single summarization strategy or model does not generalize well across all cases. This RFC proposes a **tight, extensible approach** that supports multiple podcast types while keeping system complexity under control.

## Goals

- Support diverse podcast formats with consistent output quality
- Avoid model sprawl and pipeline fragmentation
- Improve faithfulness, coverage, and structure
- Enable systematic benchmarking and future model swaps

## Non-Goals

- Selecting a single “best” summarization model
- Fully replacing the current pipeline immediately
- Introducing provider-specific dependencies into core logic

## Core Design Principle

**Standardize the pipeline and outputs; vary strategy via routing.**

Models are interchangeable components assigned to stable **roles**. Routing logic selects the appropriate strategy per episode based on measurable properties.

## Model Roles

The system is structured around a small number of stable roles:

1. **Extractor (Map Pass)**
   - High recall
   - Structured outputs (facts, bullets, entities)
   - Minimal hallucination

2. **Summarizer (Map Pass)**
   - Chunk-level narrative summaries
   - Preserves salience and context

3. **Reducer / Synthesizer**
   - De-duplication and reconciliation
   - Global coherence
   - Schema and formatting compliance

4. **Finalizer (Optional)**
   - Style, tone, and output normalization
   - Often combined with reducer

Each role may have multiple candidate models, but production usage is capped.

## Episode Profiling

Before invoking any ML models, each episode is profiled using inexpensive heuristics:

- Duration (minutes)
- Transcript token count
- Speaker count (from diarization)
- Turn-taking rate (dialogue vs monologue)
- Topic drift (semantic variance over time)
- Named entity density
- Numeric density

This profile determines the summarization strategy.

## Episode Profiles

The following profiles cover most podcast types:

1. **Short Monologue (≤15 min)**
   - Single-pass summary or light chunking

2. **Short Dialogue (≤30 min)**
   - Chunk by speaker-turn blocks
   - Emphasis on “who said what”

3. **Long Monologue (30–180+ min)**
   - Hierarchical chunking
   - Strong reducer focus

4. **Long Dialogue / Panel (60–240+ min)**
   - Topic segmentation
   - Speaker-position extraction

5. **Technical / Dense Content**
   - Extraction-first strategy
   - Conservative summarization

6. **Abstract / Philosophical Content**
   - Argument and claim mapping
   - Narrative synthesis

Initial implementation may start with 3–4 profiles and expand incrementally.

## Extraction-First Intermediate Artifacts

All strategies produce structured intermediate outputs before reduction:

- `key_points[]`
- `claims[]` with supporting evidence
- `entities[]` with roles
- `numbers[]` (value, unit, context)
- `notable_quotes[]` (optional timestamps)
- `speaker_positions[]` (dialogue only)
- `definitions[]` (technical only)

Reducers operate exclusively on these artifacts, not raw transcripts.

## Model Matrix (Initial Proposal)

To avoid uncontrolled growth, production usage is capped:

- **Extractor**: FLAN-T5 (primary), BART (fallback)
- **Long-context summarizer**: LED or LongT5
- **Reducer / Finalizer**: FLAN-T5-large (quality), smaller variant (cost)

Additional models may be evaluated offline but not promoted without justification.

## Routing Rules (Deterministic)

Routing is rule-based and logged for debuggability:

- If token count < X → single-pass strategy
- If speaker turn rate > Y → dialogue strategy
- If entity density > Z → technical strategy
- If topic drift > D → topic segmentation

Each run records the profile and routing decision.

## Evaluation Strategy

Metrics are tracked **per episode profile**, not globally:

- Faithfulness (entity and number support)
- Coverage (unique key points)
- Redundancy
- Structure / schema compliance
- Cost and latency

This avoids misleading averages across heterogeneous content.

## Next Steps

1. Implement episode profiling and routing logic
2. Introduce extraction-first artifacts for all profiles
3. Upgrade reducer model for maximum quality gain
4. Benchmark LongT5 vs LED for long-context profiles
5. Establish per-profile dashboards and regression alerts

## Open Questions

- Final output schemas and variants
- Threshold tuning for routing rules
- Automated hallucination detection depth
- Long-term model promotion criteria
