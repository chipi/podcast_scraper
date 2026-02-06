# ADR-046: Adaptive Summarization Routing

- **Status**: Proposed
- **Date**: 2026-02-05
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-053](../rfc/RFC-053-adaptive-summarization-routing.md)
- **Related PRDs**: [PRD-005](../prd/PRD-005-episode-summarization.md)

## Context & Problem Statement

The current summarization pipeline uses a uniform approach for all episodes: BART/LED models with MAP-REDUCE summarization, complex chunking logic, and two-pass aggregation. However, podcasts vary significantly:

- **Duration**: 10 minutes to multiple hours
- **Structure**: Monologue vs dialogue vs panel discussions
- **Content**: Technical vs abstract vs narrative
- **Speaker patterns**: Single host, interview format, roundtable discussions

A single summarization strategy does not generalize well:

- Short episodes (< 15 min) don't need complex chunking
- Dialogue-heavy episodes benefit from speaker-aware processing
- Technical content requires extraction-first strategies
- Long monologues need hierarchical chunking with strong reducers

## Decision

We adopt **Adaptive Summarization Routing** using rule-based routing with episode profiling.

1. **Episode Profiling**: Analyzes episode characteristics (duration, speaker count, turn-taking rate, entity density, numeric density, topic drift).
2. **Rule-Based Routing**: Deterministic rules route episodes to appropriate summarization strategies (short monologue, short dialogue, technical, long monologue, long dialogue, standard).
3. **Strategy Selection**: Each strategy uses appropriate models and processing (e.g., single-pass for short episodes, extraction-first for technical content).
4. **Extraction-First Artifacts**: All strategies produce structured intermediate outputs before reduction.

## Rationale

- **Optimization**: Different episode types benefit from different strategies, improving quality and efficiency
- **Deterministic**: Rule-based routing is debuggable and logged for each episode
- **Flexibility**: Can add new strategies without changing existing ones
- **Quality**: Episode-specific strategies produce better summaries than one-size-fits-all approach
- **Compatibility**: Works with existing summarization architecture (RFC-042 hybrid pipeline)

## Alternatives Considered

1. **One-Size-Fits-All**: Rejected as it misses optimization opportunities and produces inconsistent quality.
2. **Machine Learning Routing**: Rejected as it adds complexity and makes routing non-deterministic.
3. **Manual Strategy Selection**: Rejected as it requires user input and doesn't scale.

## Consequences

- **Positive**:
  - Better summary quality across diverse episode types
  - More efficient processing (no unnecessary chunking for short episodes)
  - Deterministic and debuggable routing
  - Easy to add new strategies
- **Negative**:
  - Initial implementation complexity
  - Requires episode profiling (adds processing overhead)
- **Neutral**:
  - Requires implementation of RFC-053

## Implementation Notes

- **Module**: `src/podcast_scraper/workflow/metadata_generation.py` - Summarization pipeline
- **Pattern**: Rule-based routing with episode profiling
- **Routing Thresholds**:
  - Token count < 2000 → Single-pass strategy
  - Speaker turn rate > 2.0 turns/min → Dialogue strategy
  - Entity density > 10.0 per 1000 tokens → Technical strategy
  - Duration > 60 min + multiple speakers → Panel strategy
- **Episode Profile**: Duration, speaker count, turn-taking rate, entity density, numeric density, topic drift
- **Strategies**: Short monologue, short dialogue, technical, long monologue, long dialogue, standard
- **Model Roles**: Extractor, Summarizer, Reducer, Finalizer (compatible with RFC-042)
- **Status**: 🟡 Draft RFC (RFC-053) - Not yet implemented

## References

- [RFC-053: Adaptive Summarization Routing Based on Episode Profiling](../rfc/RFC-053-adaptive-summarization-routing.md)
- [PRD-005: Episode Summarization](../prd/PRD-005-episode-summarization.md)
- [RFC-042: Hybrid Podcast Summarization Pipeline](../rfc/RFC-042-hybrid-summarization-pipeline.md) - Compatible architecture
