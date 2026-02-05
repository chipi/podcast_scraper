# RFC-053: Adaptive Summarization Routing Based on Episode Profiling

- **Status**: Draft
- **Date**: 2026-02-05
- **Authors**:
- **Stakeholders**: Maintainers, users processing diverse podcast types, developers
  integrating summarization
- **Related PRDs**:
  - `docs/prd/PRD-005-episode-summarization.md` (Episode summarization requirements)
- **Related RFCs**:
  - `docs/rfc/RFC-012-episode-summarization.md` (Current summarization implementation)
  - `docs/rfc/RFC-042-hybrid-summarization-pipeline.md` (Hybrid MAP-REDUCE architecture)
  - `docs/rfc/RFC-052-locally-hosted-llm-models-with-prompts.md` (Local LLM models)
- **Related ADRs**:
  - `docs/adr/ADR-010-hierarchical-summarization-pattern.md` (Hierarchical summarization)

## Abstract

This RFC proposes an **adaptive routing system** for podcast summarization that selects
optimal summarization strategies based on episode characteristics (duration, structure,
content type). Instead of using a single summarization approach for all episodes, the
system profiles each episode and routes it to the most appropriate strategy. This
enables consistent output quality across diverse podcast formats while keeping system
complexity manageable.

**Key Principle:** Standardize the pipeline and outputs; vary strategy via routing.

## Problem Statement

The current summarization pipeline (RFC-012) uses a uniform approach for all episodes:
BART/LED models with MAP-REDUCE summarization, complex chunking logic, and two-pass
aggregation. However, podcasts vary significantly:

- **Duration**: 10 minutes to multiple hours
- **Structure**: Monologue vs dialogue vs panel discussions
- **Content**: Technical vs abstract vs narrative
- **Speaker patterns**: Single host, interview format, roundtable discussions

A single summarization strategy does not generalize well across all cases:

- **Short episodes** (< 15 min) don't need complex chunking
- **Dialogue-heavy episodes** benefit from speaker-aware processing
- **Technical content** requires extraction-first strategies
- **Long monologues** need hierarchical chunking with strong reducers

**Current limitations:**

1. One-size-fits-all approach misses optimization opportunities
2. No adaptation to episode characteristics
3. Quality varies significantly across episode types
4. Evaluation metrics are averaged across heterogeneous content

## Goals

1. **Support diverse podcast formats** with consistent output quality
2. **Avoid model sprawl** and pipeline fragmentation
3. **Improve faithfulness, coverage, and structure** per episode type
4. **Enable systematic benchmarking** and future model swaps
5. **Maintain simplicity** - routing logic should be deterministic and debuggable

## Non-Goals

- Selecting a single "best" summarization model
- Fully replacing the current pipeline immediately
- Introducing provider-specific dependencies into core logic
- Real-time routing decisions (profiling happens once per episode)

## Constraints & Assumptions

**Constraints:**

- Must be backward compatible with existing summarization pipeline
- Routing decisions must be deterministic and reproducible
- Profiling must be fast (< 1s per episode)
- Must work with existing providers (ML, OpenAI, Ollama, etc.)
- Routing logic must be logged for debugging

**Assumptions:**

- Episode transcripts are available before summarization
- Speaker detection results are available (for dialogue profiling)
- Episode metadata (duration, etc.) is available
- Users accept that different episodes may use different strategies

## Design & Implementation

### 1. Episode Profiling

Before invoking any ML models, each episode is profiled using inexpensive heuristics:

**Profile Metrics:**

- **Duration** (minutes) - From episode metadata
- **Transcript token count** - From transcript analysis
- **Speaker count** - From speaker detection results
- **Turn-taking rate** - Dialogue vs monologue indicator
- **Topic drift** - Semantic variance over time (optional, requires embeddings)
- **Named entity density** - Technical content indicator
- **Numeric density** - Data-heavy content indicator

**Profiling Implementation:**

```python
@dataclass
class EpisodeProfile:
    """Episode characteristics for routing decisions."""
    duration_minutes: float
    token_count: int
    speaker_count: int
    turn_taking_rate: float  # Turns per minute
    entity_density: float  # Entities per 1000 tokens
    numeric_density: float  # Numbers per 1000 tokens
    topic_drift: Optional[float] = None  # Semantic variance (optional)
```

### 2. Episode Profiles (Routing Categories)

The following profiles cover most podcast types:

#### 2.1 Short Monologue (≤15 min)

**Characteristics:**
- Duration ≤ 15 minutes
- Single speaker or minimal dialogue
- Token count < 2000

**Strategy:**
- Single-pass summary (no chunking)
- Direct summarization with BART or LED
- Minimal processing overhead

**Models:**
- BART-large (fast, good quality for short content)
- LED-base (if context window allows)

#### 2.2 Short Dialogue (≤30 min)

**Characteristics:**
- Duration ≤ 30 minutes
- Multiple speakers (2-4)
- High turn-taking rate
- Token count < 4000

**Strategy:**
- Chunk by speaker-turn blocks
- Emphasis on "who said what"
- Speaker-aware summarization

**Models:**
- BART-large with speaker-aware chunking
- LED-base for longer dialogues

#### 2.3 Long Monologue (30-180+ min)

**Characteristics:**
- Duration > 30 minutes
- Single speaker or minimal dialogue
- Token count > 4000

**Strategy:**
- Hierarchical chunking
- Strong reducer focus
- MAP-REDUCE with LED or LongT5

**Models:**
- LED-large or LongT5-large (MAP phase)
- Instruction-tuned LLM (REDUCE phase, RFC-042)

#### 2.4 Long Dialogue / Panel (60-240+ min)

**Characteristics:**
- Duration > 60 minutes
- Multiple speakers (3+)
- High turn-taking rate
- Token count > 8000

**Strategy:**
- Topic segmentation
- Speaker-position extraction
- Hierarchical MAP-REDUCE

**Models:**
- LED-large or LongT5-large (MAP phase)
- Instruction-tuned LLM (REDUCE phase, RFC-042)

#### 2.5 Technical / Dense Content

**Characteristics:**
- High entity density (> 10 entities per 1000 tokens)
- High numeric density (> 5 numbers per 1000 tokens)
- Technical terminology

**Strategy:**
- Extraction-first approach
- Conservative summarization
- Preserve facts, numbers, entities

**Models:**
- BART-large with extraction prompts
- LED-base for long technical content

#### 2.6 Abstract / Philosophical Content

**Characteristics:**
- Low entity density
- Low numeric density
- High topic drift
- Narrative structure

**Strategy:**
- Argument and claim mapping
- Narrative synthesis
- Abstractive summarization

**Models:**
- LED-large (better for abstract content)
- Instruction-tuned LLM (REDUCE phase, RFC-042)

### 3. Routing Rules (Deterministic)

Routing is rule-based and logged for debuggability:

```python
def route_episode(profile: EpisodeProfile) -> SummarizationStrategy:
    """Route episode to appropriate summarization strategy."""

    # Short monologue
    if profile.duration_minutes <= 15 and profile.speaker_count <= 1:
        return SummarizationStrategy.SHORT_MONOLOGUE

    # Short dialogue
    if profile.duration_minutes <= 30 and profile.turn_taking_rate > 2.0:
        return SummarizationStrategy.SHORT_DIALOGUE

    # Technical content
    if profile.entity_density > 10.0 or profile.numeric_density > 5.0:
        return SummarizationStrategy.TECHNICAL

    # Long monologue
    if profile.duration_minutes > 30 and profile.speaker_count <= 2:
        return SummarizationStrategy.LONG_MONOLOGUE

    # Long dialogue/panel
    if profile.duration_minutes > 60 and profile.speaker_count > 2:
        return SummarizationStrategy.LONG_DIALOGUE

    # Default: standard MAP-REDUCE
    return SummarizationStrategy.STANDARD
```

**Routing Thresholds (Initial):**

- Token count < 2000 → Single-pass strategy
- Speaker turn rate > 2.0 turns/min → Dialogue strategy
- Entity density > 10.0 per 1000 tokens → Technical strategy
- Topic drift > threshold → Topic segmentation
- Duration > 60 min + multiple speakers → Panel strategy

### 4. Model Roles

The system is structured around stable roles (compatible with RFC-042):

1. **Extractor (Map Pass)**
   - High recall
   - Structured outputs (facts, bullets, entities)
   - Minimal hallucination
   - Models: BART, LED, LongT5

2. **Summarizer (Map Pass)**
   - Chunk-level narrative summaries
   - Preserves salience and context
   - Models: BART, LED, LongT5, PEGASUS

3. **Reducer / Synthesizer**
   - De-duplication and reconciliation
   - Global coherence
   - Schema and formatting compliance
   - Models: Instruction-tuned LLMs (RFC-042), BART-large, LED-large

4. **Finalizer (Optional)**
   - Style, tone, and output normalization
   - Often combined with reducer
   - Models: Instruction-tuned LLMs

### 5. Extraction-First Intermediate Artifacts

All strategies produce structured intermediate outputs before reduction:

```python
@dataclass
class ExtractionArtifacts:
    """Structured intermediate outputs before reduction."""
    key_points: List[str]
    claims: List[Claim]  # With supporting evidence
    entities: List[Entity]  # With roles
    numbers: List[Number]  # Value, unit, context
    notable_quotes: List[Quote]  # Optional timestamps
    speaker_positions: List[SpeakerPosition]  # Dialogue only
    definitions: List[Definition]  # Technical only
```

Reducers operate exclusively on these artifacts, not raw transcripts.

### 6. Integration with Existing Pipeline

**Backward Compatibility:**

- Default routing: Standard MAP-REDUCE (current behavior)
- Profiling is opt-in (can be disabled)
- Existing providers work unchanged
- Routing decisions are logged but don't break existing workflows

**Configuration:**

```python
# Enable adaptive routing
enable_adaptive_routing: bool = False  # Opt-in for backward compatibility

# Routing thresholds (tunable)
routing_token_threshold: int = 2000
routing_turn_rate_threshold: float = 2.0
routing_entity_density_threshold: float = 10.0
```

## Key Decisions

1. **Deterministic Routing**
   - **Decision**: Use rule-based routing, not ML-based classification
   - **Rationale**: Deterministic, debuggable, reproducible. Fast (< 1s per episode).
     No training data needed.

2. **Profile-Based Metrics**
   - **Decision**: Use inexpensive heuristics (token count, speaker count, etc.)
   - **Rationale**: Fast profiling, no ML inference required. Sufficient for routing
     decisions.

3. **Backward Compatibility**
   - **Decision**: Make routing opt-in, default to current behavior
   - **Rationale**: No breaking changes. Users can opt-in gradually.

4. **Structured Artifacts**
   - **Decision**: All strategies produce extraction artifacts before reduction
   - **Rationale**: Enables consistent reducer interface. Supports future enhancements
     (knowledge graph, etc.).

5. **Per-Profile Evaluation**
   - **Decision**: Track metrics per episode profile, not globally
   - **Rationale**: Avoids misleading averages. Enables profile-specific optimization.

## Alternatives Considered

1. **ML-Based Routing**
   - **Description**: Train a classifier to route episodes
   - **Pros**: Potentially more accurate routing
   - **Cons**: Requires training data, adds complexity, less debuggable
   - **Why Rejected**: Deterministic rules are sufficient and more maintainable

2. **Single Strategy for All**
   - **Description**: Keep current one-size-fits-all approach
   - **Pros**: Simpler, no routing logic needed
   - **Cons**: Suboptimal quality for diverse episode types
   - **Why Rejected**: Quality improvements justify added complexity

3. **Provider-Specific Routing**
   - **Description**: Different routing per provider (ML vs OpenAI vs Ollama)
   - **Pros**: Provider-specific optimizations
   - **Cons**: Fragmentation, harder to maintain
   - **Why Rejected**: Unified routing is cleaner and more maintainable

## Testing Strategy

**Test Coverage:**

- **Unit tests**: Profile calculation, routing logic, threshold validation
- **Integration tests**: End-to-end routing with real episodes
- **Quality validation**: Compare routed vs non-routed summaries per profile
- **Performance testing**: Profiling overhead (< 1s target)

**Test Organization:**

- `tests/unit/workflow/test_episode_profiling.py` - Profile calculation
- `tests/unit/workflow/test_routing.py` - Routing logic
- `tests/integration/test_adaptive_routing.py` - End-to-end routing
- `tests/integration/test_profile_quality.py` - Quality validation per profile

**Test Execution:**

- Unit tests run in CI (fast, no ML dependencies)
- Integration tests require real episodes (manual/local testing)
- Quality validation: Compare summaries for 3-5 episodes per profile

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1**: Implement profiling and routing logic (opt-in)
- **Phase 2**: Validate routing decisions on representative episodes
- **Phase 3**: Enable by default for new episodes
- **Phase 4**: Iterate on thresholds based on quality feedback

**Monitoring:**

- **Routing decisions**: Log which profile each episode gets
- **Quality metrics**: Track per-profile quality (faithfulness, coverage, etc.)
- **Performance metrics**: Profiling time, routing overhead
- **Usage tracking**: Which profiles are most common

**Success Criteria:**

1. ✅ Profiling completes in < 1s per episode
2. ✅ Routing decisions are deterministic and reproducible
3. ✅ Quality improves for at least 3 episode profiles
4. ✅ No regressions for default (non-routed) behavior
5. ✅ Documentation complete (routing guide, threshold tuning)

## Relationship to Other RFCs

This RFC (RFC-053) complements existing summarization RFCs:

1. **RFC-012: Episode Summarization** - Current implementation, provides foundation
2. **RFC-042: Hybrid Summarization Pipeline** - REDUCE phase can use instruction-tuned
   LLMs for routed episodes
3. **RFC-052: Locally Hosted LLM Models** - Local LLMs can be used in routing strategies

**Key Distinction:**

- **RFC-012**: Implements basic summarization pipeline
- **RFC-042**: Improves REDUCE phase quality with instruction-tuned LLMs
- **RFC-052**: Adds local LLM models as provider options
- **RFC-053**: Routes episodes to optimal strategies based on characteristics

Together, these provide:
- Complete summarization pipeline (RFC-012)
- High-quality REDUCE phase (RFC-042)
- Local LLM options (RFC-052)
- Adaptive routing for diverse episode types (RFC-053)

## Benefits

1. **Improved Quality**: Better summaries for diverse episode types
2. **Optimized Performance**: Right strategy for each episode
3. **Systematic Evaluation**: Per-profile metrics enable targeted improvements
4. **Extensibility**: Easy to add new profiles and routing rules
5. **Debuggability**: Deterministic routing with logging

## Migration Path

**For Users:**

1. Opt-in to adaptive routing: `enable_adaptive_routing: true`
2. System automatically profiles episodes and routes appropriately
3. Review routing decisions in logs
4. Adjust thresholds if needed (via config)

**For Developers:**

1. Review RFC-053 (this document)
2. Implement profiling logic
3. Implement routing rules
4. Add integration tests
5. Validate on representative episodes

## Open Questions

1. **Threshold Tuning**: What are optimal thresholds for routing rules?
2. **Topic Drift Calculation**: How to efficiently calculate semantic variance?
3. **Profile Expansion**: When to add new profiles vs adjust existing ones?
4. **Evaluation Metrics**: What metrics matter most per profile?
5. **Provider Integration**: How do different providers (ML, OpenAI, Ollama) affect
   routing decisions?

## References

- **Related PRD**: `docs/prd/PRD-005-episode-summarization.md`
- **Related RFC**: `docs/rfc/RFC-012-episode-summarization.md`
- **Related RFC**: `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
- **Related RFC**: `docs/rfc/RFC-052-locally-hosted-llm-models-with-prompts.md`
- **Related ADR**: `docs/adr/ADR-010-hierarchical-summarization-pattern.md`
- **Source Code**: `podcast_scraper/workflow/stages/summarization_stage.py`
