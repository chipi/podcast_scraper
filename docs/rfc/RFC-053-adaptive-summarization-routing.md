# RFC-053: Adaptive Summarization Routing Based on Episode Profiling

- **Status**: Draft
- **Date**: 2026-02-05
- **Authors**:
- **Stakeholders**: Maintainers, users processing diverse
  podcast types, developers integrating summarization
- **Execution Timing**: **Phase 4** — Implement after
  RFC-042 (Hybrid ML Platform) and RFC-049 (GIL) are
  stable. This RFC is an **optimization layer** that
  makes existing capabilities work better across diverse
  content types. It also serves as the **bridge to
  multi-content-type expansion** beyond podcasts.
- **Related PRDs**:
  - `docs/prd/PRD-005-episode-summarization.md`
    (Episode summarization requirements)
- **Related RFCs**:
  - `docs/rfc/RFC-012-episode-summarization.md`
    (Current summarization implementation)
  - `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
    (Hybrid MAP-REDUCE architecture — provides models)
  - `docs/rfc/RFC-044-model-registry.md`
    (Model Registry — model capability lookup)
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md`
    (GIL — extraction routing per content type)
  - `docs/rfc/RFC-052-locally-hosted-llm-models-with-prompts.md`
    (Local LLM models — routing targets)
- **Related ADRs**:
  - `docs/adr/ADR-010-hierarchical-summarization-pattern.md`
    (Hierarchical summarization)

**Execution Order:**

```text
Phase 1: RFC-044 (Model Registry)          ~2-3 weeks
    ▼
Phase 2: RFC-042 (Hybrid ML Platform)      ~10 weeks
    │  + RFC-052 (Local LLM Prompts)       parallel
    ▼
Phase 3: RFC-049 (GIL)                     ~6-8 weeks
    ▼
Phase 4: RFC-053 (this RFC — Routing)      ~4-6 weeks
          Optimization + multi-content bridge
```

**Why Phase 4?** RFC-053 routes to capabilities that
RFC-042 provides (MAP/REDUCE models, FLAN-T5, LLMs) and
can also optimize GIL extraction (RFC-049). It requires
those foundations to be stable first. Additionally,
the profiling data collected during Phase 3 (GIL
extraction on real episodes) provides empirical evidence
for tuning routing thresholds.

## Abstract

This RFC proposes an **adaptive routing system** for
podcast summarization that selects optimal summarization
strategies based on episode characteristics (duration,
structure, content type). Instead of using a single
summarization approach for all episodes, the system
profiles each episode and routes it to the most
appropriate strategy. This enables consistent output
quality across diverse podcast formats while keeping
system complexity manageable.

**Key Principle:** Standardize the pipeline and outputs;
vary strategy via routing.

**Beyond Podcasts:** While v1 focuses on podcast episode
profiles, the profiling and routing architecture is
**content-type-agnostic**. The same framework extends
to lectures, panel discussions, interviews, debates,
audiobooks, and other long-form audio/text content.
RFC-053 is the **bridge from "podcast scraper" to
"content intelligence platform"**.

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

**Prerequisites (must be complete before starting):**

- RFC-042 (Hybrid ML Platform) — provides model diversity
  to route to
- RFC-049 (GIL) — stable extraction pipeline for GIL
  routing

**Rollout Plan:**

- **Phase 4a**: Implement profiling and routing logic
  (opt-in, podcast profiles only)
- **Phase 4b**: Validate routing decisions on
  representative episodes (summarization + GIL)
- **Phase 4c**: Enable by default for new episodes
- **Phase 4d**: Iterate on thresholds based on quality
- **Phase 4e** (v1.1): Add interview + lecture profiles
- **Phase 4f** (v2): Multi-content expansion with
  content-type detection

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

## Integration with GIL (RFC-049)

### Routing for GIL Extraction

Episode profiling benefits GIL extraction, not just
summarization. Different content types benefit from
different extraction strategies:

| Profile | GIL Strategy | Rationale |
| --- | --- | --- |
| Short Monologue | Single-pass FLAN-T5 | Short enough for direct extraction |
| Short Dialogue | Speaker-aware extraction | "Who said what" matters for quotes |
| Long Monologue | MAP → REDUCE extraction | Chunking needed for long content |
| Long Dialogue | Topic-segmented extraction | Panel insights cluster by topic |
| Technical | Entity-first extraction | Preserve facts/numbers in insights |
| Abstract | Claim-mapping extraction | Focus on arguments and positions |

**Implementation:** RFC-053 can expose `route_episode()`
to both the summarization pipeline and the GIL extraction
pipeline, allowing RFC-049 to adapt its strategy per
episode.

### Shared Profiling

Episode profiling is computed once and reused:

```python
profile = profile_episode(transcript, metadata, speakers)

# Summarization uses profile for strategy selection
summary_strategy = route_summarization(profile)

# GIL uses profile for extraction strategy selection
extraction_strategy = route_extraction(profile)
```

This avoids duplicate work and ensures consistent
routing decisions across both pipelines.

## Beyond Podcasts: Multi-Content Expansion

RFC-053's profiling and routing architecture is
**content-type-agnostic**. The same framework extends
to any long-form audio/text content.

### Future Content Profiles

Beyond podcast-specific profiles (v1), the system can
add content-type profiles:

| Content Type | Key Characteristics | Strategy Adaptations |
| --- | --- | --- |
| **Lectures** | Single speaker, structured, technical | Section-aware chunking, definition extraction |
| **Interviews** | Two speakers, Q&A format | Question-answer pairing, interviewer filtering |
| **Panel Discussions** | 3+ speakers, topic hopping | Topic segmentation, speaker-position mapping |
| **Debates** | Opposing viewpoints, structured | Claim-counterclaim mapping, position extraction |
| **Audiobooks** | Narrative, long-form, chapters | Chapter-aware segmentation, narrative synthesis |
| **Meetings** | Multiple speakers, action items | Decision extraction, action item tracking |
| **Earnings Calls** | Structured, financial, Q&A | Financial entity extraction, guidance tracking |

### Expansion Strategy

**Phase 4a (v1 — Podcasts):**

- Implement profiling + routing for podcast profiles
- Validate on representative episodes
- Tune thresholds based on quality feedback

**Phase 4b (v1.1 — Adjacent Content):**

- Add interview and lecture profiles (closest to podcasts)
- Test with real interview/lecture transcripts
- Minimal routing rule additions

**Phase 4c (v2 — Multi-Content):**

- Add panel, debate, meeting profiles
- Introduce content-type detection (auto-classify input)
- Expand extraction strategies for new content types

**Key Insight:** The profiling metrics (duration, speaker
count, turn-taking rate, entity density, topic drift)
are universal. What changes per content type is the
**routing rules** and **strategy implementations**, not
the profiling framework itself.

### Content-Type Detection (Future)

For multi-content support, add auto-detection:

```python
def detect_content_type(
    profile: EpisodeProfile,
    metadata: dict,
) -> ContentType:
    """Auto-detect content type from profile + metadata."""
    # Heuristic-based detection
    if profile.speaker_count == 1 and profile.entity_density > 10:
        return ContentType.LECTURE
    if profile.speaker_count == 2 and profile.turn_taking_rate > 3.0:
        return ContentType.INTERVIEW
    if profile.speaker_count >= 3:
        return ContentType.PANEL
    # Default
    return ContentType.PODCAST
```

This enables the system to handle mixed input without
manual content-type specification.

## Relationship to Other RFCs

This RFC (RFC-053) is the **optimization and expansion
layer** in the overall architecture:

```text
Phase 1: RFC-044 (Model Registry)
    ▼
Phase 2: RFC-042 (Hybrid ML Platform)
    │  + RFC-052 (Local LLM Prompts)
    ▼
Phase 3: RFC-049 (GIL)
    ▼
Phase 4: RFC-053 (this RFC)
          Routing + multi-content bridge
```

**Dependency chain:**

1. **RFC-044 (Phase 1)**: Model capabilities — RFC-053
   uses registry to check which models are available
   for each routing strategy
2. **RFC-042 (Phase 2)**: Hybrid platform — provides
   the MAP/REDUCE models and FLAN-T5/LLM tiers that
   RFC-053 routes to
3. **RFC-052 (Phase 2b)**: Local LLM prompts — provides
   model-specific prompts that RFC-053 can select
   per routing strategy
4. **RFC-049 (Phase 3)**: GIL extraction — RFC-053 can
   route GIL extraction strategies per content type
5. **RFC-053 (Phase 4, this RFC)**: Routing — selects
   optimal strategies for both summarization and GIL
   extraction based on episode/content profiling

**Key Distinction:**

- **RFC-012**: Basic summarization pipeline
- **RFC-042**: Model diversity (MAP/REDUCE, FLAN-T5,
  LLMs, embedding, QA, NLI)
- **RFC-052**: Prompt quality for local LLMs
- **RFC-049**: GIL extraction orchestration
- **RFC-053**: Adaptive routing — selects the right
  strategy from the available capabilities

Together, these provide:

- Complete summarization pipeline (RFC-012)
- High-quality model platform (RFC-042)
- Local LLM options with optimized prompts (RFC-052)
- Evidence-backed insight extraction (RFC-049)
- Adaptive routing for diverse content types (RFC-053)

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

1. **Threshold Tuning**: What are optimal thresholds for
   routing rules? Use Phase 3 profiling data to calibrate.
2. **Topic Drift Calculation**: How to efficiently
   calculate semantic variance? Sentence-transformers
   (RFC-042) can provide embeddings for this.
3. **Profile Expansion**: When to add new profiles vs
   adjust existing ones? Start with 6 podcast profiles,
   expand to content-type profiles in v1.1.
4. **Evaluation Metrics**: What metrics matter most per
   profile? Per-profile quality (faithfulness, coverage).
5. ~~**Provider Integration**: How do different providers
   affect routing?~~
   **Resolved**: Routing is provider-agnostic. Strategies
   map to model roles (MAP, REDUCE), not specific
   providers. RFC-042 + RFC-044 handle provider/model
   resolution.
6. **GIL Extraction Routing**: Should GIL extraction use
   the same routing rules as summarization, or separate
   rules? Proposal: shared profiling, separate strategy
   selection.
7. **Content-Type Detection**: When should auto-detection
   of content type be implemented? Proposal: Phase 4e
   (v1.1), after podcast routing is validated.

## References

- **Related PRD**: `docs/prd/PRD-005-episode-summarization.md`
- **Related RFC**: `docs/rfc/RFC-012-episode-summarization.md`
- **Prerequisite**: `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
- **Prerequisite**: `docs/rfc/RFC-044-model-registry.md`
- **Related RFC**: `docs/rfc/RFC-049-grounded-insight-layer-core.md`
- **Related RFC**: `docs/rfc/RFC-052-locally-hosted-llm-models-with-prompts.md`
- **Related ADR**: `docs/adr/ADR-010-hierarchical-summarization-pattern.md`
- **Source Code**: `podcast_scraper/workflow/stages/summarization_stage.py`
