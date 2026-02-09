# RFC-049: Grounded Insight Layer – Core Concepts & Data Model

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, ML engineers, downstream consumers
- **Execution Timing**: **Phase 3 of 3** — Implement after
  RFC-044 (Model Registry) and RFC-042 (Hybrid ML Platform).
  This RFC consumes the model infrastructure and extraction
  capabilities built by the first two phases.
- **Related PRDs**:
  - `docs/prd/PRD-017-grounded-insight-layer.md`
    (Grounded Insight Layer)
- **Related RFCs**:
  - `docs/rfc/RFC-044-model-registry.md`
    (**prerequisite** — model metadata infrastructure)
  - `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
    (**prerequisite** — ML platform, FLAN-T5, Embedding,
    QA, NLI model loading + structured extraction)
  - `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md`
    (Use Cases & Consumption)
  - `docs/rfc/RFC-051-grounded-insight-layer-database-projection.md`
    (Database Projection)
  - `docs/rfc/RFC-052-locally-hosted-llm-models-with-prompts.md`
    (model-specific prompts for Ollama LLMs —
    extends extraction to local LLM tier)
  - `docs/rfc/RFC-053-adaptive-summarization-routing.md`
    (downstream — routes episodes to optimal
    strategies, including GIL extraction)
- **Related Documents**:
  - `docs/kg/ontology.md` - Human-readable ontology
  - `docs/kg/kg.schema.json` - Machine-readable schema
  - `docs/ARCHITECTURE.md` - System architecture

**Execution Order:**

```text
Phase 1: RFC-044 (Model Registry)        ~2-3 weeks
    │     ModelCapabilities, ModelRegistry
    ▼
Phase 2: RFC-042 (Hybrid ML Platform)    ~10 weeks
    │     FLAN-T5, Embedding, QA, NLI models
    │     Structured extraction protocol
    │     + RFC-052 (model-specific prompts, parallel)
    ▼
Phase 3: RFC-049 (this RFC — GIL)        ~6-8 weeks
    │     GIL extraction orchestration,
    │     kg.json assembly + grounding contract
    ├── Phase 3a: RFC-050 (Use Cases)    parallel
    ├── Phase 3b: RFC-051 (DB Projection) parallel
    ▼
Phase 4: RFC-053 (Adaptive Routing)      ~4-6 weeks
          Route to optimal strategies
```

## Abstract

This RFC defines the core concepts, data model, and storage principles for the **Grounded Insight Layer (GIL)**. It establishes a semantic backbone focused on **insights** (key takeaways) and **quotes** (verbatim evidence) with explicit grounding relationships.

The key innovation is treating **evidence as a first-class citizen**: every insight links to supporting quotes, and every quote links to exact transcript spans and timestamps. This grounding contract distinguishes GIL from traditional summarization and enables trustworthy downstream applications (RAG, analytics, AI agents).

The RFC intentionally avoids end-to-end user flows and UI/query experiences, which are covered in RFC-050. The goal is to define the foundational ontology, storage format, and grounding contracts that enable evidence-backed insight extraction from podcast transcripts.

**Architecture Alignment:** This RFC aligns with existing architecture by:

- Co-locating GIL outputs with existing artifacts (transcript.json, summary.json, metadata.json)
- Following the per-episode output directory structure (ADR-003, ADR-004)
- Using episode-scoped processing consistent with the workflow orchestration pattern
- Leveraging existing speaker detection (PRD-008) for quote attribution

## Problem Statement

Podcast transcripts contain rich insights and knowledge, but current summarization loses the connection to source material. Users cannot trust summaries because:

- **Evidence is disconnected**: Summaries make claims but don't link to supporting quotes
- **Timestamps are lost**: No way to jump to the exact moment something was said
- **Grounding is ambiguous**: Unclear which insights are well-supported vs speculative
- **Verification is manual**: Users must re-listen to verify claims

The existing pipeline produces transcripts and summaries, but lacks a **grounded evidence layer** that enables:

- **Trust**: Users can see the exact quote supporting any insight
- **Navigation**: Users can jump to timestamps to hear the original context
- **Quality metrics**: System can measure grounding rates and quote validity
- **RAG applications**: Downstream systems get evidence-backed knowledge

**Use Cases:**

1. **Cross-Podcast Topic Research**: Aggregate insights about a topic with supporting quotes
2. **Speaker-Centric Insight Mapping**: Understand what a speaker said with verbatim evidence
3. **Evidence-Backed Quote Retrieval**: Extract insights with exact quotes and timestamps
4. **Insight Explorer**: Query returns top insights + supporting quotes + episode timestamps

## Goals

1. **Define Insight-Centric Ontology**: Establish node types for Insights (takeaways) and Quotes (evidence)
2. **Establish Grounding Contract**: Every insight must be explicitly grounded or marked ungrounded
3. **Make Evidence First-Class**: Quotes are nodes with timestamps, not just metadata
4. **Design Co-Located Storage**: Store GIL data alongside existing outputs without separate infrastructure
5. **Enable Incremental Updates**: Support append-only, per-episode graph construction
6. **Provide Schema Validation**: Define machine-readable schema for CI validation and tooling

## Constraints & Assumptions

**Constraints:**

- Must not require global graph storage in v1 (logical union of per-episode files)
- Must be backward compatible with existing output directory structure
- Must complete extraction within reasonable time bounds (similar to summarization)
- Must not break existing workflows when disabled
- Must conform to existing module boundaries (no mixing concerns)
- **Every insight must have explicit grounding status** (`grounded=true/false`)
- **Every quote must be verbatim** (exact text from transcript, no paraphrasing)

**Assumptions:**

- **RFC-044 (Model Registry) is implemented** — provides
  `ModelCapabilities` and `ModelRegistry` for model lookup
- **RFC-042 (Hybrid ML Platform) is implemented** — provides
  FLAN-T5, extractive QA, sentence-transformers, and NLI
  models for insight/quote extraction and grounding
- Transcripts are available before GIL extraction
- Users have sufficient storage for per-episode `kg.json`
- Schema validation can be manual initially, automated
  later via CI
- Global graph queries deferred to post-v1 (mitigated by
  RFC-051 DB projection)
- Speaker detection may not always be available (quotes
  can have nullable speaker_id)

## Design & Implementation

### 1. Core Ontology

The Grounded Insight Layer uses an **insight-centric ontology** focused on takeaways, evidence, and grounding:

**Node Types (v1):**

| Node Type | Description | Key Properties |
|-----------|-------------|----------------|
| Podcast | A podcast feed | `id`, `title`, `rss_url` |
| Episode | A single episode | `id`, `title`, `publish_date`, `podcast_id` |
| Speaker | A person speaking (optional if no diarization) | `id`, `name`, `aliases` (optional) |
| Topic | An abstract subject (lightweight, mergeable) | `id`, `label`, `aliases` (optional) |
| **Insight** | A key takeaway / conclusion | `id`, `text`, `episode_id`, `grounded`, `confidence` |
| **Quote** | Verbatim transcript span (evidence) | `id`, `text`, `episode_id`, `speaker_id`, `timestamp_start_ms`, `timestamp_end_ms`, `char_start`, `char_end` |

**Node Types (v1.1 - Deferred):**

| Node Type | Description | Key Properties |
|-----------|-------------|----------------|
| Entity | Person, company, product, place | `id`, `name`, `entity_type` |

**Edge Types:**

| Edge | From → To | Description |
|------|-----------|-------------|
| HAS_EPISODE | Podcast → Episode | Podcast contains episode |
| SPOKE_IN | Speaker → Episode | Speaker participated |
| HAS_INSIGHT | Episode → Insight | Episode contains insight |
| **SUPPORTED_BY** | Insight → Quote | Quote provides evidence for insight |
| **SPOKEN_BY** | Quote → Speaker | Speaker said the quote |
| ABOUT | Insight → Topic | Insight is about topic |
| RELATED_TO | Topic ↔ Topic | Semantic relationship (optional) |

**Insight as a First-Class Node:**

Insights are key takeaways extracted from transcript content. Unlike traditional "claims," insights:

- Focus on **what users want to know** (takeaways), not just declarative statements
- Have explicit **grounding status** (`grounded=true/false`)
- Link to supporting **Quote** nodes for evidence

Required Insight Properties:

- `text`: The insight statement (can be rephrased for clarity)
- `episode_id`: Episode where insight was extracted
- `grounded`: Boolean indicating if insight has supporting quotes
- `confidence`: Extraction confidence (0.0-1.0)

**Quote as a First-Class Node (Critical Innovation):**

Quotes are **verbatim transcript spans** that serve as evidence. Making Quote a first-class node enables:

- Evidence-backed retrieval (insight → quote → timestamp)
- Trust verification (users can check quotes against transcript)
- Quality metrics (quote validity rate)
- Speaker attribution when available

Required Quote Properties:

- `text`: **Verbatim** text from transcript (no paraphrasing)
- `episode_id`: Episode containing the quote
- `speaker_id`: Speaker who said the quote (nullable if no diarization)
- `timestamp_start_ms` / `timestamp_end_ms`: Temporal location
- `char_start` / `char_end`: Character span in transcript text
- `transcript_ref`: Reference to transcript artifact

### 2. Grounding Contract (Critical - The 2025 Moat)

The grounding contract is the key differentiator that makes GIL trustworthy:

**Hard Rules (Invariants):**

1. **Every Quote MUST be verbatim**
   - Quote.text must exactly match transcript[char_start:char_end]
   - No paraphrasing, no summarization, no rewording
   - Timestamps must correspond to the quoted span

2. **Every Insight MUST have explicit grounding status**
   - `grounded=true`: Insight has ≥1 SUPPORTED_BY edge to a Quote
   - `grounded=false`: Insight is extracted but lacks supporting quote (rare, but honest)

3. **SUPPORTED_BY edges are evidence links**
   - An insight can have multiple supporting quotes
   - Each quote provides evidence for the insight's validity

**Why This Matters:**

- **Trust**: Users know exactly which insights have evidence
- **Quality Metrics**: System can measure `% insights grounded` and `quote validity rate`
- **RAG Applications**: Downstream systems can filter for grounded-only insights
- **Debugging**: Ungrounded insights are visible, not hidden

**Example Grounding:**

```
Insight: "AI regulation will lag innovation"
  grounded: true
  SUPPORTED_BY → Quote: "Regulation will lag innovation by 3-5 years..."
  SUPPORTED_BY → Quote: "We'll see laws that are already outdated when they pass..."
```

### 3. Confidence & Provenance Model

Every node derived from ML includes:

- **confidence**: `float [0.0 – 1.0]` - Extraction certainty (not factual truth)
- **model_version**: Model identifier used for extraction
- **prompt_version**: Prompt version used (enables A/B testing)

For Quote nodes specifically:

- `transcript_ref`: Pointer to transcript artifact
- `char_start` / `char_end`: Character span in transcript text
- `timestamp_start_ms` / `timestamp_end_ms`: Temporal span

**Rationale:** Confidence represents extraction certainty. Evidence pointers enable verification and debugging.

### 4. Storage & File Layout Strategy

**Guiding Idea:** GIL data should be stored alongside existing podcast_scraper outputs, not in a remote or opaque system.

**Proposed Layout (Per Episode):**

```
output/
  episode_<id>/
    metadata.json
    transcript.json
    summary.json
    kg.json          # NEW: Grounded Insight Layer data
```

**kg.json Responsibilities:**

- Contain all Insight and Quote nodes for this episode
- Reference global IDs for shared nodes (topics, speakers)
- Include explicit SUPPORTED_BY edges (grounding links)
- Include schema_version, model_version, and prompt_version for evolution tracking
- Be reprocessable (re-run extraction on single episode)

**Graph Assembly Model:**

The full Grounded Insight Layer is a logical union of all per-episode `kg.json` files.

**Advantages:**

- Easy debugging (inspect per-episode files)
- Natural sharding (no global storage required)
- Reprocessing without migration (re-run extraction on single episode)
- Co-location with existing artifacts
- Fast queries via RFC-051 Postgres projection

### 5. Node Identity & Deduplication

**Episode-Scoped IDs:**

- Episode ID must be stable and derived from RSS entry GUID if available
- Insight ID should be episode-scoped to avoid accidental global merging
- Quote ID should be episode-scoped and content-based

**Recommended Format:**

- `episode:<rss_guid>`
- `insight:<episode_id>:<sha1(text_normalized)>`
- `quote:<episode_id>:<sha1(text)>` or `quote:<episode_id>:<char_start>-<char_end>`

**Global IDs (Deduplicated Across Episodes):**

These must be stable across episodes:

- Speaker: `speaker:<slug(name)>` (optionally include podcast namespace if collisions)
- Topic: `topic:<slug(label)>`

**Deduplication Strategy:**

- Extraction and resolution are separate steps
- Extractor may emit provisional IDs
- Resolver should converge to stable IDs over time
- No hard requirement for global graph storage in v1 (RFC-051 Postgres handles queries)

### 6. Ontology as a First-Class Artifact

The GIL ontology is treated as a first-class, versioned artifact of the project, comparable to architecture diagrams or API contracts.

**Rationale:** Without an explicit ontology contract, the structure will drift based on model behavior and ad-hoc use cases. A formal ontology ensures:

- Semantic stability across iterations
- Consistent extraction and validation
- Easier onboarding and collaboration
- Safe evolution over time

**Ontology Outputs:**

1. **Human-Readable Ontology Specification**
   - Location: `docs/kg/ontology.md`
   - Purpose: Define node and edge semantics, grounding contract, required vs optional properties
   - This document is the canonical reference for contributors and reviewers

2. **Machine-Readable Schema**
   - Location: `docs/kg/kg.schema.json`
   - Purpose: Validate `kg.json` outputs, enforce grounding invariants, enable CI validation
   - All GIL outputs must conform to this schema

**Required Artifacts (v1):**

- `docs/kg/ontology.md` - Human-readable ontology specification
- `docs/kg/kg.schema.json` - Machine-readable schema for validation

These artifacts are considered part of the GIL deliverable, not optional documentation.

**Implementation Expectations:**

- RFC approval does not require these artifacts to be complete at design time
- The RFC is considered successfully implemented only once:
  - Both artifacts exist in the repository
  - Generated `kg.json` files conform to the schema
  - Grounding contract is enforced (every insight has grounded status)
  - Ontology and schema versions are explicitly declared

Schema validation may initially be manual and later enforced via CI.

### 7. Integration Points

**Prerequisite Infrastructure:**

GIL extraction depends on two prior RFCs:

| Prerequisite | What It Provides | GIL Usage |
| --- | --- | --- |
| RFC-044 (Model Registry) | `ModelCapabilities`, `ModelRegistry` | Look up model limits, family, capabilities |
| RFC-042 (Hybrid ML Platform) | FLAN-T5, QA, Embedding, NLI models | Run insight/quote extraction, grounding |

**Workflow Integration:**

GIL extraction integrates into the existing workflow:

1. **After Transcript Acquisition**: GIL extraction runs
   after transcripts are available
2. **After Speaker Detection**: GIL uses detected speakers
   (from PRD-008) for quote attribution
3. **Before/After Summarization**: GIL can run
   independently or alongside summarization
4. **Co-Located Output**: `kg.json` written to the same
   episode directory
5. **Optional DB Export**: RFC-051 Postgres export runs
   after `kg.json` generation

**Module Boundaries:**

- **Extraction Module**: New module for GIL extraction
  (follows provider pattern)
- **Storage Module**: Uses existing filesystem utilities
- **Schema Validation**: Separate validation utility
  (callable from CI)
- **Model Loading**: Uses RFC-044 `ModelRegistry` for
  model lookup and RFC-042 model loaders for
  initialization

**Configuration Integration:**

GIL extraction controlled via `Config` model:

- `generate_kg: bool` — Enable/disable GIL extraction
- `kg_extraction_provider: str` — Provider tier
  (`"ml"`, `"hybrid"`, or cloud LLM name)
- `kg_insight_model: Optional[str]` — Model for insight
  extraction (resolved via RFC-044 registry)
- `kg_qa_model: Optional[str]` — Model for quote
  extraction (resolved via RFC-044 registry)
- `kg_embedding_model: Optional[str]` — Model for
  grounding similarity (resolved via RFC-044 registry)
- `kg_nli_model: Optional[str]` — Model for grounding
  validation (resolved via RFC-044 registry)
- `kg_require_grounding: bool` — Require all insights
  to be grounded (default: `false`)

**Extraction Tier Selection:**

The provider tier determines which models are used:

| Tier | Insight Model | Quote Model | Grounding |
| --- | --- | --- | --- |
| `"ml"` | FLAN-T5 (RFC-042) | Extractive QA (RFC-042) | Similarity (RFC-042) |
| `"hybrid"` | MAP+LLM (RFC-042) | Extractive QA (RFC-042) | NLI + LLM (RFC-042) |
| Cloud LLM | API provider | Extractive QA (RFC-042) | NLI (RFC-042) |

Extractive QA for quotes is used in **all tiers** because
it guarantees verbatim spans (grounding contract).

## Key Decisions

1. **Insight-Centric Ontology**
   - **Decision**: Start with Insight + Quote + Topic (entities deferred to v1.1)
   - **Rationale**: Focus on user value (takeaways + evidence), not traditional KG completeness

2. **Quote as First-Class Node**
   - **Decision**: Quotes are nodes with timestamps, not just metadata on insights
   - **Rationale**: Evidence is a first-class citizen; enables trust, navigation, and verification

3. **Grounding Contract**
   - **Decision**: Every insight must have explicit `grounded` status
   - **Rationale**: Honest about extraction limits; enables quality metrics; builds user trust

4. **Per-Episode Storage**
   - **Decision**: Store `kg.json` per episode, logical union for global graph
   - **Rationale**: Easy debugging, natural sharding, co-location with existing outputs

5. **Entities Deferred to v1.1**
   - **Decision**: Focus v1 on Topics + Insights + Quotes; add Entities later
   - **Rationale**: Entity extraction/resolution is complex; don't let it block core value

6. **Ontology as First-Class Artifact**
   - **Decision**: Maintain both human-readable and machine-readable ontology artifacts
   - **Rationale**: Ensures semantic stability, enables validation, supports collaboration

## Alternatives Considered

1. **Traditional KG Ontology (Claim, Entity, etc.)**
   - **Description**: Use classic KG framing with Claim, Entity extraction
   - **Pros**: Standard KG terminology, more complete ontology
   - **Cons**: Doesn't focus on user value; entities are complex to resolve
   - **Why Rejected**: Insight + Quote framing better matches user needs ("takeaways + evidence")

2. **Evidence as Metadata (not nodes)**
   - **Description**: Store evidence as properties on Insight nodes, not separate Quote nodes
   - **Pros**: Simpler structure, fewer nodes
   - **Cons**: Can't query quotes independently; loses evidence as first-class citizen
   - **Why Rejected**: Quote nodes enable trust, navigation, and quality metrics

3. **Implicit Grounding (no explicit status)**
   - **Description**: Infer grounding from presence of evidence, don't require explicit flag
   - **Pros**: Simpler extraction, less work per insight
   - **Cons**: Ambiguous when insight lacks evidence; harder to measure quality
   - **Why Rejected**: Explicit grounding contract enables quality metrics and trust

4. **Entity Extraction in v1**
   - **Description**: Include Entity extraction and resolution in v1
   - **Pros**: More complete knowledge extraction
   - **Cons**: Entity resolution is complex and can slow down v1 delivery
   - **Why Rejected**: Deferred to v1.1 to keep v1 focused on core value

## Testing Strategy

**Test Coverage:**

- **Unit Tests**: Test node/edge construction, ID generation, grounding validation
- **Integration Tests**: Test extraction pipeline with real transcripts, validate schema compliance
- **E2E Tests**: Test full workflow from transcript → GIL extraction → `kg.json` generation
- **Grounding Tests**: Verify every insight has explicit grounding status; verify quote verbatim match

**Test Organization:**

- Unit tests: `tests/unit/test_kg_*.py`
- Integration tests: `tests/integration/test_kg_*.py`
- E2E tests: `tests/e2e/test_kg_*.py`

**Test Execution:**

- Run in CI as part of standard test suite
- Use existing test fixtures (transcripts, episodes)
- Validate schema compliance and grounding contract in CI

## Rollout & Monitoring

**Prerequisites (must be complete before starting):**

- RFC-044 (Model Registry) — implemented and tested
- RFC-042 (Hybrid ML Platform) — at minimum Phase 4
  (Extended Models: embedding, QA, NLI pipelines)

**Rollout Plan:**

- **Phase 1**: Implement extraction module with Insight +
  Quote extraction (using RFC-042 models)
- **Phase 2**: Add grounding contract enforcement
  (SUPPORTED_BY edges via extractive QA + NLI)
- **Phase 3**: Add Topic linking (Insight → ABOUT → Topic)
- **Phase 4**: Schema validation and CI integration

**Monitoring:**

- Track extraction success rate (episodes with valid `kg.json`)
- Monitor extraction time (should be similar to summarization)
- Track **insight grounding rate** (% insights with grounded=true)
- Track **quote validity rate** (% quotes that match transcript verbatim)
- Track schema compliance (all `kg.json` files pass validation)

**Success Criteria:**

1. ✅ All v1 node types can be extracted from transcripts (Insight, Quote, Topic)
2. ✅ All v1 edge types can be constructed (HAS_INSIGHT, SUPPORTED_BY, SPOKEN_BY, ABOUT)
3. ✅ Grounding contract is enforced (every insight has grounded status)
4. ✅ Generated `kg.json` files conform to schema
5. ✅ Extraction completes within reasonable time bounds
6. ✅ Integration with existing workflow verified

## Relationship to Other RFCs

This RFC (RFC-049) is part of a three-phase dependency chain
and the Grounded Insight Layer initiative:

**Dependency Chain (execution order):**

```text
Phase 1: RFC-044 (Model Registry)        ~2-3 weeks
    │     Prerequisite: model metadata infra
    ▼
Phase 2: RFC-042 (Hybrid ML Platform)    ~10 weeks
    │     Prerequisite: ML models + extraction
    │     + RFC-052 (model-specific prompts, parallel)
    ▼
Phase 3: RFC-049 (this RFC — GIL Core)   ~6-8 weeks
    │     Consumes RFC-044 + RFC-042
    ▼
Phase 4: RFC-053 (Adaptive Routing)      ~4-6 weeks
          Routes to optimal strategies per episode
```

**GIL Initiative RFCs (parallel within Phase 3):**

1. **RFC-049 (this RFC)**: Core ontology, grounding
   contract, storage, extraction orchestration
2. **RFC-050**: Use cases, query patterns, consumption
3. **RFC-051**: Database projection for fast queries
4. **PRD-017**: Product requirements and user value

**Prerequisite RFCs:**

1. **RFC-044: Model Registry** — Provides
   `ModelCapabilities` and `ModelRegistry` for model
   lookup. GIL uses this to discover available models
   per extraction tier.
2. **RFC-042: Hybrid ML Platform** — Provides FLAN-T5,
   extractive QA, sentence-transformers, NLI models,
   and the structured extraction protocol. GIL uses
   these for insight/quote extraction and grounding.

**Complementary RFCs:**

3. **RFC-052: Locally Hosted LLM Models** — Provides
   model-specific prompt engineering for Ollama LLMs.
   GIL extraction prompts are added in Phase 6 of
   RFC-052, extending the LLM tier with optimized
   local model prompts.
4. **RFC-053: Adaptive Summarization Routing** —
   Downstream consumer. Routes episodes to optimal
   strategies (including GIL extraction) based on
   episode profiling. Runs after GIL is stable.

**Key Distinction:**

- **RFC-044**: Model metadata and lookup infrastructure
- **RFC-042**: Model loading, hybrid pipeline, extraction
- **RFC-049**: Domain-specific GIL extraction using
  models from RFC-042 via RFC-044
- **RFC-050**: How GIL data is consumed end-to-end
- **RFC-051**: Postgres projection for fast queries
- **RFC-052**: Optimized prompts for local LLMs
- **RFC-053**: Adaptive routing across all strategies

Together, these RFCs provide:

- Complete technical design for GIL implementation
- Clear separation: infra (044) → platform (042) →
  domain (049) → consumption (050) → serving (051)
- Local LLM optimization (052) + adaptive routing (053)
- Foundation for trustworthy downstream applications

## Benefits

1. **Trust & Navigation**: Users can retrieve insights and verify evidence immediately
2. **Evidence First-Class**: Quotes are nodes, not metadata; enables quality metrics
3. **Grounding Contract**: Explicit grounding status builds user trust
4. **Co-Located**: GIL data lives alongside existing outputs, no separate infrastructure
5. **Extensible**: v1 ontology provides clear upgrade path to v1.1 (entities)

## Migration Path

N/A - This is a new feature, not a migration from an existing system.

## Resolved Questions

All design questions have been resolved. Decisions are
recorded here for traceability.

1. **Topic Normalization**: Should Topic nodes be fully
   normalized or partially episode-scoped?
   **Global normalization with exact-match IDs (v1).**
   Use `topic:<slug(label)>` as the canonical ID. Exact
   string match after lowercasing + slug normalization.
   Semantic similarity-based merging (e.g., "AI Ethics"
   ≈ "Ethical AI") is deferred to v1.1 where
   sentence-transformer embeddings (RFC-042) can power
   it. v1 may produce near-duplicate topics, which is
   acceptable — RFC-051 DB queries can GROUP BY
   normalized labels.

2. **Entity Extraction Timing**: When should entity
   extraction be added?
   **v1.1, after core GIL is validated.** Entity
   extraction (persons, companies, products) adds value
   but is not required for the core insight + quote
   value proposition. Defer to v1.1 when: (a) insight
   grounding rates exceed 80%, (b) quote validity
   exceeds 95%, and (c) the GIL pipeline is stable.
   Entity extraction uses spaCy NER (already in
   codebase) so implementation effort is low once
   prioritized.

3. **Quote Granularity**: What is the optimal quote
   length?
   **Variable, 1-3 sentences, extracted by QA model.**
   Extractive QA models (RFC-042) naturally return
   spans of variable length based on the question. The
   typical range is 1-3 sentences (20-80 words). No
   artificial truncation or padding. If the QA model
   returns a span shorter than 10 characters, discard
   it. If longer than 500 characters, split at sentence
   boundaries and keep the highest-scoring sub-span.

4. **Grounding Threshold**: What confidence threshold
   triggers `grounded=false`?
   **Grounding is binary: ≥1 SUPPORTED_BY edge =
   grounded.** The `grounded` field is determined solely
   by the existence of at least one SUPPORTED_BY edge,
   not by confidence score. Low-confidence quotes
   (QA score < 0.3 or NLI entailment < 0.5) are
   **filtered out before edge creation** — they never
   become SUPPORTED_BY edges. This means an insight with
   only low-confidence candidate quotes becomes
   `grounded=false`, which is honest and correct.

---

## Conclusion

The Grounded Insight Layer represents a strategic shift
from traditional summarization toward **evidence-backed
knowledge extraction**. By making insights and quotes
first-class nodes with explicit grounding relationships,
GIL creates a trustworthy foundation that enables:

- **Trust**: Every insight links to verbatim quotes
  that users can verify against the transcript
- **Navigation**: Timestamps on quotes let users jump
  to the exact moment an insight was spoken
- **Quality metrics**: Grounding rates and quote
  validity rates provide measurable quality signals
- **Downstream applications**: RAG systems, AI agents,
  and analytics tools get evidence-backed knowledge

The grounding contract — requiring every quote to be
verbatim and every insight to have explicit grounding
status — is the key differentiator. It makes GIL
honest about what it knows and what it doesn't.

**As Phase 3 of the three-phase build-out (RFC-044 →
RFC-042 → RFC-049), GIL consumes the model
infrastructure and extraction capabilities built by
the first two phases. RFC-050 defines how GIL data is
consumed, RFC-051 projects it for fast queries, and
RFC-052 provides optimized prompts for local LLM
extraction.**

## References

- **Prerequisite**: `docs/rfc/RFC-044-model-registry.md`
- **Prerequisite**: `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
- **Related PRD**: `docs/prd/PRD-017-grounded-insight-layer.md`
- **Related RFC**: `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md`
- **Related RFC**: `docs/rfc/RFC-051-grounded-insight-layer-database-projection.md`
- **Related RFC**: `docs/rfc/RFC-052-locally-hosted-llm-models-with-prompts.md`
- **Related RFC**: `docs/rfc/RFC-053-adaptive-summarization-routing.md`
- **Ontology**: `docs/kg/ontology.md`
- **Schema**: `docs/kg/kg.schema.json`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Source Code**: `podcast_scraper/workflow/`
