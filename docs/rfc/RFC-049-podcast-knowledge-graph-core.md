# RFC-049: Grounded Insight Layer – Core Concepts & Data Model

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, ML engineers, downstream consumers
- **Related PRDs**:
  - `docs/prd/PRD-017-podcast-knowledge-graph.md` (Grounded Insight Layer)
- **Related RFCs**:
  - `docs/rfc/RFC-050-podcast-knowledge-graph-use-cases.md` (Use Cases & Consumption)
  - `docs/rfc/RFC-051-database-export-knowledge-graph.md` (Database Projection)
- **Related Documents**:
  - `docs/kg/ontology.md` - Human-readable ontology specification
  - `docs/kg/kg.schema.json` - Machine-readable schema
  - `docs/ARCHITECTURE.md` - System architecture

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

- ML models (LLMs) are available for insight/quote extraction
- Transcripts are available before GIL extraction
- Users have sufficient storage for per-episode `kg.json` files
- Schema validation can be manual initially, automated later via CI
- Global graph queries can be deferred to post-v1 (mitigated by RFC-051 DB projection)
- Speaker detection may not always be available (quotes can have nullable speaker_id)

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

**Workflow Integration:**

GIL extraction should be integrated into the existing workflow pipeline:

1. **After Transcript Acquisition**: GIL extraction runs after transcripts are available
2. **After Speaker Detection**: GIL uses detected speakers (from PRD-008) for quote attribution
3. **Before/After Summarization**: GIL can run independently or alongside summarization
4. **Co-Located Output**: `kg.json` is written to the same episode directory as other artifacts
5. **Optional DB Export**: RFC-051 Postgres export can run after kg.json generation

**Module Boundaries:**

- **Extraction Module**: New module for GIL extraction (follows provider pattern)
- **Storage Module**: Uses existing filesystem utilities (no new I/O abstractions)
- **Schema Validation**: Separate validation utility (can be called from CI)

**Configuration Integration:**

GIL extraction should be controlled via `Config` model:

- `generate_kg: bool` - Enable/disable GIL extraction
- `kg_extraction_method: str` - Extraction method (e.g., "llm")
- `kg_model: Optional[str]` - Model identifier for extraction
- `kg_require_grounding: bool` - Require all insights to be grounded (default: false)

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

**Rollout Plan:**

- **Phase 1**: Implement extraction module with Insight + Quote extraction
- **Phase 2**: Add grounding contract enforcement (SUPPORTED_BY edges)
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

This RFC (RFC-049) is part of the Grounded Insight Layer initiative that includes:

1. **RFC-050: GIL Use Cases & Consumption** - Defines how GIL data is consumed end-to-end
2. **RFC-051: Database Projection** - Defines Postgres export for fast queries
3. **PRD-017: Grounded Insight Layer** - Defines product requirements and user value

**Key Distinction:**

- **RFC-049 (This RFC)**: Focuses on core concepts, ontology, grounding contract, and storage
- **RFC-050**: Focuses on use cases, query patterns, and consumption
- **RFC-051**: Focuses on database projection for fast queries

Together, these RFCs provide:

- Complete technical design for Grounded Insight Layer implementation
- Clear separation between ontology (RFC-049), consumption (RFC-050), and serving (RFC-051)
- Foundation for trustworthy downstream applications (RAG, agents, analytics)

## Benefits

1. **Trust & Navigation**: Users can retrieve insights and verify evidence immediately
2. **Evidence First-Class**: Quotes are nodes, not metadata; enables quality metrics
3. **Grounding Contract**: Explicit grounding status builds user trust
4. **Co-Located**: GIL data lives alongside existing outputs, no separate infrastructure
5. **Extensible**: v1 ontology provides clear upgrade path to v1.1 (entities)

## Migration Path

N/A - This is a new feature, not a migration from an existing system.

## Open Questions

1. **Topic Normalization**: Should Topic nodes be fully normalized or partially episode-scoped?
   - **Current Decision**: Global normalization with stable IDs
   - **Open**: Aggressiveness of normalization (exact match vs semantic similarity)

2. **Entity Extraction Timing**: When should entity extraction be added?
   - **Current Decision**: Deferred to v1.1
   - **Open**: Does entity extraction directly improve retrieval enough for v1?

3. **Quote Granularity**: What is the optimal quote length?
   - **Current Decision**: Extract natural quote boundaries from transcript
   - **Open**: Should quotes be sentence-level, paragraph-level, or variable?

4. **Grounding Threshold**: What confidence threshold triggers `grounded=false`?
   - **Current Decision**: No threshold; grounding is based on SUPPORTED_BY edge existence
   - **Open**: Should low-confidence quotes affect grounding status?

## References

- **Related PRD**: `docs/prd/PRD-017-podcast-knowledge-graph.md`
- **Related RFC**: `docs/rfc/RFC-050-podcast-knowledge-graph-use-cases.md`
- **Related RFC**: `docs/rfc/RFC-051-database-export-knowledge-graph.md`
- **Ontology Specification**: `docs/kg/ontology.md`
- **Schema Specification**: `docs/kg/kg.schema.json`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Source Code**: `podcast_scraper/workflow/` (integration points)
