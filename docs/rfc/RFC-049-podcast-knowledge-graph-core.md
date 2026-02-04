# RFC-049: Podcast Knowledge Graph – Core Concepts & Data Model

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, ML engineers, downstream consumers
- **Related PRDs**:
  - `docs/prd/PRD-017-podcast-knowledge-graph.md`
- **Related RFCs**:
  - `docs/rfc/RFC-050-podcast-knowledge-graph-use-cases.md` (Use Cases & Consumption)
- **Related Documents**:
  - `docs/kg/ontology.md` - Human-readable ontology specification
  - `docs/kg/kg.schema.json` - Machine-readable schema
  - `docs/ARCHITECTURE.md` - System architecture

## Abstract

This RFC defines the core Knowledge Graph (KG) concepts, data model, and storage principles for the Podcast Knowledge Graph. It establishes a stable semantic backbone that supports multiple future use cases while integrating cleanly with existing podcast_scraper outputs.

The RFC intentionally avoids end-to-end user flows and UI/query experiences, which are covered in RFC-050. The goal is to define the foundational ontology, storage format, and data contracts that enable structured knowledge extraction from podcast transcripts.

**Architecture Alignment:** This RFC aligns with existing architecture by:
- Co-locating KG outputs with existing artifacts (transcript.json, summary.json, metadata.json)
- Following the per-episode output directory structure (ADR-003, ADR-004)
- Using episode-scoped processing consistent with the workflow orchestration pattern
- Leveraging existing speaker detection (PRD-008) for attribution

## Problem Statement

Podcast transcripts contain rich structured knowledge (entities, topics, claims, relationships), but this knowledge is currently lost when transcripts are stored as unstructured text. Without a structured representation:

- **Attribution is lost**: Claims cannot be linked to specific speakers
- **Relationships are implicit**: Topics, entities, and claims exist in isolation
- **Querying is limited**: Only keyword search is possible, not semantic traversal
- **Evidence is opaque**: No way to trace claims back to transcript evidence

The existing pipeline produces transcripts and summaries, but lacks a structured knowledge layer that enables:
- Cross-episode aggregation (e.g., "all claims about AI regulation")
- Speaker-centric analysis (e.g., "what does this speaker believe?")
- Evidence-backed querying (e.g., "show me the exact transcript span for this claim")

**Use Cases:**

1. **Cross-Podcast Topic Research**: Aggregate all episodes, speakers, and claims related to a topic
2. **Speaker-Centric Insight Mapping**: Understand what a speaker believes across episodes
3. **Claim Tracking & Evidence Retrieval**: Extract concrete statements with original context
4. **Semantic Query & Question Answering**: Ask complex questions that go beyond keyword matching

## Goals

1. **Define Minimal Ontology**: Establish core node and edge types required for attribution, aggregation, and explainability
2. **Establish Evidence-First Model**: Ensure every extracted fact is traceable to transcript evidence
3. **Design Co-Located Storage**: Store KG data alongside existing outputs without requiring separate infrastructure
4. **Enable Incremental Updates**: Support append-only, per-episode graph construction without global recomputation
5. **Provide Schema Validation**: Define machine-readable schema for CI validation and tooling

## Constraints & Assumptions

**Constraints:**

- Must not require global graph storage in v1 (logical union of per-episode files)
- Must be backward compatible with existing output directory structure
- Must complete extraction within reasonable time bounds (similar to summarization)
- Must not break existing workflows when disabled
- Must conform to existing module boundaries (no mixing concerns)

**Assumptions:**

- ML models (LLMs, NER) are available for extraction
- Transcripts are available before KG extraction
- Users have sufficient storage for per-episode `kg.json` files
- Schema validation can be manual initially, automated later via CI
- Global graph queries can be deferred to post-v1

## Design & Implementation

### 1. Core Ontology

The Knowledge Graph uses a minimal ontology focused on attribution, aggregation, and evidence:

**Node Types:**

| Node Type | Description | Key Properties |
|-----------|-------------|----------------|
| Podcast | A podcast feed | `id`, `title`, `rss_url` |
| Episode | A single episode | `id`, `title`, `publish_date`, `podcast_id` |
| Speaker | A person speaking | `id`, `name`, `aliases` (optional) |
| Topic | An abstract subject | `id`, `label`, `aliases` (optional) |
| Entity | Person, company, product, place | `id`, `name`, `entity_type` |
| Claim | Declarative statement | `id`, `text`, `speaker_id`, `episode_id`, `timestamp_start_ms`, `timestamp_end_ms` |

**Edge Types:**

| Edge | From → To | Description |
|------|-----------|-------------|
| HAS_EPISODE | Podcast → Episode | Podcast contains episode |
| SPOKE_IN | Speaker → Episode | Speaker participated |
| DISCUSSES | Episode → Topic | Topic discussed |
| MENTIONS | Episode → Entity | Entity mentioned |
| ASSERTS | Speaker → Claim | Speaker made claim |
| ABOUT | Claim → Topic / Entity | Claim subject |
| RELATED_TO | Topic ↔ Topic | Semantic relationship |

**Claim as a First-Class Node:**

Claims are treated as atomic knowledge units rather than properties of episodes or speakers. This enables:
- Attribution and comparison across speakers
- Support for disagreement modeling later
- Reduced over-reliance on summaries

Required Claim Properties:
- `text`: The claim statement
- `speaker_id`: Speaker who made the claim
- `episode_id`: Episode where claim was made
- `timestamp_start_ms` / `timestamp_end_ms`: Temporal location
- `confidence`: Extraction confidence (0.0-1.0)
- `evidence`: Provenance metadata (see below)

### 2. Confidence & Provenance Model

Every node and edge derived from ML must include:

- **confidence**: `float [0.0 – 1.0]` - Extraction certainty (not factual truth)
- **evidence**: Object containing:
  - `episode_id`: Source episode
  - `transcript_ref`: Pointer to transcript artifact
  - `char_start` / `char_end`: Character span in transcript text
  - `timestamp_start_ms` / `timestamp_end_ms`: Temporal span
  - `extraction_method`: Method used (e.g., "llm", "ner", "rules")
  - `model_version`: Model version used for extraction

**Rationale:** Confidence represents extraction certainty, not factual truth. Evidence enables traceability and debugging.

### 3. Storage & File Layout Strategy

**Guiding Idea:** KG data should be stored alongside existing podcast_scraper outputs, not in a remote or opaque system.

**Proposed Layout (Per Episode):**

```
output/
  episode_<id>/
    metadata.json
    transcript.json
    summary.json
    kg.json          # NEW: Knowledge Graph data
```

**kg.json Responsibilities:**

- Contain all KG nodes and edges introduced or referenced by this episode
- Reference global IDs for shared nodes (topics, entities, speakers)
- Be append-only and reprocessable
- Include schema version for evolution tracking

**Graph Assembly Model:**

The full Knowledge Graph is a logical union of all per-episode `kg.json` files.

**Advantages:**
- Easy debugging (inspect per-episode files)
- Natural sharding (no global storage required)
- Reprocessing without migration (re-run extraction on single episode)
- Co-location with existing artifacts

### 4. Node Identity & Deduplication

**Episode-Scoped IDs:**
- Episode ID must be stable and derived from RSS entry GUID if available
- Claim ID should be episode-scoped to avoid accidental global merging

**Recommended Format:**
- `episode:<rss_guid>`
- `claim:<episode_id>:<sha1(text_normalized)>`

**Global IDs (Deduplicated Across Episodes):**

These must be stable across episodes:
- Speaker: `speaker:<slug(name)>` (optionally include podcast namespace if collisions occur)
- Topic: `topic:<slug(label)>`
- Entity: `entity:<type>:<slug(name)>`

**Deduplication Strategy:**
- Extraction and resolution are separate steps
- Extractor may emit provisional IDs
- Resolver should converge to stable IDs over time
- No hard requirement for global graph storage in v1

### 5. Ontology as a First-Class Artifact

The Knowledge Graph ontology is treated as a first-class, versioned artifact of the project, comparable to architecture diagrams or API contracts.

**Rationale:** Without an explicit ontology contract, KG structure will drift based on model behavior and ad-hoc use cases. A formal ontology ensures:
- Semantic stability across iterations
- Consistent extraction and validation
- Easier onboarding and collaboration
- Safe evolution of the KG over time

**Ontology Outputs:**

1. **Human-Readable Ontology Specification**
   - Location: `docs/kg/ontology.md`
   - Purpose: Define node and edge semantics, required vs optional properties, identity rules, examples
   - This document is the canonical reference for contributors and reviewers

2. **Machine-Readable Schema**
   - Location: `docs/kg/kg.schema.json`
   - Purpose: Validate `kg.json` outputs, enforce required fields and types, enable CI validation
   - All KG outputs must conform to this schema

**Required Artifacts (v1):**
- `docs/kg/ontology.md` - Human-readable ontology specification
- `docs/kg/kg.schema.json` - Machine-readable schema for validation

These artifacts are considered part of the Knowledge Graph deliverable, not optional documentation.

**Implementation Expectations:**
- RFC approval does not require these artifacts to be complete at design time
- The RFC is considered successfully implemented only once:
  - Both artifacts exist in the repository
  - Generated `kg.json` files conform to the schema
  - Ontology and schema versions are explicitly declared

Schema validation may initially be manual and later enforced via CI.

### 6. Integration Points

**Workflow Integration:**

KG extraction should be integrated into the existing workflow pipeline:

1. **After Transcript Acquisition**: KG extraction runs after transcripts are available
2. **After Speaker Detection**: KG uses detected speakers (from PRD-008) for attribution
3. **Before/After Summarization**: KG can run independently or alongside summarization
4. **Co-Located Output**: `kg.json` is written to the same episode directory as other artifacts

**Module Boundaries:**

- **Extraction Module**: New module for KG extraction (follows provider pattern)
- **Storage Module**: Uses existing filesystem utilities (no new I/O abstractions)
- **Schema Validation**: Separate validation utility (can be called from CI)

**Configuration Integration:**

KG extraction should be controlled via `Config` model:
- `generate_kg: bool` - Enable/disable KG extraction
- `kg_extraction_method: str` - Extraction method (e.g., "llm", "ner", "hybrid")
- `kg_model: Optional[str]` - Model identifier for extraction

## Key Decisions

1. **Minimal Ontology**
   - **Decision**: Start with minimal node/edge set (Podcast, Episode, Speaker, Topic, Entity, Claim)
   - **Rationale**: Reduces complexity, enables faster implementation, provides clear upgrade path

2. **Per-Episode Storage**
   - **Decision**: Store `kg.json` per episode, logical union for global graph
   - **Rationale**: Easy debugging, natural sharding, co-location with existing outputs, no global storage required

3. **Claim as First-Class Node**
   - **Decision**: Treat claims as nodes, not properties
   - **Rationale**: Enables attribution, comparison, and future disagreement modeling

4. **Evidence-First Model**
   - **Decision**: Require evidence metadata for all ML-derived content
   - **Rationale**: Enables traceability, debugging, and explainability

5. **Ontology as First-Class Artifact**
   - **Decision**: Maintain both human-readable and machine-readable ontology artifacts
   - **Rationale**: Ensures semantic stability, enables validation, supports collaboration

## Alternatives Considered

1. **Global Graph Storage**
   - **Description**: Store KG in a graph database (Neo4j, ArangoDB) or centralized JSON file
   - **Pros**: Faster global queries, built-in graph operations
   - **Cons**: Requires separate infrastructure, harder to debug, breaks co-location pattern
   - **Why Rejected**: Adds complexity, violates co-location principle, global queries deferred to post-v1

2. **Rich Ontology (Sentiment, Disagreement, Temporal)**
   - **Description**: Include sentiment analysis, disagreement modeling, temporal edges in v1
   - **Pros**: More expressive, supports advanced use cases immediately
   - **Cons**: Increases complexity, harder to validate, premature optimization
   - **Why Rejected**: Violates minimal ontology principle, deferred to post-v1

3. **Claim Normalization**
   - **Description**: Deduplicate similar claims across episodes globally
   - **Pros**: Enables claim-level aggregation, reduces storage
   - **Cons**: Complex normalization logic, risk of incorrect merging
   - **Why Rejected**: Episode-scoped claims are simpler and safer in v1

## Testing Strategy

**Test Coverage:**

- **Unit Tests**: Test node/edge construction, ID generation, evidence attachment
- **Integration Tests**: Test extraction pipeline with real transcripts, validate schema compliance
- **E2E Tests**: Test full workflow from transcript → KG extraction → `kg.json` generation

**Test Organization:**

- Unit tests: `tests/unit/test_kg_*.py`
- Integration tests: `tests/integration/test_kg_*.py`
- E2E tests: `tests/e2e/test_kg_*.py`

**Test Execution:**

- Run in CI as part of standard test suite
- Use existing test fixtures (transcripts, episodes)
- Validate schema compliance in CI (can be manual initially)

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1**: Implement extraction module and basic ontology (Podcast, Episode, Speaker, Topic)
- **Phase 2**: Add Entity and Claim extraction
- **Phase 3**: Add relationship extraction (edges)
- **Phase 4**: Schema validation and CI integration

**Monitoring:**

- Track extraction success rate (episodes with valid `kg.json`)
- Monitor extraction time (should be similar to summarization)
- Track schema compliance (all `kg.json` files pass validation)

**Success Criteria:**

1. ✅ All v1 node types can be extracted from transcripts
2. ✅ All v1 edge types can be constructed
3. ✅ Generated `kg.json` files conform to schema
4. ✅ Extraction completes within reasonable time bounds
5. ✅ Integration with existing workflow verified

## Relationship to Other RFCs

This RFC (RFC-049) is part of the Knowledge Graph initiative that includes:

1. **RFC-050: KG Use Cases & Consumption** - Defines how KG data is consumed end-to-end
2. **PRD-017: Podcast Knowledge Graph** - Defines product requirements and user value

**Key Distinction:**
- **RFC-049 (This RFC)**: Focuses on core concepts, data model, and storage
- **RFC-050**: Focuses on use cases, query patterns, and consumption

Together, these RFCs provide:
- Complete technical design for Knowledge Graph implementation
- Clear separation between data model (RFC-049) and consumption (RFC-050)
- Foundation for future extensions (trend detection, sentiment analysis, etc.)

## Benefits

1. **Structured Knowledge**: Transforms unstructured transcripts into queryable graph
2. **Attribution**: Links all knowledge to speakers, episodes, and evidence
3. **Evidence-Backed**: Every claim is traceable to transcript spans
4. **Co-Located**: KG data lives alongside existing outputs, no separate infrastructure
5. **Extensible**: Minimal v1 ontology provides clear upgrade path

## Migration Path

N/A - This is a new feature, not a migration from an existing system.

## Open Questions

1. **Topic Normalization**: Should Topic nodes be fully normalized or partially episode-scoped?
   - **Current Decision**: Global normalization with stable IDs
   - **Open**: Aggressiveness of normalization (exact match vs semantic similarity)

2. **Entity Resolution**: How aggressively should entity resolution be applied in v1?
   - **Current Decision**: Basic normalization (slug-based IDs)
   - **Open**: External entity linking (Wikidata, etc.) in v1?

3. **Global Graph Storage**: When does global graph storage become necessary?
   - **Current Decision**: Deferred to post-v1
   - **Open**: Performance thresholds that trigger global storage

4. **Extraction Method**: Which extraction methods are supported in v1?
   - **Current Decision**: LLM-based extraction (similar to summarization)
   - **Open**: Hybrid approaches (LLM + NER), rule-based extraction

## References

- **Related PRD**: `docs/prd/PRD-017-podcast-knowledge-graph.md`
- **Related RFC**: `docs/rfc/RFC-050-podcast-knowledge-graph-use-cases.md`
- **Ontology Specification**: `docs/kg/ontology.md`
- **Schema Specification**: `docs/kg/kg.schema.json`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Source Code**: `podcast_scraper/workflow/` (integration points)
