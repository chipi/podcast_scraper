# PRD-017: Grounded Insight Layer (GIL)

- **Status**: 📋 Draft
- **Authors**: Podcast Scraper Team
- **Related RFCs**:
  - RFC-044 (Model Registry — prerequisite)
  - RFC-042 (Hybrid ML Platform — prerequisite)
  - RFC-052 (Local LLM Prompts — prerequisite)
  - RFC-049 (Core Concepts & Data Model)
  - RFC-050 (Use Cases & Insight Explorer)
  - RFC-051 (Database Projection)
- **Related Documents**:
  - `docs/kg/ontology.md` - Human-readable ontology
  - `docs/kg/kg.schema.json` - Machine-readable schema

## Summary

The **Grounded Insight Layer (GIL)** transforms podcast content into a structured, evidence-backed system that extracts **insights** (key takeaways) and **quotes** (verbatim evidence) with full attribution and timestamps. The user-facing value is **trust and navigation**: users can retrieve insights and immediately see the exact evidence supporting them.

The internal graph structure enables semantic organization, but the primary deliverable is reliable insight retrieval with grounded evidence. Every insight links to supporting quotes, and every quote links to its exact transcript span and timestamp. This evidence-first approach distinguishes GIL from traditional summarization and enables trustworthy downstream applications (RAG, analytics, AI agents).

## Background & Context

Podcast content is rich in insights, opinions, and factual knowledge, but is currently consumed as linear, unstructured text (transcripts, summaries). This makes it difficult to:

- **Trust insights**: Users cannot verify whether a summary accurately reflects what was said
- **Navigate to evidence**: Finding the exact quote or timestamp for an insight requires manual search
- **Compare ideas across episodes**: Insights are locked in episode silos without cross-linking
- **Build downstream applications**: RAG systems need grounded evidence, not just extracted claims

The existing podcast scraper produces transcripts and summaries, but summaries lose the connection to source material. The **Grounded Insight Layer** addresses this by extracting structured insights with explicit evidence grounding—every insight links to supporting quotes with exact timestamps.

**Core Value Proposition:**

- **For researchers**: Get key takeaways and jump directly to the supporting evidence
- **For developers**: Build RAG/agent systems with provenance you can trust
- **For analysts**: Cite insights with confidence, knowing the evidence is verifiable

**How it relates to existing features:**

- **Transcripts** (PRD-001): GIL operates on existing transcript outputs
- **Summarization** (PRD-005): GIL complements summaries by providing grounded, verifiable knowledge
- **Metadata** (PRD-004): GIL data is co-located with existing metadata artifacts
- **Speaker Detection** (PRD-008): GIL leverages detected speakers for quote attribution

## Goals

1. **Extract Grounded Insights**: Identify key takeaways from transcripts as structured "insights" with confidence scores
2. **Extract Verbatim Quotes**: Capture exact quotes with timestamps that serve as evidence for insights
3. **Ensure Evidence Grounding**: Every insight must be explicitly grounded (linked to supporting quotes) or marked as ungrounded
4. **Enable Trust & Navigation**: Users can retrieve insights and immediately navigate to supporting evidence
5. **Support Downstream Applications**: Provide evidence-backed foundation for RAG, analytics, and AI agents

## Non-Goals (v1)

- Perfect factual correctness (confidence scores indicate extraction certainty, not truth)
- Real-time processing (batch processing per episode)
- Automated truth validation (evidence-backed extraction, not fact-checking)
- **Entity extraction and resolution** (deferred to v1.1; focus is on insights + quotes + topics)
- Entity normalization and linking (deferred to v1.1)
- Sentiment analysis (deferred to post-v1)
- Disagreement/contradiction modeling (deferred to post-v1)
- Trend detection and temporal analytics (deferred to post-v1)
- Advanced entity intelligence dashboards (deferred to post-v1)
- Real-time or streaming ingestion (deferred to post-v1)

## Personas

- **Knowledge Workers & Researchers**: Need to quickly find key insights and navigate to supporting quotes without listening to hours of audio
- **Founders, Investors, Analysts**: Require credible, evidence-backed insights for due diligence and market intelligence
- **Developers Building AI Agents**: Need grounded, verifiable knowledge for trustworthy RAG systems and downstream applications

## User Stories

- _As a researcher, I can query a topic and retrieve key insights with supporting quotes so that I can quickly understand what was said with evidence._
- _As an analyst, I can extract insights with verbatim quotes and timestamps so that I can cite findings with confidence._
- _As a developer, I can retrieve insights with evidence pointers programmatically so that I can build trustworthy AI agents on top of podcast data._
- _As a knowledge worker, I can explore insights by topic and navigate to the exact transcript moment so that I can verify claims and build deeper understanding._

## Functional Requirements

### FR1: Insight Extraction (Takeaways)

- **FR1.1**: Extract key insights (takeaways) from episode transcripts as structured objects
- **FR1.2**: Attach confidence scores to each extracted insight (0.0-1.0)
- **FR1.3**: Link each insight to one or more topics (lightweight, mergeable topic nodes)
- **FR1.4**: Mark each insight's grounding status (`grounded=true/false`)

### FR2: Quote Extraction (Evidence)

- **FR2.1**: Extract verbatim quotes from transcripts that serve as evidence for insights
- **FR2.2**: Attach precise timestamps (`start_ms`, `end_ms`) to each quote
- **FR2.3**: Attach transcript spans (`char_start`, `char_end`) to each quote
- **FR2.4**: Link quotes to speakers (when diarization is available)

### FR3: Evidence Grounding (Critical)

- **FR3.1**: Every insight must be either grounded (`≥1 SUPPORTED_BY` quote) or explicitly `grounded=false`
- **FR3.2**: Every quote must be verbatim and point to exact transcript span + timestamps
- **FR3.3**: Grounding relationships are explicitly modeled (Insight → SUPPORTED_BY → Quote)
- **FR3.4**: No ungrounded insights are presented as evidence-backed (honest about extraction limits)

### FR4: Graph Construction

- **FR4.1**: Deduplicate nodes (topics, speakers) across episodes using stable IDs
- **FR4.2**: Create relationships (HAS_INSIGHT, SUPPORTED_BY, SPOKEN_BY, ABOUT)
- **FR4.3**: Attach confidence scores and provenance metadata to all ML-derived nodes/edges
- **FR4.4**: Maintain episode-local Insight/Quote nodes while referencing global nodes (Topic, Speaker)

### FR5: Storage & Output

- **FR5.1**: Generate `kg.json` file per episode in the episode output directory
- **FR5.2**: Co-locate GIL data with existing artifacts (transcript.json, summary.json, metadata.json)
- **FR5.3**: Ensure all `kg.json` files conform to the machine-readable schema (`docs/kg/kg.schema.json`)
- **FR5.4**: Include schema version, model_version, and prompt_version in each `kg.json` file

### FR6: Query & Consumption (v1-Scoped)

- **FR6.1**: Support structured queries returning insights with supporting quotes
- **FR6.2**: Return evidence-backed results with links to transcript spans and timestamps
- **FR6.3**: Enable programmatic access to GIL data via JSON files
- **FR6.4**: Support key traversals: Topic → Insights → Supporting Quotes → Timestamps

## Success Metrics

- **Extraction Coverage**: >=80% of episodes successfully produce insights + quotes
- **Insight Grounding Rate**: >=90% of insights are grounded (have ≥1 supporting quote)
- **Quote Validity Rate**: >=95% of quotes have valid transcript span + timestamp references
- **Knowledge Density**: Average >=5 insights and >=10 supporting quotes per episode
- **Query Usefulness**: Positive developer feedback on insight retrieval with evidence
- **Schema Compliance**: 100% of generated `kg.json` files pass schema validation
- **Evidence Trust**: >=90% of quote → transcript lookups return verbatim match

## Dependencies

**Existing features (required inputs):**

- **PRD-001**: Transcript Acquisition Pipeline
  (GIL operates on transcripts)
- **PRD-004**: Metadata Generation (GIL data
  co-located with metadata)
- **PRD-005**: Episode Summarization (GIL complements
  summaries with grounded evidence)
- **PRD-008**: Speaker Name Detection (GIL uses
  detected speakers for quote attribution)

**ML Platform (prerequisite infrastructure):**

- **RFC-044**: Model Registry — model metadata,
  capabilities, and limits for all extraction models
- **RFC-042**: Hybrid ML Platform — provides
  extraction models (FLAN-T5, extractive QA, NLI,
  sentence-transformers) that power GIL extraction
- **RFC-052**: Local LLM Prompts — optimized prompts
  for Ollama-based GIL extraction (Tier 2)

**GIL-specific design:**

- **RFC-049**: Core GIL Concepts & Data Model
  (defines ontology, grounding contract, storage)
- **RFC-050**: GIL Use Cases & Insight Explorer
  (defines query patterns and CLI tooling)
- **RFC-051**: Database Projection (defines Postgres
  export for fast queries)

## Constraints & Assumptions

**Constraints:**

- Must not require global graph storage in v1 (logical union of per-episode files)
- Must be backward compatible with existing output directory structure
- Must complete extraction within reasonable time bounds (similar to summarization)
- Must not break existing workflows when disabled
- **Every insight must have explicit grounding status** (no ambiguous grounding)
- **Every quote must be verbatim** (no paraphrasing in Quote nodes)

**Assumptions:**

- ML platform (RFC-042 + RFC-044) is implemented
  before GIL extraction begins, providing extraction
  models at three tiers: ML-only (FLAN-T5), Hybrid
  (Ollama), and Cloud LLM
- Transcripts are available before GIL extraction
- Users have sufficient storage for per-episode
  `kg.json` files (~5-20 KB each)
- Schema validation can be manual initially, automated
  later via CI
- Speaker detection may not always be available
  (quotes can have nullable speaker_id)

## Design Considerations

### Ontology Design

- **Option A**: Insight-centric ontology (Podcast, Episode, Speaker, Topic, Insight, Quote)
  - **Pros**: User-value focused, evidence is first-class, matches "takeaways + quotes" paradigm
  - **Cons**: Different from traditional KG framing (but aligns with modern AI products)
  - **Decision**: Option A (insight-centric v1 ontology, entities deferred to v1.1)

### Storage Strategy

- **Option A**: Per-episode `kg.json` files (logical union for global graph)
  - **Pros**: Easy debugging, natural sharding, reprocessable, co-located with outputs
  - **Cons**: Global queries require scanning multiple files (mitigated by RFC-051 DB projection)
  - **Decision**: Option A (per-episode storage + Postgres projection for fast queries)

### Grounding Strategy

- **Option A**: Explicit grounding contract (every insight must declare grounding status)
  - **Pros**: Honest about extraction limits, builds trust, enables quality metrics
  - **Cons**: May have some ungrounded insights in early iterations
  - **Decision**: Option A (explicit grounding contract is the 2025 moat)

## Integration with Existing Features

The Grounded Insight Layer enhances the pipeline by:

- **Transcript Processing**: GIL extraction runs after transcript acquisition, using transcript content as input
- **Metadata Generation**: GIL data is co-located with metadata artifacts in the same episode directory
- **Summarization**: GIL complements summaries by providing grounded evidence for narrative claims
- **Speaker Detection**: GIL uses detected speakers (from PRD-008) for quote attribution
- **Database Export**: GIL data can be projected to Postgres (RFC-051) for fast queries

## Example Output

```json
{
  "schema_version": "1.0",
  "model_version": "gpt-4.1-mini-2026-01-xx",
  "prompt_version": "v2.1",
  "episode_id": "episode:abc123",
  "nodes": [
    {
      "id": "podcast:the-journal",
      "type": "Podcast",
      "properties": {
        "title": "The Journal",
        "rss_url": "https://..."
      }
    },
    {
      "id": "episode:abc123",
      "type": "Episode",
      "properties": {
        "podcast_id": "podcast:the-journal",
        "title": "AI Regulation",
        "publish_date": "2026-02-03T00:00:00Z"
      }
    },
    {
      "id": "speaker:sam-altman",
      "type": "Speaker",
      "properties": {
        "name": "Sam Altman"
      }
    },
    {
      "id": "topic:ai-regulation",
      "type": "Topic",
      "properties": {
        "label": "AI Regulation"
      }
    },
    {
      "id": "insight:episode:abc123:a1b2c3d4",
      "type": "Insight",
      "properties": {
        "text": "AI regulation will significantly lag behind the pace of innovation",
        "episode_id": "episode:abc123",
        "grounded": true,
        "confidence": 0.85
      }
    },
    {
      "id": "quote:episode:abc123:e5f6g7h8",
      "type": "Quote",
      "properties": {
        "text": "Regulation will lag innovation by 3–5 years. That's my prediction.",
        "episode_id": "episode:abc123",
        "speaker_id": "speaker:sam-altman",
        "timestamp_start_ms": 120000,
        "timestamp_end_ms": 135000,
        "char_start": 10234,
        "char_end": 10321,
        "transcript_ref": "transcript.json"
      }
    }
  ],
  "edges": [
    {
      "type": "HAS_EPISODE",
      "from": "podcast:the-journal",
      "to": "episode:abc123"
    },
    {
      "type": "SPOKE_IN",
      "from": "speaker:sam-altman",
      "to": "episode:abc123"
    },
    {
      "type": "HAS_INSIGHT",
      "from": "episode:abc123",
      "to": "insight:episode:abc123:a1b2c3d4"
    },
    {
      "type": "SUPPORTED_BY",
      "from": "insight:episode:abc123:a1b2c3d4",
      "to": "quote:episode:abc123:e5f6g7h8"
    },
    {
      "type": "SPOKEN_BY",
      "from": "quote:episode:abc123:e5f6g7h8",
      "to": "speaker:sam-altman"
    },
    {
      "type": "ABOUT",
      "from": "insight:episode:abc123:a1b2c3d4",
      "to": "topic:ai-regulation",
      "properties": {
        "confidence": 0.79
      }
    }
  ]
}
```

## Resolved Questions

All product questions have been resolved. Decisions
are recorded here for traceability.

1. **How to model disagreement between insights?**
   Deferred to post-v1. v1 extracts independent
   insights per episode. Contradiction detection
   requires cross-episode comparison with NLI models
   (RFC-042 §11.4.4), which is available infrastructure
   but not a v1 use case. When added, disagreements
   would be modeled as `CONTRADICTS` edges between
   Insight nodes.

2. **How to represent uncertainty vs opinion?**
   Deferred to post-v1. v1 uses `confidence` scores
   (extraction certainty) but does not distinguish
   factual claims from opinions. A future
   `insight_type` field (fact / opinion / prediction)
   could enable this, using NLI or LLM classification.

3. **When does entity extraction become valuable?**
   Planned for v1.1. v1 prioritizes Topics + Insights
   + Quotes. Entity extraction (people, organizations,
   products mentioned in content) adds value when
   users need cross-episode entity profiles. spaCy NER
   (already in codebase) provides baseline extraction;
   LLM-based extraction improves quality. Deferred to
   keep v1 scope manageable.

4. **When does global graph storage become necessary?**
   Not for v1. Per-episode `kg.json` files with
   Postgres projection (RFC-051) handle v1 scale.
   A graph-native store (Neo4j, Apache AGE) becomes
   valuable at ~1000+ episodes or when multi-hop
   traversals become common queries.

5. **How to handle ungroundable insights?**
   Mark `grounded=false` explicitly. The grounding
   contract (RFC-049 §5) requires every insight to
   declare its grounding status. Ungrounded insights
   are honest about extraction limits and can be
   filtered in the UI. Target: ≥90% grounding rate.

## Related Work

**ML Platform (prerequisites):**

- **RFC-044**: Model Registry — model metadata for
  all extraction models
- **RFC-042**: Hybrid ML Platform — extraction models
  (FLAN-T5, QA, NLI, embeddings)
- **RFC-052**: Local LLM Prompts — optimized prompts
  for Ollama extraction

**GIL Design:**

- **RFC-049**: Core GIL Concepts & Data Model —
  ontology, grounding contract, storage
- **RFC-050**: GIL Use Cases & Insight Explorer —
  query patterns, CLI, consumption
- **RFC-051**: Database Projection — Postgres export
  for fast queries
- **PRD-018**: Database Projection PRD — product
  requirements for DB export

**Specifications:**

- **docs/kg/ontology.md**: Human-readable ontology
- **docs/kg/kg.schema.json**: Machine-readable schema

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] RFC-049 updated with Insight + Quote ontology
- [ ] RFC-050 updated with Insight Explorer use case
- [ ] RFC-051 updated with insights/quotes/insight_support tables
- [ ] Ontology specification (`docs/kg/ontology.md`) finalized
- [ ] Schema specification (`docs/kg/kg.schema.json`) finalized
- [ ] Implementation completed
- [ ] Grounding contract enforced (every insight has grounding status)
- [ ] Tests cover insight extraction, quote extraction, and grounding
- [ ] Documentation updated (README, config examples)
- [ ] Integration with transcript pipeline verified
- [ ] Schema validation integrated (manual or CI)
