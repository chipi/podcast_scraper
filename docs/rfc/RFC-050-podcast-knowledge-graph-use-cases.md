# RFC-050: Podcast Knowledge Graph – Use Cases & End-to-End Consumption

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, downstream consumers, API users
- **Related PRDs**:
  - `docs/prd/PRD-017-podcast-knowledge-graph.md`
- **Related RFCs**:
  - `docs/rfc/RFC-049-podcast-knowledge-graph-core.md` (Core Concepts & Data Model)
- **Related Documents**:
  - `docs/kg/ontology.md` - Human-readable ontology specification
  - `docs/kg/kg.schema.json` - Machine-readable schema
  - `docs/ARCHITECTURE.md` - System architecture

## Abstract

This RFC defines how the Knowledge Graph is consumed end-to-end to deliver user value. It builds on RFC-049 (Core Concepts) and focuses on minimal v1 use cases, query patterns, output contracts, and integration with existing scraper outputs.

This document intentionally does not redefine ontology or storage mechanics (covered in RFC-049). Instead, it specifies how users and downstream systems interact with KG data to achieve concrete outcomes.

**Architecture Alignment:** This RFC aligns with existing architecture by:
- Defining consumption patterns that work with per-episode `kg.json` files
- Specifying output contracts that integrate with existing artifacts (transcript.json, summary.json, metadata.json)
- Enabling programmatic access consistent with existing API patterns
- Supporting evidence-backed queries that leverage transcript references

## Problem Statement

While RFC-049 defines *how* knowledge is extracted and stored, this RFC addresses *how* that knowledge is consumed to deliver value. Without clear consumption patterns:

- **Users don't know how to query**: No guidance on accessing KG data
- **Integration is unclear**: Unclear how KG data relates to existing outputs
- **Use cases are undefined**: No clear success criteria for v1 implementation
- **Output contracts are missing**: Downstream systems can't rely on consistent shapes

**Use Cases:**

1. **Cross-Podcast Topic Research**: Explore how a topic is discussed across episodes
2. **Speaker-Centric Insight Mapping**: Understand what a speaker talks about and claims
3. **Claim Retrieval with Evidence**: Extract concrete statements with original context
4. **Semantic Question Answering**: Ask focused questions answerable via KG structure

## Goals

1. **Define Minimal v1 Use Cases**: Establish clear, end-to-end success criteria
2. **Specify Query Patterns**: Define how users traverse the graph to answer questions
3. **Establish Output Contracts**: Define consistent output shapes for downstream systems
4. **Integrate with Existing Outputs**: Ensure KG data works alongside transcripts, summaries, metadata

## Constraints & Assumptions

**Constraints:**

- Must work with per-episode `kg.json` files (no global graph storage in v1)
- Must be evidence-backed (all answers traceable to transcript evidence)
- Must integrate cleanly with existing output directory structure
- Must support programmatic access (JSON-based, not UI-dependent)

**Assumptions:**

- Users have access to episode output directories
- Downstream systems can read JSON files
- Global graph queries can be implemented by scanning per-episode files
- Natural language query translation is deferred to post-v1

## Design & Implementation

### 1. Design Principles

1. **KG Complements Existing Outputs**: KG data augments transcripts and summaries; it does not replace them
2. **Evidence-Backed by Default**: All user-visible answers must be traceable to KG nodes and transcript evidence
3. **Structured First, Natural Language Second**: v1 prioritizes structured consumption; NL interfaces are thin wrappers
4. **Episode-Local Production, Global Consumption**: KG data is produced per episode but consumed as a logical global graph

### 2. Minimal v1 Use Cases

#### UC1. Cross-Podcast Topic Research

**User Intent:** Explore how a topic is discussed across episodes and podcasts.

**KG Traversal:**
```
Topic → Episode → Speaker
```

**Required KG Elements:**
- Topic nodes
- DISCUSSES edges
- Episode metadata

**Output Contract:**
```json
{
  "topic": "AI Regulation",
  "episodes": [
    {
      "episode_id": "episode:abc123",
      "title": "AI Regulation",
      "publish_date": "2026-02-03T00:00:00Z",
      "podcast_id": "podcast:the-journal"
    }
  ],
  "speakers": [
    {
      "speaker_id": "speaker:sam-altman",
      "name": "Sam Altman",
      "episode_count": 3
    }
  ],
  "claims": [
    {
      "claim_id": "claim:episode:abc123:...",
      "text": "Regulation will lag innovation by 3–5 years.",
      "speaker_id": "speaker:sam-altman"
    }
  ]
}
```

**Success Criteria:**
- Query returns consistent results across episodes
- Each result links back to episode metadata
- Topic nodes grow incrementally as new episodes are processed

#### UC2. Speaker-Centric Insight Mapping

**User Intent:** Understand what a speaker talks about and claims.

**KG Traversal:**
```
Speaker → Claim → Topic / Entity
```

**Required KG Elements:**
- Speaker nodes
- Claim nodes
- ASSERTS and ABOUT edges

**Output Contract:**
```json
{
  "speaker": {
    "speaker_id": "speaker:sam-altman",
    "name": "Sam Altman"
  },
  "topics": [
    {
      "topic_id": "topic:ai-regulation",
      "label": "AI Regulation",
      "claim_count": 5,
      "episode_count": 3
    }
  ],
  "claims": [
    {
      "claim_id": "claim:episode:abc123:...",
      "text": "Regulation will lag innovation by 3–5 years.",
      "episode_id": "episode:abc123",
      "timestamp_start_ms": 120000,
      "timestamp_end_ms": 135000
    }
  ]
}
```

**Success Criteria:**
- Claims are attributable and queryable
- Speaker-topic associations are stable
- Clear provenance from speaker → episode → transcript

#### UC3. Claim Retrieval with Evidence

**User Intent:** Extract concrete statements with original context.

**KG Traversal:**
```
Claim → Episode → Transcript
```

**Required KG Elements:**
- Claim nodes
- Evidence and timestamps

**Output Contract:**
```json
{
  "claim": {
    "claim_id": "claim:episode:abc123:...",
    "text": "Regulation will lag innovation by 3–5 years.",
    "speaker_id": "speaker:sam-altman",
    "episode_id": "episode:abc123",
    "timestamp_start_ms": 120000,
    "timestamp_end_ms": 135000,
    "confidence": 0.82
  },
  "evidence": {
    "episode_id": "episode:abc123",
    "transcript_ref": "transcript.json",
    "char_start": 10234,
    "char_end": 10321,
    "timestamp_start_ms": 120000,
    "timestamp_end_ms": 135000,
    "extraction_method": "llm",
    "model_version": "gpt-4.1-mini-2026-01-xx"
  },
  "transcript_span": "Regulation will lag innovation by 3–5 years. That's my prediction based on..."
}
```

**Success Criteria:**
- Claims can be cited precisely
- Evidence spans are valid (char_start/char_end match transcript)
- Transcript spans can be retrieved from transcript.json

#### UC4. Semantic Question Answering (v1-Scoped)

**User Intent:** Ask focused questions answerable via KG structure.

**Examples:**
- "Which speakers are skeptical about AI regulation?"
- "What claims mention OpenAI?"
- "Which topics does Sam Altman discuss most?"

**KG Traversal:**
- Deterministic mapping from question → graph traversal
- No free-form natural language generation

**Output Contract:**
```json
{
  "question": "Which speakers are skeptical about AI regulation?",
  "answer": {
    "speakers": [
      {
        "speaker_id": "speaker:elon-musk",
        "name": "Elon Musk",
        "claim_count": 2
      }
    ],
    "claims": [
      {
        "claim_id": "claim:episode:xyz789:...",
        "text": "AI regulation is premature and will stifle innovation.",
        "speaker_id": "speaker:elon-musk",
        "episode_id": "episode:xyz789"
      }
    ]
  },
  "explanation": "Found 1 speaker with skeptical claims about AI regulation across 2 episodes."
}
```

**Success Criteria:**
- Answers are explainable (traced to KG nodes)
- No free-form hallucinated text
- Results are evidence-backed

### 3. Minimal UI Requirements (v1)

**Design Philosophy:**

For v1, UI is about **trust, inspection, and debugging**, not "delight". The goal is to validate KG quality and enable evidence-backed queries, not to build a polished end-user product.

**What v1 UI is NOT:**
- ❌ Dashboards or charts
- ❌ Timelines or visualizations
- ❌ Fancy topic browsers
- ❌ End-user search interfaces
- ❌ Anything "product-y"

Building these now would:
- Pull ontology in the wrong direction
- Force premature query abstractions
- Optimize for presentation instead of correctness

**What v1 UI IS:**

Three minimal inspection surfaces for developers and power users:

#### 3.1. Episode KG Inspector (Non-Negotiable)

**What it is:**
- A way to view a single episode's `kg.json`
- Rendered as:
  - Nodes grouped by type (Podcast, Episode, Speaker, Topic, Entity, Claim)
  - Edges listed explicitly
  - Evidence spans visible

**Why it's required:**
- Cannot debug KG quality without seeing:
  - What claims were extracted
  - Which topics/entities were linked
  - What evidence the model chose

**Implementation bar:**
- Can be:
  - A CLI command: `kg inspect --episode <episode_id>`
  - A simple HTML page
  - A Jupyter notebook
  - **No need for React or backend**

**This answers:** "Did this episode produce sane KG data?"

**Example CLI output:**
```bash
$ kg inspect --episode episode:abc123

Episode: AI Regulation (episode:abc123)
Podcast: The Journal (podcast:the-journal)

Nodes:
  Podcasts: 1
  Episodes: 1
  Speakers: 1 (Sam Altman)
  Topics: 1 (AI Regulation)
  Entities: 0
  Claims: 3

Edges:
  HAS_EPISODE: podcast:the-journal → episode:abc123
  SPOKE_IN: speaker:sam-altman → episode:abc123
  DISCUSSES: episode:abc123 → topic:ai-regulation
  ASSERTS: speaker:sam-altman → claim:episode:abc123:...
  ABOUT: claim:episode:abc123:... → topic:ai-regulation

Claims:
  1. "Regulation will lag innovation by 3–5 years." (confidence: 0.82)
  2. "We need guardrails, not bans." (confidence: 0.75)
  3. "The EU AI Act is a good start." (confidence: 0.68)
```

#### 3.2. Claim → Evidence Viewer (Critical)

**What it is:**
- Pick a claim
- Show:
  - Claim text
  - Speaker attribution
  - Confidence score
  - Highlighted transcript span (char/timestamp-based)

**Why it's required:**
- Claims are the atomic unit of knowledge
- If users cannot:
  - See the evidence
  - Judge whether the claim is real
- Then the KG is untrustworthy

**Implementation bar:**
- Could literally be:
  - `print_claim(claim_id)` in terminal
  - Or a notebook cell that highlights text
  - Or a simple HTML page with highlighted spans

**This answers:** "Do I believe this claim?"

**Example CLI output:**
```bash
$ kg show-claim claim:episode:abc123:5d41402abc4b2a76b9719d911017c592

Claim: "Regulation will lag innovation by 3–5 years."
Speaker: Sam Altman (speaker:sam-altman)
Episode: AI Regulation (episode:abc123)
Confidence: 0.82
Timestamps: 120000ms - 135000ms (2:00 - 2:15)

Evidence (from transcript.json, chars 10234-10321):
─────────────────────────────────────────────────────
...and I think regulation will lag innovation by 3–5
years. That's my prediction based on how fast things
are moving. We need to be thoughtful about this...
─────────────────────────────────────────────────────
```

#### 3.3. One Canonical Query Surface (Only One)

**What it is:**
- Pick exactly one v1 query and make it work end-to-end
- **Recommended:** "Show me all claims about a topic"

**Why this query:**
- Exercises Topic → Claim → Episode → Transcript
- Touches almost every KG primitive
- Maps directly to UC1 (Cross-Podcast Topic Research) + UC3 (Claim Retrieval)

**Implementation bar:**
- CLI:
  ```bash
  kg query --topic "AI Regulation"
  ```
- Or notebook function:
  ```python
  get_claims_for_topic("AI Regulation")
  ```

**Output:**
- Structured JSON (for programmatic use)
- Optional pretty-print (for human inspection)

**Example CLI output:**
```bash
$ kg query --topic "AI Regulation"

Topic: AI Regulation
Found 12 claims across 5 episodes

Episodes:
  - AI Regulation (episode:abc123) - 3 claims
  - The Future of AI Policy (episode:def456) - 5 claims
  - Regulating Innovation (episode:ghi789) - 4 claims

Speakers:
  - Sam Altman (speaker:sam-altman) - 5 claims
  - Tim Cook (speaker:tim-cook) - 4 claims
  - Sundar Pichai (speaker:sundar-pichai) - 3 claims

Top Claims:
  1. "Regulation will lag innovation by 3–5 years." (Sam Altman, confidence: 0.82)
  2. "We need guardrails, not bans." (Sam Altman, confidence: 0.75)
  3. "The EU AI Act is a good start." (Sam Altman, confidence: 0.68)
  ...

Use 'kg show-claim <claim_id>' to view evidence for any claim.
```

**What a good v1 UI looks like:**

If your UI feels:
- ✅ Slightly boring
- ✅ Very explicit
- ✅ Developer-oriented

**You're doing it right.**

If it feels:
- ❌ "Polished"
- ❌ "Shareable"
- ❌ "Product-ready"

**You're too early.**

**Strong Recommendation:**

Start with:
1. A CLI-based inspector (`kg inspect`, `kg show-claim`, `kg query`)
2. A Jupyter notebook for interactive exploration
3. **No web UI at all**

Only once:
- Ontology is stable
- Claims are high quality
- Traversals feel natural

…then build a web UI (or agent interface).

**How this maps to RFCs:**
- **RFC-049 (Core KG)**: No UI required (data model only)
- **RFC-050 (Use Cases)**: Defines what must be queryable
- **UI v1**: Proves queries are real and evidence-backed

**UI is a validation tool, not a feature.**

#### 3.4. CLI Specification (Implementation Details)

**CLI Name:** `kg`

**Common Concepts:**

**Inputs:**
- `--output-dir <path>`: Root output directory produced by podcast_scraper (default: `./output`)
- `--episode-id <id>`: Episode identifier (e.g., `episode:abc123`)
- `--episode-path <path>`: Direct path to an episode folder (e.g., `output/episode_abc123/`)

Exactly one of `--episode-id` or `--episode-path` must be provided when the command targets a single episode.

**Artifact Expectations (per episode folder):**
- `metadata.json`
- `transcript.json`
- `summary.json` (optional for CLI)
- `kg.json`

**Output Formats:**
- `--format json|pretty` (default: `pretty`)
- `--out <path>`: Write output to file (optional)

**Global Options:**
- `--strict`: Fail if schema validation fails
- `--schema <path>`: Override schema path (default: `schemas/kg.schema.json`)

**Command: `kg inspect`**

**Goal:** Inspect a single episode's `kg.json` in a human-friendly way.

**Usage:**
```bash
kg inspect --episode-id episode:abc123 --output-dir ./output
# or
kg inspect --episode-path ./output/episode_abc123
```

**Options:**
- `--show nodes|edges|all` (default: `all`)
- `--group-by type` (default: enabled)
- `--stats` (default: enabled)

**Behavior:**
- Loads `kg.json`
- Validates against schema (warn by default; fail if `--strict`)
- Prints:
  - Node counts by type
  - Edge counts by type
  - Nodes grouped by type (id + core properties)
  - Edges grouped by type (from → to)
  - For ML-derived edges and claims: confidence + evidence summary

**Command: `kg claim`**

**Goal:** Resolve a claim to its transcript evidence and show it with attribution.

**Usage:**
```bash
kg claim --id claim:episode:abc123:<hash> --output-dir ./output
```

**Options:**
- `--context-chars <n>` (default: 200)
- `--highlight true|false` (default: true)

**Behavior:**
- Locates the episode containing the claim
- Loads episode `kg.json` + `transcript.json`
- Prints claim text, speaker, confidence, and highlighted evidence

**Command: `kg query`**

**Goal:** Run the single canonical v1 query proving end-to-end consumption.

**Canonical Query (v1):** "Show me all claims about a topic."

**Usage:**
```bash
kg query --topic "AI Regulation" --output-dir ./output
```

**Options:**
- `--topic <label>` (required)
- `--limit <n>` (default: 50)
- `--min-confidence <0..1>` (default: 0.0)
- `--sort confidence|time` (default: confidence)

**Behavior:**
- Builds an in-memory logical graph by scanning all `kg.json` files
- Traverses Topic → Claim → Episode → Speaker
- Returns structured, evidence-backed results

**Exit Codes:**
- `0`: success
- `2`: invalid arguments
- `3`: missing files
- `4`: schema validation failed in `--strict` mode

#### 3.5. Jupyter Notebook Template (Alternative Implementation)

For interactive exploration, a Jupyter notebook template provides an alternative to CLI commands. The template (`docs/wip/notebooks_kg_claim_inspection_template.py`) demonstrates:

- Loading episode artifacts (`kg.json`, `transcript.json`, `metadata.json`)
- Claim selection and inspection
- Evidence resolution with highlighted transcript spans
- Sanity checks (char range validation, timestamp validation)
- Topic/Entity link exploration

This template serves as a reference implementation for the "Claim → Evidence Viewer" requirement and can be adapted for other inspection tasks.

### 4. Integration with Existing podcast_scraper Outputs

**Co-Located Consumption Model:**

KG data is consumed alongside existing artifacts:

```
output/
  episode_<id>/
    metadata.json
    transcript.json
    summary.json
    kg.json          # NEW: Knowledge Graph data
```

**Consumption Patterns:**

1. **Direct kg.json Access**: Read `kg.json` directly for programmatic access
2. **Join with Summary**: Combine KG data with `summary.json` for narrative context
3. **Resolve Transcript Spans**: Use `transcript.json` to retrieve evidence spans
4. **Metadata Integration**: Use `metadata.json` for episode-level context

**Example Consumption Code:**

```python
import json
from pathlib import Path

def query_topic(topic_label: str, output_dir: Path) -> dict:
    """Query all episodes for a topic."""
    results = {
        "topic": topic_label,
        "episodes": [],
        "speakers": [],
        "claims": []
    }

    # Scan all episode directories
    for episode_dir in output_dir.glob("episode_*"):
        kg_path = episode_dir / "kg.json"
        if not kg_path.exists():
            continue

        with open(kg_path) as f:
            kg_data = json.load(f)

        # Find topic node
        topic_node = None
        for node in kg_data["nodes"]:
            if (node["type"] == "Topic" and
                node["properties"]["label"] == topic_label):
                topic_node = node
                break

        if not topic_node:
            continue

        # Find related episodes via DISCUSSES edges
        for edge in kg_data["edges"]:
            if (edge["type"] == "DISCUSSES" and
                edge["to"] == topic_node["id"]):
                episode_id = edge["from"]
                # Resolve episode node and add to results
                # ...

    return results
```

### 5. Output Shapes (Illustrative)

**Topic Query Result:**

```json
{
  "topic": "AI Regulation",
  "episodes": ["episode:abc123", "episode:def456"],
  "speakers": ["speaker:sam-altman"],
  "claims": ["claim:episode:abc123:..."]
}
```

**Claim Result:**

```json
{
  "claim": "Regulation will lag innovation by 3–5 years.",
  "speaker": "Sam Altman",
  "episode_id": "episode:abc123",
  "timestamps": [120000, 135000],
  "confidence": 0.82
}
```

**Speaker Profile:**

```json
{
  "speaker_id": "speaker:sam-altman",
  "name": "Sam Altman",
  "topics": [
    {
      "topic_id": "topic:ai-regulation",
      "label": "AI Regulation",
      "claim_count": 5
    }
  ],
  "episode_count": 3
}
```

### 6. End-to-End Success Definition (v1)

The KG implementation is considered end-to-end successful when:

- ✅ All v1 use cases (UC1–UC4) can be executed using KG data
- ✅ Outputs are evidence-backed (traceable to transcript evidence)
- ✅ KG data integrates cleanly with existing scraper artifacts
- ✅ No user-visible feature relies on raw transcript scanning
- ✅ Generated `kg.json` files conform to schema
- ✅ Query patterns are documented and reproducible

### 7. Failure Modes & Fallbacks

- **No claims found**: Return topic/episode links only (degraded but useful)
- **Low confidence extractions**: Mark explicitly, do not suppress (transparency)
- **Incomplete KG**: Fall back to summaries (graceful degradation)
- **Schema validation failure**: Log error, skip episode (non-fatal)

## Key Decisions

1. **Structured First, NL Second**
   - **Decision**: v1 prioritizes structured consumption over natural language interfaces
   - **Rationale**: Reduces complexity, enables programmatic access, provides clear contracts

2. **Evidence-Backed by Default**
   - **Decision**: All answers must be traceable to transcript evidence
   - **Rationale**: Enables trust, debugging, and explainability

3. **Co-Located Consumption**
   - **Decision**: KG data consumed alongside existing artifacts
   - **Rationale**: Maintains existing patterns, no separate infrastructure

4. **Minimal v1 Use Cases**
   - **Decision**: Focus on UC1–UC4, defer advanced use cases
   - **Rationale**: Ensures end-to-end success without over-engineering

## Alternatives Considered

1. **Natural Language Query Interface**
   - **Description**: Build NL → graph query translation in v1
   - **Pros**: More user-friendly, lower barrier to entry
   - **Cons**: Complex, error-prone, harder to validate
   - **Why Rejected**: Deferred to post-v1, structured queries are sufficient

2. **Separate Query API**
   - **Description**: Build REST API or GraphQL endpoint for queries
   - **Pros**: Standardized interface, easier for external consumers
   - **Cons**: Requires separate service, adds infrastructure
   - **Why Rejected**: File-based access is simpler and sufficient in v1

3. **Global Graph Index**
   - **Description**: Build global index for faster queries
   - **Pros**: Faster queries, better performance
   - **Cons**: Requires global storage, adds complexity
   - **Why Rejected**: Deferred to post-v1, per-episode scanning is acceptable

## Testing Strategy

**Test Coverage:**

- **Unit Tests**: Test query functions, output shape validation, evidence resolution
- **Integration Tests**: Test end-to-end use cases with real `kg.json` files
- **E2E Tests**: Test full workflow from transcript → KG → query → results

**Test Organization:**

- Unit tests: `tests/unit/test_kg_queries.py`
- Integration tests: `tests/integration/test_kg_use_cases.py`
- E2E tests: `tests/e2e/test_kg_e2e.py`

**Test Execution:**

- Run in CI as part of standard test suite
- Use existing test fixtures (transcripts, episodes)
- Validate output contracts match specifications

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1**: Implement UC1 (Topic Research) with basic query functions
- **Phase 2**: Add UC2 (Speaker Mapping) and UC3 (Claim Retrieval)
- **Phase 3**: Add UC4 (Semantic QA) with deterministic question mapping
- **Phase 4**: Documentation and example code

**Monitoring:**

- Track query success rate (queries that return results)
- Monitor query performance (time to scan per-episode files)
- Track evidence resolution accuracy (transcript spans are valid)

**Success Criteria:**

1. ✅ All v1 use cases can be executed end-to-end
2. ✅ Output contracts match specifications
3. ✅ Evidence resolution works correctly
4. ✅ Integration with existing outputs verified
5. ✅ Query patterns documented with examples

## Relationship to Other RFCs

This RFC (RFC-050) is part of the Knowledge Graph initiative that includes:

1. **RFC-049: Core KG Concepts & Data Model** - Defines ontology, storage, and schema
2. **PRD-017: Podcast Knowledge Graph** - Defines product requirements and user value

**Key Distinction:**
- **RFC-049**: Focuses on *how* knowledge is extracted and stored
- **RFC-050 (This RFC)**: Focuses on *how* knowledge is consumed and queried

Together, these RFCs provide:
- Complete technical design for Knowledge Graph implementation
- Clear separation between data model (RFC-049) and consumption (RFC-050)
- Foundation for future extensions (NL queries, global indexes, etc.)

## Benefits

1. **Clear Use Cases**: Defined success criteria for v1 implementation
2. **Structured Consumption**: Programmatic access with consistent contracts
3. **Evidence-Backed**: All answers traceable to transcript evidence
4. **Integration**: Works seamlessly with existing outputs
5. **Extensible**: Foundation for future NL queries and advanced analytics

## Migration Path

N/A - This is a new feature, not a migration from an existing system.

## Open Questions

1. **Query Performance**: How many episodes can be scanned before performance degrades?
   - **Current Decision**: Acceptable for v1, optimize in post-v1
   - **Open**: Performance thresholds that trigger global indexing

2. **Natural Language Queries**: When should NL → graph translation be added?
   - **Current Decision**: Deferred to post-v1
   - **Open**: User feedback and demand

3. **Advanced Analytics**: When should trend detection and temporal analysis be added?
   - **Current Decision**: Deferred to post-v1
   - **Open**: Use case prioritization

## References

- **Related PRD**: `docs/prd/PRD-017-podcast-knowledge-graph.md`
- **Related RFC**: `docs/rfc/RFC-049-podcast-knowledge-graph-core.md`
- **Ontology Specification**: `docs/kg/ontology.md`
- **Schema Specification**: `docs/kg/kg.schema.json`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Source Code**: `podcast_scraper/workflow/` (integration points)
