# RFC-050: Grounded Insight Layer – Use Cases & End-to-End Consumption

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, downstream consumers, API users
- **Related PRDs**:
  - `docs/prd/PRD-017-grounded-insight-layer.md` (Grounded Insight Layer)
- **Related RFCs**:
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` (Core Concepts & Data Model)
  - `docs/rfc/RFC-051-grounded-insight-layer-database-projection.md` (Database Projection)
- **Related Documents**:
  - `docs/kg/ontology.md` - Human-readable ontology specification
  - `docs/kg/kg.schema.json` - Machine-readable schema
  - `docs/ARCHITECTURE.md` - System architecture

## Abstract

This RFC defines how the **Grounded Insight Layer (GIL)** is consumed end-to-end to deliver user value. The primary user value is **trust and navigation**: users retrieve insights and immediately see supporting quotes with timestamps.

This RFC builds on RFC-049 (Core Concepts) and focuses on use cases that revolve around **insights + supporting quotes**, not just graph traversal. The key deliverable is the **Insight Explorer** pattern: query returns top insights + supporting quotes + episode links/timestamps.

**Architecture Alignment:** This RFC aligns with existing architecture by:

- Defining consumption patterns that work with per-episode `kg.json` files
- Specifying output contracts that return insights with supporting_quotes
- Enabling programmatic access consistent with existing API patterns
- Supporting evidence-backed queries that leverage transcript references and timestamps

## Problem Statement

While RFC-049 defines *how* knowledge is extracted and stored, this RFC addresses *how* that knowledge is consumed to deliver value. Without clear consumption patterns:

- **Users don't know how to query**: No guidance on accessing GIL data
- **Integration is unclear**: Unclear how GIL data relates to existing outputs
- **Use cases are undefined**: No clear success criteria for v1 implementation
- **Output contracts are missing**: Downstream systems can't rely on consistent shapes

**Core User Value:** Users want to **retrieve insights** and **see evidence**. The graph is internal plumbing; the product is **trust + navigation**.

**Use Cases:**

1. **Cross-Podcast Topic Research**: Explore insights about a topic with supporting quotes
2. **Speaker-Centric Insight Mapping**: Understand what a speaker said with verbatim evidence
3. **Evidence-Backed Quote/Insight Retrieval**: Get insights with exact quotes and timestamps
4. **Semantic Question Answering**: Ask focused questions answerable via GIL structure
5. **Insight Explorer** (NEW): Query returns top insights + supporting quotes + episode timestamps

## Goals

1. **Define Insight-Centric Use Cases**: Establish use cases that return insights + supporting quotes
2. **Specify Query Patterns**: Define how users retrieve insights with evidence
3. **Establish Output Contracts**: Define output shapes that include `insights[]` with `supporting_quotes[]`
4. **Integrate with Existing Outputs**: Ensure GIL data works alongside transcripts, summaries, metadata
5. **Prove Cross-Stack Value**: Implement one canonical query that proves the layer is valuable

## Constraints & Assumptions

**Constraints:**

- Must work with per-episode `kg.json` files (no global graph storage in v1)
- Must be evidence-backed (all answers traceable to supporting quotes)
- Must integrate cleanly with existing output directory structure
- Must support programmatic access (JSON-based, not UI-dependent)
- **Output must include insights with supporting_quotes** (not just claims)

**Assumptions:**

- Users have access to episode output directories
- Downstream systems can read JSON files
- Global graph queries can be implemented by scanning per-episode files (or via RFC-051 DB)
- Natural language query translation is deferred to post-v1

## Design & Implementation

### 1. Design Principles

1. **Insights + Quotes, Not Just Graph Traversal**: User value is insights with evidence, not graph navigation
2. **Evidence-Backed by Default**: All user-visible answers include supporting quotes with timestamps
3. **Structured First, Natural Language Second**: v1 prioritizes structured consumption; NL interfaces are thin wrappers
4. **Episode-Local Production, Global Consumption**: GIL data is produced per episode but consumed as logical global layer
5. **One Cross-Stack Feature Proves Value**: The Insight Explorer query demonstrates full system capability

### 2. Minimal v1 Use Cases

#### UC1. Cross-Podcast Topic Research

**User Intent:** Explore insights about a topic across episodes, with supporting evidence.

**GIL Traversal:**

```
Topic → Insight → Supporting Quotes → Episode
```

**Required GIL Elements:**

- Topic nodes
- Insight nodes with ABOUT edges to topics
- Quote nodes with SUPPORTED_BY edges from insights
- Episode metadata

**Output Contract (Updated for Insight + Quotes):**

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
  "insights": [
    {
      "insight_id": "insight:episode:abc123:a1b2c3d4",
      "text": "AI regulation will significantly lag behind the pace of innovation",
      "grounded": true,
      "confidence": 0.85,
      "episode_id": "episode:abc123",
      "supporting_quotes": [
        {
          "quote_id": "quote:episode:abc123:e5f6g7h8",
          "text": "Regulation will lag innovation by 3–5 years. That's my prediction.",
          "speaker_id": "speaker:sam-altman",
          "speaker_name": "Sam Altman",
          "timestamp_start_ms": 120000,
          "timestamp_end_ms": 135000
        }
      ]
    }
  ],
  "speakers": [
    {
      "speaker_id": "speaker:sam-altman",
      "name": "Sam Altman",
      "insight_count": 5
    }
  ]
}
```

**Success Criteria:**

- Query returns insights with supporting quotes (not just claims)
- Each insight links to verbatim quotes with timestamps
- Users can navigate from insight → quote → episode → timestamp

#### UC2. Speaker-Centric Insight Mapping

**User Intent:** Understand what a speaker said, with verbatim evidence.

**GIL Traversal:**

```
Speaker → Quotes (spoken by) → Insights (supported by) → Topics
```

**Required GIL Elements:**

- Speaker nodes
- Quote nodes with SPOKEN_BY edges
- Insight nodes with SUPPORTED_BY edges
- ABOUT edges to topics

**Output Contract (Updated for Insight + Quotes):**

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
      "insight_count": 5,
      "quote_count": 12,
      "episode_count": 3
    }
  ],
  "insights": [
    {
      "insight_id": "insight:episode:abc123:a1b2c3d4",
      "text": "AI regulation will significantly lag behind the pace of innovation",
      "grounded": true,
      "episode_id": "episode:abc123",
      "supporting_quotes": [
        {
          "quote_id": "quote:episode:abc123:e5f6g7h8",
          "text": "Regulation will lag innovation by 3–5 years. That's my prediction.",
          "timestamp_start_ms": 120000,
          "timestamp_end_ms": 135000
        }
      ]
    }
  ]
}
```

**Success Criteria:**

- Speaker profile shows insights they support (via their quotes)
- Each insight links to verbatim quotes from the speaker
- Clear provenance: speaker → quote → insight → episode

#### UC3. Evidence-Backed Quote/Insight Retrieval

**User Intent:** Get an insight with exact quotes, timestamps, and transcript evidence.

**GIL Traversal:**

```
Insight → Supporting Quotes → Transcript Span
```

**Required GIL Elements:**

- Insight nodes
- Quote nodes with SUPPORTED_BY edges
- Transcript references with char spans

**Output Contract (Updated for Insight + Quotes):**

```json
{
  "insight": {
    "insight_id": "insight:episode:abc123:a1b2c3d4",
    "text": "AI regulation will significantly lag behind the pace of innovation",
    "grounded": true,
    "confidence": 0.85,
    "episode_id": "episode:abc123"
  },
  "supporting_quotes": [
    {
      "quote_id": "quote:episode:abc123:e5f6g7h8",
      "text": "Regulation will lag innovation by 3–5 years. That's my prediction.",
      "speaker_id": "speaker:sam-altman",
      "speaker_name": "Sam Altman",
      "timestamp_start_ms": 120000,
      "timestamp_end_ms": 135000,
      "evidence": {
        "transcript_ref": "transcript.json",
        "char_start": 10234,
        "char_end": 10321
      }
    }
  ],
  "episode": {
    "episode_id": "episode:abc123",
    "title": "AI Regulation",
    "transcript_path": "output/episode_abc123/transcript.json"
  }
}
```

**Success Criteria:**

- Insights are returned with all supporting quotes
- Each quote has exact transcript span (char_start/char_end)
- Quote text matches transcript verbatim (verification possible)
- Users can navigate: insight → quote → timestamp → audio

#### UC4. Semantic Question Answering (v1-Scoped)

**User Intent:** Ask focused questions answerable via GIL structure.

**Examples:**

- "What insights are there about AI regulation?"
- "What did Sam Altman say about innovation?"
- "Which topics have the most insights?"

**GIL Traversal:**

- Deterministic mapping from question → graph traversal
- No free-form natural language generation

**Output Contract (Updated for Insight + Quotes):**

```json
{
  "question": "What did Sam Altman say about AI regulation?",
  "answer": {
    "insights": [
      {
        "insight_id": "insight:episode:abc123:a1b2c3d4",
        "text": "AI regulation will significantly lag behind the pace of innovation",
        "grounded": true,
        "supporting_quotes": [
          {
            "quote_id": "quote:episode:abc123:e5f6g7h8",
            "text": "Regulation will lag innovation by 3–5 years.",
            "timestamp_start_ms": 120000
          }
        ]
      }
    ],
    "episode_count": 3,
    "quote_count": 8
  },
  "explanation": "Found 5 grounded insights from Sam Altman about AI regulation across 3 episodes."
}
```

**Success Criteria:**

- Answers are explainable (traced to insights + quotes)
- No free-form hallucinated text
- Results include supporting quotes with timestamps

#### UC5. Insight Explorer (NEW - The Cross-Stack Feature)

**User Intent:** Query a topic and get a complete insight report with evidence.

This is the **canonical use case** that proves the Grounded Insight Layer delivers value. It exercises the full stack: Topic → Insights → Supporting Quotes → Episode → Timestamps.

**GIL Traversal:**

```
Topic → Insights (via ABOUT) → Supporting Quotes (via SUPPORTED_BY) → Speakers + Episodes
```

**Query Example:**

```bash
kg explore --topic "AI Regulation"
```

**Output Contract (The Complete Insight Report):**

```json
{
  "topic": {
    "topic_id": "topic:ai-regulation",
    "label": "AI Regulation"
  },
  "summary": {
    "insight_count": 12,
    "grounded_insight_count": 11,
    "quote_count": 28,
    "episode_count": 5,
    "speaker_count": 4
  },
  "insights": [
    {
      "insight_id": "insight:episode:abc123:a1b2c3d4",
      "text": "AI regulation will significantly lag behind the pace of innovation",
      "grounded": true,
      "confidence": 0.85,
      "episode": {
        "episode_id": "episode:abc123",
        "title": "AI Regulation",
        "publish_date": "2026-02-03T00:00:00Z"
      },
      "supporting_quotes": [
        {
          "quote_id": "quote:episode:abc123:e5f6g7h8",
          "text": "Regulation will lag innovation by 3–5 years. That's my prediction.",
          "speaker": {
            "speaker_id": "speaker:sam-altman",
            "name": "Sam Altman"
          },
          "timestamp_start_ms": 120000,
          "timestamp_end_ms": 135000,
          "evidence": {
            "transcript_ref": "transcript.json",
            "char_start": 10234,
            "char_end": 10321
          }
        },
        {
          "quote_id": "quote:episode:abc123:i9j0k1l2",
          "text": "We'll see laws that are already outdated when they pass.",
          "speaker": {
            "speaker_id": "speaker:sam-altman",
            "name": "Sam Altman"
          },
          "timestamp_start_ms": 142000,
          "timestamp_end_ms": 148000,
          "evidence": {
            "transcript_ref": "transcript.json",
            "char_start": 10890,
            "char_end": 10945
          }
        }
      ]
    }
  ],
  "top_speakers": [
    {
      "speaker_id": "speaker:sam-altman",
      "name": "Sam Altman",
      "quote_count": 12,
      "insight_count": 5
    }
  ]
}
```

**Why This Use Case Matters:**

- **Proves the Layer Works**: Exercises all GIL primitives in one query
- **Delivers User Value**: Users get insights + evidence + navigation in one call
- **Sets the Standard**: All other use cases are simplifications of this pattern

**Success Criteria:**

- Query returns insights sorted by confidence/relevance
- Each insight includes supporting quotes with timestamps
- Users can navigate from any insight to the exact transcript moment
- Grounding status is explicit (users know which insights have evidence)

### 3. Minimal UI Requirements (v1)

**Design Philosophy:**

For v1, UI is about **trust, inspection, and debugging**, not "delight". The goal is to validate insight quality, grounding rates, and evidence accuracy—not to build a polished end-user product.

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

Three minimal inspection surfaces for developers and power users (plus the Insight Explorer):

#### 3.1. Episode GIL Inspector (Non-Negotiable)

**What it is:**

- A way to view a single episode's `kg.json`
- Rendered as:
  - Insights with grounding status
  - Supporting quotes with timestamps
  - Topics linked to insights

**Why it's required:**

- Cannot debug GIL quality without seeing:
  - What insights were extracted
  - Which quotes support each insight
  - Grounding status (grounded vs ungrounded)

**Implementation bar:**

- Can be:
  - A CLI command: `kg inspect --episode <episode_id>`
  - A simple HTML page
  - A Jupyter notebook
  - **No need for React or backend**

**This answers:** "Did this episode produce quality insights with evidence?"

**Example CLI output:**

```bash
$ kg inspect --episode episode:abc123

Episode: AI Regulation (episode:abc123)
Podcast: The Journal (podcast:the-journal)

Summary:
  Insights: 5 (4 grounded, 1 ungrounded)
  Quotes: 12
  Topics: 3
  Speakers: 2

Insights:
  1. "AI regulation will significantly lag behind innovation" [GROUNDED]
     - confidence: 0.85
     - topic: AI Regulation
     - supporting quotes: 2
       → "Regulation will lag innovation by 3–5 years." (Sam Altman, 2:00)
       → "We'll see laws already outdated when they pass." (Sam Altman, 2:22)

  2. "Industry self-regulation is preferred over government mandates" [GROUNDED]
     - confidence: 0.72
     - topic: AI Regulation
     - supporting quotes: 1
       → "We need guardrails, not bans." (Sam Altman, 3:45)

  3. "European approach may become global standard" [UNGROUNDED]
     - confidence: 0.45
     - topic: AI Regulation
     - supporting quotes: 0 (no verbatim evidence found)
```

#### 3.2. Insight → Quote → Evidence Viewer (Critical)

**What it is:**

- Pick an insight
- Show:
  - Insight text and grounding status
  - Supporting quotes with timestamps
  - Highlighted transcript spans (char-based)

**Why it's required:**

- Insights are the atomic unit of user value
- If users cannot:
  - See the supporting quotes
  - Verify quotes match transcript verbatim
- Then the GIL is untrustworthy

**Implementation bar:**

- Could literally be:
  - `print_insight(insight_id)` in terminal
  - Or a notebook cell that highlights text
  - Or a simple HTML page with highlighted spans

**This answers:** "Do I trust this insight? Is it grounded?"

**Example CLI output:**

```bash
$ kg show-insight insight:episode:abc123:a1b2c3d4

Insight: "AI regulation will significantly lag behind innovation"
Status: GROUNDED (2 supporting quotes)
Confidence: 0.85
Episode: AI Regulation (episode:abc123)
Topic: AI Regulation

Supporting Quotes:

Quote 1: "Regulation will lag innovation by 3–5 years. That's my prediction."
Speaker: Sam Altman (speaker:sam-altman)
Timestamps: 120000ms - 135000ms (2:00 - 2:15)
Evidence (from transcript.json, chars 10234-10321):
─────────────────────────────────────────────────────
...and I think regulation will lag innovation by 3–5
years. That's my prediction based on how fast things
are moving. We need to be thoughtful about this...
─────────────────────────────────────────────────────

Quote 2: "We'll see laws that are already outdated when they pass."
Speaker: Sam Altman (speaker:sam-altman)
Timestamps: 142000ms - 148000ms (2:22 - 2:28)
Evidence (from transcript.json, chars 10890-10945):
─────────────────────────────────────────────────────
...and we'll see laws that are already outdated when
they pass. The technology moves faster than the
legislative process can keep up...
─────────────────────────────────────────────────────
```

#### 3.3. Insight Explorer (The One Canonical Query)

**What it is:**

- The single v1 query that proves end-to-end value
- **Recommended:** "Show me insights about a topic with supporting quotes"

**Why this query:**

- Exercises Topic → Insight → Quote → Episode → Transcript
- Touches almost every GIL primitive
- Maps directly to UC5 (Insight Explorer)
- Delivers the core user value: insights + evidence + navigation

**Implementation bar:**

- CLI:

  ```bash
  kg explore --topic "AI Regulation"
  ```

- Or notebook function:

  ```python
  get_insights_for_topic("AI Regulation")
  ```

**Output:**

- Structured JSON (for programmatic use)
- Optional pretty-print (for human inspection)

**Example CLI output:**

```bash
$ kg explore --topic "AI Regulation"

Topic: AI Regulation
Found 8 insights (7 grounded) across 5 episodes

Episodes:
  - AI Regulation (episode:abc123) - 3 insights, 8 quotes
  - The Future of AI Policy (episode:def456) - 3 insights, 6 quotes
  - Regulating Innovation (episode:ghi789) - 2 insights, 4 quotes

Speakers:
  - Sam Altman (speaker:sam-altman) - 12 quotes supporting 5 insights
  - Tim Cook (speaker:tim-cook) - 6 quotes supporting 2 insights

Top Insights:

1. "AI regulation will significantly lag behind innovation" [GROUNDED]
   confidence: 0.85 | episode: AI Regulation | 2 supporting quotes
   → "Regulation will lag innovation by 3–5 years." (Sam Altman, 2:00)
   → "We'll see laws already outdated when they pass." (Sam Altman, 2:22)

2. "Industry prefers self-regulation over government mandates" [GROUNDED]
   confidence: 0.72 | episode: AI Regulation | 1 supporting quote
   → "We need guardrails, not bans." (Sam Altman, 3:45)

3. "European approach may become global standard" [UNGROUNDED]
   confidence: 0.45 | episode: The Future of AI Policy | 0 quotes
   ⚠️ No verbatim evidence found

Use 'kg show-insight <insight_id>' to see full evidence for any insight.
```

**What a good v1 UI looks like:**

If your UI feels:

- ✅ Slightly boring
- ✅ Very explicit
- ✅ Developer-oriented
- ✅ Shows grounding status clearly

**You're doing it right.**

If it feels:

- ❌ "Polished"
- ❌ "Shareable"
- ❌ "Product-ready"

**You're too early.**

**Strong Recommendation:**

Start with:

1. A CLI-based inspector (`kg inspect`, `kg show-insight`, `kg explore`)
2. A Jupyter notebook for interactive exploration
3. **No web UI at all**

Only once:

- Ontology is stable
- Insights are high quality
- Grounding rates are acceptable (>80%)
- Quote validity is high (>95%)

…then build a web UI (or agent interface).

**How this maps to RFCs:**

- **RFC-049 (Core GIL)**: No UI required (data model only)
- **RFC-050 (Use Cases)**: Defines what must be queryable (insights + quotes)
- **UI v1**: Proves insights are grounded and evidence-backed

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

- `--show insights|quotes|all` (default: `all`)
- `--stats` (default: enabled)

**Behavior:**

- Loads `kg.json`
- Validates against schema (warn by default; fail if `--strict`)
- Prints:
  - Insight count (grounded vs ungrounded)
  - Quote count
  - Topic links
  - Insights with supporting quotes summary

**Command: `kg show-insight`**

**Goal:** Resolve an insight to its supporting quotes and transcript evidence.

**Usage:**

```bash
kg show-insight --id insight:episode:abc123:<hash> --output-dir ./output
```

**Options:**

- `--context-chars <n>` (default: 200)
- `--highlight true|false` (default: true)

**Behavior:**

- Locates the episode containing the insight
- Loads episode `kg.json` + `transcript.json`
- Prints insight text, grounding status, confidence, and all supporting quotes with evidence

**Command: `kg explore` (The Canonical Query)**

**Goal:** Run the Insight Explorer query proving end-to-end consumption.

**Canonical Query (v1):** "Show me insights about a topic with supporting quotes."

**Usage:**

```bash
kg explore --topic "AI Regulation" --output-dir ./output
```

**Options:**

- `--topic <label>` (required)
- `--limit <n>` (default: 50)
- `--min-confidence <0..1>` (default: 0.0)
- `--grounded-only` (default: false) - Only show grounded insights
- `--sort confidence|time` (default: confidence)

**Behavior:**

- Builds an in-memory logical graph by scanning all `kg.json` files
- Traverses Topic → Insight → Supporting Quotes → Episode → Speaker
- Returns insights with supporting quotes and timestamps

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

GIL data is consumed alongside existing artifacts:

```
output/
  episode_<id>/
    metadata.json
    transcript.json
    summary.json
    kg.json          # NEW: Grounded Insight Layer data
```

**Consumption Patterns:**

1. **Direct kg.json Access**: Read `kg.json` directly for programmatic access
2. **Join with Summary**: Combine GIL insights with `summary.json` for narrative context
3. **Resolve Transcript Spans**: Use `transcript.json` to verify quote evidence
4. **Metadata Integration**: Use `metadata.json` for episode-level context
5. **Fast Queries via DB**: Use RFC-051 Postgres projection for cross-episode queries

**Example Consumption Code (Insight Explorer):**

```python
import json
from pathlib import Path

def explore_topic(topic_label: str, output_dir: Path) -> dict:
    """Get insights about a topic with supporting quotes."""
    results = {
        "topic": topic_label,
        "insights": [],
        "episodes": [],
        "speakers": set()
    }

    # Scan all episode directories
    for episode_dir in output_dir.glob("episode_*"):
        kg_path = episode_dir / "kg.json"
        if not kg_path.exists():
            continue

        with open(kg_path) as f:
            kg_data = json.load(f)

        # Find insights about this topic
        for node in kg_data["nodes"]:
            if node["type"] == "Insight":
                # Check if insight is about this topic via ABOUT edge
                for edge in kg_data["edges"]:
                    if (edge["type"] == "ABOUT" and
                        edge["from"] == node["id"]):
                        # Resolve topic and check label
                        topic_node = _find_node(kg_data, edge["to"])
                        if topic_node and topic_node["properties"]["label"] == topic_label:
                            # Found relevant insight - get supporting quotes
                            supporting_quotes = _get_supporting_quotes(kg_data, node["id"])
                            results["insights"].append({
                                "insight": node,
                                "supporting_quotes": supporting_quotes
                            })

    return results
```

### 5. Output Shapes (Illustrative)

**Insight with Supporting Quotes (Core Pattern):**

```json
{
  "insight": {
    "insight_id": "insight:episode:abc123:a1b2c3d4",
    "text": "AI regulation will significantly lag behind innovation",
    "grounded": true,
    "confidence": 0.85
  },
  "supporting_quotes": [
    {
      "quote_id": "quote:episode:abc123:e5f6g7h8",
      "text": "Regulation will lag innovation by 3–5 years.",
      "speaker_name": "Sam Altman",
      "timestamp_start_ms": 120000,
      "timestamp_end_ms": 135000
    }
  ],
  "episode": {
    "episode_id": "episode:abc123",
    "title": "AI Regulation"
  }
}
```

**Topic Exploration Result:**

```json
{
  "topic": "AI Regulation",
  "summary": {
    "insight_count": 8,
    "grounded_count": 7,
    "quote_count": 18,
    "episode_count": 5
  },
  "insights": [
    {
      "insight_id": "...",
      "text": "...",
      "grounded": true,
      "supporting_quotes": [...]
    }
  ]
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
      "insight_count": 5,
      "quote_count": 12
    }
  ],
  "episode_count": 3
}
```

### 6. End-to-End Success Definition (v1)

The GIL implementation is considered end-to-end successful when:

- ✅ All v1 use cases (UC1–UC5) can be executed using GIL data
- ✅ Outputs include insights with supporting quotes (not just claims)
- ✅ Grounding status is explicit for every insight
- ✅ Quote text matches transcript verbatim (verifiable)
- ✅ GIL data integrates cleanly with existing scraper artifacts
- ✅ The Insight Explorer query (UC5) works end-to-end
- ✅ Generated `kg.json` files conform to schema
- ✅ Query patterns are documented and reproducible

### 7. Failure Modes & Fallbacks

- **No insights found**: Return topic/episode links only (degraded but useful)
- **Ungrounded insights**: Mark explicitly with `grounded=false`, do not suppress (transparency)
- **Quote doesn't match transcript**: Log warning, mark quote as potentially invalid
- **Low confidence extractions**: Mark explicitly, do not suppress (transparency)
- **Incomplete GIL**: Fall back to summaries (graceful degradation)
- **Schema validation failure**: Log error, skip episode (non-fatal)

## Key Decisions

1. **Insights + Quotes, Not Just Graph Traversal**
   - **Decision**: Use cases return insights with supporting quotes, not just claims
   - **Rationale**: User value is "trust + navigation"; graph is internal plumbing

2. **Evidence-Backed by Default**
   - **Decision**: All answers include supporting quotes with timestamps
   - **Rationale**: Enables trust, verification, and navigation to source material

3. **Insight Explorer as Proof of Value**
   - **Decision**: UC5 is the canonical query that proves the system works
   - **Rationale**: Exercises full stack, delivers core user value in one query

4. **Structured First, NL Second**
   - **Decision**: v1 prioritizes structured consumption over natural language interfaces
   - **Rationale**: Reduces complexity, enables programmatic access, provides clear contracts

5. **Co-Located Consumption**
   - **Decision**: GIL data consumed alongside existing artifacts
   - **Rationale**: Maintains existing patterns, no separate infrastructure

## Alternatives Considered

1. **Claims-Only Output (No Quote Nodes)**
   - **Description**: Return claims without separate Quote nodes
   - **Pros**: Simpler structure, fewer nodes
   - **Cons**: Evidence is metadata, not first-class; harder to verify
   - **Why Rejected**: Quote nodes enable trust, verification, and quality metrics

2. **Natural Language Query Interface**
   - **Description**: Build NL → graph query translation in v1
   - **Pros**: More user-friendly, lower barrier to entry
   - **Cons**: Complex, error-prone, harder to validate
   - **Why Rejected**: Deferred to post-v1, structured queries are sufficient

3. **Separate Query API**
   - **Description**: Build REST API or GraphQL endpoint for queries
   - **Pros**: Standardized interface, easier for external consumers
   - **Cons**: Requires separate service, adds infrastructure
   - **Why Rejected**: File-based access + RFC-051 DB is simpler and sufficient in v1

4. **Global Graph Index**
   - **Description**: Build global index for faster queries
   - **Pros**: Faster queries, better performance
   - **Cons**: Requires global storage, adds complexity
   - **Why Rejected**: RFC-051 Postgres projection handles this need

## Testing Strategy

**Test Coverage:**

- **Unit Tests**: Test query functions, output shape validation, evidence resolution
- **Integration Tests**: Test end-to-end use cases with real `kg.json` files
- **E2E Tests**: Test full workflow from transcript → GIL → query → results
- **Grounding Tests**: Verify insights have supporting quotes; verify quote verbatim match

**Test Organization:**

- Unit tests: `tests/unit/test_kg_queries.py`
- Integration tests: `tests/integration/test_kg_use_cases.py`
- E2E tests: `tests/e2e/test_kg_e2e.py`

**Test Execution:**

- Run in CI as part of standard test suite
- Use existing test fixtures (transcripts, episodes)
- Validate output contracts include insights with supporting_quotes

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1**: Implement UC1 (Topic Research) with insights + quotes output
- **Phase 2**: Add UC2 (Speaker Mapping) and UC3 (Evidence-Backed Retrieval)
- **Phase 3**: Add UC4 (Semantic QA) with deterministic question mapping
- **Phase 4**: Implement UC5 (Insight Explorer) - the cross-stack proof of value
- **Phase 5**: Documentation and example code

**Monitoring:**

- Track query success rate (queries that return insights with quotes)
- Monitor grounding rate (% insights with supporting quotes)
- Monitor quote validity rate (% quotes that match transcript verbatim)
- Track query performance (time to scan per-episode files or query DB)

**Success Criteria:**

1. ✅ All v1 use cases (UC1–UC5) can be executed end-to-end
2. ✅ Output contracts include insights with supporting_quotes
3. ✅ Grounding status is explicit for every insight
4. ✅ Quote evidence resolution works correctly (verbatim match)
5. ✅ Insight Explorer (UC5) demonstrates full cross-stack value
6. ✅ Integration with existing outputs verified
7. ✅ Query patterns documented with examples

## Relationship to Other RFCs

This RFC (RFC-050) is part of the Grounded Insight Layer initiative that includes:

1. **RFC-049: Core GIL Concepts & Data Model** - Defines ontology, grounding contract, and storage
2. **RFC-051: Database Projection** - Defines Postgres export for fast queries
3. **PRD-017: Grounded Insight Layer** - Defines product requirements and user value

**Key Distinction:**

- **RFC-049**: Focuses on *how* knowledge is extracted and stored (ontology, grounding)
- **RFC-050 (This RFC)**: Focuses on *how* knowledge is consumed (use cases, output shapes)
- **RFC-051**: Focuses on *how* knowledge is served at scale (database projection)

Together, these RFCs provide:

- Complete technical design for Grounded Insight Layer implementation
- Clear separation between ontology (RFC-049), consumption (RFC-050), and serving (RFC-051)
- Foundation for trustworthy downstream applications (RAG, agents, analytics)

## Benefits

1. **User-Centric Output**: Insights with supporting quotes, not just graph data
2. **Evidence-Backed**: All answers include verbatim quotes with timestamps
3. **Explicit Grounding**: Users know which insights have evidence
4. **Cross-Stack Proof**: Insight Explorer demonstrates full system value
5. **Integration**: Works seamlessly with existing outputs and RFC-051 DB

## Migration Path

N/A - This is a new feature, not a migration from an existing system.

## Open Questions

1. **Query Performance**: How many episodes can be scanned before performance degrades?
   - **Current Decision**: Acceptable for v1; RFC-051 DB handles scale
   - **Open**: Performance thresholds that trigger DB-only queries

2. **Natural Language Queries**: When should NL → graph translation be added?
   - **Current Decision**: Deferred to post-v1
   - **Open**: User feedback and demand

3. **Grounding Rate Thresholds**: What grounding rate is acceptable for v1?
   - **Current Decision**: >80% grounded insights is acceptable
   - **Open**: How to improve grounding for difficult transcripts

## References

- **Related PRD**: `docs/prd/PRD-017-grounded-insight-layer.md`
- **Related RFC**: `docs/rfc/RFC-049-grounded-insight-layer-core.md`
- **Related RFC**: `docs/rfc/RFC-051-grounded-insight-layer-database-projection.md`
- **Ontology Specification**: `docs/kg/ontology.md`
- **Schema Specification**: `docs/kg/kg.schema.json`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Source Code**: `podcast_scraper/workflow/` (integration points)
