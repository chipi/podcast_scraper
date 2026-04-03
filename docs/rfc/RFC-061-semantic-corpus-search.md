# RFC-061: Semantic Corpus Search

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, GIL/KG consumers, downstream API/digest users
- **Related PRDs**:
  - `docs/prd/PRD-021-semantic-corpus-search.md` (product requirements)
  - `docs/prd/PRD-017-grounded-insight-layer.md` (GIL — primary indexed content)
  - `docs/prd/PRD-019-knowledge-graph-layer.md` (KG — secondary indexed content)
- **Related RFCs**:
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` (GIL artifacts)
  - `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md` (UC4/UC5 — semantic QA, Insight Explorer)
  - `docs/rfc/RFC-051-database-projection-gil-kg.md` (complementary SQL serving)
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md` (KG artifacts)
  - `docs/rfc/RFC-056-knowledge-graph-layer-use-cases.md` (KG use cases — entity/topic roll-ups)
- **Related Documents**:
  - [GitHub #466](https://github.com/chipi/podcast_scraper/issues/466) — GI + KG depth roadmap
  - `docs/architecture/gi/ontology.md` — GIL ontology (node types, text fields)

## Abstract

This RFC defines the technical design for **Semantic Corpus Search**: a vector index over
GIL insights, quotes, summary bullets, and transcript chunks that enables meaning-based
retrieval across the podcast corpus. The design introduces a `VectorStore` protocol with
a FAISS implementation for CLI/local use (Phase 1) and a Qdrant implementation for
platform/service mode (Phase 2). Search results preserve full GIL provenance — grounding
status, supporting quotes, timestamps, and transcript references.

**Architecture Alignment:** This feature is purely additive. It does not change existing
artifacts (`gi.json`, `kg.json`, summaries, transcripts), pipeline stages, or CLI commands.
It adds a new optional pipeline stage (embed-and-index), a new CLI command (`podcast search`),
and transparently upgrades `gi explore` / `gi query` to use semantic matching when an index
is available.

## Problem Statement

GIL and KG produce rich, structured artifacts per episode, but consumption is limited to
**exact-match and substring filtering**:

- `gi explore --topic "AI Regulation"` does `key in insight_text.lower()` — misses
  "Government AI Policy," "tech oversight," "regulatory impact"
- `gi query` maps regex patterns to the same substring path — not semantic
- `kg entities` / `kg topics` match by exact string — "Elon Musk" and "Musk" are separate
- No cross-corpus question like "what do my podcasts say about X?"
- RFC-050 explicitly defers UC4 (Semantic QA) as "post-v1, after Insight Explorer validated"
- `gi explore` hits a ~100 episode performance ceiling (file scan)

The project already loads `sentence-transformers` (`all-MiniLM-L6-v2`) and has
`embedding_loader.py` with `encode()` and `cosine_similarity()` — but these are only used
for GIL grounding and eval metrics, not user-facing search.

**Use Cases:**

1. **Cross-Corpus Semantic Search**: "What do my podcasts say about supply chain disruptions?"
   — finds insights about logistics, shipping delays, port congestion across feeds
2. **Transcript Deep-Dive**: "Where was quantum computing discussed?" — returns timestamped
   chunks even if the speaker said "qubits" or "quantum advantage"
3. **Evidence-Backed Discovery**: Search returns GIL Insight nodes with their full provenance
   chain (Insight → Quotes → transcript spans → timestamps) — not generated text
4. **Semantic `gi explore` Upgrade**: `gi explore --topic "climate"` matches insights about
   "global warming," "carbon emissions," "net zero" without explicit topic labels
5. **Digest Clustering**: Weekly digest groups similar Insight embeddings to find themes
   and deduplicate cross-feed coverage

## Goals

1. **`VectorStore` protocol**: Clean abstraction supporting FAISS (Phase 1) and Qdrant
   (Phase 2) behind the same interface
2. **Embed-and-index pipeline stage**: Produces and maintains a vector index as part of
   the pipeline, incremental by default
3. **`podcast search` CLI**: Meaning-based corpus queries with filtering and structured output
4. **Transparent `gi explore` upgrade**: Semantic matching when index available, substring
   fallback when not
5. **Reuse existing infrastructure**: Same embedding model, same `embedding_loader.py`,
   same output directory conventions

## Constraints & Assumptions

**Constraints:**

- CLI-first: no server process required (FAISS in-process for Phase 1)
- Must not break existing behavior when `vector_search` is disabled (default: `false`)
- Pipeline runtime increase < 30% when indexing is enabled
- Index files live alongside corpus outputs (no external database for Phase 1)
- Must work with the existing `all-MiniLM-L6-v2` model (384-dim)

**Assumptions:**

- Corpus scale: 10-50 feeds, up to ~5,000 episodes (~1.2M vectors with transcript chunks)
- GIL and/or summary artifacts exist before search is enabled
- English-only content

## Design & Implementation

### 1. VectorStore Protocol

A minimal protocol that both FAISS and Qdrant backends implement:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class SearchResult:
    doc_id: str
    score: float
    metadata: dict


@dataclass
class IndexStats:
    total_vectors: int
    doc_type_counts: dict[str, int]
    feeds_indexed: list[str]
    embedding_model: str
    embedding_dim: int
    last_updated: str
    index_size_bytes: int


class VectorStore(Protocol):
    def upsert(
        self, doc_id: str, embedding: list[float], metadata: dict
    ) -> None: ...

    def batch_upsert(
        self, doc_ids: list[str], embeddings: list[list[float]],
        metadata_list: list[dict]
    ) -> None: ...

    def search(
        self, query_embedding: list[float], top_k: int = 10,
        filters: dict | None = None
    ) -> list[SearchResult]: ...

    def delete(self, doc_ids: list[str]) -> None: ...

    def persist(self) -> None: ...

    def stats(self) -> IndexStats: ...
```

**Key design decisions:**

- `doc_id` is a string like `insight:<episode_id>:<hash>` or `chunk:<episode_id>:<index>`
  — stable, deterministic, aligns with GIL/KG ID conventions
- `metadata` is a flat dict with known keys (`doc_type`, `episode_id`, `feed_id`,
  `publish_date`, `speaker_id`, `grounded`, `char_start`, `char_end`,
  `timestamp_start_ms`, `source_id`)
- `filters` is a dict of metadata field → value (or list of values) for pre-/post-filtering
- `batch_upsert` for efficient bulk indexing (FAISS benefits from batched adds)

### 2. FaissVectorStore Implementation

```text
src/podcast_scraper/search/
    __init__.py
    protocol.py         # VectorStore protocol + SearchResult + IndexStats
    faiss_store.py       # FaissVectorStore implementation
    chunker.py           # Transcript chunking
    indexer.py           # Embed-and-index pipeline logic
```

**Index structure on disk:**

```text
<output_dir>/search/
    vectors.faiss        # FAISS IndexIDMap wrapping IndexFlatIP (or IVF-PQ at scale)
    metadata.json        # doc_id → metadata mapping (or .sqlite for large corpora)
    index_meta.json      # embedding_model, dim, created_at, last_updated, version
```

**FAISS index type selection:**

| Corpus size | Index type | Notes |
| ----------- | ---------- | ----- |
| < 100K vectors | `IndexFlatIP` wrapped in `IndexIDMap` | Exact search; fast enough |
| 100K - 1M vectors | `IndexIVFFlat` (nlist=256) + `IndexIDMap` | ~10x faster, slight accuracy loss |
| > 1M vectors | `IndexIVFPQ` (nlist=1024, m=48) | Compressed; ~20x faster |

Auto-selection based on vector count at persist time. Rebuild threshold configurable.

**Metadata filtering (FAISS):**

FAISS has no built-in metadata filtering. Strategy:

1. FAISS search returns top `k * 3` candidates (over-fetch)
2. Post-filter by metadata predicates (type, feed, date, speaker, grounded)
3. Return top `k` from filtered set
4. If fewer than `k` results after filtering, warn user

This is simple and sufficient for CLI-scale corpora. Qdrant Phase 2 replaces this with
native payload filtering.

### 3. Transcript Chunker

```python
@dataclass
class TranscriptChunk:
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    timestamp_start_ms: int | None
    timestamp_end_ms: int | None


def chunk_transcript(
    text: str,
    target_tokens: int = 300,
    overlap_tokens: int = 50,
    timestamps: list[dict] | None = None,
) -> list[TranscriptChunk]:
    """Split transcript into overlapping sentence-boundary chunks."""
    ...
```

**Strategy:**

1. Split text into sentences (simple regex: `(?<=[.!?])\s+` with fallback on `\n`)
2. Group sentences into chunks targeting `target_tokens` (estimated via whitespace split)
3. Overlap: carry last ~`overlap_tokens` worth of sentences from previous chunk
4. Track `char_start` / `char_end` per chunk
5. If `timestamps` provided (from Whisper segments), interpolate `timestamp_start_ms` /
   `timestamp_end_ms` per chunk based on character position alignment

**Why sentence boundaries:** Avoids splitting mid-sentence, which degrades embedding quality.
Simpler and more predictable than recursive splitting. No external dependency.

### 4. Embed-and-Index Pipeline Stage

**When it runs:**

```text
Existing:  RSS → download → transcribe → metadata → summarize → GIL → (KG)
New stage:                                                             → embed & index
```

Runs after GIL and KG (if enabled), or after summarization if GIL is not enabled.
Trigger: `vector_search: true` in config.

**What gets indexed:**

| Document type | Source | doc_id pattern | Requires |
| ------------- | ------ | -------------- | -------- |
| Insight | `gi.json` → `insight.properties.text` | `insight:<episode_id>:<hash>` | `generate_gi: true` |
| Quote | `gi.json` → `quote.properties.text` | `quote:<episode_id>:<hash>` | `generate_gi: true` |
| Summary bullet | `SummarySchema.bullets[i]` | `bullet:<episode_id>:<i>` | `generate_summaries: true` |
| Transcript chunk | Transcript file → chunked | `chunk:<episode_id>:<i>` | Transcript on disk |

**Incremental logic:**

1. Load `index_meta.json` → get set of already-indexed episode IDs + content hashes
2. For each episode in the output directory:
   a. Compute content hash (SHA-256 of `gi.json` + summary + transcript paths)
   b. If hash matches → skip (already indexed)
   c. If new or changed → delete old vectors for that episode, embed new content, upsert
3. Persist updated index

**Embedding:** Uses `embedding_loader.encode()` with the configured model (default
`all-MiniLM-L6-v2`). Batch encoding for efficiency (~14K sentences/sec on GPU).

### 5. Search CLI Command

```bash
podcast search "<query>" [options]
```

**Options:**

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `--type` | `insight\|quote\|summary\|transcript` | all | Filter by document type |
| `--feed` | string | all | Filter by feed name |
| `--since` | date | none | Filter by publish date |
| `--speaker` | string | none | Filter quotes/insights by speaker |
| `--grounded-only` | flag | false | Only grounded insights |
| `--top-k` | int | 10 | Number of results |
| `--format` | `json\|pretty` | pretty | Output format |
| `--index-path` | path | `<output_dir>/search/` | Index location |

**Implementation flow:**

1. Load `FaissVectorStore` from `--index-path`
2. Encode query using same embedding model
3. Search with over-fetch → post-filter by metadata → return top-k
4. For each result, resolve full context:
   - Insight → load `gi.json`, resolve supporting quotes
   - Quote → load `gi.json`, resolve parent insight and speaker
   - Summary bullet → load metadata, resolve episode
   - Transcript chunk → resolve episode and timestamps
5. Format and output

### 6. Index Management CLI

```bash
podcast index --rebuild [--output-dir ./output]
podcast index --stats [--output-dir ./output]
```

- `--rebuild`: Delete existing index, re-index all episodes from scratch
- `--stats`: Print `IndexStats` (vector count, type breakdown, feeds, model, size)

### 7. Enhanced `gi explore` and `gi query`

**`gi explore` upgrade:**

In `explore.py`, modify `_insight_matches_topic()`:

```python
def _insight_matches_topic(
    artifact, insight_id, insight_text, topic, vector_store=None
):
    if not topic or not topic.strip():
        return True
    if vector_store is not None:
        return _semantic_topic_match(insight_id, topic, vector_store)
    # Existing substring fallback
    key = topic.strip().lower()
    ...
```

When `gi explore` starts, check if a vector index exists at the default path. If yes,
load it and pass to matching functions. If no, use existing substring logic. Zero behavior
change when no index is present.

**`gi query` upgrade:**

`run_uc4_semantic_qa` currently maps patterns to `run_uc5_insight_explorer`. With a
vector index, it can:

1. Encode the user's question as a vector
2. Search for top-K matching insights
3. Return the same `ExploreOutput` contract — just better results

### 8. Configuration

New fields in `Config` (all optional, search disabled by default):

```python
vector_search: bool = False
vector_backend: Literal["faiss", "qdrant"] = "faiss"
vector_index_path: Optional[str] = None  # auto: <output_dir>/search/
vector_index_types: list[str] = [
    "insights", "quotes", "summary_bullets", "transcript_chunks"
]
vector_chunk_size_tokens: int = 300
vector_chunk_overlap_tokens: int = 50
```

CLI flags: `--vector-search`, `--vector-backend`, `--vector-index-path`,
`--vector-chunk-size`, `--vector-chunk-overlap`.

## Key Decisions

1. **FAISS for Phase 1, Qdrant for Phase 2**
   - **Decision**: Ship with FAISS; add Qdrant when platform mode ships
   - **Rationale**: FAISS is in-process, zero server overhead, sufficient for CLI-scale
     corpora. Qdrant adds native filtering and upserts but requires Docker for production.
     The `VectorStore` protocol makes the switch transparent.

2. **Global corpus index, not per-feed**
   - **Decision**: One index for the entire corpus with feed metadata for filtering
   - **Rationale**: Cross-feed discovery is a primary use case. Per-feed would require
     multi-index coordination.

3. **Post-filter metadata (FAISS Phase 1)**
   - **Decision**: Over-fetch from FAISS, post-filter by metadata
   - **Rationale**: Simple, no external dependency. Sufficient for < 1M vectors.
     Qdrant Phase 2 replaces with native payload filtering.

4. **Sentence-boundary chunking**
   - **Decision**: Sentence-boundary windows, not fixed-token or paragraph-based
   - **Rationale**: Preserves semantic coherence; predictable chunk sizes; no external
     dependency (simple regex splitter).

5. **Transparent `gi explore` upgrade**
   - **Decision**: Detect index existence; use semantic matching when available; substring
     fallback when not
   - **Rationale**: Zero breaking change. Users who don't enable search get identical
     behavior. Users who do get better results from the same command.

6. **`faiss-cpu` only (no GPU variant in default deps)**
   - **Decision**: `faiss-cpu` in `[project.dependencies]`; `faiss-gpu` as optional
   - **Rationale**: CPU is sufficient for CLI-scale corpora. GPU variant has CUDA
     dependency that complicates installation.

## Alternatives Considered

1. **Qdrant-only (skip FAISS)**
   - **Description**: Use Qdrant local mode for Phase 1
   - **Pros**: Built-in filtering; native upserts; same API for local and server
   - **Cons**: Heavier dependency; local mode is "for small-scale / demos" per docs;
     adds Rust binary to the Python package
   - **Why Rejected**: FAISS is lighter, battle-tested, and sufficient for Phase 1

2. **ChromaDB**
   - **Description**: Use Chroma as an all-in-one embedded vector DB
   - **Pros**: Simple API; built-in metadata filtering; embedded mode
   - **Cons**: Heavier than FAISS; SQLite-based storage adds fragility; less mature
     at scale; another dependency to maintain
   - **Why Rejected**: FAISS is more predictable and lighter; Qdrant is better for
     production scale

3. **Postgres pgvector (via RFC-051)**
   - **Description**: Add vector columns to the RFC-051 Postgres projection
   - **Pros**: Single database for structured + vector queries; native SQL filtering
   - **Cons**: Requires Postgres server (violates CLI-first); pgvector performance
     lags FAISS/Qdrant at scale; couples search to database projection
   - **Why Rejected**: Good for Phase 3 (platform) but not for CLI Phase 1

4. **No abstraction (FAISS directly)**
   - **Description**: Call FAISS API directly without `VectorStore` protocol
   - **Pros**: Less code; simpler
   - **Cons**: Locks in FAISS; no migration path to Qdrant/platform
   - **Why Rejected**: The protocol is tiny (~20 lines) and enables clean Phase 2

## Testing Strategy

**Test Coverage:**

- **Unit tests**: `VectorStore` protocol, `FaissVectorStore` CRUD, chunker, metadata
  sidecar, search with filters, incremental indexing logic
- **Integration tests**: Full round-trip: embed sample artifacts → build index → search →
  verify results include correct GIL provenance
- **E2E tests**: Pipeline run with `vector_search: true` → `podcast search` → verify
  results; `gi explore` with and without index

**Test Organization:**

- `tests/unit/podcast_scraper/search/` — unit tests for search module
- `tests/integration/test_search_integration.py` — index + query round-trip
- `tests/e2e/test_search_cli_e2e.py` — CLI end-to-end

**Test Execution:**

- Unit + integration: `make ci-fast`
- E2E: `make ci` (full suite)
- Fixtures: small `gi.json` + transcript + summary fixtures for deterministic tests
- Mock `embedding_loader.encode()` in unit tests (return fixed vectors)

## Rollout & Monitoring

**Rollout Plan (Option A — post-hardening slice):**

- **Step 1**: `VectorStore` protocol + `FaissVectorStore` + unit tests — **done**
- **Step 2**: Transcript chunker + unit tests — **done**
- **Step 3**: Embed-and-index pipeline stage + unit tests (`test_indexer.py`); integration-style coverage in search unit tests — **done**
- **Step 4**: `podcast search` CLI + `podcast index` CLI + E2E tests — **done**
- **Step 5**: Config fields + YAML support (`config.py`, `config/examples/config.example.yaml`) — **done**
- **Step 6**: `gi explore` semantic upgrade (transparent: `<output_dir>/search/vectors.faiss` + `--topic`) — **done**
- **Step 7**: Documentation update (README, Development Guide, `docs/guides/SEMANTIC_SEARCH_GUIDE.md`, MkDocs nav) — **done**

**Phase 2 (platform, separate RFC/issue):**

- `QdrantVectorStore` implementation
- Service-mode API endpoint
- Digest clustering integration

**Monitoring:**

- Index build time per episode (logged as `vector_index_sec` in JSONL metrics)
- Search latency (logged per query)
- Index size on disk (reported by `podcast index --stats`)

**Success Criteria:**

1. `podcast search` returns semantically relevant results for paraphrased queries
2. `gi explore --topic` produces better results with index than without (manual eval)
3. Zero regression when `vector_search: false` (default)
4. `make ci-fast` passes with search module included
5. Index build + search round-trip works in integration tests

## Relationship to Other RFCs

This RFC (RFC-061) is part of the GIL/KG **depth** initiative ([#466](https://github.com/chipi/podcast_scraper/issues/466)):

```text
RFC-049 (GIL Core)              → artifacts to index
RFC-050 (GIL Use Cases)         → UC4/UC5 that search enables
    ↓
RFC-061 (this RFC)              → semantic search over corpus
    ↓
RFC-051 (DB Projection)         → complementary structured serving
Platform megasketch              → digest, API, multi-tenant search
```

**Key Distinction:**

- **RFC-049/050**: Define *what* GIL extracts and *how* it's consumed (structured)
- **RFC-061**: Adds *meaning-based discovery* over GIL + KG + summary + transcript content
- **RFC-051**: Adds *SQL-based serving* for structured queries (complementary, not competing)

Together, semantic search (RFC-061) and database projection (RFC-051) provide two
complementary query paths: "find by meaning" (vectors) and "filter by structure" (SQL).

## Benefits

1. **Unlocks UC4 (Semantic QA)**: The explicitly deferred RFC-050 use case becomes functional
2. **Removes scale ceiling**: `gi explore` goes from ~100 episode file scan to vector index
3. **Cross-feed discovery**: "What do all my podcasts say about X?" becomes answerable
4. **Preserves GIL provenance**: Search results carry grounding, quotes, timestamps — not
   hallucinated text
5. **Minimal new dependencies**: `faiss-cpu` (~20 MB); everything else already in the tree
6. **Foundation for platform features**: Digest clustering, recommendations, and API search
   all build on the same index

## Migration Path

N/A — this is a new additive feature. Existing behavior is unchanged when
`vector_search: false` (default). No artifacts, schemas, or CLI commands are modified.

## Open Questions

1. **Metadata sidecar format**: JSON file vs SQLite for the FAISS metadata mapping?
   JSON is simpler; SQLite handles larger corpora better. Recommendation: start with JSON,
   add SQLite option when corpora exceed ~50K vectors.
2. **Exact dedup**: Should the index deduplicate near-identical insights across episodes?
   (Recommend: defer to digest layer; index stores all, clustering deduplicates at query time.)
3. **Re-ranking**: Should search results be re-ranked with a cross-encoder for higher
   precision? (Recommend: defer to Phase 2; bi-encoder retrieval is sufficient for v1.)

## References

- **Related PRD**: `docs/prd/PRD-021-semantic-corpus-search.md`
- **Related RFC**: `docs/rfc/RFC-049-grounded-insight-layer-core.md`
- **Related RFC**: `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md`
- **Related RFC**: `docs/rfc/RFC-051-database-projection-gil-kg.md`

- **Source Code**: `podcast_scraper/providers/ml/embedding_loader.py`
- **Source Code**: `podcast_scraper/gi/explore.py`
