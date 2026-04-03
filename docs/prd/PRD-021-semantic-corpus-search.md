# PRD-021: Semantic Corpus Search

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Related RFCs**:
  - RFC-061 (Semantic Corpus Search — technical design)
  - RFC-049 (GIL Core — prerequisite, provides indexable artifacts)
  - RFC-050 (GIL Use Cases — prerequisite, defines UC4/UC5 that this feature unlocks)
  - RFC-051 (Database Projection — complementary serving layer)
  - RFC-055/056 (KG Core / Use Cases — KG artifacts are also indexable)
- **Related PRDs**:
  - [PRD-017: Grounded Insight Layer](PRD-017-grounded-insight-layer.md) (GIL artifacts are the primary search corpus)
  - [PRD-019: Knowledge Graph Layer](PRD-019-knowledge-graph-layer.md) (KG topics/entities benefit from semantic matching)
  - [PRD-018: Database Projection](PRD-018-database-projection-gil-kg.md) (complementary — SQL for structured, vectors for semantic)
- **Related Documents**:
  - [GitHub #466](https://github.com/chipi/podcast_scraper/issues/466) — GI + KG depth roadmap
  - `docs/wip/platform-corpus-service-megasketch.md` — Platform context

## Summary

**Semantic Corpus Search** adds meaning-based retrieval over the podcast corpus — insights,
quotes, summaries, and transcript chunks — using sentence embeddings and a vector index.
Users query in natural language and get ranked results with full GIL provenance (supporting
quotes, timestamps, grounding status). This transforms GIL and KG from "write-heavy,
read-weak" artifact stores into a navigable, question-driven knowledge layer.

## Background & Context

The podcast scraper pipeline produces rich structured artifacts: GIL insights with grounded
quotes (`gi.json`), KG entities and topics (`kg.json`), summaries with bullets, and full
transcripts. However, **consumption is limited to exact-match filtering**:

- `gi explore --topic "AI Regulation"` uses **substring matching** on insight text — "Government
  AI Policy" or "tech oversight" won't match
- `gi query` maps fixed English patterns to the same explore path — not semantic
- `kg entities` and `kg topics` roll up by exact string — "Elon Musk" vs "Musk" are separate
- No way to ask "what do my podcasts say about X?" across the corpus

The project already loads `sentence-transformers` (`all-MiniLM-L6-v2`) for GIL grounding
evidence (QA + NLI), but this capability is not exposed for user-facing search. The
megasketch (Part A.7) explicitly defers "optional search later."

**Why now:** Shallow v1 (GIL + KG) is hardening. The artifacts are stable, the CLI commands
exist, and the evidence stack works. Semantic search is the highest-leverage "depth" feature
because it:

- Immediately makes `gi explore` and `gi query` useful at scale (removes the ~100 episode ceiling)
- Unlocks RFC-050 UC4 (Semantic QA) which is explicitly deferred as "post-v1"
- Provides infrastructure for digest clustering (megasketch Part C), topic alignment, and
  entity resolution
- Reuses the embedding model already in the dependency tree

## Goals

1. **Meaning-based retrieval**: Users query in natural language and find relevant
   insights/quotes/summaries/transcript moments regardless of exact wording
2. **Evidence preservation**: Search results carry full GIL provenance — grounding status,
   supporting quotes, timestamps, transcript spans
3. **CLI-first**: Works as a CLI command with no server process, consistent with the project's
   "CLI stays first-class" constraint
4. **Incremental indexing**: New episodes are added to the index without full rebuild;
   existing indexes grow as the corpus grows
5. **Cross-feed discovery**: Find related content across different podcast feeds
6. **Abstracted backend**: Vector store protocol supports FAISS (CLI/v1) and Qdrant
   (platform/service mode) behind the same interface

## Non-Goals

- Full-text keyword search (BM25) or hybrid keyword+vector search (defer to later)
- Web UI or REST API for search (platform mode, not this PRD)
- RAG-style answer generation (results are existing artifacts, not generated text)
- Entity resolution or disambiguation (separate concern; embeddings help but don't solve)
- Real-time / streaming index updates (batch after pipeline run)
- Replacing `gi explore` or `kg entities` — search enhances them, not replaces
- Multi-language support (English-only, matching current pipeline)

## Personas

- **Knowledge Worker / Researcher**: Subscribes to 10-50 podcasts. Wants to ask "what have
  my shows said about X?" and get evidence-backed answers without scanning episode by episode.
  - Needs: meaning-based queries, ranked results, links to source material
  - Value: saves hours of manual searching; discovers connections across feeds

- **Developer Building on the Corpus**: Uses GIL/KG artifacts programmatically. Wants
  structured search results as JSON for downstream pipelines (RAG, agents, analytics).
  - Needs: CLI + JSON output, `SearchResult` contract, filter by type/feed/date
  - Value: enables evidence-backed RAG without hallucination

- **Power User / Operator**: Manages a large corpus. Wants to understand what the corpus
  covers, find gaps, validate quality across feeds.
  - Needs: index stats, cross-feed topic exploration, temporal filtering
  - Value: corpus intelligence without manual file scanning

## User Stories

- _As a researcher, I can search "AI regulation impact on startups" and get ranked insights
  with supporting quotes from across my podcast library so that I find relevant content by
  meaning, not exact keywords._
- _As a developer, I can query the vector index programmatically and get structured
  `SearchResult` JSON with `insight_id`, `episode_id`, `grounding_status`, and
  `supporting_quotes` so that I can build evidence-backed applications._
- _As a power user, I can filter search by feed, date range, speaker, and document type
  (insight/quote/summary/transcript) so that I narrow results to what matters._
- _As an operator, I can run `podcast index --stats` to see how many vectors are indexed,
  which feeds are covered, and when the index was last updated so that I know my corpus
  is searchable._
- _As a researcher, I can search "where was quantum computing discussed?" and get
  transcript chunks with timestamps so that I can jump to the exact moment in the audio._

## Functional Requirements

### FR1: Vector Store Abstraction

- **FR1.1**: Define a `VectorStore` protocol with `upsert()`, `batch_upsert()`, `search()`,
  `delete()`, `persist()`, and `stats()` operations
- **FR1.2**: Implement `FaissVectorStore` using `faiss-cpu` for CLI/local use (Phase 1)
- **FR1.3**: Index persisted as files on disk (`vectors.faiss` + metadata sidecar)
- **FR1.4**: Metadata sidecar stores document type, episode_id, feed_id, publish_date,
  speaker_id, grounding status, and source references per vector

### FR2: Embedding and Indexing Pipeline Stage

- **FR2.1**: New pipeline stage runs after GIL (or independently when `vector_search: true`)
- **FR2.2**: Embeds and indexes four document types: GIL Insight text, GIL Quote text,
  summary bullets, and transcript chunks
- **FR2.3**: Transcript chunking uses sentence-boundary windows (~300 tokens, ~50 token
  overlap) with `char_start`, `char_end`, and `timestamp_start_ms` per chunk
- **FR2.4**: Incremental: only embeds new/changed episodes; skips already-indexed episodes
  (tracked by episode_id + content hash in metadata)
- **FR2.5**: Uses the same embedding model as GIL grounding (`all-MiniLM-L6-v2` default,
  configurable via `gi_embedding_model`)
- **FR2.6**: Index location defaults to `<output_dir>/search/` (configurable)

### FR3: Search CLI Command

- **FR3.1**: `podcast search "<query>"` — encode query, search index, return ranked results
- **FR3.2**: Filter flags: `--type insight|quote|summary|transcript`, `--feed <name>`,
  `--since <date>`, `--speaker <name>`, `--grounded-only`
- **FR3.3**: `--top-k <n>` (default 10) controls result count
- **FR3.4**: `--format json|pretty` (default `pretty`) for output format
- **FR3.5**: Results include: rank, similarity score, document text, document type,
  episode title, episode_id, feed, publish_date, and source references
- **FR3.6**: For Insight results, include `grounded` status and `supporting_quotes` summary
- **FR3.7**: For transcript chunk results, include `timestamp_start_ms` and `char_start`

### FR4: Index Management CLI

- **FR4.1**: `podcast index --rebuild` — full re-index (e.g. after model change)
- **FR4.2**: `podcast index --stats` — show vector count, document type breakdown, feeds
  indexed, last updated timestamp, embedding model, index size on disk
- **FR4.3**: Auto-detect embedding model mismatch between index metadata and config;
  warn user and suggest `--rebuild`

### FR5: Enhanced `gi explore` and `gi query`

- **FR5.1**: When a vector index exists, `gi explore --topic <query>` uses semantic
  matching instead of substring matching (transparent upgrade)
- **FR5.2**: When no index exists, fall back to current substring matching (backward
  compatible)
- **FR5.3**: `gi query` semantic QA uses the vector index for real meaning-based
  question answering (RFC-050 UC4)

### FR6: Configuration

- **FR6.1**: `vector_search: bool` (default `false`) — enable embedding and indexing
- **FR6.2**: `vector_backend: faiss | qdrant` (default `faiss`)
- **FR6.3**: `vector_index_path: str | null` — auto: `<output_dir>/search/`
- **FR6.4**: `vector_index_types: list` — which document types to index
  (default: `[insights, quotes, summary_bullets, transcript_chunks]`)
- **FR6.5**: `vector_chunk_size_tokens: int` (default 300) and
  `vector_chunk_overlap_tokens: int` (default 50) for transcript chunking

## Success Metrics

- `podcast search` returns semantically relevant results for paraphrased queries (manual
  validation on 20+ query/corpus pairs)
- Synonym / rephrase queries match content that substring search misses (e.g. "government
  AI policy" finds insights labeled "AI Regulation")
- Search latency < 100 ms for a 100K-vector corpus on standard hardware
- Index build time < 60 seconds for 500 episodes (incremental: < 5 seconds per new episode)
- Zero regression on existing `gi explore` / `gi query` behavior when no index present
- Index size < 500 MB for 500 episodes (with IVF-PQ compression at scale)

## Dependencies

- PRD-017 / RFC-049: GIL artifacts (`gi.json`) provide Insight and Quote text to index
- PRD-005: Summaries provide bullet text to index
- PRD-001: Transcripts provide text for chunked indexing
- `sentence-transformers` (already in dependency tree)
- `faiss-cpu` (new dependency, ~20 MB)

## Constraints & Assumptions

**Constraints:**

- Must work without a server process (CLI-first; FAISS in-process)
- Must not increase pipeline runtime by more than 30% when enabled
- Must reuse the existing `embedding_loader.py` / `all-MiniLM-L6-v2` (no new large model
  downloads unless user explicitly configures a different model)
- Index is append-friendly; full rebuild only required on embedding model change

**Assumptions:**

- Corpus scale for CLI use is 10-50 feeds, up to ~5,000 episodes (1M+ vectors with
  transcript chunks)
- Users have existing GIL and/or summary artifacts before enabling search
- English-only content (matching current pipeline language support)

## Design Considerations

### Vector Backend: FAISS vs Qdrant

- **Decision**: FAISS for Phase 1 (CLI); Qdrant for Phase 2 (platform/service)
- **Rationale**: FAISS is in-process, no server, minimal dependency. Qdrant adds built-in
  metadata filtering and native upserts but requires a server for production use. The
  `VectorStore` protocol abstracts the choice.

### Index Scope: Per-Feed vs Global

- **Decision**: Global corpus index with feed metadata for filtering
- **Rationale**: Cross-feed discovery is a primary use case. Per-feed indexes would require
  multi-index search for cross-feed queries.

### Embedding Model: Current vs Upgrade

- **Decision**: Keep `all-MiniLM-L6-v2` for Phase 1; evaluate upgrades for Phase 2
- **Rationale**: Already loaded for GIL grounding. Zero marginal cost. 384-dim is efficient
  for FAISS. Upgrade path (e.g. `bge-base-en-v1.5`) available via config when needed.

## Integration with GIL and KG

Semantic Corpus Search enhances the GIL and KG layers by adding a **discovery** dimension:

- **GIL Integration**: Search results for Insights carry full GIL provenance (grounding,
  quotes, timestamps). `gi explore` transparently upgrades to semantic matching when an
  index exists. UC4 (Semantic QA) becomes functional.
- **KG Integration**: Topic and entity labels can be semantically matched across episodes,
  improving the quality of `kg entities` and `kg topics` roll-ups without requiring
  entity resolution.
- **Digest Integration**: Embedding similarity enables the weekly digest clustering
  described in the platform megasketch (Part C) — group similar Insights, deduplicate
  cross-feed coverage, rank by novelty.

## Example Output

```bash
$ podcast search "impact of AI regulation on startups"

Search: "impact of AI regulation on startups"
Index: 47,832 vectors | Model: all-MiniLM-L6-v2 | Last updated: 2026-04-01

Results (top 5):

1. [insight] score=0.87 | GROUNDED
   "AI regulation will significantly lag behind the pace of innovation"
   Episode: AI Regulation (The Journal) | 2026-02-03
   Quotes: 2 supporting → "Regulation will lag innovation by 3–5 years." (Sam Altman, 2:00)
   → gi show-insight --id insight:episode:abc123:a1b2c3d4

2. [insight] score=0.82 | GROUNDED
   "Startups face disproportionate compliance burden compared to big tech"
   Episode: Startup Policy (The Information) | 2026-01-15
   Quotes: 1 supporting → "Small companies can't afford compliance teams..." (CEO, 14:30)

3. [summary] score=0.79
   "European AI Act creates new requirements for high-risk applications"
   Episode: Tech Policy Update (Pivot) | 2026-03-10

4. [transcript] score=0.74
   "...the real question is whether startups can even survive the
    regulatory overhead. We're seeing companies move to jurisdictions..."
   Episode: Innovation vs Regulation (a16z) | 2026-02-20 | timestamp: 23:45

5. [quote] score=0.71 | speaker: Mary Smith
   "Every dollar spent on compliance is a dollar not spent on R&D"
   Episode: Founder Stories (YC) | 2026-03-01 | timestamp: 8:12
```

## Related Work

- Issue #466: Post-v1 depth backlog (NL consumption, richer aggregation)
- PRD-017: Grounded Insight Layer
- PRD-019: Knowledge Graph Layer
- RFC-050: GIL Use Cases (UC4 Semantic QA, UC5 Insight Explorer)


## Release Checklist

**Tracking:** [#485](https://github.com/chipi/podcast_scraper/issues/485) (foundation prerequisites), [#484](https://github.com/chipi/podcast_scraper/issues/484) (Phase 1 implementation), epic [#466](https://github.com/chipi/podcast_scraper/issues/466)

- [ ] PRD-021 reviewed and approved
- [ ] RFC-061 created with technical design
- [ ] `VectorStore` protocol + `FaissVectorStore` implemented
- [ ] Transcript chunker implemented
- [ ] Embed-and-index pipeline stage implemented
- [ ] `podcast search` CLI command implemented
- [ ] `podcast index` CLI command implemented
- [ ] `gi explore` semantic upgrade implemented
- [ ] Config fields added (`vector_search`, `vector_backend`, etc.)
- [ ] Unit + integration tests cover search round-trip
- [ ] Documentation updated (README, config examples, guides)
- [ ] `make ci-fast` passes with search enabled and disabled
