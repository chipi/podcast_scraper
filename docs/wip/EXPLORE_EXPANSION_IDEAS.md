# Explore Expansion: What Can We Do With What We Have

Phase 3b-d of the insight enrichment plan (speaker profiles, consensus
detection, temporal tracking) are parked — they need diarization data
(SPOKEN_BY edges) and a larger re-ingested production corpus.

This note explores what we CAN do with `gi explore` today, using the
capabilities already built:
- Multi-quote per insight (1-5 quotes from real podcasts)
- Insight clustering (15 cross-episode clusters on production corpus)
- Topic clustering (112 clusters at 0.75)
- Cluster context expansion (cross-episode quotes in explore results)
- FAISS semantic search (insights, quotes, topics, transcripts indexed)

---

## What gi explore does today

```bash
gi explore --topic "quantum computing" --output-dir /path/to/corpus
```

Returns insights matching the topic query, ranked by semantic relevance.
Each insight has 1-5 supporting quotes with char offsets + timestamps.

**Limitations:**
- No awareness of insight clusters (treats each insight independently)
- No aggregation across episodes for the same claim
- No quote-level search (can't search for specific evidence)
- Topic-only entry point (no "show me everything about X concept")

---

## Expansion ideas (with existing infrastructure)

### 1. Cluster-expanded explore (ready to wire)

`insight_cluster_context.py` is built but not wired into the CLI.
Wire it: when `gi explore` returns results, run them through
`expand_with_cluster_context()` before output.

```bash
# Before: 3 insights from 1 episode
gi explore --topic "investment risk" --output-dir corpus/

# After: same 3 insights, but each shows cross-episode evidence
gi explore --topic "investment risk" --output-dir corpus/ --expand-clusters
```

**Effort:** ~30 min. Add flag to CLI, call the wrapper.

### 2. Quote-level search

Instead of searching insights, search quotes directly. FAISS already
indexes `doc_type: "quote"`. A new `gi explore-quotes` command would
find verbatim evidence across the corpus.

```bash
gi explore-quotes --query "92% of active managers" --output-dir corpus/
# Returns: all quotes mentioning this statistic, with episode + speaker context
```

**Effort:** ~2 hours. New CLI subcommand using existing FAISS search.

### 3. Cluster browse

Instead of querying, browse all insight clusters. Show the top N clusters
by member count, each with its canonical insight + episode list.

```bash
gi clusters --output-dir corpus/ --top 10
# Returns: top 10 insight clusters, sorted by cross-episode evidence strength
```

**Effort:** ~1 hour. CLI command that reads insight_clusters.json + formats.

### 4. Topic × Insight matrix

Combine topic clusters and insight clusters into a structured map.
For each topic cluster, show which insight clusters are associated.

```bash
gi topic-insights --topic "quantum computing" --output-dir corpus/
# Returns: all insight clusters where member insights have ABOUT edges
# to topics in the "quantum computing" topic cluster
```

This is the "what is CLAIMED about this TOPIC" query — the corpus
knowledge map. Uses existing ABOUT edges + both cluster artifacts.

**Effort:** ~2 hours. Cross-reference topic_clusters.json + insight_clusters.json
via ABOUT edges in gi.json.

### 5. Evidence density scoring

For each insight cluster, compute an "evidence density" score:
- Number of unique episodes
- Number of unique quotes
- Average quote diversity (char distance)
- Number of unique speakers (when SPOKEN_BY available)

Higher evidence density = more trustworthy claim. Surface this in
explore results as a confidence signal.

```bash
gi explore --topic "AI safety" --sort-by evidence-density
```

**Effort:** ~1 hour. Post-processing on cluster data.

---

## Recommended execution order

1. **Cluster-expanded explore** (30 min) — immediate value, just wiring
2. **Cluster browse** (1 hr) — standalone discovery tool
3. **Quote-level search** (2 hrs) — new search dimension
4. **Topic × Insight matrix** (2 hrs) — the knowledge map
5. **Evidence density scoring** (1 hr) — confidence signal

Total: ~half day for all 5. Each is independently useful.

---

## What this does NOT require

- No new ML models
- No new LLM calls
- No diarization / SPOKEN_BY edges
- No re-ingestion of production corpus
- Just wiring existing data + search infrastructure

---

## Parked: Phase 3b-d (needs diarization + re-ingestion)

| Feature | Blocker | Resume when |
| ------- | ------- | ----------- |
| Speaker profiles | SPOKEN_BY edges | Deepgram provider (#597) deployed |
| Consensus detection | NLI between cluster members | Larger corpus with opposing views |
| Temporal tracking | publish_date on cluster members | Production re-ingestion with provider-mode GI |
