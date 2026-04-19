# Explore Expansion: Production Validation (pending re-ingestion)

Tested all 5 explore expansion features (#601) on 199-episode production
corpus (`corpus_rss_registry_openai_gemini_10`). Features work correctly
but surface data quality gaps from the old corpus configuration.

## Results summary (2026-04-19)

| Command | Status | Notes |
| ------- | :----: | ----- |
| `gi clusters --top N` | PASS | 88 clusters, 15 cross-episode, top has 26 insights across 21 episodes |
| `gi explore --expand-clusters` | PASS | Cross-episode cluster context displayed correctly |
| `gi explore-quotes --query` | PASS | Returns results, but low scores (0.35-0.64) and sparse matches |
| `gi topic-insights --topic` | PASS | 15 clusters linked via ABOUT edges for "quantum" |
| `gi explore --sort evidence-density` | PASS | Sort works, but density = cluster_size only (no quotes) |

## Data quality observations

### 1. grounded=False on most insights

Explore results show `grounded=False` with 0 supporting quotes. The old
corpus was likely run without provider-mode grounding or with
`gi_insight_source=bullets` which doesn't always produce quotes.

**Impact:** Evidence density scoring degrades to cluster_size × 1 (no
quote multiplier). Cluster context expansion has no cross-episode quotes
to show.

**Fix:** Re-ingest with `gi_insight_source=provider` and all providers
running evidence extraction (multi-quote #600).

### 2. Only 336 quotes indexed in FAISS (out of 646 insights)

Many insights have no quotes in the vector index. Quote-level search
(`gi explore-quotes`) is limited by coverage.

**Impact:** Sparse results for quote search queries.

**Fix:** Re-ingestion with multi-quote extraction will produce 3-5
quotes per insight → ~2000-3000 indexed quotes.

### 3. Quote search relevance is low

Scores 0.35-0.44 for semantically matching queries. Quotes are short
fragments indexed without surrounding transcript context — the embedding
may not capture enough meaning for standalone semantic search.

**Potential improvement:** Consider indexing quotes with surrounding
transcript context (±200 chars) to improve embedding quality. Or use
the insight text as a prefix for the quote embedding.

## Validation plan after re-ingestion

When production corpus is re-ingested with latest pipeline:

1. **Re-run `insight-clusters`** — expect more cross-episode clusters
   with better grounding data
2. **Re-test `gi explore --expand-clusters`** — should show cross-episode
   quotes from other episodes
3. **Re-test `gi explore-quotes`** — should have ~3x more indexed quotes,
   higher relevance scores
4. **Re-test `gi explore --sort evidence-density`** — should show
   grounded=True with meaningful density scores (cluster × quote count)
5. **Re-test `gi topic-insights`** — should show richer clusters with
   supporting evidence

## Corpus stats for reference

```
Corpus: corpus_rss_registry_openai_gemini_10
Episodes: 199
GI artifacts: 199
Insights: 646
Clusters: 88 (15 cross-episode)
Singletons: 402
FAISS index: 3641 transcript, 916 summary, 617 kg_entity, 541 insight, 518 kg_topic, 336 quote
```
