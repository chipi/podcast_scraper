# Post Re-Ingestion Validation Plan

After production corpus is re-ingested with latest pipeline
(multi-quote #600, provider-mode GI, KG v3 prompt, topic alignment).

## Pre-condition

New production corpus available with:

- `gi_insight_source: provider` (not stub)
- Multi-quote extraction (uncapped, expect 3-5 quotes/insight)
- KG v3 prompt (noun-phrase topics)
- GI/KG topic alignment (KG labels replace GI bullet-slugs before bridge)
- Bridge fuzzy reconciliation enabled

## Step 1: Validate explore expansion features (~30 min)

Re-run all 5 CLI commands on fresh corpus (see `EXPLORE_EXPANSION_VALIDATION.md`):

```bash
# Build insight clusters
podcast insight-clusters --output-dir <corpus>

# 1. Cluster browse
podcast gi clusters --output-dir <corpus> --top 10

# 2. Explore with cluster expansion
podcast gi explore --topic "quantum" --output-dir <corpus> --expand-clusters

# 3. Quote-level search
podcast gi explore-quotes --query "index funds" --output-dir <corpus> --top-k 10

# 4. Topic × Insight matrix
podcast gi topic-insights --topic "investing" --output-dir <corpus>

# 5. Evidence density sort
podcast gi explore --topic "AI" --output-dir <corpus> --sort evidence-density
```

### Expected improvements over old corpus

| Metric | Old corpus | Expected |
| ------ | :--------: | :------: |
| grounded insights | 0% | ~100% |
| quotes per insight | 0 | 3-5 |
| explore-quotes results | 1 (sparse) | 10+ per query |
| evidence density scores | cluster_size only | cluster × quote count |
| cross-episode cluster quotes | none | real cross-episode evidence |

## Step 2: Insight clustering quality assessment (~1 hr)

With grounded insights and multi-quote, clustering should be richer:

- Re-run `insight-clusters` — expect more cross-episode clusters
- Compare cluster count and cross-episode rate vs old corpus
  (88 clusters, 15 cross-episode on old ungrounded data)
- Spot-check top 5 clusters: do cross-episode quotes make sense?
- Check if threshold 0.75 is still optimal or needs re-sweep

## Step 3: Bridge merge quality (~30 min)

With KG v3 noun-phrase labels + topic alignment:

- Check bridge merge rate (was 100% after alignment fix)
- Verify fuzzy reconciliation produces sensible merges
- Check Person entities have correct host/guest roles

## Step 4: Update eval baseline (~30 min)

Score new corpus against silver refs:

- GI insight coverage (target: maintain 65-80%)
- KG topic coverage with v3 prompt (target: maintain 65-79%)
- Quote verbatim rate (target: ≥90%)
- Quote diversity (target: quotes from different transcript sections)

## Step 5: Update docs and close validation

- Update `EXPLORE_EXPANSION_VALIDATION.md` with fresh results
- Update eval report if scores changed significantly
- Close any remaining validation items

## Step 6: NER provider validation at scale (~1 hr)

Current smoke eval (5 eps / 15 entities, `data/eval/runs/ner_*_smoke_v1/`)
puts spaCy trf at F1=1.000 tied with all 6 cloud LLMs — but 15 entities is
a ceiling where everyone ties. `cloud_balanced.yaml` was switched from
Gemini NER to spaCy trf on this basis (free, local, deterministic) but
the data is thin.

After re-ingestion, with a larger/harder corpus:

- Run NER scoring across spaCy trf + 6 cloud LLMs on ≥20 episodes
- Include harder entities: less-famous names, title-cased common words,
  multilingual edge cases, ORG vs PERSON ambiguity
- If spaCy trf still within ~3pp of best cloud LLM → switch stays
- If spaCy trf lags materially → revert cloud_balanced NER to a cloud
  provider (gemini or openai) and document the delta
