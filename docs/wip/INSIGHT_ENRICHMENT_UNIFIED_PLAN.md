# Unified Plan: Multi-Quote + Insight Clustering + Corpus Q&A

Blends #600, #599, #601 into a single build-verify-build pipeline where
each step is validated with data before the next begins.

---

## Gaps found reviewing the three issues together

### 1. Quote quality verification missing from #600

We say "change prompt from 1 to 2-3 quotes" but never verify: are the 2nd
and 3rd quotes actually good? They could be overlapping, low-relevance, or
not-verbatim. Need silver-based scoring after the prompt change.

**Fix:** after implementing #600, score multi-quote output against GI silver
refs (which have verified verbatim quotes with char offsets). Measure:
- Quote verbatim match rate (do quotes exist in transcript?)
- Quote overlap rate (do quotes overlap in char range?)
- Quote diversity (are they from different parts of the transcript?)

### 2. Insight deduplication within episodes needed before #599

With gi_max_insights=12, the same episode may produce near-duplicate insights
("index funds beat active" AND "passive investing outperforms"). These should
be merged BEFORE cross-episode clustering to avoid inflating cluster sizes.

**Fix:** add a dedup pass in build_artifact or post-processing. If two insights
from the same episode have cosine similarity > 0.85, keep only the one with
higher grounding confidence.

### 3. FAISS already indexes insights — #599 doesn't need new indexing

Issue #599 says "add doc_type: gi_insight to FAISS." But the indexer already
indexes `doc_type: "insight"` (line 364 in indexer.py). The infrastructure
exists. #599 only needs the clustering layer on top.

### 4. Speaker attribution depends on diarization quality

#601's speaker profiles require SPOKEN_BY edges on quotes. Current Whisper API
doesn't provide diarization. Deepgram (#597) would add this. Until then,
speaker profiles are limited to episodes that have diarization data.

**Mitigation:** speaker profiles work with whatever SPOKEN_BY edges exist.
Incomplete data shows partial profiles. Don't gate #601 on #597.

### 5. Consensus detection is harder than clustering

#601 mentions consensus vs dissent detection. Clustering tells you insights
are ABOUT the same topic. It doesn't tell you if they AGREE or DISAGREE.
"AI risk is overstated" vs "AI risk requires immediate action" might cluster
together (both about AI risk) but express opposite positions.

**Fix:** Phase 3 of #601 needs a stance classifier or polarity signal, not
just similarity clustering. Options:
- NLI between clustered insights (entailment vs contradiction)
- LLM call: "Do these insights agree or disagree?"
- Manual annotation for MVP

### 6. ML provider returns only 1 QA span — needs top_k=3

The ML provider's `extract_quotes()` calls `extractive_qa.answer()` (singular)
not `answer_candidates()` (plural, returns top_k). For multi-quote, ML
provider needs to call `answer_candidates(top_k=3)`.

---

## Unified execution plan

### Phase 1: Multi-quote extraction (#600) — half day + autoresearch loop

**Step 1a: Generate multi-quote silver refs (~$3)**

Prompt Sonnet 4.6 to produce 2-3 verbatim quotes per insight for each of the
5 held-out episodes. Verify char offsets programmatically. This becomes the
reference quality standard for multi-quote — same approach as our GI insight
silver and KG topic silver.

Store as `data/eval/references/silver/silver_sonnet46_gi_multiquote_benchmark_v2/`

**Step 1b: Implement prompt change — variant A (1 hr)**

All 7 LLM providers: change `extract_quotes()` prompt from "a single short
quote" to "2-3 short verbatim quotes." JSON schema: `{"quotes": [...]}`.
ML provider: change `answer()` to `answer_candidates(top_k=3)`.

**Step 1c: Score variant A against silver (30 min)**

Run GI pipeline on 5 held-out episodes. Score against multi-quote silver:
- Avg quotes per insight (target: ≥ 1.8, was 1.0)
- Quote verbatim rate (target: ≥ 90%, check against transcript)
- Quote overlap rate (target: ≤ 5%, quotes should not overlap)
- Quote coverage: how many silver quotes does the provider find?
- Quote diversity: avg char distance between quotes (target: ≥ 1000 chars)

**Step 1d: Prompt tuning — variants B + C if A underperforms (1 hr)**

If variant A scores below targets, iterate:
- Variant B: add few-shot example of good multi-quote output
- Variant C: stronger instruction ("quotes MUST be from different parts of
  the transcript, at least 500 characters apart")
Score each against silver, pick winner. Same ratchet as v2 summarization.

**Step 1e: Per-provider validation (30 min)**

Run `make pipeline-validate` on at least 3 providers (gemini, openai, qwen)
to confirm multi-quote works across the matrix. Check:
- All providers return 2-3 quotes (not 1)
- Grounding rate stays ≥ 95%
- No providers break on the new JSON schema

**Step 1f: Provider comparison matrix (30 min)**

Score all 7 providers on multi-quote quality against silver. Same matrix
format as GI insight coverage and KG topic coverage. Identify if any
provider is systematically worse at multi-quote.

**Step 1g: Validate on production corpus (30 min)**

Run multi-quote on 3-5 real podcast episodes from the production corpus
(different feeds). Real podcasts have more speaker repetition — expect
higher multi-quote rate than synthetic. Measured: Odd Lots episode
produced **3-5 unique quotes/insight (100% of insights)** vs 1.30 on
synthetic. Quote cap removed — LLM finds all supporting passages naturally.

**Phase 1 results (2026-04-19):**

| Dataset | Avg unique quotes/insight | With 2+ | With 3+ |
| ------- | :-----------------------: | :-----: | :-----: |
| Synthetic held-out | 1.30 | 18% | 12% |
| Real podcast (Odd Lots) | 3.00-5.00 | 100% | 100% |

**Gate PASSED.** Multi-quote works on real content. Proceed to Phase 2.

**Gate:** proceed to Phase 2 only if avg quotes/insight ≥ 1.8 AND verbatim
rate ≥ 90% AND overlap rate ≤ 5% on the winning prompt variant.

---

### Phase 2: Insight dedup + clustering (#599) — 1-2 days + verification

**Step 2a: Within-episode insight deduplication (2 hrs)**

Before clustering across episodes, deduplicate within:
- Embed all insights per episode
- If cosine similarity > 0.85 between two insights from same episode:
  keep the one with more/better grounding quotes
- Log dedup count per episode

**Step 2b: Cross-episode insight clustering (half day)**

Mirrors topic_clusters.py:
- Collect insight vectors from FAISS (already indexed as doc_type: "insight")
- Aggregate per unique insight text across episodes
- Run `cluster_indices_by_threshold()` at 0.75 (same as topics)
- Select canonical insight via centroid-closest
- Output `insight_clusters.json`

**Step 2c: Threshold sweep on production corpus (1 hr)**

Run insight clustering at [0.70, 0.75, 0.80, 0.85] on the 112-episode
production corpus. Same sweep methodology as topic clustering:
- Cluster count + singleton rate per threshold
- Max cluster size (controls UI readability)
- Cross-episode cluster count (the metric that matters)

Pick threshold that maximizes cross-episode clusters while keeping max
cluster size ≤ 20.

**Step 2d: Cluster coherence scoring (1 hr)**

For top 10 clusters:
- Manual spot-check: are all members semantically the same claim?
- Automated check: avg pairwise cosine within cluster (should be ≥ threshold)
- Quote check: do quotes from different episodes in the same cluster
  support the same claim? (Sample 5 clusters, read 2-3 quotes each)

Score: coherence_rate = coherent_clusters / checked_clusters.

**Step 2e: Re-run pipeline-validate with insight clustering (30 min)**

Verify insight_clusters.json is valid across providers. Check schema,
cluster counts, singleton rates.

**Gate:** proceed to Phase 3 only if ≥10 cross-episode clusters form
AND coherence_rate ≥ 80% AND selected threshold produces manageable
cluster sizes.

---

### Phase 3: Explore expansion via gi explore (#601) — ~1 day

Phase 3 focuses on maximising value from existing infrastructure.
No new ML models, no new LLM calls, no diarization — just wiring
existing data + search infrastructure into the CLI.

Speaker profiles (3b), consensus detection (3c), and temporal tracking
(3d) from the original plan are **parked** — they need diarization data
(SPOKEN_BY edges) and a larger re-ingested production corpus. See
`docs/wip/EXPLORE_EXPANSION_IDEAS.md` for the parked items table.

**Step 3a: Cluster-expanded explore (30 min) — ✅ DONE**

`insight_cluster_context.py` wired into explore results.
`expand_with_cluster_context()` adds cross-episode evidence to each
matched insight.

**Step 3b: Cluster browse (1 hr)**

New `gi clusters` CLI command. Browse all insight clusters — top N
by member count, each with canonical insight + episode list.

```bash
gi clusters --output-dir corpus/ --top 10
```

Reads `insight_clusters.json`, formats for terminal. No search needed.

**Step 3c: Quote-level search (2 hrs)**

New `gi explore-quotes` command. Search quotes directly via FAISS
(already indexes `doc_type: "quote"`). Find verbatim evidence across
the corpus.

```bash
gi explore-quotes --query "92% of active managers" --output-dir corpus/
```

**Step 3d: Topic × Insight matrix (2 hrs)**

Combine topic clusters and insight clusters into a structured map.
For each topic cluster, show associated insight clusters via ABOUT edges.

```bash
gi topic-insights --topic "quantum computing" --output-dir corpus/
```

Cross-references `topic_clusters.json` + `insight_clusters.json` via
ABOUT edges in gi.json.

**Step 3e: Evidence density scoring (1 hr)**

For each insight cluster, compute an "evidence density" score:
- Number of unique episodes
- Number of unique quotes
- Average quote diversity (char distance)

Surface as confidence signal in explore results.

```bash
gi explore --topic "AI safety" --sort-by evidence-density
```

**Viewer UI integration** for all 5 features is tracked separately
(see GitHub issue for viewer integration).

---

## Verification summary (data gates between phases)

| Gate | Metric | Target | Blocks |
| ---- | ------ | :----: | ------ |
| Post-1 | Avg quotes/insight | ≥ 1.8 | Phase 2 |
| Post-1 | Verbatim rate | ≥ 90% | Phase 2 |
| Post-1 | Overlap rate | ≤ 5% | Phase 2 |
| Post-2 | Cross-episode clusters | ≥ 10 | Phase 3 |
| Post-2 | Cluster coherence (manual) | ≥ 80% | Phase 3 |
| Post-3 | Explore expansion features work on production corpus | Qualitative | Ship |

---

## What "done" looks like

When all phases complete:

```
Corpus (112 episodes) →
  646 insights (12/ep) →
    ~1300 grounding quotes (all per insight) →
      ~15 cross-episode insight clusters →
        Queryable via: gi explore --topic/--expand-clusters
                       gi explore-quotes --query
                       gi clusters --top N
                       gi topic-insights --topic
```

Each query returns **multi-source evidence** — not summaries, not generated
text, but actual verbatim quotes from real speakers with timestamps,
aggregated across episodes that independently support the same claim.

---

## Estimated total effort

| Phase | Effort | Output |
| ----- | :----: | ------ |
| 1a: Multi-quote silver refs | 30 min + ~$3 | Reference standard for multi-quote quality |
| 1b: Implement prompt variant A | 1 hr | Updated providers |
| 1c: Score variant A vs silver | 30 min | Data gate (verbatim, overlap, diversity) |
| 1d: Prompt tuning if needed | 1 hr | Best prompt variant (ratchet) |
| 1e: Per-provider validation | 30 min | Pipeline pass |
| 1f: Provider comparison matrix | 30 min | Multi-quote quality per provider |
| 2a: Within-episode dedup | 2 hrs | Cleaner insights |
| 2b: Cross-episode clustering | half day | insight_clusters.json |
| 2c: Threshold sweep on production corpus | 1 hr | Optimal threshold with data |
| 2d: Cluster coherence scoring | 1 hr | Data gate (≥80% coherent) |
| 2e: Pipeline revalidation | 30 min | Integration pass |
| 3a: Cluster-expanded explore | 30 min | ✅ Done — `--expand-clusters` flag |
| 3b: Cluster browse | 1 hr | `gi clusters` command |
| 3c: Quote-level search | 2 hrs | `gi explore-quotes` command |
| 3d: Topic × Insight matrix | 2 hrs | `gi topic-insights` command |
| 3e: Evidence density scoring | 1 hr | `--sort-by evidence-density` |
| **Total** | **~4-5 days** | Full corpus knowledge base |

---

## Parked: original Phase 3b-d (needs diarization + re-ingestion)

| Feature | Blocker | Resume when |
| ------- | ------- | ----------- |
| Speaker profiles | SPOKEN_BY edges | Deepgram provider (#597) deployed |
| Consensus detection | NLI between cluster members | Larger corpus with opposing views |
| Temporal tracking | publish_date on cluster members | Production re-ingestion with provider-mode GI |

---

## Dependencies outside this plan

| Dependency | Issue | Impact if missing |
| ---------- | :---: | ----------------- |
| Profile loader | #593 | Users manually set gi_insight_source=provider (works, less elegant) |
| KG label prompt tuning | #590 | Topic clusters slightly less clean (enforcement helps, prompt is better) |

None of these block execution. All are "better with, works without."
