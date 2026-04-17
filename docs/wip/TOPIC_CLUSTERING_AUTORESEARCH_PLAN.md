# Topic Clustering Autoresearch Plan

Validate and optimize the topic clustering pipeline that groups semantically
similar KG topics across episodes. This is the final layer in the chain:
**Summary → KG topics → FAISS embeddings → Clustering → UI graph view.**

**Depends on:** KG autoresearch (do KG first so clustering input is optimized).

**Related:** `docs/wip/KG_AUTORESEARCH_PLAN.md`, RFC-075 (topic clustering),
RFC-072 (CIL bridge)

---

## Current state

### Pipeline (already built)

- **Algorithm:** greedy average-linkage hierarchical clustering on cosine similarity
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (same as semantic search)
- **Key threshold:** `--threshold 0.75` (min mean cosine similarity to merge clusters)
- **Output:** `topic_clusters.json` with clusters + singletons + CIL alias mapping
- **CLI:** `podcast_scraper topic-clusters --threshold 0.75 --output-dir <corpus>`
- **UI integration:** Cytoscape compound parent nodes (`tc:` prefix), sibling episode
  auto-load (cap=10), overlay in graph viewer

### CIL bridge (already built)

- **bridge.json:** per-episode identity dedup across GI and KG layers
- **cil_lift_overrides.json:** manual + auto-generated topic aliases
- **Auto-merge:** `--merge-cil-overrides` flag writes cluster aliases into CIL overrides

### Parameters (all current "best guesses")

| Parameter | Value | Location | Confidence |
|-----------|-------|----------|------------|
| **Clustering threshold** | 0.75 | CLI `--threshold` | Low — initial guess, never validated |
| **Embedding model** | all-MiniLM-L6-v2 | FAISS store default | Medium — standard choice, but untested alternatives |
| **Sibling episode cap** | 10 | `VITE_CLUSTER_SIBLING_EPISODE_CAP` | Low — arbitrary |
| **kg_max_topics** | 20 | config.py | Medium — recently bumped from 5 |
| **kg_max_entities** | 15 | config.py | Low — default guess |
| **Topic label length** | 2-8 words | KG extraction prompt | Medium — prompt guidance |

### Validation (minimal)

- Optional operator-supplied YAML with `expected_merge_pairs` / `expected_distinct`
- Unit tests for clustering algorithm correctness
- No corpus-level quality metrics committed
- No systematic threshold optimization

### Known production issue (#580)

Issue #580 documents the real-world problem: Gemini KG produces 156-char sentence
labels (vs OpenAI's 21-char noun phrases). Result: 90% singletons at threshold 0.75.
Root cause is poor KG input (sentence labels don't cluster well). Our KG autoresearch
(switching to `kg_extraction_source="provider"` with noun-phrase prompting) should fix
the input quality. The threshold sweep below validates whether the current 0.75 is
right once input quality is fixed.

### Known risks (from RFC-075)

- Geography-qualified topics may cluster incorrectly ("Cuban Economic Crisis" ↔
  "Iranian Economic Crisis" — high embedding similarity, should NOT merge)
- Single-linkage chaining mitigated by average-linkage, but not tested at scale
- Embedding model may not distinguish domain-specific topic nuances

---

## What needs validation

### Implementation details (from code analysis)

- **Algorithm:** greedy average-linkage in `cluster_indices_by_threshold()`
  (topic_clusters.py:125-173). Repeatedly merges the pair of clusters with
  highest **mean pairwise cosine similarity** while above threshold.
- **Centroid selection:** `pick_centroid_closest_label()` picks the member
  whose embedding has highest cosine to cluster mean — avoids picking
  overly narrow labels.
- **Vector aggregation:** `collect_topic_rows_from_faiss()` groups FAISS
  `kg_topic` vectors by `source_id`, computes mean vector per topic across
  episodes, L2-normalizes.
- **Embedding text:** label + description concatenated (from `indexer.py:165-174`).
  Long sentence labels (#580) dilute the embedding with surface-form noise.
- **Auto-aliases:** `topic_id_aliases_from_clusters_payload()` maps each
  non-centroid member to the centroid's canonical `topic:slug`. Merged into
  `cil_lift_overrides.json` with hand-edits taking precedence.

### 1. Threshold sweep — is 0.75 the right number?

**Experiment:** run clustering at thresholds [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
on a real corpus. For each threshold, measure:

- **Cluster count:** how many non-singleton clusters form?
- **Avg cluster size:** are clusters too large (over-merging) or too small (under-merging)?
- **Known-good merges:** do expected topic pairs (from validation YAML or manual
  inspection) end up in the same cluster?
- **Known-bad merges:** do topics that should NOT cluster get merged?
  (geography-qualified pairs, unrelated topics with similar embeddings)

**Gold standard:** manually annotate 20-30 topic pairs as "should merge" / "should not
merge" on a real corpus. Score each threshold's F1.

### 2. Embedding model comparison

**Experiment:** compare clustering quality with:
- `all-MiniLM-L6-v2` (current, 384-dim, fast)
- `all-mpnet-base-v2` (768-dim, higher quality, slower)
- `bge-small-en-v1.5` (384-dim, newer, potentially better for short texts)

Same threshold sweep per model. Does a better embedding model shift the optimal threshold?

### 3. Topic label quality impact

**Experiment:** run clustering on KG topics from 3 quality tiers (same as GI/KG):
- bart-led topics (poor quality labels)
- qwen3.5:9b topics (mid quality)
- gemini topics (high quality)

Measure: do better topic labels produce better clusters? (Same chain hypothesis as GI.)

### 4. CIL bridge accuracy

**Experiment:** on a real corpus with both GI and KG artifacts:
- How many identities appear in both layers?
- How many get correctly merged vs incorrectly merged?
- Are there false negatives (same person in GI and KG but not bridged)?

### 5. End-to-end UI validation

**Manual inspection:** load a 10-20 episode corpus in the viewer, check:
- Do clustered topics visually make sense?
- Do sibling episode links lead to relevant content?
- Is the cluster count manageable for navigation (not too many, not too few)?

---

## Suggested experiment order

```
1. KG autoresearch first (silver refs + quality measurement)
   ↓ (KG topics optimized)
2. Threshold sweep on current corpus (0.60-0.90, 7 points)
   ↓ (optimal threshold identified)
3. Embedding model comparison (3 models × optimal threshold range)
   ↓ (best model + threshold pair)
4. Topic label quality impact (3 tiers × best config)
   ↓ (confirms chain or shows clustering is robust to input quality)
5. CIL bridge accuracy (manual spot-check, 20-30 identities)
6. End-to-end UI validation (manual, 10-20 episodes)
```

Steps 1-4 are automated. Steps 5-6 require manual inspection.

---

## Deliverables

1. **Optimal threshold** — replace the 0.75 guess with a data-backed number
2. **Best embedding model** — confirm all-MiniLM-L6-v2 or recommend upgrade
3. **Clustering quality report** — F1 on gold merge/distinct pairs
4. **Updated defaults** in CLI + docs if threshold or model changes
5. **Validation YAML** — a committed, tested set of expected_merge_pairs and
   expected_distinct for our production corpus

---

## Estimated budget

- Clustering runs are local + instant (FAISS operations, no API calls)
- Only cost is KG silver generation (covered in KG plan, ~$5)
- Embedding model comparison: download 2 additional models (~500MB each)
- Manual inspection: ~1-2 hours of human time
- **Total $: ~$0 beyond KG plan** (clustering itself is free)
