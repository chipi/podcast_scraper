# KG (Knowledge Graph) Autoresearch Plan

Evaluate and optimize KG extraction quality using the same silver-reference
methodology proven on GI. KG extracts topics + entities from episodes; quality
depends on summary bullets (or direct transcript extraction) and the LLM prompt.

**Related:** #579 (GI autoresearch), #578 (NER), `docs/wip/GI_AUTORESEARCH_PLAN.md`

---

## Current state

### KG pipeline (already built)

- **Pipeline:** `src/podcast_scraper/kg/pipeline.py` — `build_artifact()`
- **Two extraction paths:**
  - `kg_extraction_source="summary_bullets"` (default) — derives topics/entities from
    summary bullets via LLM prompt
  - `kg_extraction_source="provider"` — full transcript extraction via LLM
- **Node types:** Episode, Entity (person/org), Topic
- **Edge types:** MENTIONS (Entity→Episode, Topic→Episode)
- **Config:** `kg_max_topics` (default 20), `kg_max_entities` (default 15),
  `kg_merge_pipeline_entities` (default True — merges speaker-detected hosts/guests)

### Quality metrics (PRD-019, already built)

- Extraction coverage: % artifacts with non-empty extraction
- Avg nodes/edges per artifact
- Tool: `scripts/tools/kg_quality_metrics.py`
- Scorer: `src/podcast_scraper/evaluation/kg_scorer.py`

### What's missing (same gap as GI had)

- **No silver KG references** — can't answer "are these the RIGHT topics/entities?"
- **No cross-provider KG comparison**
- **No measurement of how summary quality affects KG quality** (chain hypothesis for KG)

---

## Approach: mirror GI methodology

GI autoresearch proved:
1. Generate silver refs via Sonnet 4.6
2. Score pipeline output against silver using embedding similarity
3. Test summary-derived vs direct extraction
4. Measure count scaling (more topics = more coverage?)

Apply the same pattern to KG.

---

## Phase 1: Silver KG references + baseline measurement

### Step 1 — Generate silver KG refs (~$3-5)

Prompt Sonnet 4.6 with full transcript, ask for:
```json
{
  "topics": [
    {"label": "2-8 word noun phrase", "description": "1-2 sentence context"}
  ],
  "entities": [
    {"name": "Full Name", "kind": "person|org", "role": "host|guest|mentioned",
     "description": "1-2 sentence context"}
  ]
}
```

Generate for 5 held-out + 10 dev episodes. Verify entity names appear in transcript.

### Step 2 — Score current pipeline output against silver

Measure:
- **Topic coverage:** what fraction of silver topics are covered by pipeline topics?
  (embedding similarity, same as GI insight coverage)
- **Entity coverage:** what fraction of silver entities are found by pipeline?
  (text match, similar to NER entity_set scoring)
- **Precision:** does pipeline extract topics/entities NOT in silver? (noise)

### Step 3 — Compare 3 tiers (same as GI)

| Tier | Summary source | Expected KG quality |
|------|---------------|-------------------|
| Low | bart-led | Poor topics (summary is weak) |
| Mid | qwen3.5:9b bundled | Reasonable |
| High | gemini-2.5-flash-lite | Best from summary |

### Step 4 — Test direct extraction vs summary-derived

Same experiment as GI: `kg_extraction_source="provider"` vs `"summary_bullets"`.
GI showed +10pp for direct extraction — expect similar for KG.

### Step 5 — Count scaling

Test `kg_max_topics` at 5, 10, 15, 20. Does coverage plateau like GI did at 12?

---

## Phase 2: Provider comparison

Run all 7 LLM providers at the optimal count. GI showed provider variance (78-88%);
KG may show similar or different ranking.

---

## Phase 3: Optimize (based on findings)

- **If summary-derived is close to direct:** keep default, save the extra LLM call
- **If direct wins big:** switch `kg_extraction_source="provider"` for quality deployments
- **If topic labeling is inconsistent:** tune the KG extraction prompt
  (current: `shared/kg_graph_extraction/v1.j2` and `from_summary_bullets_v1.j2`)
- **If entity extraction overlaps with NER:** consolidate NER + KG entity paths

---

## Connection to topic clustering

KG topics feed directly into the topic clustering pipeline
(`src/podcast_scraper/search/topic_clusters.py`). Better KG topics → better clusters.
See `docs/wip/TOPIC_CLUSTERING_AUTORESEARCH_PLAN.md` for the clustering-specific plan.

The chain: **Summary → KG topics → FAISS embeddings → Clustering → UI**

Optimizing KG topic quality has downstream effects on clustering. Do KG autoresearch
BEFORE topic clustering autoresearch so the clustering input is as good as it can be.

---

## Estimated budget

- Silver generation: 15 episodes × ~$0.30/episode = ~$5
- KG pipeline runs: 3 tiers × 15 episodes = 45 runs (mostly free, LLM calls ~$2)
- Provider comparison: 7 providers × 5 episodes = 35 runs (~$3)
- **Total: under $15**
