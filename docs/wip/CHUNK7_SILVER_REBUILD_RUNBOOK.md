# RFC-097 chunk 7 — silver rebuild + scoreboard re-baseline runbook

**Status**: Code-half ready. Eval-run half operator-attended (multi-hour DGX time).
**Branch**: `feat/corpus-ontology-v2`
**Date**: 2026-06-20

## Code-half summary (done in chunk 7)

The scorers in `src/podcast_scraper/evaluation/` are **shape-agnostic** —
they operate at the counts / grounding-contract level and don't enumerate
node types. RFC-097 v2.0 (KG Entity → Person/Organization/Podcast +
HAS_EPISODE) and v3.0 (GI MENTIONS_PERSON/_ORG + insight_type + position_hint)
shape changes therefore pass through them unchanged.

Confirmed by `tests/unit/podcast_scraper/evaluation/test_v2_v3_shape_compatibility.py`:

- `compute_kg_prediction_stats` aggregates correctly for v2.0 payloads
- Mixed v1.x + v2.0 corpus aggregates uniformly
- v3.0 GI artifact with the full new shape passes strict schema validation

**Net code change for chunk 7**: 1 new test file. The scorers themselves
needed no edits — by design.

## Eval-run half (operator-attended)

The remaining work is LLM execution time on DGX, ~half-day per spec
(silver regen) + multi-hour (candidate sweep). Steps below in order.

### Pre-flight

```bash
# 1. Confirm DGX is in research mode (the project's slot — never `code`).
ssh dgx-llm-1 "~/bin/gpu-mode-swap.sh status"
# Expect: "✓ autoresearch vLLM up on :8003"

# If not in research mode:
ssh dgx-llm-1 "~/bin/gpu-mode-swap.sh research"

# 2. Confirm v2/v3 pipeline is the live emitter (post chunks 3-5).
.venv/bin/python -c "
from podcast_scraper.kg.pipeline import build_artifact
art = build_artifact('test', 'x', podcast_id='podcast:test', episode_title='T')
assert art['schema_version'] == '2.0'
assert any(n['type'] == 'Podcast' for n in art['nodes'])
print('KG v2.0 emit confirmed')
"
```

### Phase A — Silver regeneration (Opus 4.7 + Sonnet 4.6)

Regenerate the four silvers the autoresearch programme depends on
**against the v2/v3-emitting pipeline** so the silvers carry the new
shape (Person/Organization typed nodes, MENTIONS_PERSON/_ORG edges,
`insight_type` + `position_hint`):

```bash
# Opus 4.7 dev_v1 (KG + GI silvers)
make autoresearch-silver \
  PROFILE=silver_opus47_kg_dev_v1 \
  CORPUS=.test_outputs/manual/prod-v2/corpus

make autoresearch-silver \
  PROFILE=silver_opus47_gi_dev_v1 \
  CORPUS=.test_outputs/manual/prod-v2/corpus

# Sonnet 4.6 benchmark_v2
make autoresearch-silver \
  PROFILE=silver_sonnet46_kg_dev_v1 \
  CORPUS=.test_outputs/manual/prod-v2/corpus

make autoresearch-silver \
  PROFILE=silver_sonnet46_gi_dev_v1 \
  CORPUS=.test_outputs/manual/prod-v2/corpus
```

Expected: ~half-day of LLM time per the RFC-097 §Rollout estimate.

### Phase B — Candidate sweep + re-baseline

Re-score every candidate model in the autoresearch programme cohort
against the **new** silvers:

```bash
# Candidates per memory:
#   - Gemini 2.5 family
#   - Sonnet 4.6
#   - Opus 4.7
#   - DGX Qwen3-30B-A3B-Instruct-2507 (vLLM :8003)
make autoresearch-finale \
  COHORT=v2-baseline \
  SILVERS="silver_opus47_kg_dev_v1 silver_opus47_gi_dev_v1 silver_sonnet46_kg_dev_v1 silver_sonnet46_gi_dev_v1"
```

### Phase C — Vendor-bias compliance check (#939 lesson)

Per `feedback_silver_judge_vendor_bias.md`: the silver+judge vendor
must NOT match any single candidate vendor in the same cohort.

Verify before publishing the scoreboard:

| Silver vendor | Judge vendor | Candidate vendors in cohort | OK? |
|---|---|---|---|
| Anthropic (Opus 4.7) | TBD | Gemini, Sonnet 4.6, Opus 4.7, Qwen3-30B | ✗ Opus same vendor; rotate judge or split cohort |
| Anthropic (Sonnet 4.6) | TBD | Gemini, Sonnet 4.6, Opus 4.7, Qwen3-30B | ✗ Sonnet same vendor; rotate judge or split cohort |

**Decision rule**: drop Anthropic candidates from the cohort where the
silver is Anthropic, OR change the judge vendor to a disjoint one
(e.g., Gemini judge with Anthropic silver, Anthropic judge with Gemini
silver, etc.). Documented in `autoresearch/JUDGING.md` per memory.

### Phase D — Publish scoreboards

```bash
make autoresearch-report COHORT=v2-baseline OUT=docs/guides/eval-reports/EVAL_RFC097_V2_BASELINE_$(date +%Y_%m_%d).md
```

Operator-readable report goes into `docs/guides/eval-reports/` (canonical
home alongside the 50+ other promoted eval reports); the scorer-level JSON
artifacts persist under `data/eval/runs/`.

### Phase E — Acceptance criteria (per RFC-097 §Success Criteria)

1. ✓ Full silver rebuild scoreboards published with non-regressive
   KG/GI coverage on the new shape
2. ✓ Migration scripts dry-run clean on `.test_outputs/manual/prod-v2/corpus`
   (separate from silver regen; chunk 6 deliverable)
3. ✓ Grounding contract preserved bit-for-bit — every Insight with
   `grounded=true` has ≥1 SUPPORTED_BY edge; no descriptive edge
   promotes ungrounded Insights
4. (Chunk 9 gate) Schemas reject legacy after 2-4 weeks of v2 in prod

## What's NOT in chunk 7's scope

- Per-provider `insight_extraction/v1.j2` prompt updates — parked v2.1
  (see RFC-097 §3 retroactive-sweep status note). Megabundle path
  emits structured `insight_type`; per-provider path defaults to `unknown`.
- Cross-layer Podcast id derivation unification (KG `slug(feed_id)` vs
  GI `slug(show_title)`) — parked v3 (see corpus/ontology.md
  Known limitations section).
- `backfill_gi_insight_type.py` LLM-based reclassification — vocab
  normalisation in `migrate_gi_document_v3` covers the deterministic
  part; LLM backfill is a separate v2.1 if quality signal demands.

## Notes & gotchas

- **GPU swap rule** (per `feedback_never_use_coder_next`): NEVER
  `gpu-mode-swap.sh code`. That's the operator's IDE slot. Use
  `research` for project work; `idle` to release.
- **DGX vLLM is `autoresearch`** (Qwen3-30B-A3B-Instruct-2507 on `:8003`).
  Don't conflate with `coder-next` (off-limits).
- **Foreground `make`** (per `feedback_foreground_make`): run silver
  + sweep targets in foreground so VS Code output appears live.
- **Never push during this work** (per `feedback_never_push_early`).
  PR creation is task 11; operator-authorised after chunk 9.

## Rollback

If silver regen produces unexpected scores, the prior silvers (v1.x
shaped) are intact under `data/eval/references/silver/`. Roll back via
git checkout on the silver dirs — no data loss.
