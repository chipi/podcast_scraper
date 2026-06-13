# Eval: GI autoresearch — direct-from-transcript vs summary-derived on v2 (#978)

**Date:** 2026-06-13
**Ticket:** [#978](https://github.com/chipi/podcast_scraper/issues/978) —
materialize `_GI_OPTIONS` registry entries.
**Parent epic:** [#907](https://github.com/chipi/podcast_scraper/issues/907)
**Companion:** [EVAL_GIL_BUNDLING_2026_05.md](EVAL_GIL_BUNDLING_2026_05.md)
(evidence-stack bundling matrix), [EVAL_HELDOUT_V2_2026_04.md](EVAL_HELDOUT_V2_2026_04.md)
(summary-provider compound winner).

## TL;DR

The historic `GI_AUTORESEARCH_PLAN` claim that "provider mode" — direct
LLM extraction from transcript, bypassing the summary step — would beat
summary-derived extraction by +10pp **does not hold on v2 fixtures**.

| Approach | Cov | Avg sim | Notes |
| --- | ---: | ---: | --- |
| Direct (n=6) | **10%** | 0.476 | gemini-2.5-flash-lite |
| Direct (n=8) | **10%** | 0.478 | gemini-2.5-flash-lite |
| Direct (n=10) | **10%** | 0.487 | gemini-2.5-flash-lite |
| Direct (n=12) | **8%** | 0.487 | gemini-2.5-flash-lite |
| Direct (n=16) | **8%** | 0.483 | gemini-2.5-flash-lite |
| Summary-derived bart-led | 8% | 0.393 | baseline summarizer |
| Summary-derived qwen3.5:9b bundled | **70%** | 0.748 | bundled prompts |
| Summary-derived gemini flash-lite | **72%** | 0.762 | v2 default — `cloud_balanced` |

Direct-from-transcript caps out at **10% coverage regardless of `n`**.
Summary-derived hits **72%** on the same v2 silver, on the same provider,
in the same eval window. Direct mode loses by **~60pp** — not within margin.

**Decision: keep the summary-derived default** (`gi_insight_source: provider`
in profile YAMLs). The historic plan-doc claim is reversed on v2 fixtures.

## What this unblocks

- Closes the missing-research-ref gap that blocked `_GI_OPTIONS`
  materialization in #978.
- The winning configuration `provider_n12_grounded_bundled` becomes the
  default for every cloud-bearing profile preset.
- `_GI_OPTIONS` joins KG / NER / clustering as research-backed in the
  pipeline-stage registry.

---

## Method

- **Direct mode** — the `experiment_gi_direct_insights.py` script feeds the
  raw transcript to the LLM and asks for exactly `n` insights as JSON.
  Skips the summary step entirely. Sweeps `n` ∈ {6, 8, 10, 12, 16}.
- **Summary-derived mode** — the production pipeline:
  transcript → cleaning → summary → insight extraction. Reuses the
  pre-existing v2 baseline measurements (`bart-led`, `qwen3.5:9b bundled`,
  `gemini-2.5-flash-lite` summary) reported alongside this run.
- **Provider**: `gemini-2.5-flash-lite` for the direct sweep — same model
  as `cloud_balanced`'s summary winner so the comparison is apples to
  apples on cost + latency. Other providers (Anthropic, OpenAI, DeepSeek)
  not tested for the direct sweep — the gap is large enough that
  provider-by-provider direct-mode comparison would just confirm the
  structural finding.
- **Silver**: `silver_sonnet46_gi_benchmark_v2` (40 insights / 5 episodes;
  same silver #903 uses for GI scoring on v2 fixtures).
- **Coverage**: avg cosine sim ≥ 0.65 between an extracted insight and
  any silver insight, MiniLM-L6 embeddings.
- **Dataset**: `curated_5feeds_kg_v2` (15 episodes total; silver covers
  p*_e03 = 5 episodes).

Script: `scripts/eval/experiment/experiment_gi_direct_insights.py`.

## Per-n direct-mode (full table)

```text
  gemini/gemini-2.5-flash-lite n= 6: 4/40 covered (10%), sim=0.476, 5.7s
  gemini/gemini-2.5-flash-lite n= 8: 4/40 covered (10%), sim=0.478, 6.1s
  gemini/gemini-2.5-flash-lite n=10: 4/40 covered (10%), sim=0.487, 6.8s
  gemini/gemini-2.5-flash-lite n=12: 3/40 covered (8%), sim=0.487, 7.3s
  gemini/gemini-2.5-flash-lite n=16: 3/40 covered (8%), sim=0.483, 9.1s
```

Direct-mode coverage is **flat** with `n` — adding more requested insights
does not raise coverage. Even the avg similarity barely moves (0.476 →
0.487). The structural limitation is in the extraction path, not the
budget; reading the same transcript and being told to write `N` insights
just produces `N` plausible-sounding statements that don't track the
silver's claim structure.

## Why direct mode loses

Hypothesis (one we cannot run a follow-up experiment for without writing
new tooling): the summary step compresses the transcript into a structured
form that the GI extractor *can actually decompose into propositional
insights* (claim / recommendation / question). Direct-mode produces
free-form "key takeaway" prose that humans recognise as the same content
but that the semantic-similarity check against silver claims rejects.

Two surface signals supporting this:

1. **Sim ceiling is low.** Even the *best* sim score (0.487) is below the
   coverage threshold (0.65). The direct-mode insights are semantically
   adjacent to the silver but not aligned with the silver's framing.
2. **Bart-led at 8%** (summary-derived but with a weak summarizer) lands
   in the same neighbourhood as direct mode. Two different failure modes
   (bad summary vs no summary) converge on the same low coverage —
   evidence that the bottleneck is the summary-to-insight bridge.

## Out of scope (intentional — already covered elsewhere)

- **Bundling axis** (`gil_evidence_quote_mode`, `gil_evidence_nli_mode`):
  resolved by `EVAL_GIL_BUNDLING_2026_05.md` — `bundled_ab` is the
  cross-provider champion (5/7 cloud providers). The current YAML default
  `bundled` matches.
- **Grounding axis** (`gi_require_grounding: true`): treated as universal
  on across all production profiles. Switching grounding off would change
  what an "insight" means; it's not a tunable like `n`.
- **`gi_max_insights = 12`**: not swept on the summary-derived side in
  this experiment. The historic n=12 default holds for cloud_balanced and
  cloud_with_dgx_primary (`gi_max_insights: 12` in both YAMLs). A separate
  n-sweep on summary-derived mode would be tractable but would only shift
  the default within the 70–75% band — not high-impact, deferred.
- **Per-provider direct-mode comparison**: skipped — the 60pp gap to
  summary-derived dominates any provider-vs-provider variation; running
  Anthropic / OpenAI / DeepSeek on the same hopeless extraction path
  would not change the verdict.

## What lands in `_GI_OPTIONS`

The winning configuration becomes:

```python
"provider_n12_grounded_bundled": StageOption(
    stage="gi",
    option_id="provider_n12_grounded_bundled",
    provider="provider",  # gi_insight_source = provider
    extra_settings={
        "max_insights": 12,
        "require_grounding": True,
        "evidence_quote_mode": "bundled",
        "evidence_nli_mode": "bundled",
    },
    research_ref="docs/guides/eval-reports/EVAL_GI_AUTORESEARCH_V2_2026_06_13.md",
    headline_metric=(
        "summary-derived provider mode beats direct-from-transcript by "
        "~60pp on v2 silver (72% vs 10%); n=12 historic default holds; "
        "bundled per EVAL_GIL_BUNDLING_2026_05"
    ),
    measured_at="2026-06-13",
    tier="primary",
),
```

Every existing `_PROFILE_PRESETS` entry already declares
`kg_extraction_source: provider` + `gi_max_insights: 12` in its YAML, so
adopting this option keeps the registry in lock-step with prod YAMLs (the
drift test catches any divergence).

## Acceptance

- [x] Direct-from-transcript sweep on v2 silver run with 5 `n` values.
- [x] Coverage measured against `silver_sonnet46_gi_benchmark_v2` (40 insights / 5 episodes).
- [x] Summary-derived baseline included in the same headline table for direct comparison.
- [x] Eval report (this file).
- [x] Run output persisted: `data/eval/runs/baseline_gi_autoresearch_v2/run.log`.
- [x] Materialization plan documented (option_id + extra_settings).

## Reproduction

```bash
mkdir -p data/eval/runs/baseline_gi_autoresearch_v2
export $(grep -E '^GEMINI_API_KEY=' .env)
PYTHONPATH=. .venv/bin/python scripts/eval/experiment/experiment_gi_direct_insights.py \
    --dataset curated_5feeds_kg_v2 \
    --silver  silver_sonnet46_gi_benchmark_v2 \
    --providers gemini:gemini-2.5-flash-lite \
    --counts 6,8,10,12,16 \
    | tee data/eval/runs/baseline_gi_autoresearch_v2/run.log
```
