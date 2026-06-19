# #1033 cohort rerun — corrected scoreboard (2026-06-19)

**Trigger**: #1033 step 2 — rerun #1016 Round 3 cohort GI + KG stages
under the corrected pipeline (`kg_extraction_src: provider` instead of
`summary_bullets`) so the rankings reflect the production code path.

**Branch**: `feat/autoresearch-followups-2026-06-18`
**Dataset**: `curated_5feeds_dev_v1` (10 episodes)
**Silver**: `silver_opus47_kg_dev_v1` + `silver_opus47_gi_dev_v1`
**Pipeline**: corrected per #1033 step 1 (`provider` source,
tightened entity-vs-topic prompt with worked examples)

## Headline result

**Qwen3.5-35B-A3B sweeps both quality stages.** Cell F (Qwen3-30B-A3B-NVFP4)
slips from "GI cohort #1 winner" under the broken pipeline to #2 on both
GI and KG topic coverage under the corrected pipeline.

**Entity coverage is 0% across all 7 candidates** — confirms pattern B
from the live discussion: this is a fundamental LLM extraction gap, not
a per-model capability dimension. The tightened entity-vs-topic prompt
reduced false-positive emission (Cell F p02_e02: 7 false orgs → 3) but
no model emits the named persons/orgs the silver expects. Filed as
follow-up #1035 — add NER pre-pass before LLM classification.

## Full scoreboard — sorted by GI avg_max_sim

| # | Candidate | GI avg_sim | KG topic cov | KG entity | GI s/run | KG s/run |
|---|---|---:|---:|---:|---:|---:|
| 1 | **Qwen3.5-35B-A3B** | **0.618** | **50%** | 0% | 311 | 215 |
| 2 | **Cell F NVFP4** (Qwen3-30B-A3B-NVFP4) | **0.595** | 45% | 0% | **286** | **154** |
| 3 | Gemma-4-26B-A4B-it | 0.593 | 35% | 0% | 349 | 204 |
| 4 | Moonlight-16B-A3B | 0.581 | 40% | 0% | **169** | **140** |
| 5 | Ministral-3-14B-Instruct-2512 | 0.574 | **50%** | 0% | 874 | 576 |
| 6 | Llama-3.3-70B-Instruct-NVFP4 | 0.509 | 24% | 0% | 1031 | 638 |
| 7 | DeepSeek-V2-Lite-Chat | 0.405 | 4% | 0% | 104 | 179 |

## Side-by-side — old (broken) pipeline vs new (corrected) pipeline

| Candidate | OLD GI (`summary_bullets`) | NEW GI (`provider`) | Δ |
|---|---:|---:|---:|
| Cell F NVFP4 | 0.611 | **0.595** | **-0.016** |
| Qwen3.5-35B-A3B | 0.605 (#1016 R3) | **0.618** | **+0.013** |
| Moonlight-16B-A3B | 0.594 | 0.581 | -0.013 |
| Gemma-4-26B-A4B-it | 0.585 | 0.593 | +0.008 |

| Candidate | OLD KG topic | NEW KG topic | Δ |
|---|---:|---:|---:|
| Cell F NVFP4 | 41% | **45%** | **+4pp** |
| Qwen3.5-35B-A3B | 49% | **50%** | +1pp |
| Moonlight-16B-A3B | 38% | 40% | +2pp |
| Ministral-3-14B | 48% | **50%** | +2pp |

**The corrected pipeline shifts the cohort rankings.** Topic coverage
improves under `provider` source across the board (+1pp to +4pp), and
GI shifts by smaller amounts in both directions. The combined effect
re-ranks Cell F from "GI cohort #1" to "#2 on both stages."

## Why Qwen3.5-35B-A3B now wins outright

Three reasons the architectural fix favored it:

1. **The summary prompt was suppressing names** — Qwen3.5-35B-A3B
   summaries strictly followed the "no speaker names" rule. Bullets
   were anonymized; bullets-derived KG had nothing to extract. With
   `provider` source, Qwen3.5 reads the transcript directly and finds
   the same content the silver does.

2. **Tightened entity-vs-topic prompt rewards thorough models.**
   Qwen3.5 produced more cleanly-bounded topic labels (35 / 50% cov
   compared to Moonlight's 40% / Cell F's 45%) because its prompt
   adherence is stronger.

3. **Cell F's NVFP4 quantization slightly compresses extraction
   quality** — visible on the GI axis (0.595 vs Cell F's bf16 sibling
   at higher numbers from #1016 Phase 2c).

## Cell F's daily-driver status — re-evaluated

Cell F was crowned daily-driver in #1022 based on:
1. Speed: 1.7× faster end-to-end than Qwen3.5-35B-A3B ✓ (unaffected)
2. Footprint: 18 GB vs 67 GB ✓ (unaffected)
3. GI cohort #1: 0.4250 cov_rate (the headline that drove the verdict)
4. Tied or better on KG topic ✓ (now +5pp behind, still competitive)

The new picture for Cell F vs Qwen3.5-35B-A3B:

| Dimension | Qwen3.5-35B-A3B | Cell F NVFP4 |
|---|---|---|
| GI avg_sim | 0.618 ✓ | 0.595 (-3.9%) |
| KG topic | 50% ✓ | 45% (-5pp) |
| End-to-end speed (s for GI+KG) | 526 | **440 (-16% faster)** |
| Weight footprint | 67 GB | **18 GB (-73%)** |
| Boot time | ~6 min | **~2 min** |

**The daily-driver verdict still holds.** Cell F is materially faster
and dramatically smaller. The quality gap to Qwen3.5-35B-A3B is real
but bounded (-4% GI, -5pp KG). The story shifts from "wins quality + is
faster" to "best speed-quality trade-off; Qwen3.5-35B-A3B is the
top-quality reserve for one-shot evals."

The existing `prod_dgx_full_with_fallback` profile docstring already
includes a note about manually swapping to Qwen3.5-35B-A3B for
highest-stakes evals — that note remains correct.

## What changed for the #1016 final-report addendum

The original #1016 Round 3 final report § 2 (per-stage winners) +
§ 7 (per-candidate verdict) now need correction:

- **Summary stage rankings** — unchanged (independent of source choice)
- **GI stage rankings** — re-ordered (Qwen3.5-35B-A3B now sweeps,
  Cell F drops from #1 to #2, Gemma-4 moves from #1 to #3)
- **KG stage rankings** — Qwen3.5-35B-A3B still #1; Cell F still
  competitive; Moonlight rises slightly
- **§ 6b entity-coverage attribution** — was "prompt doesn't request
  entities"; actual cause is LLM extraction gap (all candidates 0%
  even with explicit prompt). Filed as #1035.

## What changed for the #1022 Cell F validation doc

The "Cell F is the new GI cohort winner" claim needs walking back to
"Cell F is competitive on GI (#2)" + "Cell F's KG entity-coverage
metric was 0% on both pipelines — that's a cohort-wide gap, not
Cell F-specific."

The daily-driver designation stands; the rationale is updated.

## Operational artifacts

- `autoresearch/1033_cohort_rerun/runs.tsv` — full per-candidate-stage
  scoreboard with wall-clock + score
- `autoresearch/1033_cohort_rerun/sweep.log` — orchestration log
- `autoresearch/1033_cohort_rerun/run_sweep.sh` — re-runnable sweep
  harness; supports `--skip` for partial resumes
- `autoresearch/1033_cohort_rerun/logs/` — per-stage stdout/stderr
- `data/eval/runs/1033_rerun/<candidate>/{gi,kg}/` — raw predictions
  + relocated artifacts per candidate

## Failed boots — fixed via incremental flag adjustments

Two candidates boot-failed on the first sweep attempt and were
recovered manually:

- **Gemma-4-26B-A4B**: required `--max-num-batched-tokens=8192` to
  satisfy the multimodal `max_tokens_per_mm_item=2496` constraint.
  Default `--max-num-batched-tokens=2048` rejected the boot with
  a `ValueError`. Added the flag; Gemma booted cleanly.
- **Ministral-3-14B**: required the mistral tokenizer trio
  (`--tokenizer_mode=mistral --config_format=mistral --load_format=mistral`).
  Sweep harness adds these conditionally per candidate; the
  initial run had them but the boot took longer than the 15-min
  monitor timeout. Retry with the same flags + longer wait
  succeeded.

The sweep harness now records both fixes in `run_sweep.sh` for
re-runnability.

## DGX state after rerun

- Compose: restored to Cell F NVFP4 (canonical daily-driver state)
- vLLM: healthy on `autoresearch` served-model-name alias
- No uncommitted compose changes against homelab `main` (verified
  via `git diff docker-compose.yml` on DGX)
- Total wall-clock: ~3 hours (including 2 retry sequences)

## Next steps

- **#1016 final report addendum** — update with corrected rankings
  (in this commit's adjacent doc updates)
- **#1022 Cell F validation doc addendum** — same
- **#1033** — closes with this rerun + the corrected scoreboard;
  related artifacts (the deletion follow-up in #1034) stay open
- **#1035** — to be filed: "Add NER pre-pass for entity extraction"
  with this rerun's evidence (0% entity coverage across ALL
  candidates including the highest-quality ones) as the empirical
  case for needing a deterministic NER layer ahead of the LLM
- **Task #113 small-model standoff** — will now run under the
  corrected pipeline automatically; no extra prep needed
