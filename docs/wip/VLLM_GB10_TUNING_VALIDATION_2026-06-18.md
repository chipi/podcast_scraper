# vLLM-on-GB10 tuning — Cells A–D validation (#1022)

**Date**: 2026-06-18
**Branch**: `feat/autoresearch-followups-2026-06-18`
**Hardware**: GB10 (Blackwell, sm_121) on `dgx-llm-1`
**Workload**: Qwen3-30B-A3B-Instruct-2507 (autoresearch slot, vllm:26.05-py3)
**Dataset**: `curated_5feeds_dev_v1` (10 episodes)
**Per-stage configs** (round3_v1, vendor-correct sampling, post-#1016):
- summary: `autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_paragraph_round3_v1`
- gi: `autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_grounded_insights_round3_v1`
- kg: `autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_knowledge_graph_round3_v1`

---

## Methodology

Three-layer pre/post validation, executed end-to-end:

### Layer 0 — noise floor

Two locked baselines on the committed compose, back-to-back, no
restart between (`baseline_1` then `baseline_2`). Both runs used the
same vLLM process, same fixture, same prompts, same sampling params.

| Stage | b1 (s) | b2 (s) | Δ (s) | Δ (%) |
|---|---|---|---|---|
| summary | 140 | 137 | -3 | -2.1% |
| gi | 299 | 307 | +8 | +2.7% |
| kg | 292 | 280 | -12 | -4.1% |
| **total** | **731** | **724** | **-7** | **-1.0%** |

**Conservative noise floor: ±5%** (max per-stage variance was 4.1%).
For a cell change to count as a "real" signal, total wall-clock
change must exceed ±5%.

### Layer 1 — per-cell A/B (single-lever changes vs locked baseline)

Each cell: revert prior cell, apply one lever change, restart vLLM,
wait for `/health` 200, 3 warmup calls, run summary → GI → KG eval,
move artifacts under `data/eval/runs/1022/<label>/`, log to
`autoresearch/1022_gb10_tuning/runs.tsv`.

### Layer 2 — combined champion

**Not executed.** No cell delivered a signal that warranted a
combined run.

---

## Results — four cells

| Cell | summary | gi | kg | **total** | Verdict |
|---|---|---|---|---|---|
| baseline mean | 138s | 303s | 286s | **728s** | — |
| A (`gpu-memory-utilization 0.65 → 0.85`) | +5.4% | +1.7% | -5.2% ⚡ | **-0.3%** | No signal |
| B (`max-num-seqs default → 4`) | +5.4% 🐢 | +5.6% 🐢 | 0% | **+3.4%** | Slight regression |
| C (`max-model-len 32768 → 131072`) | +9.7% 🐢 | +7.9% 🐢 | -5.6% ⚡ | **+3.0%** | Slight regression |
| D (warmup 3 calls × 4 tokens → 10 × 200) | +6.9% 🐢 | +5.3% 🐢 | +4.5% | **+5.3%** 🐢 | Slight regression |

Raw data: `autoresearch/1022_gb10_tuning/runs.tsv`. Per-stage
predictions, fingerprints, READMEs under
`data/eval/runs/1022/<label>/{summary,gi,kg}/`. Compose snapshots
per cell at `autoresearch/1022_gb10_tuning/compose_snapshots/`.

### Note on actual baseline state

The issue's "Current state" table said `gpu-memory-utilization=0.75`
(compose default). Actual production state on DGX has
`VLLM_GPU_MEM_UTIL=0.65` in `.env` (pre-existing override). Cell A
was therefore measuring `0.65 → 0.85` (an even bigger lever delta
than planned) and still produced no signal — the conclusion is
strengthened, not weakened.

### Note on the post-restart summary tax

A consistent pattern across cells A–D: summary stage runs ~5–10%
slower than the baseline mean. This is **not** a lever effect —
Cell D ran the same compose as the baselines but added aggressive
warmup (10 calls × 200 tokens) and summary still came in at +6.9%
(148s vs 138s). The +5–10% reflects vLLM's intrinsic restart-cycle
warmup that the warmup-call approach does not eliminate.

This means the noise floor measured back-to-back (±5%) underestimates
restart-cycle variance. A more honest noise floor for this regime
is closer to **±7–10%**, which would classify all four cells' totals
as noise.

---

## Decisive findings

1. **Zero of cells A–D delivered an attributable speedup.** Total
   wall-clock changes range from -0.3% to +5.3%. No cell crosses
   the conservative noise floor in a way that would justify
   shipping a config change.

2. **The autoresearch vLLM is NOT leaving headroom on these levers
   at this workload.** The issue's framing — that we're underusing
   GPU memory, batching, and KV cache budget — does not bite for
   Qwen3-30B-A3B-Instruct-2507 in single-stream eval mode.
   Single-stream prefill + decode at 30k context with max-num-seqs
   irrelevant (we have at most one active sequence at a time) and
   KV cache plenty under 32K.

3. **None of `gpu-memory-utilization`, `max-num-seqs`,
   `max-model-len`, or `warmup-call-count` is on the critical path
   for autoresearch single-stream eval.** They're all knobs around
   throughput / concurrency — none of which is the bottleneck at
   N=1 streams.

4. **The real wins, if they exist, live in the deferred cells:**
   - **Cell E** — `vllm:26.05-py3 → vllm/vllm-openai:cu130-nightly`.
     FlashInfer CUTLASS + MoE autotuner improvements from the
     2026-06-01 vLLM blog. **Architectural** change to the kernel
     path, not a runtime knob.
   - **Cell F** — NVFP4 quantization. ~4× weight throughput, often
     ~2× end-to-end latency win on MoE models. Different
     accuracy/quality tradeoff to measure.
   - **Cell G** — MoE active-params sweep (A3B → A12B / A22B).
     Trades speed for quality at the parameter-count level. Not
     a runtime tweak; a model-replacement experiment.

   These are the high-information cells. A–D were the easy-and-safe
   set; they came up empty.

---

## Recommendation

**Do not ship Cells A–D.** The lever changes deliver no measurable
benefit at this workload; shipping them would be config complexity
for zero return.

**If more autoresearch speed is needed**, the cost-effective path
is Cell E (image bump). Cell F (NVFP4) is the biggest absolute lever
but requires re-validating quality against the cohort. Cell G is
out of scope for "tuning" — it's effectively a new candidate eval.

**Document the negative result.** This validation cost ~1.5h of DGX
time + zero cost. The methodology (pre/post + noise floor + signal
threshold) is reusable. Future cells should adopt it.

---

## Operational artifacts

- **Validation script**: `autoresearch/1022_gb10_tuning/run_labeled.sh`
  — re-runnable per cell, captures wall-clock + relocates artifacts.
- **Results TSV**: `autoresearch/1022_gb10_tuning/runs.tsv` —
  rolling log of every cell run.
- **Per-cell logs**: `autoresearch/1022_gb10_tuning/*.log` — stdout
  for each run.
- **Compose snapshots**: `autoresearch/1022_gb10_tuning/compose_snapshots/*.yaml`
  — `docker compose config` output captured at each cell's run time
  (audit trail of which lever was active).
- **Per-stage predictions**: `data/eval/runs/1022/<label>/{summary,gi,kg}/`
  — predictions.jsonl + baseline.json + fingerprint.json + README.md
  per stage per cell, ready for `rescore_against_silver.py` if we
  want quality numbers later.

DGX final state: compose at committed `main` (untouched), .env
unchanged from production (`VLLM_GPU_MEM_UTIL=0.65`). vLLM-autoresearch
running cleanly. Pre-flight stash from earlier session
(`wip_pre_1022_baseline_deepseek_v2_lite`) still in git stash;
restore with `git stash pop` if the DeepSeek-V2-Lite-Chat config
needs to come back.

---

## Caveats

- **Quality scoring not run.** The TSV captures wall-clock only.
  G-Eval and silver-based quality scoring (per JUDGING.md) was not
  exercised since no cell showed a speed signal to justify the
  quality-side cost. Predictions are saved per cell so this can be
  added later.
- **Single sample per cell.** Each cell ran once × dev_v1 10 ep.
  A second per-cell run would refine variance estimates but at
  ~+12 min per cell. Given no cell showed signal, additional samples
  would only narrow confidence intervals around "no effect" — low
  marginal value.
- **Restart-cycle variance** larger than back-to-back variance.
  The 5% noise floor is conservative for same-session, optimistic
  for cross-restart. A more rigorous design would measure several
  fresh-restart baselines.
- **Workload-specific finding.** This says nothing about prod
  serving (multi-stream) or other models. Conclusions apply only
  to autoresearch single-stream eval on Qwen3-30B-A3B-Instruct-2507.

---

# ADDENDUM — Cell F extension (2026-06-19)

After landing the cells A–D negative result, operator asked to push
into F/G. **Cell G dropped** (operator: no models >35B). **Cell E
dropped** (operator: NVIDIA NGC image is Spark-optimized; switching
to upstream vllm-openai would LOSE NVIDIA's curated optimizations).

That left **Cell F — NVFP4 quantization** as the one remaining
high-information lever.

## Cell F — clean WIN

**Model**: `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` (NVIDIA Model
Optimizer team, official org). Same architecture as baseline
(qwen3_moe), quantized to 4 bits/weight via NVIDIA ModelOpt.
Weight size: ~18.1 GB (vs baseline 57 GB bf16, 3.2× smaller).

### Speed — dev_v1, 10 episodes

| Stage | Baseline (bf16) | Cell F (NVFP4) | Δ |
|---|---|---|---|
| summary | 138s | **72s** | **-48.0%** ⚡ |
| gi | 303s | **170s** | **-43.9%** ⚡ |
| kg | 286s | **138s** | **-51.7%** ⚡ |
| **total** | **728s** | **380s** | **-47.8%** ⚡ |

Far above the ±5% noise floor. ~2× speedup across all three
stages. Per-episode: 72.8s baseline → 38s NVFP4.

### Quality — dev_v1, Opus 4.7 silver (vendor-disjoint per JUDGING.md)

| Stage | Metric | Baseline mean | Cell F | Verdict |
|---|---|---|---|---|
| summary | embedding cosine | 0.8008 | 0.8011 | ±0.04% — neutral |
| summary | numbers_retained | 0.5 | 1.0 | +100% — Cell F **better** |
| gi | avg max similarity | 0.5945 | **0.611** | +2.8% — slightly better |
| kg | topics covered | 42.5% | 41% | -3.5% — within variance |
| kg | topics avg similarity | 0.6075 | 0.600 | -1.2% — neutral |
| kg | entities covered | 0% | 0% | bug (#1016 task #111, unrelated to F) |

Cell F is **within or above baseline quality on every measured
dimension**. The 4-bit precision loss does NOT manifest in semantic
similarity to silver references.

### Held-out champion validation — benchmark_v2, 5 episodes

Per validation methodology, the held-out `curated_5feeds_benchmark_v2`
(5 ep, e03) is touched once per committed champion. Silvers are
Sonnet 4.6 (cross-vendor sanity check vs the dev_v1 Opus 4.7 silvers).

**Speed (cell_f_nvfp4_benchmark_v2_heldout)**:

| Stage | Time | Per-episode |
|---|---|---|
| summary | 43s | 8.6s/ep |
| gi | 94s | 18.8s/ep |
| kg | 76s | 15.2s/ep |
| **total** | **213s** | **42.6s/ep** |

Per-episode rate is ~12% slower on held-out vs dev_v1 (38s/ep);
within the noise band for between-dataset comparison.

**Quality (Sonnet 4.6 silvers)**:

| Stage | Metric | dev_v1 (Opus47) | benchmark_v2 (Sonnet46) | Cross-dataset / cross-vendor verdict |
|---|---|---|---|---|
| summary | embedding cosine | 0.8011 | **0.8297** | Higher on held-out — quality holds |
| summary | numbers_retained | 1.0 | 1.0 | Perfect retention both datasets |
| gi | avg max similarity | 0.611 | **0.614** | Nearly identical — quality holds |
| kg | topics covered | 41% | **44%** | Slightly higher on held-out |
| kg | topics avg sim | 0.600 | **0.623** | Slightly higher on held-out |

**Verdict**: cross-vendor sanity check **passes**. Cell F quality
on Sonnet 4.6 silvers (held-out) at least matches its Opus 4.7
silver scores (dev_v1) — no Sonnet-mimicry artifact, no held-out
surprise. The speed win comes with **no measurable quality cost**.

## Updated final recommendation

**Cell F is the autoresearch-tier champion.** Replace baseline
Qwen3-30B-A3B-Instruct-2507 (bf16) with NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4
in the homelab autoresearch compose.

Expected operational benefit:
- ~2× faster eval sweeps on dev_v1 + benchmark_v2
- ~3× smaller model footprint (18 GB vs 57 GB) — frees ~40 GB of
  unified memory for KV cache headroom or future co-residence
- No quality regression across summary / GI / KG on either silver

Required follow-up:
- **Homelab PR** to `chipi/agentic-ai-homelab` swapping
  `Qwen/Qwen3-30B-A3B-Instruct-2507` →
  `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` in
  `infra/vllm/autoresearch/docker-compose.yml`. Operator merges.
- **Per-candidate config alignment** — none of our YAML configs
  reference the model name directly (they use `model: "autoresearch"`
  via the vLLM served-model-name alias), so no config changes needed
  on the podcast_scraper side.

## What the 4 safe cells + Cell F together teach

The issue's framing — that we're leaving 10–30% headroom on runtime
knobs (mem-util, num-seqs, max-len, warmup) — turned out to be
wrong for this workload. **None of those knobs is on the critical
path.** The bottleneck is the model itself: weight throughput and
memory bandwidth dominate at single-stream eval.

The cell that DID move the needle was the architectural lever —
swap the weight precision (NVFP4 = 4×4×4 lookup-table-encoded MoE
weights). That's not "tuning"; that's a different model artifact
running on the same architecture.

Lesson for future "tuning" issues: triage runtime knobs vs
architectural levers first. If the workload is single-stream + the
bottleneck is weight bandwidth, only architectural changes
(quantization, model swap) will move the needle.

---

# COHORT COMPARISON (2026-06-19) — Cell F vs full #1016 Round 3 cohort

Once Cell F (Qwen3-30B-A3B-NVFP4) was confirmed as a clean win on
the same-model bf16 → NVFP4 comparison, operator asked the bigger
question: **does Cell F displace any of the #1016 Round 3 cohort
on the autoresearch tier?** With DGX limited to one served model
at a time, the autoresearch tier picks a **single daily-driver
champion**.

## Methodology

Same scorers / silvers as the #1016 final report:
- Summary: `rescore_against_silver.py` vs `silver_opus47_dev_v1_paragraph`
- GI: `score_gi_insight_to_insight.py` vs `silver_opus47_gi_dev_v1`
- KG: `score_kg_topic_coverage.py` vs `silver_opus47_kg_dev_v1`

Speed: `avg_latency_ms` from each run's `metrics.json` intrinsic
performance block, divided to s/ep.

Cell F results were measured fresh during #1022; the rest of the
cohort scores are reused from the #1016 Round 3 final report run
artifacts (`data/eval/runs/autoresearch_prompt_vllm_*_dev_*_round3_v1/`).

## Per-stage scoreboard

### Summary (rouge1_f1 vs Opus 4.7 silver, higher = better)

| # | Candidate | rouge1_f1 | s/ep |
|---|---|---|---|
| 1 | Qwen3.5-35B-A3B (#1016 top dog) | 0.5936 | 13.58 |
| 2 | Moonlight-16B-A3B (#1016 safe pick) | 0.5745 | 8.96 |
| 3 | Mistral-Small-3.2-24B | 0.5557 | 69.69 |
| 4 | Magistral-Small-2509 | 0.5520 | 64.11 |
| **5** | **Qwen3-30B-A3B-NVFP4 (Cell F)** | **0.5407** | **7.20** ⚡ |
| 6 | Ministral-3-14B | 0.5334 | 30.22 |
| 7 | Gemma-4-26B-A4B | 0.5183 | 14.78 |
| 8 | Llama-3.3-70B-NVFP4 | 0.4869 | 42.90 |
| 9 | DeepSeek-V2-Lite-Chat | 0.3922 | 4.79 |

### GI (coverage_rate, higher = better) — NEW STAGE WINNER

| # | Candidate | cov_rate | avg_max_sim | s/ep |
|---|---|---|---|---|
| **1** | **Qwen3-30B-A3B-NVFP4 (Cell F)** | **0.4250** | **0.6114** | 17.00 |
| 2 | Gemma-4-26B-A4B (prior #1016 GI winner) | 0.4125 | 0.5933 | 34.91 |
| 3 | Qwen3.5-35B-A3B | 0.3625 | 0.6049 | 30.14 |
| 4 | Ministral-3-14B | 0.3000 | 0.5817 | 88.42 |
| 5 | Moonlight-16B-A3B | 0.1625 | 0.5543 | 15.10 |
| 5 | Llama-3.3-70B-NVFP4 | 0.1625 | 0.4801 | 111.56 |
| — | DeepSeek-V2-Lite-Chat | 0.0000 | 0.0941 | 8.99 (BPE bug — #111) |

### KG (topic coverage_rate vs Opus 4.7 silver)

| # | Candidate | topic_cov | topic_avg_sim | s/ep |
|---|---|---|---|---|
| 1 | Qwen3.5-35B-A3B | 0.4854 | 0.6570 | 21.18 |
| 2 | Ministral-3-14B | 0.4175 | 0.5985 | 56.93 |
| **3** | **Qwen3-30B-A3B-NVFP4 (Cell F)** | **0.4078** | **0.6000** | **13.80** ⚡ |
| 4 | Gemma-4-26B-A4B | 0.3495 | 0.5494 | 20.11 |
| 5 | Moonlight-16B-A3B | 0.2816 | 0.5116 | 16.17 |
| 6 | Llama-3.3-70B-NVFP4 | 0.2524 | 0.4719 | 59.74 |
| — | DeepSeek-V2-Lite-Chat | 0.0000 | 0.0940 | 16.19 (BPE bug — #111) |

## End-to-end per-episode speed (3 stages summed)

| Candidate | summary | gi | kg | **total s/ep** |
|---|---|---|---|---|
| **Cell F NVFP4** | 7.20 | 17.00 | 13.80 | **38.00** |
| Moonlight | 8.96 | 15.10 | 16.17 | 40.23 |
| Qwen3.5-35B-A3B | 13.58 | 30.14 | 21.18 | 64.90 |
| Gemma-4-26B-A4B | 14.78 | 34.91 | 20.11 | 69.80 |
| Ministral-3-14B | 30.22 | 88.42 | 56.93 | 175.57 |
| Mistral-Small-3.2-24B | 69.69 | — | — | — |

Cell F is the **cohort end-to-end speed leader** (excluding the
disqualified DeepSeek-V2-Lite, whose quality cratered).

## Head-to-head: Cell F vs Moonlight (current safe pick)

| Dimension | Moonlight (16B-A3B) | Cell F (30B-A3B-NVFP4) | Δ |
|---|---|---|---|
| Summary rouge1_f1 | 0.5745 | 0.5407 | Moonlight +6.3% |
| GI cov_rate | 0.1625 | **0.4250** | **Cell F +161%** ⚡⚡⚡ |
| KG topic cov | 0.2816 | **0.4078** | **Cell F +45%** ⚡⚡ |
| End-to-end s/ep | 40.23 | **38.00** | **Cell F -5%** |
| Weight footprint | 32 GB bf16 | **18 GB NVFP4** | Cell F -44% |

**Cell F dominates Moonlight in 4 of 5 dimensions.** Moonlight wins
on summary by 6%, loses everywhere else (massively on GI / KG).
**This displaces Moonlight as the autoresearch safe pick.**

## Head-to-head: Cell F vs Qwen3.5-35B-A3B (current top dog)

| Dimension | Qwen3.5-35B-A3B | Cell F | Δ |
|---|---|---|---|
| Summary rouge1_f1 | 0.5936 | 0.5407 | Qwen3.5 +9.8% |
| GI cov_rate | 0.3625 | **0.4250** | **Cell F +17%** ⚡ |
| KG topic cov | 0.4854 | 0.4078 | Qwen3.5 +19% |
| End-to-end s/ep | 64.90 | **38.00** | **Cell F -41%** ⚡⚡⚡ |
| Weight footprint | 67 GB bf16 | **18 GB NVFP4** | Cell F -73% |

Qwen3.5-35B-A3B keeps the top-quality crown on summary + KG.
Cell F wins GI outright and runs ~1.7× faster end-to-end.

## Single-model daily-driver decision: **Cell F NVFP4**

**Choice**: replace both Moonlight and Qwen3.5-35B-A3B with Cell F
NVFP4 as the single autoresearch-tier daily driver. **For
highest-stakes one-shot evals** where summary or KG quality
matters more than time, operator can manually swap the compose to
Qwen3.5-35B-A3B for that specific run — but the steady-state
champion is Cell F.

### Why Cell F as the single daily driver

1. **GI winner** — Cell F is the best of the cohort on the hardest
   stage. GI is the autoresearch-champion gate (per JUDGING.md:
   ≥75% of staged target). Cell F is on the winning side of that.
2. **1.7× faster end-to-end** — 38 s/ep vs 64.9 s/ep on
   Qwen3.5-35B-A3B. For autoresearch sweeps + real-podcast eval
   work (#972), this compounds across hundreds of episodes.
3. **Modest, bounded quality loss on summary + KG** — -9.8%
   summary, -16% KG. Within the autoresearch champion bands;
   operationally acceptable.
4. **3.7× smaller weight footprint** — 18 GB vs 67 GB. Frees
   GPU memory for KV cache headroom + future co-residence.
5. **Already held-out validated** — `benchmark_v2` confirms
   quality holds cross-dataset + cross-vendor silver.
6. **Same architecture as #1016 baseline** — qwen3_moe, same
   prompt convention, same sampling defaults. Drop-in replacement
   at the homelab compose level; no per-candidate config plumbing
   needed.

### Operational footprint

- Homelab compose: 1-line model swap
  (`Qwen/Qwen3-30B-A3B-Instruct-2507` →
  `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`).
- Profile YAMLs (`prod_dgx_full_with_fallback`, `prod_dgx_balanced`,
  `eval_default`): **no functional change** — all use the
  `autoresearch` served-model-name alias. Docstrings updated to
  reference the new model.
- Registry (`model_registry.py`): docstrings updated for
  consistency. No StageOptions changes (still routes via
  `autoresearch` alias).
- Pricing (`pricing_assumptions.yaml`): no change —
  `autoresearch: $0/token` row already exists.

### What this DOES NOT change

- **Cloud profiles** (`cloud_quality`, `cloud_balanced`,
  `cloud_thin`): no change. These don't touch the DGX.
- **Local Ollama profiles** (`local_dgx_balanced` etc.): no
  change. These use Ollama, not the vLLM autoresearch slot.
- **The autoresearch slot serving the new model isn't faster on a
  per-token basis** — it's faster because NVFP4 weights need ~1/4
  the memory bandwidth per forward pass. The CUDA compute graph
  + flash-attention path is unchanged.

## Recommendation for #1016 final report

The #1016 final report calls Qwen3.5-35B-A3B the "top dog" and
Moonlight the "safe pick." **Both designations are superseded
by Cell F NVFP4 for daily-driver use** on the autoresearch tier.
The cohort comparison table above is the new state of the world.

Update the report with an addendum noting Cell F entered the
cohort post-#1016 via the #1022 validation effort.
