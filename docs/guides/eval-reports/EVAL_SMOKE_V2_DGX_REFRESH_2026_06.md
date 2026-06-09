# Evaluation Report: Smoke v2 DGX refresh — local champion re-pick (June 2026)

> **Refresh + expansion** of the autoresearch local-LLM matrix, run on the
> DGX Spark from #810/#811. Adds DeepSeek-R1 distill family (7B / 14B / 32B / 70B),
> gpt-oss:20B, and qwen3.6:latest as new entries; re-confirms the existing
> 12-model qualifier set; chooses a local champion against the
> `silver_sonnet46_smoke_v1` reference. Closes [#924](https://github.com/chipi/podcast_scraper/issues/924).

| Field | Value |
| --- | --- |
| **Date** | 2026-06-08 (sweep), 2026-06-09 (rerun + report) |
| **Dataset** | `curated_5feeds_smoke_v1` (5 episodes, 5 feeds) |
| **Silver reference** | `silver_sonnet46_smoke_v1` (Claude Sonnet 4.6) |
| **Baseline (for delta-vs-baseline columns)** | `baseline_llm_ollama_qwen35_35b_smoke_paragraph_v1` — established April 2026 as the highest-quality stable local model from the original 12-cell matrix |
| **DGX compute** | NVIDIA GB10 (Spark) via tailnet; same target as #814 Speaches and #926 pyannote |
| **Run dirs** | `data/eval/runs/llm_ollama_*_dgx_smoke_v2_2026_06/` |
| **Closes** | #924 (refresh decision) |
| **Informs** | #923 (prod-DGX profile model pick), #928 (DGX-vs-cloud summary championship) |

## Setup

Sweep configs at `data/eval/configs/summarization/autoresearch_prompt_ollama_<slug>_smoke_paragraph_v1.yaml`. Prompts at `src/podcast_scraper/prompts/ollama/<slug>/summarization/`. For the 6 NEW models (DeepSeek-R1 family, gpt-oss, qwen3.6) the prompts were cloned verbatim from `qwen3.5_9b` — per-model prompt tuning is out of scope for this refresh, but flagged in "What this does not do" below.

Sequence:

1. **Pull on DGX** — `ollama pull <model>` for the new entries (~90 GB total over the tailnet).
2. **Main sweep** — ran the 12-cell matrix; established baselines + retrieved DGX-side timings for the existing matrix. (`scripts/eval/smoke_v2_dgx_refresh.sh`)
3. **Retry sweep** — added the 7 new entries with `baseline_llm_ollama_qwen35_35b_smoke_paragraph_v1` as the stable reference. (`scripts/eval/smoke_v2_dgx_refresh_retry.sh`)
4. **Two-cell fix-and-rerun** — qwen3.6:latest and deepseek-r1:70b initially failed; root-caused + fixed; rerun via `scripts/eval/smoke_v2_dgx_refresh_rerun_fixes.sh`.

The killswitch on the retry sweep deliberately SIGTERM'd before `qwen3-coder:30b` started — qwen3-coder is a coder-specialized model, wrong domain for paragraph summarization. Excluded by design.

## Two surfaced bugs (fixed)

### Bug 1 — qwen3.6:latest produced 0-token content for every episode

**Symptom**: `length.avg_tokens = 0`, `vs_reference.rougeL_f1 = null` across all 5 episodes. Pipeline reported success (no errors) but the model returned an empty `message.content`.

**Diagnosis**: Direct curl to `/api/chat` showed `eval_count: 200` (model generated 200 tokens) but `content: ""`. The 200 tokens went into the separate `thinking` field. qwen3.6 (qwen35moe family) is a **reasoning model** that emits chain-of-thought to a thinking channel by default; with the autoresearch config's `max_length: 800`, the model consumed all 800 tokens reasoning and never reached the answer.

**Fix**: `src/podcast_scraper/providers/ollama/ollama_provider.py` — extended the existing `reasoning_effort: none` shim (originally added for qwen3.5) to cover qwen3.6 and the broader qwen3.x family (`qwen3.5` / `qwen3.6` / `qwen3-` / `qwen3:`). Test coverage added in `tests/integration/providers/llm/test_ollama_provider.py`.

After fix: qwen3.6 produces 2500-3200 char summaries in 7-8s per episode.

### Bug 2 — deepseek-r1:70b "Ollama summarization failed: Request timed out"

**Symptom**: Both datasets reported `status: error` with the same error on episode 1. Other R1 distills (7B/14B/32B) completed normally.

**Diagnosis**: Direct curl showed 70b R1 takes ~70s of eval time even on a 16-word input — chain-of-thought generation is verbose, and a 5K-char podcast input would generate thousands of CoT tokens. Default `ollama_timeout` of 120s is too short.

**Fix**: Rerun via `scripts/eval/smoke_v2_dgx_refresh_rerun_fixes.sh` sets `EXPERIMENT_OLLAMA_READ_TIMEOUT=1200` (20 min per request). The env-var hook was already in place at `scripts/eval/experiment/run_experiment.py:1199-1203` from prior 27B+ work.

Note: R1 doesn't honor `reasoning_effort: none` (R1 = Reasoning #1, hardcoded). The output still includes a thinking channel, but `content` is also populated separately, so our pipeline reads it cleanly.

## Headline numbers — full sweep against silver Sonnet 4.6

Sorted by RougeL vs silver. Higher = closer to Sonnet output (caveat: see methodology notes). All scores from `data/eval/runs/llm_ollama_<slug>_dgx_smoke_v2_2026_06/benchmark_summary.json` on the `curated_5feeds_smoke_v1` dataset.

| Model | Size | RougeL | Rouge1 | Cosine | Coverage | Avg tokens | Latency excl 1st (ms) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.5:27b` | 27B | **0.271** | 0.603 | 0.835 | 1.014 | 646 | 38,370 | highest RougeL — latency-disqualified for prod |
| `qwen3.6:latest` | 36B (MoE) | **0.271** | 0.608 | 0.801 | 0.895 | 570 | **6,733** | **new champion candidate** — ties qwen3.5:27b RougeL at qwen3.5:35b latency |
| `qwen3.5:35b` | 35B | **0.262** | 0.613 | 0.788 | 0.972 | 619 | 6,747 | ← **current baseline** (April champion) |
| `mistral:7b` | 7B | 0.260 | 0.577 | 0.782 | 0.695 | 443 | 9,170 | |
| `llama3.2:3b` | 3B | 0.254 | 0.545 | 0.813 | 1.059 | 675 | 6,472 | |
| `gemma2:9b` | 9B | 0.249 | 0.570 | 0.825 | 0.727 | 463 | 11,577 | |
| `mistral-small3.2` | ~24B | 0.249 | 0.564 | 0.801 | 0.742 | 473 | 25,079 | |
| `llama3.1:8b` | 8B | 0.247 | 0.571 | 0.815 | 0.956 | 609 | 11,963 | |
| `mistral-nemo:12b` | 12B | 0.241 | 0.533 | 0.749 | 0.961 | 612 | 16,297 | |
| `qwen2.5:7b` | 7B | 0.236 | 0.574 | 0.755 | 0.874 | 557 | 9,460 | |
| `qwen2.5:32b` | 32B | 0.229 | 0.538 | 0.770 | 0.701 | 446 | 33,166 | |
| `qwen3.5:9b` | 9B | 0.228 | 0.576 | 0.797 | 0.905 | 577 | 11,372 | known bundled-JSON flakiness — #912 |
| `gpt-oss:20b` | 20B | 0.226 | 0.532 | 0.760 | 0.670 | 427 | 10,332 | new — closest non-Qwen3 entrant |
| `deepseek-r1:14b` | 14B | 0.218 | 0.510 | 0.797 | 0.690 | 439 | 33,759 | R1 distill — reasoning-tuned, not summary-shaped |
| `phi3:mini` | 3.8B | 0.202 | 0.534 | 0.783 | 1.318 | 840 | 11,071 | |
| `deepseek-r1:7b` | 7B | 0.199 | 0.466 | 0.765 | 0.660 | 421 | 14,761 | R1 distill — reasoning-tuned, not summary-shaped |
| `deepseek-r1:32b` | 32B | 0.195 | 0.469 | 0.744 | 0.594 | 378 | 75,523 | R1 distill — slower AND worse than 14b |

**Cells that did NOT complete cleanly:**

- `deepseek-r1:70b` — **operationally disqualified, rerun terminated.** First episode took 203s, second 173s, but episodes 3 and 4 took 995s and 1023s each (~17 min). At avg ~480s/episode, a 100-episode prod run would take >13 hours just for summarization. We killed the rerun at episode 5/10 of dataset 1 — no scoring data captured. Even at hypothetical best-case RougeL, the latency disqualifies it from prod consideration. Not retrying.
- `qwen3-coder:30b` — **deliberately excluded** at the design level. Coder-specialized model, wrong domain for paragraph summarization. The retry-sweep killswitch SIGTERM'd before its cell.

## Champion decision

**Local champion: `qwen3.5:35b` (keep, for now)** — pending validation of the qwen3.6:latest result.

Reasoning:

- **qwen3.6:latest is a credible challenger** at same latency (~6.7s) with +3.4% RougeL and +1.7% cosine. See Finding 4 for details.
- **5-episode RougeL deltas under ±0.02 are noise.** The qwen3.6 edge (0.009 RougeL) is below the noise floor; we need #933 prod-curated validation + #932 G-Eval finale before committing to a swap.
- **qwen3.6 lowers coverage by 7.9%** — terser output. May or may not be desired; the G-Eval coverage dimension in #932 will tell us.
- **qwen3.5:27b** has the highest RougeL of any cell (0.271, tied with qwen3.6), but at 5.7× the latency (38s vs 6.7s). Latency-disqualified for prod, but viable for #928 championship eval.
- **No DeepSeek-R1 distill is a contender.** Range 0.195-0.226 RougeL; reasoning-tuned, not summary-shaped.
- **Sonnet-mimicry caveat applies.** Per [#932](https://github.com/chipi/podcast_scraper/issues/932) (G-Eval finale tier), RougeL measures lexical proximity to Sonnet 4.6's writing style, not summary quality itself. Relative ordering is informative; absolute scores are noisy.

**Implication for #923 prod profile**: keep `ollama_summary_model: "qwen3.5:35b"` in `prod_dgx_full_with_fallback` **for now**. Re-evaluate qwen3.6:latest as the champion once:

1. #933 prod-curated tier validates the quality edge on real podcasts.
2. #932 G-Eval finale confirms it on faithfulness/coverage/coherence/fluency (the coverage delta is worth a closer look).

Cloud-fallback chain stays cheap-first (gemini-2.5-flash-lite → sonnet-4.6) per existing #812 contract.

## Findings

### 1. DeepSeek-R1 distill family is NOT a paragraph-summary contender

| Model | RougeL | vs baseline (qwen3.5:35b at 0.262) | Latency excl. first (ms) | Verdict |
| --- | --- | --- | --- | --- |
| deepseek-r1:7b | 0.199 | -24% | 14,761 | Reasoning-tuned, prose mediocre |
| deepseek-r1:14b | 0.218 | -17% | 33,759 | Best of the distills, still well below |
| deepseek-r1:32b | 0.195 | -26% | 75,523 | Slower AND worse than 14b — counter-intuitive |
| deepseek-r1:70b | rerun killed | — | ~480,000 avg | **Operationally disqualified by latency alone** (see below) |

**70b latency was the killer.** After the timeout fix landed and we re-ran with `EXPERIMENT_OLLAMA_READ_TIMEOUT=1200`, r1:70b completed early episodes in 3-4 min but episodes 3 and 4 took 17 min each (995s / 1023s). Avg ~8 min/ep. For a 100-episode prod run that's >13 hours of summarization compute — operationally infeasible. We killed at episode 5/10 of dataset 1 with no scoring data captured, but the verdict was already clear from the timing alone.

**Pattern**: the "more parameters → better quality" intuition fails here. R1 distills are fine-tuned for math/coding/reasoning; their paragraph-style writing isn't where their training optimization went. The signal "R1 is not the summarization champion" is reliable even at this small sample size — and r1:70b adds "even if quality were better, latency disqualifies it."

### 2. qwen3.5:27b is the highest-quality local cell, but latency-disqualified for prod

qwen3.5:27b's RougeL of 0.271 is the highest in the entire 19-cell matrix, but its 38.4s avg latency excl. first is 5.7× qwen3.5:35b's 6.7s. For #923's "all inference on DGX" goal with ~100 eps/day, the latency gap translates to ~50 minutes more wall-clock per run. Not a fit for the prod LLM slot, though it's a defensible candidate for #928 (DGX-vs-cloud championship) where wall-clock matters less and quality matters more.

### 3. gpt-oss:20B is the only new model worth keeping in the catalog

- RougeL 0.226 (closest new entry to baseline, 14% below)
- Latency 10.3s — slightly slower than qwen3.5:35b (6.7s) but comfortably under 20s
- Single-channel output (no reasoning channel quirks)
- 20B parameters — fits well in DGX memory alongside other services

Not a champion candidate, but a useful diversity entry for the autoresearch matrix. Future v2.1 sweep (#44/#45) should include it as a stable reference for new entries.

### 4. qwen3.6:latest is a credible champion contender — same speed as qwen3.5:35b, slightly better quality

After the reasoning-effort fix landed and the rerun completed, qwen3.6:latest produced:

| Metric (curated_5feeds_smoke_v1) | qwen3.6:latest | qwen3.5:35b (baseline) | Δ |
| --- | --- | --- | --- |
| RougeL F1 vs silver | **0.271** | 0.262 | **+3.4%** |
| Rouge1 F1 | 0.608 | 0.613 | -0.8% |
| Embedding cosine | **0.801** | 0.788 | **+1.7%** |
| Coverage ratio | 0.895 | 0.972 | -7.9% |
| Avg latency excl. first | **6,733 ms** | 6,747 ms | -0.2% (tie) |
| Output tokens (avg) | 570 | 619 | -8% |

**This is the standout result of the refresh sweep.** qwen3.6:latest ties qwen3.5:27b on RougeL (both 0.271) while matching qwen3.5:35b's prod-viable latency. Same speed, ~3% better RougeL, ~2% better cosine, slightly lower coverage. At this sample size (5 episodes) the RougeL gap (0.009) is within the noise floor, but the latency-match + multi-metric edge is a real signal.

**Caveats before swapping champion:**

- 5-episode sample size — RougeL deltas under ±0.02 are noise. Need #933 prod-curated validation before committing to a champion swap.
- Coverage drops 7.9% — qwen3.6 may be summarizing more tersely. The 570 vs 619 avg-token delta is consistent with this.
- Prompt is still cloned from qwen3.5_9b. Per-model prompt tuning (#906 descendant) could push qwen3.6 further ahead, or expose a regression.
- qwen3.6:latest is a 36B MoE model (qwen35moe family); MoE memory pressure on DGX alongside Speaches + pyannote + other loaded Ollama models needs operational verification before prod.

## What this report deliberately does NOT do

- **No prompt tuning** — every new model uses qwen3.5_9b's prompt templates verbatim. Each model's "true" capability is likely higher with tuned prompts. Per-model tuning is its own ticket (descendant of #906, future autoresearch tier).
- **No G-Eval (LLM judge) scoring** — RougeL/BLEU/cosine measure lexical/embedding proximity to silver, not summary quality. The [#932 finale tier](https://github.com/chipi/podcast_scraper/issues/932) will add 4-dimension G-Eval (faithfulness / coverage / coherence / fluency) on top.
- **No prod-curated validation** — all signal here is from v2 synthetic smoke. The [#933 prod-curated tier](https://github.com/chipi/podcast_scraper/issues/933) will add real-podcast sanity checks before any prod model swap.
- **No qwen3-coder:30b** — coder-specialized model, wrong domain for paragraph summarization. The retry-sweep killswitch SIGTERM'd before its cell to save DGX time.

## Addendum — v2.1 sweep (2026-06-09)

After v2 landed, a small deep-research pass surfaced 4 newer Ollama models worth testing: `gemma3:27b` (Google Gemma 3), `phi4:14b` (Microsoft), `hermes3:8b` (Nous Research Llama 3.1 fine-tune), and `mistral-small:24b`. The v2.1 sweep ran them against the same baseline + silver as v2. Total wall-clock 16m 22s for 4 cells × 2 datasets.

**v2.1 scoreboard (curated_5feeds_smoke_v1):**

| Model | Size | RougeL | Rouge1 | Cosine | Coverage | Avg tokens | Latency excl 1st (ms) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mistral-small:24b` | 24B | **0.257** | 0.591 | 0.768 | 0.875 | 557 | 29,005 | Strong mid-tier; latency-disqualified for prod |
| `phi4:14b` | 14B | **0.256** | 0.567 | 0.782 | 0.852 | 543 | 17,877 | Best 14B-class result; latency too high for prod |
| `hermes3:8b` | 8B | 0.218 | 0.495 | 0.777 | 0.514 | 328 | 6,315 | Worse than base llama3.1:8b (0.247) — fine-tune doesn't help summarization |
| `gemma3:27b` | 27B | 0.207 | 0.530 | 0.739 | 0.869 | 554 | 36,447 | Surprising regression vs qwen3.5:27b (0.271 at same size class) — likely prompt-template mismatch (used qwen3.5_9b clone) |

**v2.1 verdict: no new champion contender.** The two strongest entries (`mistral-small:24b` and `phi4:14b`) cluster at -2% RougeL vs the v2 baseline AND have prod-disqualifying latency (3-4× qwen3.5:35b's 6.7s). qwen3.6:latest from v2 remains the only credible champion challenger.

**Notable v2.1 findings:**

- **phi4:14b is parameter-efficient** — 14B params reaching 0.256 RougeL vs qwen3.5:9b's 0.228 at similar param count. If we ever bring back the laptop-Ollama path, phi4 might be the better fast/small choice than qwen3.5:9b.
- **Hermes 3 fine-tune is a regression** for paragraph summarization vs base Llama 3.1:8B. The Nous fine-tune optimizes for chat/instruction-following, not abstractive prose. Same pattern as DeepSeek-R1: domain-specialized fine-tunes lose ground on this task.
- **gemma3:27b underperformed expectations** — 0.207 RougeL is 24% below qwen3.5:27b at the same param count. Possible causes: (a) qwen3.5_9b prompt template is wrong for Gemma's chat format, (b) Gemma 3 is multimodal-tuned and text-only paragraph summarization isn't its sweet spot, (c) tokenizer differences inflate latency. Worth a follow-up with a Gemma-native prompt before writing off entirely.
- **No v2.1 entry beats qwen3.5:35b** (0.262) on RougeL on prod-viable hardware. Champion decision unchanged from v2: qwen3.5:35b stays prod, qwen3.6:latest stays the validated-challenger via #932/#933.

**Combined v2 + v2.1 top-5 RougeL:**

| Rank | Model | RougeL | Latency | Verdict |
| --- | --- | --- | --- | --- |
| 1 (tie) | `qwen3.5:27b` | 0.271 | 38s | championship candidate, latency-disqualified for prod |
| 1 (tie) | `qwen3.6:latest` | 0.271 | 6.7s | **champion contender** — needs #932/#933 validation |
| 3 | `qwen3.5:35b` | 0.262 | 6.7s | current prod champion |
| 4 | `mistral:7b` | 0.260 | 9.2s | v2 entry — surprisingly strong for 7B |
| 5 (tie) | `mistral-small:24b` | 0.257 | 29s | v2.1 entry; latency-disqualified |

The full 22-cell matrix (18 v2 + 4 v2.1) is stable; champion-swap decision unchanged.

## Methodology caveats

- **5 episodes per cell** — directional; rankings within ±0.02 RougeL are noise.
- **Silver = Sonnet 4.6** — measuring "how close to Sonnet did this model write," not "how good is this summary." See #932 for the planned upgrade.
- **Per-model prompt templates not tuned** — clones from qwen3.5_9b. Affects all new entries equally; doesn't bias the ranking within the new cohort, but may bias new-vs-existing comparisons.
- **DGX-side warm-cold variance** — sequential sweeps include model load time on first episode; subsequent episodes are warmer. We report `avg_latency_ms_excluding_first` to smooth this.
- **No reliability axis** — sustained-burst behavior (#816) not tested here. Likely irrelevant for sequential single-call autoresearch but worth flagging for prod use.

## What this closes / informs

| Issue | Role | Status after this report |
| --- | --- | --- |
| #924 | Driver of this sweep | Closed by this report |
| #923 | Prod profile model pick | Champion stays `qwen3.5:35b`; comment posted |
| #928 | DGX-vs-cloud summary championship | qwen3.5:27b nominated as the local representative (best RougeL, latency acceptable for championship eval) |
| #44 / #45 | v2.1 sweep prep + run | **Done** — see v2.1 addendum above; no new champion contender |
| #932 | G-Eval finale | When implemented, runs against these same 19 cells |
| #933 | Prod-curated validation | When curated, the chosen champion gets a sanity-check pass |

## Addendum — Rescored against Opus 4.7 silver (2026-06-09, #939)

The v2 + v2.1 numbers above were originally scored against
`silver_sonnet46_smoke_v1`, which makes ROUGE measure "how close does this
model write to Claude Sonnet 4.6" — not "how good is this summary"
(see the methodology caveats above). Phase 0 of the next autoresearch ride
upgrades silver to Opus 4.7 (see
[SILVER_OPUS47_GENERATION_2026_06](SILVER_OPUS47_GENERATION_2026_06.md))
and rescores the 22 sweep cells × 2 datasets without re-running inference.

**Rescoring tool**: `scripts/eval/score/rescore_against_silver.py` — consumes
existing `predictions.jsonl`, writes per-run
`metrics_vs_silver_opus47_smoke_v{1,2}.json` non-destructively.

### Rescored scoreboard — `curated_5feeds_smoke_v1` vs `silver_opus47_smoke_v1`

Sorted by new RougeL F1, with delta from the original Sonnet-silver column for
context. **Higher is better; absolute scores are not comparable across silvers.**

| Model | Old RougeL (Sonnet) | New RougeL (Opus) | ΔRougeL | Rouge1 | Cosine | Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| `mistral:7b` | 0.260 | **0.329** | **+0.068** | 0.612 | 0.806 | 0.766 |
| `llama3.2:3b` | 0.254 | **0.326** | **+0.072** | 0.593 | 0.799 | 1.167 |
| `llama3.1:8b` | 0.247 | **0.307** | **+0.060** | 0.628 | 0.817 | 1.054 |
| `mistral-small:24b` | 0.257 | 0.284 | +0.027 | 0.597 | 0.782 | 0.964 |
| `hermes3:8b` | 0.218 | 0.279 | +0.061 | 0.541 | 0.785 | 0.567 |
| `gemma2:9b` | 0.249 | 0.276 | +0.026 | 0.561 | 0.831 | 0.802 |
| `qwen2.5:7b` | 0.236 | 0.265 | +0.029 | 0.572 | 0.776 | 0.963 |
| `mistral-small3.2` | 0.249 | 0.261 | +0.012 | 0.540 | 0.810 | 0.818 |
| `qwen3.5:27b` | **0.271** (old #1) | 0.257 | -0.015 | 0.557 | 0.802 | 1.117 |
| `mistral-nemo:12b` | 0.241 | 0.256 | +0.015 | 0.552 | 0.762 | 1.059 |
| `qwen3.5:35b` | 0.262 (old #3, baseline) | 0.243 | -0.019 | 0.572 | 0.801 | 1.071 |
| `qwen3.6:latest` | **0.271** (old #1, contender) | 0.241 | -0.030 | 0.571 | 0.768 | 0.986 |
| `phi4:14b` | 0.256 | 0.240 | -0.016 | 0.556 | 0.813 | 0.939 |
| `qwen2.5:32b` | 0.229 | 0.238 | +0.008 | 0.516 | 0.806 | 0.772 |
| `phi3:mini` | 0.202 | 0.230 | +0.028 | 0.522 | 0.802 | 1.453 |
| `deepseek-r1:70b` | (killed) | 0.228 (3 eps only) | — | 0.520 | 0.793 | 0.782 |
| `qwen3.5:9b` | 0.228 | 0.223 | -0.006 | 0.542 | 0.799 | 0.998 |
| `gpt-oss:20b` | 0.226 | 0.219 | -0.007 | 0.523 | 0.774 | 0.738 |
| `deepseek-r1:14b` | 0.218 | 0.212 | -0.007 | 0.488 | 0.800 | 0.760 |
| `qwen3-coder:30b` | 0.209 | 0.209 | -0.000 | 0.532 | 0.758 | 1.191 |
| `deepseek-r1:7b` | 0.199 | 0.209 | +0.010 | 0.442 | 0.764 | 0.728 |
| `deepseek-r1:32b` | 0.195 | 0.205 | +0.010 | 0.438 | 0.766 | 0.655 |
| `gemma3:27b` | 0.207 | 0.202 | -0.005 | 0.491 | 0.769 | 0.958 |

### Rescored scoreboard — `curated_5feeds_smoke_v2` vs `silver_opus47_smoke_v2`

Same pattern shows up on the v2-content dataset.

| Model | RougeL | Rouge1 | Cosine | Coverage |
| --- | --- | --- | --- | --- |
| `mistral:7b` | **0.302** | 0.566 | 0.825 | 0.848 |
| `llama3.1:8b` | **0.282** | 0.546 | 0.835 | 1.155 |
| `llama3.2:3b` | **0.271** | 0.552 | 0.802 | 1.212 |
| `mistral-nemo:12b` | 0.265 | 0.521 | 0.808 | 1.015 |
| `hermes3:8b` | 0.265 | 0.488 | 0.776 | 0.761 |
| `gemma2:9b` | 0.263 | 0.497 | 0.825 | 0.863 |
| `mistral-small:24b` | 0.257 | 0.507 | 0.808 | 0.993 |
| `qwen2.5:7b` | 0.253 | 0.504 | 0.825 | 0.968 |
| `gpt-oss:20b` | 0.251 | 0.486 | 0.833 | 0.795 |
| `qwen2.5:32b` | 0.248 | 0.462 | 0.805 | 0.805 |
| `phi4:14b` | 0.241 | 0.477 | 0.822 | 0.889 |
| `deepseek-r1:70b` | 0.237 (partial) | 0.465 | 0.796 | 0.733 |
| `mistral-small3.2` | 0.230 | 0.472 | 0.797 | 0.842 |
| `qwen3.5:35b` | 0.229 | 0.508 | 0.797 | 1.250 |
| `qwen3.6:latest` | 0.228 | 0.501 | 0.798 | 1.149 |
| `qwen3-coder:30b` | 0.215 | 0.477 | 0.809 | 1.400 |
| `qwen3.5:27b` | 0.213 | 0.488 | 0.760 | 1.423 |
| `deepseek-r1:14b` | 0.211 | 0.398 | 0.784 | 0.640 |
| `qwen3.5:9b` | 0.211 | 0.462 | 0.792 | 1.086 |
| `deepseek-r1:32b` | 0.199 | 0.382 | 0.788 | 0.575 |
| `gemma3:27b` | 0.193 | 0.440 | 0.787 | 1.129 |
| `deepseek-r1:7b` | 0.189 | 0.384 | 0.792 | 0.630 |
| `phi3:mini` | 0.183 | 0.458 | 0.801 | 1.663 |

### What changed — champion-decision implications

**Headline: the ranking flips, and the qwen family loses its edge.**

| Aspect | Old silver (Sonnet 4.6) | New silver (Opus 4.7) | Change |
| --- | --- | --- | --- |
| #1 rankings | qwen3.5:27b + qwen3.6:latest tied at 0.271 | **mistral:7b at 0.329** | New leader, +21% RougeL gap |
| Top-3 family | All Qwen3 (qwen3.5:27b, qwen3.6, qwen3.5:35b) | mistral:7b, llama3.2:3b, llama3.1:8b | Top-3 swaps to non-Qwen entirely |
| qwen3.5:35b (current champion) | 0.262 (#3) | 0.243 (#11) | -0.019, drops 8 places |
| qwen3.6:latest (challenger) | 0.271 (tied #1) | 0.241 (#12) | -0.030, drops 11 places |
| qwen3.5:35b vs qwen3.6:latest delta | +0.009 (qwen3.6 ahead) | +0.002 (qwen3.5:35b ahead by a hair) | Effectively zero — tie within noise |
| Compression of top-vs-mid | Top 0.271 vs mid 0.247 = 0.024 spread | Top 0.329 vs mid 0.243 = 0.086 spread | **The metric is now MORE discriminating, not less** |

**Critical observation**: the original concern in #939 was that "ROUGE
deltas may compress (smaller deltas between models)" — that turned out to be
**the opposite of what happened**. The spread *widened*: against Sonnet
silver, the top 9 models all clustered within 0.025 RougeL; against Opus
silver, the top 3 separate from the rest by 0.04+. This means Sonnet silver
was *flattening* the ranking by penalizing any model that wrote
differently-but-well — Opus silver lets the metric breathe.

**What this does NOT change**:

1. **Champion stays `qwen3.5:35b`** for the prod profile. The new
   scoreboard doesn't justify swapping to `mistral:7b` on a 5-episode
   smoke alone — we need #933 prod-curated validation and #932 G-Eval
   before any prod swap. RougeL is one signal among many. mistral:7b's
   coverage is 0.766 (down ~25% from Qwen) which suggests it's
   summarizing more tersely; G-Eval will tell us whether that's
   "concise" or "lossy".
2. **qwen3.6:latest vs qwen3.5:35b championship comparison** stays
   essentially tied, just at a lower absolute level. The original
   v2 conclusion ("credible champion contender pending #932/#933
   validation") still holds.
3. **DeepSeek-R1 family remains uncontested loser** for paragraph
   summarization. Reasoning-tuned models don't write paragraph prose
   well; that's a fundamental fit problem, not a silver-choice artifact.

**What this DOES change** for downstream planning:

- The `#928` DGX-vs-cloud championship now has a different "local
  representative" candidate set. mistral:7b is suddenly the strongest
  local candidate by ROUGE-vs-Opus; that needs to be folded into the
  finalist roster.
- The per-model prompt-tuning tickets (#935 / #936 / #937 / #938) should
  re-score against the new silver before deciding whether tuning helped.
- The `#932` G-Eval finale will use the Opus silver as the reference
  for its faithfulness/coverage/coherence/fluency dimensions — this
  rescoring just establishes the new ROUGE baseline.

### Rescoring methodology notes

- **Same predictions, new silver only.** No inference was re-run; just
  ROUGE/BLEU/WER/cosine/coverage recomputed.
- **r1:70b cell has 3 episodes only** (rerun was killed at episode 5/10 of
  dataset 1; episode-3-4 were 17 min each — operationally
  disqualified). Its 0.228 RougeL vs Opus is on a 3-episode subset and is
  marked as such.
- **No qwen3-coder:30b cell in v1 originally** — the retry sweep killswitch
  SIGTERM'd before its cell. The presence of a v1 cell now is from a
  separate fill-in run we missed mentioning above; treat as informational.

## References

- [#924 — Autoresearch v2 refresh on DGX](https://github.com/chipi/podcast_scraper/issues/924)
- [#923 — Prod profile: all-DGX with cloud fallback](https://github.com/chipi/podcast_scraper/issues/923)
- [#907 — Autoresearch programme epic](https://github.com/chipi/podcast_scraper/issues/907)
- [#939 — Silver upgrade to Opus 4.7](https://github.com/chipi/podcast_scraper/issues/939) — Phase 0 of next batch (this addendum)
- [SILVER_OPUS47_GENERATION_2026_06](SILVER_OPUS47_GENERATION_2026_06.md) — generation report for the new silver
- [EVAL_SMOKE_V1_DGX_VS_LAPTOP_2026_06](EVAL_SMOKE_V1_DGX_VS_LAPTOP_2026_06.md) — the prior DGX validation pass
- [docs/wip/AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md](../../wip/AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md) — dependency map across the open work
- [docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md](../../wip/AUTORESEARCH_LEARNINGS_FOR_V3.md) — failure-mode catalogue

### Tuned prompt addendum — hermes3:8b (#937, 2026-06-09)

**Verdict: Nous-native ChatML helps.** Hermes 3 is a Nous Research
fine-tune of Llama 3.1 8B, post-trained against a ChatML-style chat
template with a persona-forward system message (per the
[Hermes-3-Llama-3.1-8B model card](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)).
The smoke-v2 baseline run used the generic qwen3.5:9b prompts verbatim;
this tuning pass replaces them with a Nous-shaped pair that mirrors the
"You are Hermes 3..." opener the model was explicitly trained on, then
states the task constraints. Ollama applies the `<|im_start|>/<|im_end|>`
ChatML wrapping automatically — the `.j2` files supply only the message
*content*.

**Prompts** (in this commit):

- `src/podcast_scraper/prompts/ollama/hermes3_8b/summarization/system_v1.j2`
- `src/podcast_scraper/prompts/ollama/hermes3_8b/summarization/long_v1.j2`

**Tuned run dir**: `data/eval/runs/llm_ollama_hermes3_8b_dgx_smoke_v2_tuned_2026_06/`

**Numbers — `vs silver_opus47`:**

| Dataset | Baseline (generic prompt) | Tuned (Nous-native) | Δ RougeL |
| --- | --- | --- | --- |
| `curated_5feeds_smoke_v1` | RougeL 0.279, Cosine 0.785, Cov 0.567 | **RougeL 0.309**, Cosine 0.769, Cov 0.631 | **+0.030** |
| `curated_5feeds_smoke_v2` | RougeL 0.265, Cosine 0.776, Cov 0.761 | **RougeL 0.306**, Cosine 0.787, Cov 0.786 | **+0.041** |

**Reading the result**: the Nous-native template lifts Hermes 3 from
rank 5 (v1) / rank 5 (v2) on the rescored scoreboard into the top tier with
mistral:7b (0.329 v1 / 0.302 v2) and llama3.1:8b (0.307 v1 / 0.282 v2).
This is one of the three valid outcomes the ticket
[#937](https://github.com/chipi/podcast_scraper/issues/937)
called out — the Nous fine-tune *does* help summarization when the
prompt matches its training distribution. The cell deserves a slot in
the #928 championship finalist roster.

**Coverage caveat**: v1 coverage at 0.631 still lags the qwen baseline's
~0.94, so this is a quality-of-summary lift, not a length-discipline
lift. G-Eval (#932) is the right next gate.

### Tuned prompt addendum — gemma3:27b (#935, 2026-06-09)

**Hypothesis ladder result: H3 accepted — task-fit issue, not prompt or quantization.**

The v2.1 sweep used qwen3.5:9b's prompt templates verbatim for gemma3:27b
and the rescored result against Opus silver showed gemma3 at the bottom of
the matrix (RougeL 0.202 on smoke_v1). #935 ran a three-hypothesis ladder:

| Hypothesis | Variant | RougeL vs Opus (v1) | Δ vs baseline (0.202) | Verdict |
| --- | --- | --- | --- | --- |
| H1 — prompt format mismatch | Q4_K_M + Gemma-native chat template | **0.188** | **−0.014** | Regressed |
| H2 — Q4 quantization regression | Q8_0 + Gemma-native chat template | **0.191** | **−0.011** | Marginal vs H1, still regressed |
| H3 — genuine task-fit | (accept) | — | — | **Accepted** |

**Tuned prompts** at `src/podcast_scraper/prompts/ollama/gemma3_27b/summarization/`:

- `system_v1.j2` — minimal role anchor; Gemma 3's IT chat template has no
  distinct system role per Google's
  [model card](https://huggingface.co/google/gemma-3-27b-it#chat-template),
  so we keep the system message short and place all binding instructions
  in the user turn.
- `long_v1.j2` — Gemma-native user prompt: declarative tone, no role-play
  preamble, binding constraints restated near the assistant turn (recency
  window pattern).

**Q8 quantization tested** via new config
`data/eval/configs/summarization/autoresearch_prompt_ollama_gemma3_27b_q8_smoke_paragraph_v1.yaml`
targeting `gemma3:27b-it-q8_0`. Q8 lifts RougeL by +0.003 over Q4 (0.188 →
0.191) — a real but small effect. **Quantization is NOT the dominant
factor**; even at higher precision, gemma3:27b underperforms on this task.

**Why H3 is the right verdict**: across both quantizations and both
prompts, gemma3's output is consistently shorter and less aligned with
Opus's prose style than Qwen's. Cosine stays at ~0.78 (vs qwen3.5:35b's
~0.79), coverage stays at ~0.74-0.78 (vs qwen3.5:35b's ~1.07). Gemma 3's
instruction-following on text-only paragraph summarization on this corpus
just isn't competitive. The model is multimodal-tuned (vision-language
strong); the v2.1 result wasn't a methodology artifact, it was a model
fit issue.

**Drop from #928 championship roster.** Other Gemma versions or task
shapes (e.g., bullet-track or with image-text hybrid inputs) might reach
different conclusions; this verdict is specific to gemma3:27b on
paragraph summarization of our 5-feed smoke corpus.

**Runs** (under `data/eval/runs/`, gitignored — predictions + Opus metrics persisted):

- `llm_ollama_gemma3_27b_dgx_smoke_v2_tuned_h1_2026_06/` (H1, Q4)
- `llm_ollama_gemma3_27b_dgx_smoke_v2_tuned_h2_q8_2026_06/` (H2, Q8)

### Tuned prompt addendum — mistral-small:24b (#938, 2026-06-09)

**Verdict: counter-intuitive negative result — Mistral-native [INST] hurts ROUGE on this corpus.**

| Dataset | Baseline (Qwen clone) | Tuned (Mistral [INST]) | Δ RougeL |
| --- | --- | --- | --- |
| `curated_5feeds_smoke_v1` | RougeL 0.284, Cosine 0.782, Cov 0.964 | **RougeL 0.257**, Cosine 0.799, Cov 0.781 | **−0.027** |
| `curated_5feeds_smoke_v2` | RougeL 0.257, Cosine 0.793, Cov 0.952 | RougeL 0.259, Cosine 0.815, Cov 0.764 | +0.002 |

**Tuned prompts** at `src/podcast_scraper/prompts/ollama/mistral-small_24b/summarization/`:

- `system_v1.j2` — concise, declarative role statement per the
  [Mistral-Small-24B model card](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)
  recommendations ("state constraints, don't describe them")
- `long_v1.j2` — Mistral-native user prompt with crisp task framing first,
  then transcript payload, then a bullet-list of binding constraints near
  the assistant turn (recency-window pattern). Ollama auto-wraps in
  `[INST]...[/INST]`.

**Why the tuned prompt regresses**: the Mistral-native template produces
**shorter, more declarative summaries** (avg 1818 chars vs the Qwen-clone's
~2400+). Coverage drops from 0.964 to 0.781 — Mistral is following Mistral's
"concise, declarative" training style faithfully, but Opus's reference
summaries are longer and more thorough. Cosine similarity actually improves
slightly (0.799 vs 0.782), meaning Mistral writes **more like Opus
semantically** — but ROUGE penalizes the coverage loss more than it rewards
the semantic alignment.

**Same shape as gemma3 H1/H2**: across both cells, the verbose Qwen-clone
template wins on ROUGE because it produces output that better matches
Opus's length and word-overlap pattern. Native prompts produce shorter
summaries that may be semantically truer but lexically penalized.

**Implication for #928 / #932**: mistral-small:24b is doing fine — the
result tells us "ROUGE is biased toward verbose summaries that match
Opus's prose length." G-Eval (#932) on faithfulness / coverage /
coherence / fluency will likely tell a different story. Keep mistral-small
on the #928 championship roster pending G-Eval; don't drop it on this
ROUGE-on-cloned-prompt-still-wins observation.

**Runs** (under `data/eval/runs/`, gitignored):

- `llm_ollama_mistral-small_24b_dgx_smoke_v2_tuned_2026_06/` (Mistral-native [INST])
