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
- **No v2.1 candidates** — gemma3, phi4, hermes3, etc. are queued for a separate v2.1 sweep (#44/#45) to land after this report ships.

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
| #44 / #45 | v2.1 sweep prep + run | Unblocked — DGX free, candidate list informed by what's missing here |
| #932 | G-Eval finale | When implemented, runs against these same 19 cells |
| #933 | Prod-curated validation | When curated, the chosen champion gets a sanity-check pass |

## References

- [#924 — Autoresearch v2 refresh on DGX](https://github.com/chipi/podcast_scraper/issues/924)
- [#923 — Prod profile: all-DGX with cloud fallback](https://github.com/chipi/podcast_scraper/issues/923)
- [#907 — Autoresearch programme epic](https://github.com/chipi/podcast_scraper/issues/907)
- [EVAL_SMOKE_V1_DGX_VS_LAPTOP_2026_06](EVAL_SMOKE_V1_DGX_VS_LAPTOP_2026_06.md) — the prior DGX validation pass
- [docs/wip/AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md](../../wip/AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md) — dependency map across the open work
- [docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md](../../wip/AUTORESEARCH_LEARNINGS_FOR_V3.md) — failure-mode catalogue
