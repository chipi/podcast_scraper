# Session status — #1033 / #1034 / #1035 / #116 / #113 (2026-06-20)

## Done this session

Branch `feat/autoresearch-followups-2026-06-18` — 5 unpushed commits:

| Commit | Subject |
|---|---|
| `c1109272` | feat(eval): #1033 step 2 — corrected cohort scoreboard (7/7 candidates) under provider pipeline |
| `01863d03` | docs(eval): #1033 — correct #1016 + #1022 addenda + registry headline_metric (Cell F no longer "GI cohort #1") |
| `7135b618` | feat(#1034) chunk 1 — remove summary_bullets value from Literal + flip default |
| `1750dcbf` | feat(#1034) chunk 2 — delete bullet-derived KG/GI dispatch arms |
| `6b03d945` | feat(#1034) chunk 3 — delete provider bullets methods + corpus-side artifacts |

**Tests**: 1294 unit pass; 1244 integration pass (2 pre-existing Gemini Client mock signature failures unrelated to #1033/#1034 — confirmed failing on parent commit `01863d03` before any chunk).

**GH activity**:
- #1033 corrected scoreboard committed; awaiting close authorization from operator
- #1034 — all 3 chunks committed locally; awaiting close authorization
- **#1035 filed**: https://github.com/chipi/podcast_scraper/issues/1035 — "Add NER pre-pass for KG entity extraction (Pattern B follow-up to #1033)"

## Paused — blocked on operator authorization

### #116 — Cell C re-baseline against Cell F NVFP4 (30B vs 35B drift)

**Scope**: Run Cell C (Ollama Qwen3.5-35b, #928 local DGX winner) through the corrected `provider`-source pipeline. Compare to Cell F NVFP4 (vLLM Qwen3-30B-A3B-NVFP4, current daily-driver) on the same dev_v1 + benchmark_v2 fixtures.

**What's already in hand**:
- vLLM-bf16 Qwen3.5-35B-A3B vs Cell F NVFP4: measured in #1033 rerun (GI 0.618 vs 0.595; KG topic 50% vs 45%)
- Ollama-vs-vLLM drift for Qwen3.5-35B-A3B alone: measured in task #100 (apples-to-apples)

**What's missing**:
- Cell C-on-Ollama vs Cell F-on-vLLM under the corrected `provider` pipeline. Specifically:
  - Does Ollama Qwen3.5-35b still beat NVFP4 Qwen3-30B-A3B on the corrected pipeline?
  - Is the "Ollama tax" smaller or larger than the "NVFP4 quant tax"?

**Blocker — explicit operator authorization needed**:

1. **Ollama daemon start** — per `feedback_ollama_manual_start.md`: user starts Ollama; agent must not `ollama serve`. Cell C runs on Ollama.

2. **DGX serving-mode swap** — autoresearch slot currently serves Cell F NVFP4 via vLLM. Running Cell C in parallel means either:
   - Operator authorizes a compose stop on vLLM, then runs the Cell C sweep via Ollama. Re-starts vLLM after.
   - OR Cell C runs on a separate port without touching vLLM's slot, and the eval harness is pointed at the Ollama endpoint for the duration.

3. **Compose changes outside the podcast_scraper repo** — `infra/vllm/autoresearch/docker-compose.yml` lives in the `agentic-ai-homelab` repo; I do not edit there per `feedback_never_touch_infra.md`.

**Estimated wall-clock**: ~1.5 h once Ollama is up (similar profile to a single #1033 candidate stage pair: ~440s for GI + KG on the dev_v1 fixture, plus warm-up + scoring + repeat on benchmark_v2 for held-out validation).

**Proposed approach when operator returns**:
- Operator starts Ollama on DGX with Qwen3.5:35b cached
- Operator picks: stop vLLM during Cell C sweep, or run Cell C on a separate port
- I write a #116-specific runner script under `autoresearch/116_cell_c_rebaseline/`
- Reuse the #1033 rerun harness (`autoresearch/1033_cohort_rerun/run_sweep.sh`) with a Cell-C-specific candidate entry
- Score against the same `silver_opus47_*_dev_v1` silvers used for cohort
- Land verdict in `docs/wip/EVAL_116_CELL_C_VS_CELL_F.md`

### #113 — Small-model standoff (9B-class vs top dog vs safe pick)

**Scope**: Add 9B-class candidates (Qwen3.5:9b being the prime candidate) and re-baseline against the corrected-pipeline cohort. Outcome: a per-stage routing table that includes the 9B tier (e.g. summary-quality vs latency-quality vs cost-quality decision points).

**Status**: corrected-pipeline harness is ready. Same blocker pattern — most 9B candidates likely Ollama; need operator to start Ollama and authorize vLLM swap or co-residence on DGX.

**Estimated wall-clock**: ~3 h end-to-end (2 candidates × 2 stages × dev_v1 + benchmark_v2 held-out).

## Branch policy

Per `feedback_never_push_early`: no push. When operator returns and confirms, push `feat/autoresearch-followups-2026-06-18` and decide PR scope (single #1033 + #1034 + corrected scoreboard PR, or split per-issue).

Per `feedback_make_docs_before_push`: run `make docs` before push given the WIP markdown additions in this session.

## Open WIP doc list (this session)

- `docs/wip/EVAL_1033_COHORT_RERUN_2026-06-19.md` — corrected scoreboard
- `docs/wip/EVAL_1016_FINAL_REPORT_2026_06_17.md` § 11 — addendum
- `docs/wip/VLLM_GB10_TUNING_VALIDATION_2026-06-18.md` — #1033 corrected-pipeline addendum
- `docs/wip/SESSION_2026-06-20_1033_FOLLOWUPS_STATUS.md` — this file

Untracked (not part of this session): `docs/wip/LLM-FUNDAMENTALS-READING-LIST.md`.
