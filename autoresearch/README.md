# Autoresearch

Thin automation layer on top of `scripts/eval/run_experiment.py` and
`src/podcast_scraper/evaluation/`.

**Current framework: v2** — see [RFC-073](../docs/rfc/RFC-073-autoresearch-v2-framework.md).
**Original framework: v1** — see [RFC-057](../docs/rfc/RFC-057-autoresearch-optimization-loop.md)
(closed via [ADR-073](../docs/adr/ADR-073-rfc057-autoresearch-closure.md); Track B still v1).

**Active programme:** [#907 — Autoresearch programme epic](https://github.com/chipi/podcast_scraper/issues/907).
v2-driven tuning programme covering entity canonicalization, cleaning, CIL,
topic clusters, NER, Whisper, prompts, and **reliability under sustained
load**. All 7 children closed as of 2026-06-08 — see
[Eval reports](#eval-reports-907-programme) below.

## v2 headline: dev + held-out, not smoke + benchmark

| Tier | Dataset | Size | Role | Touch during iteration? |
| ---- | ------- | ---- | ---- | ----------------------- |
| **Dev** | `curated_5feeds_dev_v1` | 10 ep (e01+e02) | Ratchet iteration | Yes (every experiment) |
| **Held-out** | `curated_5feeds_benchmark_v2` | 5 ep (e03, ~32 min) | Champion validation | **Never** — once per committed champion |

`curated_5feeds_smoke_v1` and `curated_5feeds_benchmark_v1` are preserved **as-is** for reproducibility of
v1 runs, baselines, and silvers. Do not modify. New work uses the v2 datasets.

See [JUDGING.md](JUDGING.md) for the dual-judge system (rubric, fraction-based contestation, prose
extraction, seed plumbing).

---

## Current champions (OpenAI v2, held-out scores)

| Track | Approach | Config | Silver | Held-out final | ROUGE-L |
| ----- | -------- | ------ | ------ | -------------- | ------- |
| Bullets | **Non-bundled** (winner) | `autoresearch_prompt_openai_benchmark_bullets_v2.yaml` | `silver_sonnet46_benchmark_v2_bullets` | **0.566** | 39.6% |
| Bullets | Bundled | `autoresearch_prompt_openai_bundled_benchmark_bullets_v2.yaml` | `silver_sonnet46_benchmark_v2_bullets` | 0.505 | 33.2% |
| Paragraph | **Non-bundled** (winner) | `autoresearch_prompt_openai_benchmark_paragraph_v2.yaml` | `silver_sonnet46_benchmark_v2_paragraph` | **0.481** | 31.7% |
| Paragraph | Bundled | `autoresearch_prompt_openai_bundled_benchmark_paragraph_v2.yaml` | `silver_sonnet46_benchmark_v2_paragraph` | 0.469 | 29.5% |

Full comparison: [`openai_v2_comparison_2026-04-14.md`](openai_v2_comparison_2026-04-14.md).

---

## Running the ratchet (v2)

### Dev iteration (the normal loop)

Run the orchestrator directly with explicit `CONFIG` and `REFERENCE`:

```bash
# Bundled bullets ratchet (gpt-4o, 10 ep dev, scoring_output_field: bullets)
make autoresearch-score-bundled \
  CONFIG=data/eval/configs/summarization_bullets/autoresearch_prompt_openai_bundled_dev_bullets_v2.yaml \
  REFERENCE=silver_sonnet46_dev_v1_bullets

# Non-bundled bullets ratchet (gpt-4o, 10 ep dev)
AUTORESEARCH_EVAL_N=10 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization_bullets/autoresearch_prompt_openai_dev_bullets_v2.yaml \
  --reference silver_sonnet46_dev_v1_bullets

# Bundled paragraph ratchet
AUTORESEARCH_EVAL_N=10 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization/autoresearch_prompt_openai_bundled_dev_paragraph_v2.yaml \
  --reference silver_sonnet46_dev_v1_paragraph

# Non-bundled paragraph ratchet
AUTORESEARCH_EVAL_N=10 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization/autoresearch_prompt_openai_dev_paragraph_v2.yaml \
  --reference silver_sonnet46_dev_v1_paragraph
```

Output: one `final` scalar on stdout (higher is better); `rougeL_f1`, `judge_mean`,
`contested`, per-episode contestation count on stderr.

**Accept rule** (dev): `+1%` delta vs the last committed champion. For prompts shared between
tracks (bundled system+user cover both bullets and paragraph), apply dual-metric:
target track `+1%` AND the other track `−1%` or better.

### Held-out validation (after accepting a champion)

Run **once** per committed champion — never iterate against held-out:

```bash
# Bundled bullets held-out
AUTORESEARCH_EVAL_N=5 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization_bullets/autoresearch_prompt_openai_bundled_benchmark_bullets_v2.yaml \
  --reference silver_sonnet46_benchmark_v2_bullets

# Non-bundled paragraph held-out (example)
AUTORESEARCH_EVAL_N=5 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization/autoresearch_prompt_openai_benchmark_paragraph_v2.yaml \
  --reference silver_sonnet46_benchmark_v2_paragraph
```

If held-out score is within `dev ± noise floor` (~±3-5%), the champion generalises — commit it.
If held-out collapses, the change was overfitting dev — revert.

### Dry re-score (ROUGE only, no judge API calls)

Useful to re-check ROUGE from existing `predictions.jsonl` without re-running summarization:

```bash
make autoresearch-score-bundled \
  CONFIG=... REFERENCE=... DRY_RUN=1
```

Not sufficient to evaluate a *new* prompt — judges don't run. Run the full harness after template changes.

---

## Allowlisted files for prompt tuning

### Bundled mode

| File | Role |
| ---- | ---- |
| `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_system_v1.j2` | Bundled system prompt (OpenAI-specific) |
| `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_user_v1.j2` | Bundled user prompt (OpenAI-specific) |

### Non-bundled mode

| File | Role |
| ---- | ---- |
| `src/podcast_scraper/prompts/shared/summarization/system_bullets_v1.j2` | Shared bullets system prompt |
| `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2` | Shared bullets user prompt |
| `src/podcast_scraper/prompts/openai/summarization/system_v1.j2` | OpenAI paragraph system prompt |
| `src/podcast_scraper/prompts/openai/summarization/long_v1.j2` | OpenAI paragraph user prompt |

## Files you must NOT edit during a ratchet

| Path | Reason |
| ---- | ------ |
| `autoresearch/bundled_prompt_tuning/eval/score.py` | Immutable harness |
| `autoresearch/bundled_prompt_tuning/eval/rubric.md` | Immutable during a run |
| `autoresearch/bundled_prompt_tuning/eval/judge_config.yaml` | Pinned judge models (human-only between rounds) |
| `data/eval/**` | Gold inputs, references, datasets — read-only |
| Benchmark v2 configs (`*_benchmark_bullets_v2.yaml`, `*_benchmark_paragraph_v2.yaml`) | Held-out — run once per champion only |

---

## Environment

The score CLI loads `.env` then `.env.autoresearch` (optional overrides) from the repo root. Required
for a full run:

- `AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY` — OpenAI for `run_experiment` summarization
- `AUTORESEARCH_JUDGE_OPENAI_API_KEY` — judge A
- `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY` — judge B
- Optional: `AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1` — fall back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`

All v2 ratchet configs set `params.seed: 42` — OpenAI's approximately-deterministic sampling. See
[JUDGING.md](JUDGING.md) §Seed & determinism for the empirical characterisation.

---

## Prerequisites

- Materialized transcripts (both datasets):
  - `data/eval/materialized/curated_5feeds_dev_v1/` (10 ep) — `make dataset-materialize DATASET_ID=curated_5feeds_dev_v1`
  - `data/eval/materialized/curated_5feeds_benchmark_v2/` (5 ep) — `make dataset-materialize DATASET_ID=curated_5feeds_benchmark_v2`
- Silver references:
  - `silver_sonnet46_dev_v1_bullets`, `silver_sonnet46_dev_v1_paragraph`
  - `silver_sonnet46_benchmark_v2_bullets`, `silver_sonnet46_benchmark_v2_paragraph`
- All seeded automatically by `silver_candidate_sonnet46_<dataset>_<track>.yaml` + `promote_run.py`.

---

## Git workflow

1. Use a dedicated branch, e.g. `autoresearch/<short-tag>` — not `main`.
2. Before each experiment: state a one-sentence hypothesis (in chat or results TSV notes).
3. After each ratchet run:
   - **Improved** vs last committed champion: `git add` only allowlisted files + results TSV →
     commit with `[autoresearch-<track>] <exp-id>: <hypothesis> (+X%)`.
   - **Not improved**: `git checkout HEAD -- <template paths>`.
4. Append one row per experiment to the relevant results TSV.
5. After accepting a champion, run held-out validation and record the held-out score in the TSV
   as a separate `held-out` row.

### Results TSVs

| Track | File |
| ----- | ---- |
| Bundled bullets (dev v2) | `autoresearch/bundled_prompt_tuning/results/results_openai_r1.tsv` (rolling) |
| Bundled paragraph (dev v2) | `autoresearch/bundled_prompt_tuning/results/results_openai_paragraph_r1.tsv` |
| Non-bundled bullets (dev v2) | `autoresearch/prompt_tuning/results/results_openai_nonbundled_bullets_dev_v2.tsv` |
| Non-bundled paragraph (dev v2) | `autoresearch/prompt_tuning/results/results_openai_nonbundled_paragraph_dev_v2.tsv` |

TSV columns: `experiment_id  score  delta  status  notes  judge_a_model  judge_b_model  rubric_hash  eval_dataset_ref`.

---

## Next work (deferred)

1. **Multi-run averaging** (N=3) to eliminate OpenAI API non-determinism as a signal noise source. See [RFC-073](../docs/rfc/RFC-073-autoresearch-v2-framework.md) §Future Work.
2. **Other providers** — Anthropic, Gemini, Mistral under v2. Port OpenAI champion prompts as starting template, re-use Sonnet 4.6 silvers.
3. **Grow held-out dataset** from 5 episodes if champion discrimination becomes the limiting factor.

For any of the above, start by reading [RFC-073](../docs/rfc/RFC-073-autoresearch-v2-framework.md) §Replication for other providers.

---

## Eval reports (#907 programme)

The #907 autoresearch programme contributed 6 new eval reports (plus 7 new
scoring scripts) covering quality, cost, latency, and — newly — **reliability
under sustained load**. All reports live under `docs/guides/eval-reports/`.

| Child | Report | Outcome |
| ----- | ------ | ------- |
| [#853](https://github.com/chipi/podcast_scraper/issues/853) | [EVAL_ENTITY_CANON_2026_06_08.md](../docs/guides/eval-reports/EVAL_ENTITY_CANON_2026_06_08.md) | Token/overall ratio thresholds tuned (0.78→0.65, 0.85→0.70); nickname + title-prefix awareness shipped |
| [#594](https://github.com/chipi/podcast_scraper/issues/594) | [EVAL_CLEANING_AUTORESEARCH_2026_06_08.md](../docs/guides/eval-reports/EVAL_CLEANING_AUTORESEARCH_2026_06_08.md) | Anthropic + Gemini cleaning temp 0.2 → 0.4 shipped; documented gpt-4o cleaning regression vs gpt-4o-mini |
| [#904](https://github.com/chipi/podcast_scraper/issues/904) | [EVAL_FIXTURES_V2_TIER1_TUNING_2026_06_08.md](../docs/guides/eval-reports/EVAL_FIXTURES_V2_TIER1_TUNING_2026_06_08.md) | CIL predicate redesign +18pp recall; sponsor + topic-cluster threshold sweeps |
| [#905](https://github.com/chipi/podcast_scraper/issues/905) | [EVAL_FIXTURES_V2_TIER2_TUNING_2026_06_08.md](../docs/guides/eval-reports/EVAL_FIXTURES_V2_TIER2_TUNING_2026_06_08.md) | Cleaning profile sweep + MAP-REDUCE chunking behavior; current `cleaning_v4` default validated as judge-suboptimal vs `cleaning_v3` (not shipped due to hardcoded fallbacks — follow-up) |
| [#906](https://github.com/chipi/podcast_scraper/issues/906) | [EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md](../docs/guides/eval-reports/EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md) | NER (`en_core_web_trf` +13pp recall vs `_sm`, not shipped pending install verification); Whisper accent WER (`base.en` prod default validated, 2.8× more accurate than `tiny.en`); **`long_v2.j2` Anthropic prompt shipped (5-0 sweep)** |
| [#816](https://github.com/chipi/podcast_scraper/issues/816) | [EVAL_SUMMARY_MODEL_RELIABILITY_2026_06_08.md](../docs/guides/eval-reports/EVAL_SUMMARY_MODEL_RELIABILITY_2026_06_08.md) | Reliability axis added (success-rate floor, effective $/successful-call, p50/p95 under load). 4-candidate panel re-ranked; **`gemini-2.5-flash-lite` kept** by 4-10× cost dominance |

## Eval reports (#927 — DGX-vs-cloud programme)

The #927 epic synthesized DGX vs cloud across the four pipeline stages.
All four children resolved into the `cloud_with_dgx_primary` profile
already shipping in `config/profiles/`. Routing-decision PROD_RUNBOOK
entry: §"Provider model selection — DGX vs cloud per stage".

| Child | Report | Outcome |
| ----- | ------ | ------- |
| [#928](https://github.com/chipi/podcast_scraper/issues/928) | [EVAL_SUMMARY_DGX_LOCAL_2026_06.md](../docs/guides/eval-reports/EVAL_SUMMARY_DGX_LOCAL_2026_06.md) | Cell C: vLLM-served Qwen3.6-35B-A3B ties Ollama qwen3.5:35b within scoring noise. **Ollama kept** as `cloud_with_dgx_*` summary (operationally simpler) |
| [#929](https://github.com/chipi/podcast_scraper/issues/929) | [EVAL_TRANSCRIPTION_3WAY_2026_06.md](../docs/guides/eval-reports/EVAL_TRANSCRIPTION_3WAY_2026_06.md), [EVAL_WHISPER_CONTENTION_2026_06.md](../docs/guides/eval-reports/EVAL_WHISPER_CONTENTION_2026_06.md) | whisper-openai on `dgx:8002` (post-#929 temperature fix) matches MPS within noise at ~3× speed. **Transcription routed to DGX** for DGX-equipped profiles. Contention re-tests (#963, 2026-06-11 + 2026-06-14): mean WER stable, but active vLLM serving can trigger catastrophic single-episode failure (operator-gated rule) |
| [#930](https://github.com/chipi/podcast_scraper/issues/930) | [EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06.md](../docs/guides/eval-reports/EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06.md) | pyannote on DGX ties MPS within noise (~13× realtime). **Diarization routed to DGX** for DGX-equipped profiles. Gemini speaker-detector 3-way completion deferred to follow-up |
| [#931](https://github.com/chipi/podcast_scraper/issues/931) | [EVAL_HYBRID_ROUTING_2026_06.md](../docs/guides/eval-reports/EVAL_HYBRID_ROUTING_2026_06.md) | Hybrid synthesis: `cloud_with_dgx_primary` = whisper+diarize on DGX, summary+speaker-detector on Gemini. PROD_RUNBOOK entry documents per-stage rationale + the load-bearing operator gate (no sweep-vs-transcription overlap) |

### Reliability axis (#816) — methodology change

Summary-model autoresearch now measures reliability as a hard floor, not
a tiebreaker. New metrics:

- **`success_rate_pct`** at the eval-scale operating point — hard floor (default ≥95)
- **`cost_usd_per_successful_call`** — replaces nameplate $/call (same when clean, meaningfully different under load)
- **`latency_p50_s` / `latency_p95_s` under sustained burst** — replaces single-call latency

Harness: `scripts/eval/score/summary_model_reliability_v1.py`
Evidence dir: `autoresearch/data/reliability_evidence/`

The harness measures provider-side behavior (SDK call → success/failure)
before the application-level circuit breaker (#697). Composite ranking is
reliability-floor-first, then cost, then latency; operators can re-weight
at evaluation time.

### Rolling notes → v3 fixtures (#921)

Each #907 child appends failure-mode learnings to
[`docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md`](../docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md).
The v3 fixtures rebuild (#921) pulls from there so v3 simulates real-prod
defects directly in the fixture corpus instead of relying on prod
encounters.
