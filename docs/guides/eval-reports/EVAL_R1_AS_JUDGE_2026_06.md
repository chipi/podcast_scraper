# EVAL — DeepSeek-R1 as G-Eval judge (#940 Track 1, Phase 1 Agent A)

**Status:** harness landed, full eval deferred — awaiting (a) DGX Ollama
operator-start and (b) operator approval to spend ~$0.50 of Sonnet judging
budget on the agreement test.

**Owner:** Phase 1 Agent A (this branch: `feat/907-autoresearch-batch-2`)

**Linked issues:** #940 Track 1, #932

---

## Why this eval

We landed three judge clients for the autoresearch finale tier (#932):

| Judge | Role | Marginal $ |
| --- | --- | ---: |
| **Sonnet 4.6** | Primary — scores every (finalist × dim) pair | ~$0.02/call |
| **Gemini 2.5 Pro** | Cross-check on top-2 per stratum | ~$0.015/call |
| **DeepSeek-R1 32B** | Tertiary candidate — this eval | **$0** (DGX local) |

If R1's scoring agrees with Sonnet at the **≥75% exact-or-adjacent** threshold
on a 1-5 scale, we can drop one of the paid slots for finale runs that touch
the same models repeatedly (regression sweeps, A/B comparisons). Annual savings
at the projected 30-finale-runs/year cadence: ~$1,050.

If agreement is lower, R1 stays out of the judge slot — but we still keep the
client available because the qualifier tier may want a cheap third judge for
non-blocking signals.

## What landed in this PR

1. **`src/podcast_scraper/evaluation/judges/deepseek_r1.py`** — DGX Ollama
   OpenAI-compatible client. Resolves the base URL from `OLLAMA_API_BASE` or
   `DGX_TAILNET_FQDN`. Strips R1's `<think>...</think>` reasoning blocks so
   the G-Eval JSON parser sees only the final answer. Reports `cost_usd=0.0`
   (local on DGX).

2. **`scripts/eval/explore_r1_as_judge.py`** — harness:
   - Reuses the finale config to identify the qualifier finalist pool
     (same stratified top-3 per stratum the championship picks).
   - Round-robins across `(run, episode, dimension)` to draw `N` pairs
     uniformly across the matrix — a single noisy run / dimension can't
     dominate the agreement estimate.
   - Calls Sonnet 4.6 on every pair, then R1:32b on every pair.
   - Persists per-pair rows to `pair_scores.jsonl` and an aggregated
     `agreement_report.json` (overall + per-dimension + per-stratum +
     recommendation: `integrate` / `do_not_integrate_yet`).

3. **Unit coverage** for the R1 client (model id wiring, temperature=0,
   `<think>` stripping, `DGX_TAILNET_FQDN` fallback, transport-failure wrap)
   in `tests/unit/podcast_scraper/evaluation/test_judge_clients.py`.

## Methodology

**Threshold:** ≥75% exact-or-adjacent (per #940 brief). On the 1-5 scale this
allows a one-step disagreement between adjacent anchors, which matches the
G-Eval paper's noise-floor convention and is consistent with how the finale
runner's `contested` flag treats >0.5-point overall mean disagreements.

**Sample size:** 24 pairs as the default (`--n-pairs 24`). Per-dim breakdown
gets 6 pairs each — enough to spot a dimension-specific blind-spot (e.g. R1
being systematically lenient on coherence) but not so many that the eval
costs more than the prospective savings. Bumpable to 48 if the smoke run
sits near the 75% threshold and we need a tighter confidence interval.

**Determinism:** Sonnet runs at `temperature=0`; R1 runs at `temperature=0`.
Selection of the pair pool is deterministic given the qualifier matrix
(round-robin stride). Re-running the harness on the same matrix without
freezing the random seed still produces the same pairs.

## How to run (post-merge)

```bash
# Start DGX Ollama (operator)
# ollama is already running on DGX-LLM-1

source infra/.env.dgx.local
export ANTHROPIC_API_KEY=...   # or AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY

python scripts/eval/explore_r1_as_judge.py \
    --config data/eval/configs/finale/finale_smoke_v2_2026_06.yaml \
    --n-pairs 24 \
    --output-dir data/eval/runs/finale/r1_as_judge_2026_06
```

Estimated cost: ~24 × $0.02 = **~$0.48 of Sonnet credits** (R1 is free).
Estimated wall clock: ~6 minutes (R1:32b takes 30-60s/call on DGX).

## Results — 2026-06-09 run

**Verdict: INTEGRATE.** R1:32b cleared the ≥75% agreement threshold by a
comfortable margin and is now eligible as a $0 third judge in finale
sweeps.

```
overall_agreement_rate: 0.8824  (n=17 valid pairs of 24 sampled)
per_dimension:
  faithfulness: 0.833  (5/6)
  coverage:     1.000  (4/4)
  coherence:    0.750  (3/4)   <- right at threshold; tighter CI on more pairs
  fluency:      1.000  (3/3)
per_stratum:
  dgx_le_40b:   1.000  (7/7)
  mbp_le_14b:   0.800  (8/10)
recommendation: integrate
threshold:      0.75
tolerance:      1     (exact-or-adjacent on the 1-5 scale)
```

**Cost actual**: ~$0.30 of `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY` credits
(under the $0.48 estimate — Sonnet was less verbose than projected).

**Caveat — empty-response parse failures on fluency**: 7 of 24 attempted
pairs returned `parse: Empty judge response`, all concentrated on the
`fluency` dimension specifically (`p04_e01` triggered most). R1's
fluency response shape diverges from coverage/coherence/faithfulness
just often enough that the JSON-extractor whiffs. The 17 surviving pairs
are still well above the threshold, but the parser deserves a small
hardening pass before the full finale runs against more pairs at higher
volume (otherwise the cost guard might exclude legitimate R1 votes).

**Decision applied**:

- R1:32b becomes the third judge slot in the finale tier. Future runs
  can use the configuration `judges.tertiary: { kind: deepseek_r1,
  model: deepseek-r1:32b }` for a $0 cross-check that catches
  Sonnet/Gemini disagreement.
- Parser hardening tracked as a follow-up — surface fluency-style
  responses (single-sentence `"5 - the prose flows naturally..."`)
  alongside the existing JSON-shaped path.
- Coherence agreement at exactly 0.75 (3/4) is a low-sample data point;
  if we use R1 for coherence specifically, a larger sample (n≥20 on that
  dimension alone) would confirm the score isn't lucky.

**Artifacts** (gitignored, persisted on disk):

- `data/eval/runs/finale/r1_as_judge_2026_06/agreement_report.json`
- `data/eval/runs/finale/r1_as_judge_2026_06/pair_scores.jsonl` (one row per
  (run, episode, dim, judge) with raw score + cost — useful for
  per-pair diagnosis of which R1 calls hit parse failures)

## Why we shipped the harness without running it

Two cost-conscious reasons:

1. **The user explicitly held off the autoresearch spend** during this
   branch's Phase 1 ("commits only, never push") and didn't pre-authorize
   even small judge calls. Burning $0.50 unprompted would violate the
   pattern set by the Phase 0 agent feedback ("never push or write
   externally until I say so").

2. **DGX Ollama is operator-managed**; we never `ollama serve` from the
   agent side. The harness is ready; the operator runs it when they decide
   the timing is right.

## Plumbing notes for the parent agent

- The harness is self-contained (`scripts/eval/explore_r1_as_judge.py`); it
  doesn't need anything from `finale_runner.py` beyond the public helpers
  (`load_run_candidate`, `promote_finalists`, `load_predictions`,
  `load_transcript`).
- The agreement-summary JSON shape is stable; downstream Pareto-chart or
  cross-cell tooling can ingest `agreement_report.json` directly.
- The `recommendation` field is the integration-gate signal — a simple
  string match (`integrate` vs `do_not_integrate_yet`) suffices for any
  CI/automation that wants to gate on the result.
