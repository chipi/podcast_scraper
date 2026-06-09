# EVAL — Autoresearch Finale Tier Methodology (#932)

**Branch:** `feat/907-autoresearch-batch-2`
**Status:** harness landed; first end-to-end finale run gated on operator approval to spend.
**Companion docs:**
- `docs/guides/eval-reports/EVAL_SMOKE_V2_DGX_REFRESH_2026_06.md` — qualifier
  sweep results that motivated the finale
- `docs/guides/eval-reports/EVAL_R1_AS_JUDGE_2026_06.md` — #940 Track 1
  agreement test that may feed back into the judge selection

---

## Why a finale tier exists

The qualifier sweep (`scripts/eval/score/rescore_against_silver.py` against
`silver_opus47_smoke_v{1,2}`) tells us whether a model is **fundamentally
suitable** for paragraph summarization. After the v2.1 and #945 refresh
runs, ROUGE-L on the Opus silver gave us a clean screening signal — models
in the bottom third (deepseek-r1 distills, qwen3-coder, gemma3:27b) were
ruled out for paragraph summarization regardless of prompt template.

What the qualifier **cannot** tell us:

1. **Which finalist is actually best.** Two models with rougeL=0.25 and
   rougeL=0.27 are statistically indistinguishable under ROUGE noise — but
   one may write substantially more grounded summaries than the other.
2. **Whether the prompt-induced verbosity from Phase 0.5 is a true quality
   lift.** The cross-cutting addendum in the v2 refresh report showed that
   5/7 native-prompt experiments regressed on ROUGE despite plausibly
   producing higher-quality summaries (richer language, more concrete
   detail). ROUGE rewards Opus-mimicry, not absolute quality.
3. **Faithfulness, coverage, coherence, fluency.** All four are first-class
   product concerns that ROUGE / BLEU / WER cannot measure.

The finale tier replaces ROUGE with **LLM judges scoring four behavior-
grounded rubrics**, applied only to the top-3-per-stratum qualifier
finalists. This is the championship round — ROUGE got us to the finalists,
G-Eval crowns them.

---

## Stratification

```
strata:
  - name: cloud         match: anthropic_/openai_/gemini_/grok_/mistral_/deepseek_/...
  - name: dgx_le_40b    match: ollama_qwen35_35b/27b, mistral-small_24b, gemma3_27b, phi4, ...
  - name: mbp_le_14b    match: ollama_qwen35_9b, mistral_nemo_12b, llama32_3b, hermes3_8b, ...
```

**Why first-match-wins (not most-specific-wins):** simpler config, easier
to reason about. Ordered substrings let us put specialised rules (cloud
APIs, big DGX models) before generic catch-alls (`ollama_`). This is
asserted by a dedicated test
(`test_load_run_candidate_specific_stratum_wins_over_catchall`).

**Why 3 strata, not more:** deployment realities. Each stratum is a
concrete deploy target the user actually considers:

- **cloud**: the "we'll pay for it" tier — anyone with an internet
  connection and an API key
- **dgx_le_40b**: workloads we'd run on the home DGX (tailnet-only, low
  marginal cost, ~40B param ceiling for sane TTFT)
- **mbp_le_14b**: workloads we'd run on the laptop in offline mode (the
  airgapped profile target)

A hypothetical "hybrid_system" stratum (e.g. extractive QA + small LLM) is
intentionally out-of-scope for this finale — it would need its own
qualifier sweep first.

## Promotion rule

Per stratum:

1. Rank by `vs_reference.rougeL_f1` from the existing
   `metrics_vs_silver_opus47_smoke_v1.json` file in each run dir.
2. Take the top 3.
3. Drop any whose `rougeL_f1 < 0.8 × stratum_leader_rougeL`.

Then trim the union to a **global cap of 12** by global ROUGE descending.

**Why 3:** balances "enough finalists per stratum to expose intra-stratum
quality differences" against "small enough to keep cost tractable".

**Why 0.8 × leader floor:** prevents promoting weak finalists from sparse
strata. If the mbp stratum's leader is rougeL=0.31 and #3 is at rougeL=0.20,
that #3 is materially worse than #1 (35% relative drop) and shouldn't
share a stage with it. 0.8 is a soft cliff — empirically the v2 refresh
spread had natural breaks around 80% of the leader.

**Why a global cap of 12:** $35/finale-run budget (per the #932 spec).
At 30 articles × 4 dimensions × 12 finalists × $0.02/Sonnet-call ≈ $28.80;
add the Gemini cross-check on top-2/stratum (6 × 30 × 4 × $0.015 ≈ $10.80)
and we're at $39.60, comfortably under the operator's $50 hard cap.

## G-Eval dimensions

We score each (candidate summary, episode transcript) pair on four
independent dimensions, each in the integer range `[1, 5]`:

| Dimension | What it measures | Why |
| --- | --- | --- |
| **Faithfulness** | No hallucinated names/numbers/quotes/claims | The pipeline's #1 product risk — fabrications surfaced in viewer UIs are non-recoverable |
| **Coverage** | Captures key decisions, arguments, lessons | The summary's primary use case — "what is this episode about" |
| **Coherence** | Reads as a single piece, not a list | Determines whether downstream tasks (KG extraction, search) get clean structure |
| **Fluency** | Sentence-level grammar/idiom | The lowest-stakes dimension but a reliable proxy for "how well the model speaks English" |

Each dimension carries explicit 1-5 anchors in `_RUBRICS` (see
`src/podcast_scraper/evaluation/g_eval.py`). The anchors are
behavior-grounded ("at most one minor ambiguity", "captures all major
themes") rather than vibes-grounded ("good", "excellent"), so two judges
should land within ±1 anchor on the same summary even when their tone
differs.

**Why one dimension per judge call:** three reasons.

1. **Smaller context per call → cheaper.** A four-dimension mega-prompt
   triples the token bill on shared frame (rubric headers, transcript
   reminder) while the judge does the same number of attention sweeps over
   the transcript.
2. **Less score-leakage.** When all four rubrics share attention, the
   judge tends to anchor on the first-scored dimension and let the others
   drift toward it. Per-dimension isolation kills this effect.
3. **Per-dim retry.** If a single dimension reply fails to parse (the
   judge prepended commentary, used wrong JSON shape), we re-run just
   that one call instead of all four.

## Judges

| Judge | Role | When invoked | Why |
| --- | --- | --- | --- |
| **Sonnet 4.6** (`claude-sonnet-4-6`) | Primary | Every (finalist × episode × dim) | High-quality baseline, no thinking-mode temperature constraint, reliable JSON output |
| **Gemini 2.5 Pro** (`gemini-2.5-pro`) | Cross-check | Top-2 finalists per stratum | Cap cost while still detecting Sonnet-specific stylistic bias |
| **DeepSeek-R1:32b** (DGX local) | Optional tertiary (#940) | Conditional on agreement-test result | Free local inference; if R1 agrees with Sonnet ≥75% on a pilot sample, it can substitute for one paid slot in regression sweeps |

**Why not Opus 4.7 as primary:** Opus 4.7's thinking-mode deprecates the
`temperature` parameter; deterministic scoring is harder to wire. Sonnet
4.6 is the standard model, no thinking-mode, supports `temperature=0.0`,
and is cheaper per call. The track_a precedent (`autoresearch_track_a.py`)
also uses Sonnet for the existing dual-judge pattern.

**Why Gemini Pro for the cross-check (not Opus 4.7 or Mistral Large):**
diversity. Sonnet and Opus share the same RLHF lineage so their failure
modes correlate; Gemini Pro is from a different training pipeline and
catches Anthropic-specific biases. Mistral Large was considered but is
weaker on long-context reasoning per the smoke v2 results.

## Promotion → finale data flow

```
qualifier run dirs (existing)            ─┐
  predictions.jsonl                       │
  metrics_vs_silver_opus47_smoke_v1.json  │
                                          ├──► load_run_candidate
strata config (finale yaml)              ─┘         │
                                                    ▼
                                          promote_finalists
                                                    │
                                                    ▼
                                          12 finalists × 5 episodes (smoke)
                                                    │   or 30 episodes (full)
                                                    ▼
                                          Sonnet primary scoring
                                                    │
                                                    ▼
                                          top-2/stratum → Gemini cross-check
                                                    │
                                                    ▼
                                          aggregate_finalist
                                                    │
                                                    ▼
                                          finale_report.{json, md}
                                          + finalists.jsonl
                                          + promotion.json
```

## Contested-pair handling

A finalist is flagged `contested: true` when **primary overall mean** and
**cross-check overall mean** differ by **> 0.5** on the 1-5 scale (≈10% of
the full range). This mirrors the spirit of `autoresearch_track_a.py`'s
0.25 threshold on the 0-1 scale.

Contested finalists are still scored and reported — the flag triggers
manual review in the report. We do NOT auto-collapse to ROUGE the way
track_a does, because the whole point of the finale tier is to break ties
that ROUGE can't.

We also expose a **pairwise agreement rate** (exact-or-adjacent on a 1-5
scale): for every (episode, dim) pair where both judges scored, how often
do they agree? This drives the #940 Track 1 integration decision for R1
and is reported in `finale_report.json` for all top-2 finalists.

## Cost guard

`cost_cap_usd` (default 50) is enforced inline. The runner:

- Tracks running cost across primary + cross-check phases
- Aborts further judge calls when the running total exhausts the cap
- **Persists partial artifacts anyway** — the operator gets `promotion.json`,
  whatever portion of `finalists.jsonl` finished, and a report annotated
  with what's missing

This matters because finale sweeps will land deep into long-tail eval
budgets. A silent abort that loses 80% of the work is unacceptable.

## What runs end-to-end today

- ✅ `--dry-run` against the existing 25-cell #939-rescored matrix:
  validated. Promotes 6 finalists (3 dgx_le_40b + 3 mbp_le_14b) with the
  expected leader / floor math.
- ⏳ Cloud-stratum finalists: blocked on rescoring cloud-API run dirs
  against `silver_opus47_smoke_v1`. Run:

  ```bash
  python scripts/eval/score/rescore_against_silver.py \
      --reference silver_opus47_smoke_v1 \
      --runs-glob 'data/eval/runs/autoresearch_prompt_anthropic_*'
  ```

  (also openai/gemini/grok/mistral/deepseek). This is a $0 op — no LLM
  calls, just ROUGE re-scoring against the new silver.

- ⏳ First paid finale run: blocked on operator approval to spend ~$40
  on Sonnet + Gemini Pro for the first full finale.

## What gets reported

`finale_report.md` for each sweep is structured as:

```markdown
# Finale sweep — <tag>
_Total spend: $X.XX (cap $YY.YY)_

## Verdicts by stratum

### cloud
| Rank | Run | Faith | Cov | Coh | Flu | Mean | Contested? |

### dgx_le_40b
…

### mbp_le_14b
…

## Promotion details
(per stratum: leader RougeL, floor, promoted runs, rejected runs + reasons)
```

`finale_report.json` carries the machine-readable equivalent plus the
pairwise agreement rates, total cost, and per-finalist error counts.

`finalists.jsonl` is the per-(finalist, episode, dim, judge) row dump so
follow-up analyses (e.g. "is faithfulness uncorrelated with coverage in
the mbp stratum?") can run without re-judging.

## Out of scope for this PR

- **Pareto chart**: planned in the #932 spec but deferred because (a) the
  budget didn't allow for a real finale run yet, so there's no data to
  plot, and (b) matplotlib is dependency-heavy for a one-chart deliverable
  that the report already covers as a Markdown table.
- **Auto-escalation to manual review queue on contested pairs**: future
  work, would require a queue model + viewer surface.
- **Hybrid-system stratum**: needs its own qualifier first.

## Open methodology questions for the next finale run

1. **Sample size:** 5 episodes is fine for smoke; 30 per the spec for the
   real championship. Re-evaluate after the first full run whether 30 is
   tight enough on confidence intervals for the per-dim scores — variance
   on faithfulness is likely higher than on fluency.
2. **R1 integration timing:** wait for the #940 Track 1 agreement
   result before committing to R1 as a finale judge slot. If R1 lands
   ≥75% agreement, we'd add it as a third judge for regression sweeps
   only (championship still uses Sonnet + Gemini).
3. **Tie-breakers within stratum:** currently the top-3 are ranked by
   `primary_overall_mean`. If two finalists tie at the 0.1 level we may
   want to fall back to coverage (the highest-stakes dimension). Worth
   revisiting after the first real data.
