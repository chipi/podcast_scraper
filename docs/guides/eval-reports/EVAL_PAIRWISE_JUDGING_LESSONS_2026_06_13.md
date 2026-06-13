# Pairwise LLM judging — lessons learned and methodology

**Date:** 2026-06-13
**Trigger:** #989 (cleaning_v3 vs cleaning_v4) found that 5 of #905's
original "ties" were actually v3 wins once A/B positions were swapped.
Position bias in pairwise judging is real and non-negligible.

This doc is **mandatory reading before designing any pairwise-judge
eval for fixtures v3, the finale tier (#932), or any future autoresearch
ticket that decides a production default**. The reusable harness is at
`scripts/eval/score/pairwise_judge_v2.py`.

## The headline finding

Issue #905 Tier 2's pairwise Sonnet 4.6 judge on 5 v2 episodes reported v3
beating v4 **10W-0L-5T**. Five ties were enough to defer the production
flip — "broader judge pass" became the gate.

In issue #989, the same comparison was rerun on 15 v2 episodes (3× the sample)
with one methodology change: **each pair was judged twice, once with
A=v3 and once with A=v4**. The result: **v3 wins 15/15**. The 5 ties
disappeared.

The ties weren't real ties. They were positional artifacts — Sonnet
sometimes preferred whichever output sat in slot A, and 5 of 15 tested
pairs happened to land in the bias's failure zone. With position-swap
neutralisation, the verdict became unambiguous.

This is not a Sonnet-specific bug. It's a calibration artifact common
across modern frontier LLM judges. **You cannot prompt-engineer it out**
— "be unbiased" in the system prompt doesn't shift the prior.

Smoke-test confirmation (2 v2 episodes × 3 judges × 2 orderings,
2026-06-13): **gpt-4o-mini flipped its p02_e01 verdict** when v3 and v4
swapped slots, even with explicit JSON-output mode. Sonnet 4.6 and
Gemini 2.5 Flash held position-stable on both items. Two of three judges
agreeing kept the multi-judge consensus correct — but the
single-judge-without-swap configuration would have shipped a false TIE.

## Bias taxonomy

Position is the most-discussed bias but not the only one. Eval design
needs to defend against all five:

| Bias | What it does | Failure mode |
| --- | --- | --- |
| **Position** | Judge has a per-call learned preference for slot A or slot B | Tier-1 ties that disappear under swap |
| **Length** | Judge prefers longer outputs | Summary-eval verdicts dominated by `max_tokens`, not quality |
| **Verbosity** | Judge prefers confident / technical-sounding prose | Hedging or honest "I don't know" outputs systematically lose |
| **Recency** | Judge weights whichever content appeared most recently in the prompt | First few items of a list-comparison get under-credited vs the last |
| **Self-preference** | Judge from lab X prefers outputs from lab X's other models | Sonnet judging Anthropic-vs-OpenAI summary outputs is poisoned |

Position is the cheapest to defend against (just swap). The others
require structural design — different rubric, different prompt shape,
different judge pool.

## The three-tier framework

Pick a tier by the **stake of the decision**, not the cost you're
comfortable with. Cheap eval that ships a wrong default flip is the
expensive eval.

### Tier 1 — production default flips and customer-facing behaviour changes

**When**: any change that flips a production default
(`DEFAULT_PROFILE`, NER model, summary provider, diarization backend,
…). Anything that affects what real users see.

**Methodology**:

- **Multi-judge ensemble** — at least 3 judges from different labs.
  Anthropic + Gemini + OpenAI is the canonical triad. Different labs
  have different bias directions; biases partially cancel.
- **Position swap** — each pair judged twice with A/B reversed. Use
  consensus across orderings as the per-judge verdict. Tag
  `TIE_POSITIONAL` when orderings disagree.
- **Strict majority across judges** — at least ⌈N/2⌉+1 judges must
  agree on the verdict. Anything else is `DISAGREEMENT`, treated as
  "not ready to flip".
- **Anonymise candidates** — judges see A/B labels, never the real
  candidate names. Otherwise the model name leaks in.
- **Save full audit log** — every raw judge response, every reason,
  every token count, every cost. So a future reviewer can audit the
  decision without rerunning.

**Cost**: 3 judges × 2 orderings × N items. For N=15 items the cost is
roughly 90 API calls × ~$0.005/call mix = ~$0.40-1.50 total. Cheap
enough to do for every production-default flip; the implicit cost of
shipping a wrong flip is far higher.

**Harness**: `scripts/eval/score/pairwise_judge_v2.py`. Example:

```bash
export $(grep -E '^(ANTHROPIC|OPENAI|GEMINI)_API_KEY=' .env)
PYTHONPATH=. .venv/bin/python scripts/eval/score/pairwise_judge_v2.py \
    --items data/eval/runs/<your_run>/items.jsonl \
    --judges anthropic:claude-sonnet-4-6 \
    --judges openai:gpt-4o-mini \
    --judges gemini:gemini-2.5-flash \
    --orderings swap \
    --rubric scripts/eval/score/rubrics/<your_rubric>.md \
    --output data/eval/runs/<your_run>
```

### Tier 2 — autoresearch picking a winner among N variants

**When**: prompt-engineering sweeps, model-variant tournaments, anything
where you have lots of pairwise comparisons across many candidates and
want a ranking, not a verdict.

**Methodology**:

- **Randomised A/B order per pair** — each pair gets one call but with
  a coin-flip on which candidate goes in slot A. Bias averages out in
  aggregate across many comparisons.
- **Single judge sufficient** — Sonnet 4.6 is the default; the
  randomisation handles position bias statistically.
- **Bradley-Terry-style aggregation** — fit a BT model over all the
  pairwise outcomes to extract a ranking. Win-count rankings work too
  for small N.
- **No multi-judge** unless cost is no object — the per-call cost
  matters when N² comparisons are involved.

**Cost**: ~1 call per comparison; for an N-candidate tournament that's
N·(N-1)/2 pair calls.

### Tier 3 — continuous quality monitoring

**When**: trend tracking, regression detection on already-shipped
defaults. "Did our cleaning quality regress this week?"

**Methodology**:

- **Reference-free rubric scoring** — judge gives each candidate a
  0–5 score against a fixed rubric, no head-to-head comparison.
- **Single judge, single ordering** — there's no position to swap;
  there's no second candidate.
- **Anchors matter** — the rubric needs worked examples of each score
  level so judge calibration stays stable across sample windows.
- **Track trends, not absolute scores** — week-over-week changes
  matter more than the absolute value.

**Cost**: 1 call per output. Cheapest tier; suitable for daily
monitoring.

## The always-do checklist

Apply to every pairwise eval regardless of tier:

- [ ] **Anonymise candidate labels** — never put the real names in the
  judge prompt. Use A/B (or X/Y) and decode on output.
- [ ] **Anonymise the producing provider** — for summary eval, don't
  identify which model generated which output. Self-preference is real.
- [ ] **Save the judge model id + temperature + system prompt + rubric**
  in the metrics output. Reproducibility depends on this.
- [ ] **Save the raw verdict text + reason** in an audit JSONL alongside
  the parsed verdict. A one-line "TIE — both look similar" carries
  different signal than "TIE — A is better at X but B is better at Y".
- [ ] **Quote the cost** in the report. Future researchers need to
  choose tier consciously.
- [ ] **Report the gate config** in the eval report's TL;DR (judges,
  orderings, sample size, rubric). Don't bury it.

## What this changes about how we run autoresearch on v3 fixtures

When v3 fixtures land (#921 and the broader rebuild), every winner-picking
gate inside autoresearch should run on the new tier framework:

- **Default flips during v3 autoresearch** — Tier 1. Multi-judge + swap.
  Codified by the new harness.
- **Prompt-tuning sweeps and model tournaments** — Tier 2. Randomised
  order, single judge.
- **Continuous cleaning / summary quality watch on prod_validation_v1**
  (#933) — Tier 3. Rubric scoring, single judge, weekly cadence.

The harness handles Tier 1 today. Tier 2's BT-tournament aggregation and
Tier 3's rubric-scoring path are clean follow-ups (each ~half-day) when
they become load-bearing — both can reuse the same judge client
abstractions in `pairwise_judge_v2.py`.

## When to escalate beyond multi-judge + swap

If multiple Tier-1 runs produce `DISAGREEMENT` across judges for the
same pair, that's the signal that **the comparison isn't decidable by
this judge pool**. Three options at that point:

1. **Human review** — fall back to operator judgement. Multi-judge
   `DISAGREEMENT` is the correct point to invoke human attention.
2. **Better-anchored rubric** — the rubric may be too vague for
   judges to align. Rewrite with worked examples.
3. **A larger judge pool** — Claude Opus + GPT-5 + Gemini Pro instead
   of the cheap-tier triad, at 10×+ cost. Last resort.

Multi-judge disagreement is not an embarrassment. It's the eval
system correctly surfacing "this is harder than the cheap tier can
decide." Treat it as information, not failure.

## Anti-patterns to avoid

- **Single-judge single-order ("just run Sonnet on the cleaned pairs")**
  for production-default decisions. This is what #905 did; #989 caught
  the 5 false ties. Don't.
- **Confusing self-consistency (high-temp resampling of the same judge)
  with bias reduction**. Self-consistency reduces stochastic noise. It
  does *not* address position bias. The same biased judge stays biased
  across re-rolls.
- **Reporting "X% v3 wins" without disclosing the judge config**.
  Future readers can't tell if the verdict survived position-swap.
- **Reusing a silver generated from a candidate to judge that candidate**.
  This was almost certainly the GI "+10pp direct mode wins" trap — the
  silver was direct-mode-shaped, so direct-mode "won" by construction.
  Silvers need a provenance chain that's independent of the candidates
  being compared.

## See also

- `scripts/eval/score/pairwise_judge_v2.py` — the reusable harness
- `scripts/eval/score/cleaning_v3_vs_v4_broader_judge_v1.py` — the
  position-swap-only single-judge precursor (still useful for cheap
  re-validation)
- `docs/guides/eval-reports/EVAL_CLEANING_V3_V4_BROADER_JUDGE_2026_06_13.md`
  — the #989 result that motivated this doc
- `docs/guides/eval-reports/EVAL_GI_AUTORESEARCH_V2_2026_06_13.md` — the
  60pp GI verdict reversal (silver-shape failure mode) that this doc
  also closes against
