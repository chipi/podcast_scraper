# Autoresearch Judging System

How the dual-LLM judge layer works, why it is designed the way it is, and how to
update it between rounds.

---

## Overview

The autoresearch harness combines two signals into a single scalar:

```text
final = rouge_weight * ROUGE-L_f1 + (1 - rouge_weight) * judge_mean
```

Default `rouge_weight = 0.70` (set via `AUTORESEARCH_SCORE_ROUGE_WEIGHT` env var).

ROUGE-L measures lexical overlap against a silver reference. The judge layer adds
semantic quality signal — coverage, accuracy, efficiency — that ROUGE cannot measure.

---

## Two judges, one rubric

Two independent LLM judges score each episode summary against the transcript:

| Role | Provider | Model (see `judge_config.yaml`) |
| ---- | -------- | ------------------------------- |
| Judge A | OpenAI | `gpt-4o-mini` (default) |
| Judge B | Anthropic | `claude-haiku-4-5-20251001` (default) |

Both judges receive the **same prompt**: rubric + transcript + candidate summary.
Each returns a score 0–1 per the JSON contract in the rubric.

The **episode score** is the midpoint of the two judge scores: `(judge_a + judge_b) / 2`.
The **run score** is the mean of all episode midpoints.

Judge models are pinned in `bundled_prompt_tuning/eval/judge_config.yaml` and
**must only be changed between autoresearch rounds** (not mid-session).

---

## Judge modes: scalar vs pairwise

The judge_config's optional top-level `mode:` field picks between two
scoring paths. Default is scalar (backward-compatible with every existing
config); `mode: pairwise` opts into the discrimination-boosted variant.

**Scalar (default):**

+ Prompt: rubric + transcript + **candidate summary** → judge scores 0–1
+ Reply schema: `{"score": 0.85, "notes": "..."}`
+ Judge saturation: prone on smoke sets (W27 saw judge_mean in
  [0.925, 0.975] across every candidate)
+ Contest fires when abs(judge_a - judge_b) > 0.25 on ≥40% of episodes
  (magnitude disagreement)
+ Silver not directly required — ROUGE-L handles the reference axis
+ Provider support: openai / anthropic / ollama / vllm

**Pairwise:**

+ Prompt: rubric + transcript + **Summary A** + **Summary B** → judge
  picks preference + magnitude
+ Slot randomization: candidate randomized into A or B per episode via
  `hash(episode_id)` — kills first-slot bias
+ Reply schema: `{"preference": "A" or "B" or "tie", "magnitude": 1-5, "rationale": "..."}`
+ Discrimination: direction is binary → judges can't hide in the top of
  a 5-point scale
+ Contest fires when both judges pick OPPOSITE parties on ≥40% of
  episodes (directional disagreement — magnitude alone doesn't count)
+ Silver **required** — silver `predictions.jsonl` loaded per episode;
  pairwise fails loudly if missing
+ Provider support: ollama / vllm only (DGX-focused). Cloud-API pairwise
  raises NotImplementedError

Both modes feed into the same
`final = rouge_weight * ROUGE-L + (1 - rouge_weight) * judge_mean`
scoring formula. In pairwise mode `judge_mean` is the mean of per-episode
pairwise scores (candidate + magnitude → [0.6, 1.0]; silver + magnitude
→ [0.0, 0.4]; tie → 0.5).

The pairwise summary (per-judge `win_rate` / `tie_rate` / `decisive_rate`)
lands under `scores.pairwise` in the output JSON for the leaderboard audit
column. Under scalar mode `scores.pairwise` is null.

Implementation:

+ `src/podcast_scraper/evaluation/pairwise.py` — primitives
+ `autoresearch_track_a.mean_pairwise_scores` — dispatch
+ `mode: pairwise` in the judge_config yaml
+ `judge_config_vllm.yaml` — working pairwise config
  (Qwen3-30B-A3B via vLLM + llama3.3:70b via Ollama)

---

## Rubric

Rubric file: `bundled_prompt_tuning/eval/rubric.md`

Three dimensions, each scored independently, then averaged into a single `score`:

| Dimension | What it measures |
| --------- | ---------------- |
| **Coverage** | All main themes present; nothing central missing |
| **Accuracy** | No contradictions or invented facts vs. the transcript |
| **Efficiency** | Each sentence adds unique information; no padding or repetition; length is appropriate to content depth — not penalised for being long if the content warrants it |

The **Efficiency** dimension deliberately does not penalise longer summaries. An episode
with 10 distinct topics warrants a longer summary than a short focused one. What is
penalised is *redundancy and filler*, not *length per se*.

Judge output format (JSON, no markdown):

```json
{
  "coverage": 0.9,
  "accuracy": 1.0,
  "efficiency": 0.85,
  "score": 0.917,
  "notes": "One short sentence explaining the rating."
}
```

`score` should equal the mean of the three dimension scores. The per-dimension fields
are logged at DEBUG level for visibility; `score` is the authoritative value used in
the harness.

---

## Contestation logic

**When judges disagree on an episode**, it is marked as *contested*:

```python
contested = abs(judge_a_score - judge_b_score) > DIVERGENCE_THRESHOLD  # 0.25
```

**When a run is considered contested**, it falls back to ROUGE-only:

```python
any_contested = (contested_episode_count / total_episodes) > CONTEST_FRACTION_THRESHOLD  # 0.40
```

Key design decision: contestation is **fraction-based, not a binary OR**. If one
episode out of five contests, the run still uses the full blend. At least 40 % of
episodes must contest before the harness discards judge scores entirely.

> **Why not binary OR?** At 5-episode smoke scale, a single unusual episode flips the
> entire metric from `0.70*ROUGE + 0.30*judge` to pure ROUGE — a ~20-point swing.
> This made the metric too brittle for meaningful prompt comparisons. Fraction-based
> logic requires multiple episodes to agree before treating the run as contested.

The 0.40 threshold means:

| Dataset size | Episodes needed to contest |
| ------------ | -------------------------- |
| 5 (smoke) | ≥ 3 |
| 10 | ≥ 5 |
| 25 (benchmark) | ≥ 11 |

---

## Summary extraction before judging

Bundled-mode summaries are stored as a JSON string in `summary_final`:

```json
{"title": "...", "summary": "...", "bullets": ["...", "..."]}
```

Before passing to judges, the harness extracts the prose fields and presents them in
a clean, readable format:

```text
Title: ...

Summary:
<prose paragraphs>

Key takeaways:
- bullet 1
- bullet 2
```

This ensures both judges evaluate the same content rather than interpreting the raw
JSON blob differently (one judge might treat JSON length as a conciseness signal).

---

## Seed & determinism (v2)

v2 adds a `seed` plumbing path for OpenAI calls. All v2 autoresearch configs set
`params.seed: 42`. The seed is threaded:

```text
YAML params.seed=42
  → ExperimentConfig.params["seed"]
  → run_experiment.py: openai_summary_seed=params_dict_raw.get("seed")
  → Config.openai_summary_seed
  → OpenAIProvider.summary_seed
  → client.chat.completions.create(..., seed=42)
```

Non-bundled and bundled paths both pass the seed. See
`src/podcast_scraper/providers/openai/openai_provider.py` lines around the
`_make_api_call` helpers.

### Empirical behaviour

OpenAI's seed is *approximately* deterministic, per their API docs. Tested during v2:

+ Stable `system_fingerprint` across runs → good sign.
+ `predictions.jsonl` md5 still differs between two seeded runs → seed does NOT give
  byte-identical outputs.
+ Final score variance reduced but not eliminated — roughly ~5% swing remains from API
  non-determinism alone, before judge variance.

**Practical implication**: seed helps but does not fix the fundamental problem. For a
principled fix, implement multi-run averaging (N=3 per experiment) — deferred to future
work per RFC-073. In the meantime, seed + the v2 dev (N=10) + held-out validation are the
combined mitigation.

---

## Silver / judge vendor-bias rule (cross-vendor cohorts)

**Rule.** When a candidate cohort spans multiple vendor families (e.g. Anthropic +
OpenAI + Google models all compared in one sweep), the silver reference and the
primary judge **MUST NOT** be the same vendor family as any *single* candidate. If
they are, that candidate gets a free same-vendor style boost — judges score writing
that resembles the silver higher, and writing that resembles the judge's own style
higher. Both effects compound.

Concrete failure mode observed in #939 (commit `56f95727`): against Sonnet 4.6
silver, Qwen3 family topped the ROUGE leaderboard (qwen3.5:27b 0.271, qwen3.6
0.271 tied #1). Against an **Opus 4.7** silver (different model, even within
Anthropic family — Opus writes meaningfully differently from Sonnet), the same
predictions reshuffled: mistral:7b 0.329, llama3.2:3b 0.326, qwen3.5:35b dropped
to #11 (0.262 → 0.243). The ROUGE spread also **widened** (0.024 → 0.086) — the
flat Sonnet baseline was masking real quality differences.

The phenomenon is documented as the **Sonnet-mimicry artifact**: Qwen3 family
trains heavily on Sonnet-style outputs, so it scores artificially high against a
Sonnet silver and mid-pack against Opus.

**What to do.**

1. For any cross-vendor candidate cohort, generate **two silver references** with
   models from disjoint vendor families:
   + Anthropic side: Opus 4.7 (`silver_candidate_anthropic_opus47_*`)
   + OpenAI side: GPT-5.4 or GPT-4o (`silver_candidate_openai_gpt54_*`)
   + (Google Gemini silver optional — we don't use it as silver because Gemini
     2.5 Pro is on the judge panel.)
2. Run the candidates' predictions through `rescore_against_silver.py` for BOTH
   silvers and report ROUGE side-by-side. Disagreements between the two silvers
   are diagnostic — they identify candidates that are silver-style-mimicking
   rather than genuinely better.
3. The **judge panel** for the finale tier should likewise be cross-vendor:
   + Primary: Sonnet 4.6 (high precision, low cost)
   + Cross-check: GPT-5.4 (cross-vendor)
   + Optional second cross-check: Gemini 2.5 Pro (third vendor) — adds cost but
     necessary when one of the candidates IS a Gemini family member.

This rule applies to **all autoresearch tiers** that compare cross-vendor
candidates (#928 reframe, #1016 per-stage eval, and any future per-vendor
landscape comparisons). It does **not** apply to single-vendor sweeps (e.g.
prompt tuning within Qwen family) where same-vendor silver is fine.

See `docs/guides/eval-reports/SILVER_OPUS47_GENERATION_2026_06.md` for the full
generation procedure + the `EVAL_SMOKE_V2_DGX_REFRESH_2026_06.md` addendum for
the rescored-against-Opus numbers that surfaced the Sonnet-mimicry artifact.

---

## Distribution + output-length sanity checks (mandatory)

Two failure modes burned hours in #1016 Phase 2b: a misleading mean (Phase 1
batch-total log line slipping into a per-episode aggregator → all vLLM
candidates looked 2× slower than they actually were), and a silent output
inflation (the `cloud_structured_max_output_tokens` floor in
`openai_provider.py:1095` clamped max_tokens up to 4096 → Qwen models
generated 4-6× over the requested spec, dragging their judge scores).

**Both were preventable with mandatory sanity checks. Apply these every
autoresearch round:**

### 1. Every aggregate must report min / p50 / mean / p99 / max + spread

A bare "mean=132s" is useless. The required reporting shape:

```text
candidate            N  min  p50  mean  p95  max  spread(max/min)  cv(sd/mean)
example_vllm        10  17s  22s   22s  27s  27s  1.6×              0.15
```

+ **spread > 3× or cv > 0.5** → flag a suspicious data point. Investigate
  the outlier episode before drawing conclusions.
+ **mean diverges from p50 by > 30%** → the mean is being dragged by an
  outlier. Use p50 as the primary headline, mean as secondary.
+ **Always tag the dataset / config version** the numbers came from so a
  later session can't mix prior-run timings into a new-run table.

### 2. Validate output size against spec on every prediction

Cloud LLM providers (and the OpenAI provider in this repo) have silent
clamps that can override `max_length`. The harness may also have prompts
that don't carry the spec to the model (e.g. `{{ paragraphs_max }}` ranges
that derive from `max_length // 100` and can mislead the model).

Required per-prediction-set table:

| candidate | spec (chars) | min ch | p50 ch | mean ch | max ch | in spec? |
|-----------|-------------:|-------:|-------:|--------:|-------:|:--------:|
| example   | 800-3200     | 1800   | 2100   | 2150    | 2400   | ✓        |
| broken    | 800-3200     | 3000   | 4200   | 4400    | 5500   | ✗ over   |

+ **mean chars > 1.5 × upper-spec** → there's a clamp, a floor, or a
  prompt issue. Diagnose before scoring; running judges on out-of-spec
  outputs invalidates the comparison.
+ **mean chars < 0.5 × lower-spec** → model is terminating early. Could
  be EOS injection, prompt issue, or quantization-induced premature
  stopping. Diagnose before scoring.

### 3. Sample-output sanity check

Before publishing G-Eval results, eyeball at least 1 sample per candidate.
Look for:

+ Reasoning preamble leaking through (e.g. `<think>` blocks in DeepSeek-R1,
  reasoning prose in Qwen3 family if `enable_thinking=False` isn't honored).
+ Byte-level BPE artifacts (`Ġ`, `Ċ`, `âĢĻ` in DeepSeek-R1 on vLLM 26.05).
+ Markdown / JSON in plain-text-spec outputs (prompt says "plain text only"
  but model returns bullets or `**bold**`).
+ Truncated mid-sentence outputs (finish_reason="length" → max_tokens hit
  before natural EOS; output is incomplete and shouldn't be scored).
+ Same / templated openings across episodes (model is over-relying on
  prompt scaffolding rather than the transcript).

If any of these are present in the sample, the cohort's judge scores are
not comparable. Patch the harness / prompt / config and re-run.

### 4. Run direct-API probe to confirm the harness sends what the config says

For any unexpected output (length, format, content), bypass the harness
and `curl` the provider directly with the exact max_tokens / params the
config claims. If the direct probe matches the spec but the harness output
doesn't, the harness is mutating the request between config and call.
This is exactly how the `cloud_structured_max_output_tokens` floor was
caught in Phase 2b — direct probe at `max_tokens=500` returned 500-token
output, but the harness produced 800+ token output despite the same config.

### 4b. Preliminary Result Gate (mandatory before any judge call)

Each prediction set MUST pass a preliminary gate before any expensive
downstream step (judge calls, finale promotion, report writing). The gate
is a fixed checklist — if ANY check fails, STOP. Diagnose, fix, re-run.
Do NOT proceed to judges on a set that hasn't passed the gate.

**Treat every result as suspicious until proven otherwise.** Confirmation
bias is the dominant failure mode in eval work — once a number lands in
a table, it propagates into reports, decisions, and follow-up tickets
before anyone questions whether it's correct.

The gate (run in order, stop at first failure):

```text
┌─────────────────────────────────────────────────────────────────────┐
│ PRELIMINARY RESULT GATE                                              │
├─────────────────────────────────────────────────────────────────────┤
│ Check 1: predictions.jsonl exists with N == expected episode count   │
│   FAIL → inference incomplete; check run.log for errors              │
│                                                                      │
│ Check 2: per-prediction finish_reason ∈ {"stop", "eos_token"}        │
│   FAIL on "length" → truncated mid-generation; output incomplete     │
│   FAIL on "content_filter" → moderation blocked; flag, don't score  │
│                                                                      │
│ Check 3: char-count distribution fits spec                           │
│   spec = (max_length × 4 chars/tok) ± 50%                            │
│   mean_chars within spec: PASS                                       │
│   mean_chars > 1.5 × spec_upper: FAIL — config / floor / clamp bug  │
│   mean_chars < 0.5 × spec_lower: FAIL — premature termination       │
│                                                                      │
│ Check 4: cross-candidate length parity                               │
│   max(mean_chars) / min(mean_chars) across cohort < 2.0: PASS        │
│   ratio ≥ 2.0: FAIL — judges will penalize verbose responses;        │
│     either tighten spec to match the shortest or run with disclosure │
│                                                                      │
│ Check 5: latency distribution sanity                                 │
│   per-candidate cv(latency) < 0.5: PASS                              │
│   cv ≥ 0.5: FAIL — investigate outlier episode (transcript pathology,│
│     network, model state — fix before scoring)                       │
│                                                                      │
│ Check 6: sample-eyeball (1 random prediction per candidate)          │
│   no reasoning preamble, no byte-level BPE artifacts,                │
│   no markdown when spec says plain text, no templated openings,      │
│   no mid-sentence truncation: PASS                                   │
│   any visible artifact: FAIL — patch harness/prompt, re-run          │
│                                                                      │
│ Check 7: harness-vs-direct-API spot check                            │
│   curl the provider directly with the same params as the config     │
│   claims. Output length should match within 20%.                     │
│   mismatch > 20%: FAIL — harness is mutating the request; trace     │
│   (clamps, floors, default overrides) before scoring                  │
└─────────────────────────────────────────────────────────────────────┘
```

Pass the gate → proceed to judges. Fail any check → stop, diagnose, fix.
**Document every gate failure** in the eval report's methodology section,
even after it's been fixed — the failure trail prevents the same trap
from recurring and signals the limits of the methodology to future readers.

This gate would have caught both #1016 Phase 2b failures (the Phase 1
timing artifact via Check 5, and the cloud-floor verbose-output bug via
Check 3 + Check 7). Cost of running the gate: ~5 minutes of analysis per
cohort. Cost of skipping it: hours of judge spend on uncomparable results
plus wrong conclusions in the eval report.

### 5. Cross-candidate output-length parity

If candidate A produces 2000 mean chars and candidate B produces 4000 mean
chars on the same dataset + same prompts, **they are NOT being judged
fairly** — judges penalize verbose responses on coherence + efficiency.
Either tighten the spec until both produce comparable length, or disclose
the length disparity prominently in the verdict (so the result can't be
read as "A is more concise / better").

These rules apply to ALL eval reports under `docs/guides/eval-reports/`
and to every autoresearch round (#928, #1016, future tickets).

---

## Known limitations

1. **Cheap judge models.** `gpt-4o-mini` and `claude-haiku` are the cheapest models
   available. They produce less calibrated, higher-variance scores than flagship models.
   If judge disagreement remains high after rubric fixes, upgrading to `gpt-4o` +
   `claude-sonnet-4` is the next lever. See `judge_config.yaml` — models are pinned there.

2. **Single-score output.** Even with per-dimension scoring, the harness uses only the
   final `score` for optimization. If you need to understand *which* dimension is causing
   rejection, run with `--log-level DEBUG` and inspect the per-dimension logs.

3. **Rubric calibration.** The rubric was written before the bundled-mode output format
   existed. If you change the output shape (e.g., add a new field), revisit the rubric
   to confirm the Efficiency dimension still applies correctly.

4. **Transcript truncation.** Judges receive at most `MAX_TRANSCRIPT_CHARS = 28,000`
   characters of the transcript. For very long episodes, the judge may miss content
   discussed late in the transcript and penalise the summary for "missing" themes it
   cannot see.

5. **OpenAI temperature=0 is not deterministic.** Documented by OpenAI as "approximately
   deterministic." Seed plumbing (§Seed & determinism) reduces variance but does not
   eliminate it. Smoke-scale (N=5) scores can flip across runs for borderline
   configurations — use v2 dev (N=10) as the primary iteration set and validate champions
   on held-out for any decision that matters.

6. **Held-out benchmark is small (v2).** `curated_5feeds_benchmark_v2` has only 5 episodes.
   Own noise ~±5%. Big enough to catch major overfitting (>10pp drops), too small to
   discriminate between similarly-good champions. Acceptable for this stage; tracked for
   expansion in RFC-073 §Future Work.

---

## How to update between rounds

**Rubric changes** — edit `bundled_prompt_tuning/eval/rubric.md`. The rubric hash in
`results_openai_r1.tsv` will change for new experiments, marking a methodological
boundary. Previous rows with the old hash are not directly comparable.

**Judge model changes** — edit `bundled_prompt_tuning/eval/judge_config.yaml`. Must be
done between autoresearch sessions (the file is pinned during a run).

**Threshold changes** — edit `DIVERGENCE_THRESHOLD` or `CONTEST_FRACTION_THRESHOLD`
in `src/podcast_scraper/evaluation/autoresearch_track_a.py`. These are code constants.

**ROUGE weight changes** — set `AUTORESEARCH_SCORE_ROUGE_WEIGHT` in `.env.autoresearch`.
Default 0.70. Range 0–1.

---

## Score formula reference

```python
# Per-episode judge score
midpoint = (judge_a + judge_b) / 2.0

# Run-level judge mean
judge_mean = mean(all episode midpoints)

# Contestation check
fraction_contested = contested_episode_count / total_episodes
any_contested = fraction_contested > CONTEST_FRACTION_THRESHOLD  # 0.40

# Final scalar
if any_contested or judge_mean is None:
    final = rouge_l_f1
else:
    final = rouge_weight * rouge_l_f1 + (1 - rouge_weight) * judge_mean
```
