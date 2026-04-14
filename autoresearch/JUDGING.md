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
