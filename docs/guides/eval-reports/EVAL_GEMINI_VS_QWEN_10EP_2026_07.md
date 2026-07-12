# Gemini vs qwen3.5:35b — 10 prod episodes, DGX transcripts, one pinned judge

**Date:** 2026-07-13
**Dataset:** `prod_v3_10ep_v1` — 10 episodes curated from the existing prod-v3 DGX corpus
(`tailnet_dgx_whisper` / `faster-whisper-large-v3`, `tailnet_dgx` diarization). Pinned by
transcript path + sha256. **Not re-fetched from the live feed.**
**Runs:** `data/eval/runs/h2h_gemini_10ep`, `data/eval/runs/h2h_qwen_10ep`

## Question

Can qwen3.5:35b on the DGX match `gemini-2.5-flash-lite` on grounded-insight quality, so prod v3
can be built entirely on our own hardware?

## Answer

**On insight quality, yes — qwen is at parity.** It produces as many CORE insights per episode as
gemini, with less than half the filler. It delivers 82% of gemini's grounded insights per episode
and clears the ADR-053 grounding contract.

The remaining gaps are **evidence density** (1.31 vs 2.08 quotes per insight) and **speed**
(275s vs 33s per episode).

## Method — what was held constant

Only the LLM varies. Everything else is identical:

| held constant | how |
| --- | --- |
| episodes | pinned by path + sha256 (a live feed silently drifts — see §Pitfalls) |
| ASR + diarization | the same DGX transcripts feed both arms |
| code | same commit |
| insight ceiling | 50, non-binding for both |
| summary params | identical (they were **not**, until this run — see §Pitfalls) |
| value gate | ON, `min_tier=2` |
| **gate judge** | **pinned to `claude-haiku-4-5`** — vendor-disjoint from both candidates |

The pinned judge is the load-bearing control. Letting each model grade its own output is the #939
same-vendor bias, and it is not small: qwen rejects **4%** of its own insights where an independent
judge rejects **26%** of the same ones.

## Results

| per episode | gemini-2.5-flash-lite | qwen3.5:35b (DGX) |
| --- | --- | --- |
| insights emitted | 44.0 | 25.9 |
| insights kept (post-gate) | 24.4 | 20.3 |
| **grounded insights** | **20.1** | **16.4** |
| grounding rate | 82.4% | 80.8% |
| quotes / episode | 49.8 | 26.7 |
| quotes per insight | 2.08 | 1.31 |
| wall-clock | 33s | 275s |

Both clear the ADR-053 contract (grounding ≥ 80%). **qwen delivers 82% of gemini's grounded
insights per episode.**

### The judge's verdict — gemini pads, qwen does not

Same judge, both arms, so the rejection rates are directly comparable:

| arm | emitted | rejected as filler | reject rate |
| --- | --- | --- | --- |
| gemini | 440 | 196 | **45%** |
| qwen | 259 | 56 | **22%** |

**Nearly half of gemini's insights are filler. Only a fifth of qwen's are.** Gemini is prolific and
noisy; qwen is sparser and disciplined. This is why the raw insight counts flattered gemini and the
gated counts do not.

### Two-judge panel (blind, shuffled, tiered)

Judges are vendor-disjoint from both candidates. Scored on post-gate insights.

| metric | judge | gemini | qwen |
| --- | --- | --- | --- |
| CORE / ep | anthropic | 6.6 | **8.9** |
| CORE / ep | openai | 11.6 | 11.6 |
| USEFUL+ / ep | anthropic | 18.4 | 17.8 |
| USEFUL+ / ep | openai | 24.0 | 20.2 |
| FILLER / ep | anthropic | 6.0 | **2.5** |
| FILLER / ep | openai | 0.4 | **0.1** |

**qwen matches or beats gemini on CORE insights, with less than half the filler.**

Inter-judge agreement: exact 52–61%, within-1 **92–94%**, keep/drop 76–88%. The judges disagree on
absolute strictness (openai is markedly more lenient) but **rank the two arms identically** — so the
conclusion is judge-independent even though the absolute tiers are not. Do not quote a single
judge's absolute numbers as fact.

## What this means for the DGX

The gap is not insight quality. It is:

1. **Evidence density** — 1.31 vs 2.08 quotes per insight. qwen finds less supporting evidence for
   each claim it makes. This is the next thing to tune.
2. **Speed** — 275s vs 33s per episode (8x). Known, expected, and not a quality question.

## Pitfalls found while building this comparison

Every one of these would have produced a confident, wrong number. They are recorded because the
comparison is only worth what the method is worth.

1. **A live RSS feed silently changes the episode set.** `--max-episodes N` takes the newest N, so
   two runs an hour apart compared *different episodes*. An earlier grounding delta (66.7% → 72.7%)
   was reported as a fix and was actually different content. **Retracted.** The eval dataset pins by
   path + sha256 and is immune.
2. **Summary params were ollama-only.** `_episode_summary_params` passed `max_length` for the ollama
   backend and *nothing* for every cloud backend. qwen ran at the configured budget, gemini at the
   internal default. Two different configurations, reported as like-for-like — and it truncated
   gemini's summary on long episodes, aborting the run.
3. **The pinned judge inherited a 404 model.** `gi_value_gate_provider: anthropic` swapped the
   provider but inherited its *default* model (`claude-3-5-sonnet-20241022`, deprecated). Every
   classify call threw, the gate failed open, and a full 10-episode run completed **ungated** while
   reporting a healthy 44.3 insights/episode. Fail-open saved the episodes and hid the
   misconfiguration — a broken gate and a permissive gate produce the same artifact. The only tell
   was zero drop-lines across ten episodes.
4. **The eval allowlist silently drops unknown params.** It voided three separate gate settings in
   turn (`enabled`, `provider`, `model`). Each time the cell ran and looked like a result.
5. **A transcript set destroyed by the cleaning bug.** Three of the six 10-episode prod-v3 runs have
   `.cleaned.txt` files of ~144 chars (the `CommercialDetector` bug, fixed 14:55). A dataset built on
   those would be silently worthless. The builder now asserts transcript length.

The common shape: **the pipeline reports success while a stage is inert or destroyed.** Every check
here exists because that happened.

## Caveats

- **10 episodes from ONE feed** (Hard Fork). This says which stack is better *for this show*. It is
  not a general result.
- Grounding rate carried **±10pp run-to-run noise** at n=3; 10 episodes reduces but does not remove it.
- The gate judge (`claude-haiku`) and one panel judge (`claude-sonnet`) share a vendor. The
  openai panel judge is the independent check, and it agrees on the ranking.
