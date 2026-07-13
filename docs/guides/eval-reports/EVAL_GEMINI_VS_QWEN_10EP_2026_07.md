# Gemini vs qwen3.5:35b on the DGX — can we build prod v3 on our own hardware?

**Date:** 2026-07-13
**Runs:** `h2h_gemini_10ep`, `h2h_qwen_10ep`, `gen3_gemini`, `gen3_qwen`, `gate_allprov_*`

## Answer

**On one show, yes. Across nine shows, no — qwen delivers about 70% of gemini's grounded
insights.** Both clear the 80% grounding floor everywhere, and qwen is the more *reliable* of the
two (gemini fell back to a stub on one episode; qwen never did). But qwen extracts less.

| | qwen as % of gemini's grounded insights |
| --- | --- |
| Hard Fork only (10 episodes, one show) | **94%** |
| Nine shows (17 episodes, 2 per feed) | **70%** |

Parity is show-specific. It does not generalize.

## Read this first: everything before today was measured on a broken pipeline

Every provider hardcoded `temperature=0.3` for insight generation and ignored the configured value.
The same config, on the same three episodes, run twice:

| | insights/ep | grounding | quotes/insight |
| --- | --- | --- | --- |
| run A | 28.0 | 79.8% | 1.51 |
| run B | 18.3 | 94.5% | 6.00 |

Grounding landed on either side of the 80% floor on a re-run of an *identical* configuration.
Nothing this repo has ever measured was free of that noise.

Fixed: the temperature is now honoured, and a re-run drifts by **0.7 insights and 0.9pp grounding**
instead of 9.7 and 14.7pp. Every number below was measured after the fix, with temperature pinned
to 0.

## Method — what is held constant

Only the LLM varies:

| held constant | how |
| --- | --- |
| episodes | pinned by transcript path + sha256 (a live feed silently drifts) |
| transcripts | cleaned, as production feeds them to the insight stage |
| ASR + diarization | identical DGX transcripts to both arms |
| insight ceiling | 50, non-binding for both |
| temperature | 0 (it was ignored entirely until today) |
| value gate | on, keeping tier >= 2 |
| **gate judge** | **one pinned judge (`claude-haiku`) for every arm** |

The pinned judge is load-bearing. Letting each model grade its own output is a same-vendor bias, and
it is large: qwen rejects **4%** of its own insights where an independent judge rejects **26%** of
the same ones. Unpinned, each arm would be filtered by a different strictness.

## Result 1 — Hard Fork, 10 episodes

qwen ran on the actual DGX (verified: `/api/ps` shows `qwen3.5:35b` resident in 29 GB of VRAM while
serving).

| per episode | gemini-2.5-flash-lite | qwen3.5:35b (DGX) |
| --- | --- | --- |
| insights kept | 19.2 | 19.4 |
| **grounded insights** | **17.1** | **16.1** |
| grounding rate | 89.1% | 83.0% |
| quotes per insight | 3.01 | 1.53 |
| wall-clock | 48s | 114s |

qwen produces **the same number of insights** as gemini and delivers **94%** of the grounded ones.
On this show they are equivalent.

## Result 2 — nine shows, 18 episodes (2 per feed)

This is the test that matters, and it is the one that says no.

| per episode | gemini | qwen |
| --- | --- | --- |
| grounded insights | 20.5 | 14.4 |
| grounding rate | 88.5% | 84.4% |

**qwen = 70% of gemini's grounded insights.** (Excluding one episode where *gemini* fell back to a
stub — including it flatters qwen to 76%, which would be crediting it for gemini's failure.)

Two per feed was deliberate: with one episode per feed you cannot tell a hard *show* from a noisy
*episode*. The per-episode spread is wide (44%–212%), so individual feed numbers are not
trustworthy; the aggregate over 17 episodes is.

**Reliability cuts the other way:** gemini produced a stub fallback on 1 of 18 episodes. qwen
produced none.

## Result 3 — all providers, same 3 episodes, one pinned judge

| provider | insights/ep | grounded/ep | grounding | quotes/insight | s/ep |
| --- | --- | --- | --- | --- | --- |
| anthropic (haiku-4.5) | 27.0 | 26.0 | 96.3% | 2.01 | 140 |
| **qwen3.5:35b (DGX)** | 24.3 | **20.3** | 83.6% | 1.48 | 131 |
| deepseek-chat | 18.3 | 18.0 | 98.2% | 1.91 | 48 |
| gemini-2.5-flash-lite | 23.0 | 18.0 | **78.3%** | 5.65 | 55 |
| openai (gpt-4o-mini) | 14.7 | 13.0 | 88.6% | 1.32 | 86 |

Three episodes is too small to rank on — treat this as directional. Note gemini falls *below* the
80% floor here while qwen clears it, which is the opposite of the head-to-head and a good
illustration of why n=3 decides nothing.

## What is left to close

**Evidence density.** qwen finds 1.5 quotes per insight against gemini's 3.0. Decomposed on
identical insights and transcript, the gap is **supply, not the gate**: gemini finds 4.75 candidate
quotes per insight (range 1–18, tracking how much the episode actually discusses the claim); qwen
finds 2.00 (range 1–3). The entailment threshold is already permissive, so it is not the lever.
Sharpening the prompt lifted qwen 2.00 → 2.50 and the ceiling did not move — it looks structural.

Insights with **zero** evidence are at parity (18% vs 19%), so qwen's evidence is *thin*, not
*missing*. Tracked separately; deferred unless a consumer actually uses multiple quotes per insight.

## Nine ways this comparison nearly lied to us

Every one produced a confident, wrong number. The comparison is only worth what the method is worth.

1. **A live RSS feed changes the episode set between runs.** `--max-episodes N` takes the newest N,
   so two runs an hour apart compared *different episodes*. A grounding "improvement" (66.7% → 72.7%)
   was reported as a fix and was actually different content. Retracted.
2. **Summary params were ollama-only.** Every cloud backend silently got the internal default. Two
   different configurations, reported as like-for-like — and it truncated gemini's summary on long
   episodes, aborting the run.
3. **The pinned judge inherited a deprecated model and 404'd.** Every classify call threw, the gate
   failed open, and a full 10-episode run completed **ungated** while reporting a healthy 44.3
   insights/episode. A broken gate and a permissive gate produce the same artifact; the only tell
   was zero drop-lines across ten episodes.
4. **The experiment allowlist silently drops unknown params.** It voided three gate settings in turn
   (enabled, provider, model). Each time the cell ran and looked like a result.
5. **A transcript set destroyed by the cleaning bug** (~144 chars per episode). A dataset built on
   those would be silently worthless. The builder now asserts transcript length.
6. **The ollama backend config had no `base_url` field** — every ollama eval cell in this repo's
   history ran on localhost. This machine has a local qwen3.5:35b, so the "DGX" arm ran on the
   MacBook and reported success. The first version of this report used those numbers.
7. **Half the prod-v2 corpus is a single unbroken line** (speaker turns concatenated, no separator).
   The insight stage produces **zero quotes** on those, for every provider, while reporting success.
   A dataset that mixed them in showed 8 of 20 episodes with zero grounded insights in *both* arms.
8. **The prompt store caches templates.** An A/B that swapped a template on disk compared the same
   prompt twice; byte-identical counts gave it away.
9. **Insight temperature was hardcoded** — see the top of this report. The worst of the nine,
   because it made every measurement unreliable rather than one stage.

The common shape: **the pipeline reports success while a stage is inert, misdirected, or destroyed.**

## Caveats

- Nine shows is broader than one, but it is still one corpus. Grounding rate carries real
  per-episode variance; individual feed numbers are noisy and only the aggregate is meaningful.
- The gate judge and one panel judge share a vendor. The openai panel judge is the independent
  check, and it agrees on the ranking.
- The 3-episode provider table is directional only.
