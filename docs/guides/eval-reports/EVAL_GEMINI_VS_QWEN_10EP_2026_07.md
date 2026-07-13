# Gemini vs qwen3.5:35b on the DGX — 10 prod episodes, one pinned judge

**Date:** 2026-07-13
**Dataset:** `prod_v3_10ep_v1` — 10 episodes curated from the existing prod-v3 DGX corpus
(`tailnet_dgx_whisper` / `faster-whisper-large-v3`, `tailnet_dgx` diarization). Pinned by
transcript path + sha256. **Not re-fetched from the live feed.**
**Runs:** `data/eval/runs/h2h_gemini_10ep`, `data/eval/runs/h2h_qwen_10ep`

## Question

Can qwen3.5:35b on the DGX match `gemini-2.5-flash-lite` on grounded-insight quality, so prod v3
can be built entirely on our own hardware?

## Answer

**Yes, on quality.** qwen delivers **90%** of gemini's grounded insights per episode, clears the
ADR-053 grounding contract, and a two-judge panel scores it **at or above gemini on CORE insights
with half the filler**.

The remaining gap is **evidence density** — 1.27 vs 2.08 quotes per insight. qwen finds fewer
supporting passages for each claim it makes.

## Method — what was held constant

Only the LLM varies:

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
same-vendor bias, and it is large: qwen rejects **4%** of its own insights where an independent
judge rejects **26%** of the same ones.

## Results

qwen ran on the **actual DGX** (verified: `/api/ps` shows `qwen3.5:35b` resident in 29 GB of VRAM
while the request is served). An earlier version of this report used numbers from a run that
silently executed on the laptop — see §Pitfalls 6.

| per episode | gemini-2.5-flash-lite | qwen3.5:35b (DGX) |
| --- | --- | --- |
| insights kept (post-gate) | 24.4 | 22.2 |
| **grounded insights** | **20.1** | **18.0** |
| grounding rate | 82.4% | 81.1% |
| quotes / episode | 49.8 | 28.1 |
| quotes per insight | 2.08 | 1.27 |
| wall-clock | 33s | 119s |

Both clear ADR-053 (grounding ≥ 80%). **qwen delivers 90% of gemini's grounded insights.**

### The judge's verdict — gemini pads, qwen does not

Same judge, both arms, so the rejection rates are directly comparable:

| arm | emitted | rejected as filler | reject rate |
| --- | --- | --- | --- |
| gemini | 440 | 196 | **45%** |
| qwen | 259 | 56 | **22%** |

**Nearly half of gemini's insights are filler. Only a fifth of qwen's are.** Gemini is prolific and
noisy; qwen is sparser and disciplined. The raw insight counts flattered gemini because we were
counting its padding as knowledge.

### Two-judge panel (blind, shuffled, tiered)

Judges vendor-disjoint from both candidates. Scored on post-gate insights.

| metric | judge | gemini | qwen (DGX) |
| --- | --- | --- | --- |
| CORE / ep | anthropic | 6.6 | **9.0** |
| CORE / ep | openai | 11.6 | 11.2 |
| USEFUL+ / ep | anthropic | 18.4 | **19.0** |
| USEFUL+ / ep | openai | 24.0 | 21.8 |
| FILLER / ep | anthropic | 6.0 | **3.2** |
| FILLER / ep | openai | 0.4 | 0.4 |

**qwen matches or beats gemini on CORE, with half the filler.**

Inter-judge agreement: exact 52–61%, within-1 **92–96%**, keep/drop 76–88%. The judges disagree on
absolute strictness (openai is markedly more lenient) but **rank the arms identically** — the
conclusion is judge-independent even though the absolute tiers are not. Do not quote a single
judge's absolute numbers as fact.

## What is left to close

1. **Evidence density** — 1.27 vs 2.08 quotes per insight. Decomposed on identical insights and
   transcript, the gap is **supply, not the gate**: gemini finds 4.75 candidates per insight
   (range 1–18, tracking how much the episode discusses the claim); qwen finds 2.00 (range 1–3).
   Both arms ran the NLI threshold at 0.5, which already accepts everything above "related but does
   not support" — so the threshold is not the lever. Making the prompt's expectation concrete
   ("sweep the whole transcript, two quotes is usually too few") lifted qwen to 2.50 (+25%), which
   is shipped. qwen still looks structurally reluctant past ~3.
2. **Speed** — 119s vs 33s per episode. Known, expected, not a quality question.

## Pitfalls found while building this comparison

Every one would have produced a confident, wrong number. The comparison is only worth what the
method is worth.

1. **A live RSS feed silently changes the episode set.** `--max-episodes N` takes the newest N, so
   two runs an hour apart compared *different episodes*. An earlier grounding delta (66.7% → 72.7%)
   was reported as a fix and was actually different content. **Retracted.**
2. **Summary params were ollama-only.** `_episode_summary_params` passed `max_length` for ollama and
   *nothing* for every cloud backend. Two different configurations, reported as like-for-like — and
   it truncated gemini's summary on long episodes, aborting the run.
3. **The pinned judge inherited a 404 model.** `gi_value_gate_provider: anthropic` swapped the
   provider but inherited its *default* model (deprecated, 404). Every classify call threw, the gate
   failed open, and a full 10-episode run completed **ungated** while reporting a healthy 44.3
   insights/episode. A broken gate and a permissive gate produce the same artifact; the only tell was
   zero drop-lines across ten episodes.
4. **The eval allowlist silently drops unknown params.** It voided three gate settings in turn
   (`enabled`, `provider`, `model`). Each time the cell ran and looked like a result.
5. **A transcript set destroyed by the cleaning bug.** Three of six 10-episode prod-v3 runs have
   `.cleaned.txt` files of ~144 chars. A dataset built on those would be silently worthless. The
   builder now asserts transcript length.
6. **`OllamaBackendConfig` had no `base_url` — every ollama eval cell ever ran on localhost.** The
   schema declared only `type` and `model`, so a `base_url` in the YAML was dropped by pydantic, and
   run_experiment's ollama branch never applied one. This machine has a local ollama with
   qwen3.5:35b, so the "DGX" arm ran on the MacBook and reported success. The first published
   version of this report used those numbers.
7. **The prompt store lru_caches templates.** An A/B that swapped a template on disk compared the
   same prompt twice; byte-identical counts gave it away.

The common shape: **the pipeline reports success while a stage is inert, misdirected, or
destroyed.** Every check here exists because that happened.

## Caveats

- **10 episodes from ONE feed** (Hard Fork). This says which stack is better *for this show*. It is
  not a general result.
- Grounding rate carried **±10pp run-to-run noise** at n=3; 10 episodes reduces but does not remove it.
- The gate judge (`claude-haiku`) and one panel judge (`claude-sonnet`) share a vendor. The openai
  panel judge is the independent check, and it agrees on the ranking.
