# prod-v3: gemini vs qwen, on the pipeline we would actually ship

**Date:** 2026-07-14
**Dataset:** `prod_v3_adr110_18ep_v1` — 18 episodes, 9 shows, relabelled with ADR-110 + #1188
**Runs:** `final_gemini_18ep`, `final_qwen_18ep`
**Judge:** `claude-haiku-4-5` pinned, tier floor 2 — cross-vendor to both candidates (#939)
**Supersedes:** `EVAL_PROD_V3_TWO_AXIS_18EP_2026_07.md`, whose numbers were measured through a
pipeline with three defects (below) and must not be used.

## Why the earlier run does not count

The 18-episode comparison run earlier the same day was measured through:

1. **an eval harness that never passed the diarized segments to GI** — so the ad-drop, `surfaceable`
   and person-node gates were all dead in eval while live in production;
2. **a value gate that graded a bare sentence, before grounding** — it could not see the quote, the
   speaker, or that nothing supported the insight at all;
3. **a corpus whose speaker labels predate ADR-110 and #1188.**

All three are fixed. Both arms below run the same code, on the same 18 transcripts, with the same
pinned judge. Only the LLM differs.

## Results

| | gemini-2.5-flash-lite | qwen3.5:35b (DGX) |
| --- | ---: | ---: |
| insights generated | 787 | 368 |
| flagged `EVIDENCE: NONE` | 114 (**14%**) | 73 (**20%**) |
| dropped by the judge | 242 | 109 |
| kept | 545 | 259 |
| **surfaceable / episode** | **21.6** | **10.4** |
| unsurfaceable / episode | 8.7 | 3.9 |
| quotes / episode | 59.8 | 25.0 |
| distinct people named | 24 | 25 |
| ungrounded insights shipped | **0** | **0** |
| **judge KEEP rate** | **69%** | **70%** |

**gemini delivers 2.06x the surfaceable knowledge per episode.**

## The finding that reverses this morning's conclusion

Earlier today, measured through the blind judge, qwen kept **84%** of its insights against gemini's
**53%**, and the conclusion was: *"qwen's insights are not worse — there are simply fewer of them."*

**That gap was an artefact of the blind judge.** Shown the evidence — the verbatim quote, who said
it, and whether anything supports the claim at all — the judge keeps **69% of gemini and 70% of
qwen**. The quality difference the old measurement reported does not exist.

What remains is volume, and it is not close:

* gemini generates **2.1x** more insights,
* grounds **2.4x** more evidence (59.8 vs 25.0 quotes/episode),
* and **hallucinates less**: 14% of its output has no verbatim support, against qwen's **20%**.

A blind judge is lenient toward short, safe, generic output — which is precisely what qwen produces
more of. Giving the judge its evidence removed the flattery.

## What this means for prod-v3

The overarching question was: *can qwen3.5:35b on the DGX build prod-v3 at least as well as gemini
built prod-v2?*

**On this evidence, no.** It delivers **half the knowledge per episode** and a **higher hallucination
rate**. The corpus would be materially thinner and less trustworthy.

Cost is not the counter-argument: gemini built the entire 99-episode corpus for **$0.51**. The case
for the DGX rests on privacy, unmetered eval sweeps, and independence from a cloud vendor — all real,
none of them worth halving the corpus.

**The pipeline itself is in far better shape than either model.** Zero ungrounded insights ship in
either arm (44 did this morning), the ad narrators are gone from the graph, and speaker naming went
from 65.2% to 71.9% of talk across the 90-episode corpus.

## Open

* **A 100-episode reprocess requires operator approval and has not been run.**
* If local-only is the goal, the lever is qwen's **volume**, not its quality: it is under-generating
  by ~2x at the same judged quality. That is a prompt/sampling problem, and it is the one worth
  attacking before this comparison is repeated.
