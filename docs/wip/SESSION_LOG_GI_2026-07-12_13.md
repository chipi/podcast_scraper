# Session log — grounded insights, gemini vs qwen, 2026-07-12 → 07-13

A chronological record of what we did, what we claimed, what we retracted, and — most importantly
— **what we still have not verified.** Written as a checkpoint before deciding the 100-episode
reprocess.

---

## THE BIGGEST GAP: we measured a configuration production does not run

| setting | production (`prod_dgx_only.yaml`) | what every eval measured |
| --- | --- | --- |
| `gi_value_gate_enabled` | **False — the gate is OFF** | ON |
| `gi_value_gate_provider` | unset → each model grades itself | pinned to claude-haiku |
| `gi_max_insights` | **12** | 50 |
| `gi_insight_chunk_chars` | **0 — no chunking** | 30000 (in the chunked runs) |
| `ollama_temperature` | **0.3 — not reproducible** | 0 |

**Production today emits at most 12 insights, ungated, at a random temperature.** Every quality
number in this session describes a stack we have never actually configured. Before any corpus
build we have to decide, deliberately:

- What is the real insight ceiling? 12 was never derived from anything.
- Is the value gate on in production? If yes, **which judge** — self-grading is ~6x more lenient
  (qwen rejects 4% of its own insights; an independent judge rejects 26% of the same ones), and an
  independent judge means an API call per episode on a local-only stack.
- Is chunking on? It is a large quality lever and a large cost.
- Is temperature pinned? A corpus build that cannot be reproduced cannot be debugged.

Nothing else in this document matters until those four are answered.

---

## Timeline

### Part 1 — nine silent defects

Every one produced plausible output and reported success.

1. **Insight clamp.** All 7 providers did `max_insights = min(max(1, max_insights), 10)`. Every
   profile asked for 12; every prompt said 10. Dead knob repo-wide since March. (`f16d1f30`)
2. **50k transcript cut.** 6 providers truncated the transcript before quote extraction. Episodes
   run 67–117k chars, so the last third was invisible. Of 1418 grounded quotes in the v2 corpus,
   **4** fall beyond the cut; the last 20% of every episode was empty. We had read that as "insight
   density is front-loaded". It was the truncation. (`848448bd`)
3. **A 3-quote cap in our own prompt**, plus gemini's anti-duplicate wording never propagated to
   the other five providers. (`5e1ae3f2`)
4. **A 512-token reply cap** that would have silently eaten the uncapped quote prompt — an overrun
   is cut mid-JSON and yields *zero* quotes. (`bb8f7787`)
5. **Stub fallback.** A truncated insight list threw away the whole episode; the guardrail fired,
   the exception was swallowed, and the episode landed with 1 placeholder insight while the run
   reported success. Zeros went 8/15 → 0/8. (`ab3da1b9`)
6. **`classify_insights` broken on anthropic** — it fences its JSON. Found by a per-provider smoke
   test; 5 of 7 providers had never executed that code path. (`61f467d6`)
7. **The pinned judge inherited a deprecated model and 404'd.** Every classify call threw, the gate
   failed open, and a full 10-episode run completed **ungated** while reporting a healthy 44.3
   insights/episode. A broken gate and a permissive gate produce identical artifacts. (`ecae6a2a`)
8. **The ollama backend config had no `base_url`.** Every ollama eval cell in this repo's history
   ran on localhost. This machine has a local qwen3.5:35b, so the "DGX" arm ran on the laptop and
   reported success. (`e6a1b501`)
9. **Insight temperature hardcoded to 0.3**, ignoring config. The same config on the same 3
   episodes gave 28.0 vs 18.3 insights/ep and grounding on either side of the 80% floor (79.8% vs
   94.5%). **Nothing this repo has ever measured was free of that noise.** The operator caught it —
   a 28-vs-16 swing was too large to be variance, and it was not. (`f3cf7b0f`)

Plus: the eval allowlist silently dropped four separate settings in turn; the prompt store's cache
made an A/B compare the same template twice; a transcript set destroyed by the cleaning bug.

### Part 2 — claims made and retracted

| claim | status |
| --- | --- |
| "The clamp fix lifted grounding 66.7% → 72.7%" | **RETRACTED** — loop3 and loop4 ran on different episodes (feed drift) |
| "Insight density is front-loaded in podcasts" | **RETRACTED** — it was `[:50000]` |
| "The candidate-supply gap is self-inflicted" | **OVERSTATED** — the 3-quote cap was worth ~+0.15/call, not 2x |
| "qwen is at parity with gemini" | **SHOW-SPECIFIC** — 94% on Hard Fork, 70% across nine feeds |
| "The DGX arm ran on the DGX" | **FALSE** — it ran on the laptop |
| "Chunked insights ground better (95–98%)" | **WRONG** — my offline test bypassed the QA and NLI gates. In-pipeline, chunked qwen grounds at **75.1%, below the floor** |
| "Chunking will close the gap" | **WRONG** — it widens it: 94% → 64% |

### Part 3 — what the evidence currently says

Reproducible pipeline, cleaned transcripts, pinned judge, temperature 0:

| | gemini | qwen (DGX) |
| --- | --- | --- |
| Hard Fork, 10 eps — grounded/ep | 17.1 | 16.1 (**94%**) |
| Nine shows, 17 eps — grounded/ep | 20.5 | 14.4 (**70%**) |
| grounding rate | 88–89% | 83–84% |

**Root cause of the 94-vs-70 gap:** qwen's insight count is **flat** against episode length (+2.3
insights going from a <40k to a >=40k transcript) while gemini scales (+6.6). qwen saturates at
~18 insights **per call**. Hard Fork episodes happen to sit right at that ceiling, which is why
they looked equal. The target format (45–90 min) is exactly where the ceiling bites.

**Chunking** (more passes, not a bigger window — context was never the limit):

| | 1-call | chunked |
| --- | --- | --- |
| gemini grounded/ep | 17.1 | 43.2 (**+153%**) |
| qwen grounded/ep | 16.1 | 27.7 (**+72%**) |
| qwen grounding | 83.0% | **75.1% — out of contract** |
| qwen as % of gemini | 94% | **64%** |

Chunking is a large win for the pipeline and **not** the fix for the DGX gap. It amplifies whatever
each model can do per pass, so the stronger model gains more.

---

## Gaps — what we have NOT verified

Ordered by how much they could change the decision.

1. **The production config question above.** Nothing else matters first.
2. **Why does chunked qwen ground at 75% in-pipeline but 95%+ in an offline probe?** The offline
   probe only asked whether `extract_quotes` returned anything; the pipeline also applies the QA
   and NLI gates. Which gate rejects chunk-local insights, and why? This is the whole "solve
   grounding for qwen" problem and it is not yet diagnosed.
3. **Is the chunked CORE gain real post-gate?** The blind judge scored chunked insights +62% CORE,
   but that was measured **pre-gate**. Post-gate CORE is unknown.
4. **The value gate's cost and vendor story in production.** An independent judge means a cloud API
   call per episode on a stack whose entire point is to be local. Self-grading is 6x too lenient.
   There is no third option identified yet.
5. **Judge disagreement.** The two-judge panel agreed on keep/drop for only 76–88% of insights. The
   value gate uses **one** judge. That imprecision is unmeasured in the final numbers.
6. **The all-provider table is 3 episodes** — far too noisy to rank providers on. It is directional
   only, and was presented as such.
7. **The Omny feed has no diarized run at all** and was excluded from the generalization set. Its
   behaviour is unknown, and it is in the production corpus.
8. **Are there blob transcripts in prod-v3?** 58% of prod-v2 is single-line blobs (the pre-diarization
   May runs). We never checked v3.
9. **Chunking cost is unmeasured in-pipeline.** Roughly 3x the calls; the wall-clock impact on a
   100-episode DGX run has not been measured.
10. **Chunk boundaries are naive** (equal splits on line/sentence boundaries). A claim spanning a
    boundary could be missed by both chunks. Untested.
11. **Two datasets, two corpora.** Hard Fork uses prod-v3 transcripts; the generalization set uses
    prod-v2 June. ASR/diarization may differ between them, which weakens cross-comparison.
12. **Evidence density** (qwen 1.5 vs gemini 3.0 quotes/insight) — deferred as *thin, not missing*
    (zero-evidence insights are at parity, 18% vs 19%), with explicit reopen conditions.

---

## The pattern worth remembering

**The pipeline reports success while a stage is inert, misdirected, or destroyed.** Nine times in
one session. Every defect was caught by a human staring at 3–20 episodes by hand — which is exactly
what nobody will do for 100 episodes. That is why the per-episode quality telemetry is a
precondition for the corpus build, not a follow-up.
