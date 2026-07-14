# prod-v3: gemini vs qwen, and v2 vs v3 — 18 episodes, 9 shows

**Date:** 2026-07-14
**Dataset:** `prod_v3_crossshow_18ep_v1` — 18 episodes, 2 from each of 9 shows, 826k chars
**Runs:** `h2h_gemini_18ep`, `h2h_qwen_18ep`, `h2h_gemini_v2settings_18ep`
**Question:** can qwen3.5:35b on the DGX build prod-v3 at least as well as gemini built prod-v2?

---

## Design: two axes, one variable each

The corpus-battle pilot compared gemini-on-v2-transcripts against qwen-on-v3-transcripts and
measured **transcript x LLM together** — its own `caveat` field says so. This run fixes that. All
three arms read the **same 18 transcripts** with the **same speaker labels**:

| axis | arms | what varies | what is held constant |
| --- | --- | --- | --- |
| **1 — the MODEL** | `v3/gemini` vs `v3/qwen` | the LLM | pipeline, transcripts, judge |
| **2 — the PIPELINE** | `v2/gemini` vs `v3/gemini` | the GI layer | LLM, transcripts, speaker labels |

**The v2 arm is not a guess.** Its settings were read off `9debc8d2` — the commit that actually
produced the prod-v2 corpus (2026-05-22, per `snapshot.manifest.json`):

- `gi_max_insights` = 20 (`DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX` at that commit)
- **no value gate** — `gi_value_gate_*` did not exist
- **no voice gates** — `gi/pipeline.py` had **zero** references to `voice_type` / `surfaceable`, so an
  advertisement could be grounded and an unnameable voice could mint a `Person` node

Reproduced by stripping `voice_type` from the segments while keeping the labels. v2 *did* attribute
speakers, just badly.

**Scope limit, stated up front:** axis 2 isolates the **GI layer** (cap, value gate, voice gates). It
does **not** re-measure the roster/naming rewrite, because both arms are given v3's speaker labels.
The roster was measured separately — the 160-episode warrant audit ("every named speaker has a
warrant") and coverage 75.60% -> 77.45%.

---

## Results

| metric | v2 / gemini | v3 / gemini | v3 / qwen |
| --- | ---: | ---: | ---: |
| episodes | 18 | 18 | 18 |
| insights generated | 344 | 756 | 354 |
| dropped by the pinned judge | — (no gate) | 358 | 70 |
| **kept** | 344 | 398 | 298 |
| **surfaceable / episode** | **19.1** | **18.7** | **13.7** |
| unsurfaceable / episode | 0.0 | 3.4 | 2.9 |
| quotes / episode | 43.5 | 46.3 | 23.3 |
| Person nodes / episode | 2.8 | 1.6 | 1.6 |
| **fake `SPEAKER_NN` Persons** | **19** | **0** | **0** |
| distinct real people named | 24 | 23 | 24 |
| avg latency / episode | 24s | 46s | 91s |

Judge: `claude-haiku-4-5` pinned, tier floor 2 (>= USEFUL) — cross-vendor to both candidates, per
the same-vendor bias rule (#939).

---

## Axis 1 — the model: gemini wins on volume, qwen on density

**gemini delivers 1.37x the surfaceable knowledge per episode** (18.7 vs 13.7).

But the interesting number is the other one:

| | insights generated / ep | judge KEEP rate |
| --- | ---: | ---: |
| gemini | 42.0 | **53%** |
| qwen | 19.7 | **84%** |

**qwen's insights are not worse — there are simply fewer of them.** Gemini writes roughly twice as
much and the judge throws away nearly half of it as filler; qwen writes less and 84% of it survives.
This reproduces the 10-episode finding exactly, on 9 shows instead of one.

Where qwen genuinely lags is **evidence**: 23.3 quotes/episode against gemini's 46.3. It grounds half
as much of what it claims.

## Axis 2 — the pipeline: same volume, categorically better provenance

v3 surfaces **18.7** insights/episode against v2's **19.1** — flat. That flatness is the story, and it
needs one caveat to read correctly: **v2 was hitting its cap.** With `gi_max_insights` = 20 it
produced 19.1/ep, i.e. it was truncated, not exhausted. v3 raises the cap to 50, generates 42/ep, and
then *removes* 358 insights the judge called filler — landing on the same surfaced volume.

So at equal volume, v3's insights are:

- **judge-approved** — 358 filler insights that v2 would have shipped are gone
- **speaker-attributable** — **19 fake `SPEAKER_NN` Person nodes** in v2, **zero** in v3
- **honestly marked** — 62 insights spoken by voices nobody names are flagged unsurfaceable. v2
  published every one of them **as somebody's opinion**

### The one real cost, stated plainly

v3 names **2 real people fewer** than v2 in the GI graph (`Patrick Healy`, `Patrick O'Shaughnessy`),
and one more (`Henry Hartevelt`). Net **-1 real person across 18 episodes, against -11 fake ones**.

It is not a gating failure — the segments type both men as `person` (318 / 206 / 14 segments under
their real names), so no gate touched them. In v3's insight set no grounded quote landed on their
turns: O'Shaughnessy is the *interviewer* on Invest Like the Best, and the substantive claims belong
to his guest. The roster still names him; he simply has no insight of his own that episode.

**A GI `Person` node means "this person said something we grounded", not "this person was in the
episode".** Worth remembering before anyone reads Person-node counts as a naming metric.

---

## What this run also caught

The first attempt at this comparison would have been **invalid**, and the reason is worth recording.

`run_experiment.py` never passed `transcript_segments` to `build_artifact`. Every voice gate — the
ad-drop, `surfaceable`, the person-node guard — was **dead in eval while live in production**. The
head-to-head that was about to decide prod-v3 would have scored a pipeline **neither model would ever
run under**.

That matters more than it sounds. Ad copy is *written* to be quotable; it is the most fluent, most
confident false insight available. An ungated harness does not merely omit the gate — it actively
rewards whichever model hallucinates most eagerly on advertisements.

The offsets were the trap underneath it: `char_start`/`char_end` index the **raw** screenplay, and the
eval preprocesses the text before summarising. Handing GI the cleaned text with raw offsets leaves
only **0-8%** of segments resolving — every voice lookup lands on whichever speaker the shift
happened to hit. So the loader now refuses segments that do not index the text it is about to ground
(>= 95% must resolve): **a mis-attributing gate is worse than an absent one.** All 18 episodes in
every arm reported 100% alignment.

Guarded by `tests/unit/scripts/eval/experiment/test_eval_reads_diarized_segments.py`, mutation-tested
(4 mutations, 4 red). Full write-up: `docs/wip/CORPUS-V4-FIXTURE-LADDER.md` §A/§D.

---

## Verdict

**gemini is ahead on delivered knowledge (1.37x) and on evidence (2x the quotes). qwen is ahead on
precision (84% vs 53% judge keep-rate) and is the only option that keeps the corpus off the cloud.**

The v3 *pipeline* is a clear improvement over v2 regardless of model: same surfaced volume, no fake
people, no filler, no ad copy, and unattributable claims honestly marked.

The open decision is the model, and it is a product call, not a metrics call:

- **13.7 vs 18.7 surfaceable insights/episode** — is a ~27% smaller corpus acceptable in exchange
  for no cloud dependency?
- Cost is **not** the argument for the DGX: gemini built the entire 99-episode corpus for **$0.51**.
  The case for local rests on privacy, unmetered eval sweeps, and independence.
- Latency: 91s/ep (qwen) vs 46s/ep (gemini) — over 100 episodes, ~2.5h vs ~1.3h. Not decisive.

**A 100-episode reprocess requires operator approval and has not been run.**
