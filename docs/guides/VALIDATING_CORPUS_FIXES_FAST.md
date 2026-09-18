# Validating Corpus Fixes Fast

**Status:** Reference Guide
**Applies to:** Any change to speaker attribution, entity resolution, enrichment, or
anything else whose correctness is a property of the CORPUS rather than of a function
**Last updated:** September 2026

---

## The problem this guide exists for

A bug in this system usually is not "this function returns the wrong value". It is
**"the corpus says something untrue about an episode"** — a guest's name on the host's
voice, an organisation published as a person, a name nobody in the recording ever says.

That shape makes the obvious validation loop unusable:

- **The unit test can pass while the corpus is wrong.** Every function behaves as
  written; the defect lives in how ten of them compose over real audio.
- **The pipeline is long.** Reaching a naming decision means ingest, ASR, diarization,
  detection, resolution, record building, cleaning, ad removal, GI, KG, summary. A
  naming change touches two of those stages and pays for all eleven.
- **The corpus is big.** 4,543 stored episodes across 55 feeds. Evidence at that scale
  is why conclusions here are trustworthy; it is also why a naive loop costs a day.
- **Ground truth does not exist.** Nobody has hand-labelled who speaks in 4,543
  episodes. The corpus's own labels are the thing under repair, so they cannot be the
  answer key. (See "Never score against the roster" below — it has cost this project
  three reverted changes.)

Measured, that loop was: **~8.5 hours** for one pass of the DGX harness over 292
episodes. One iteration per working day, and a mistake in the measurement costs the
whole day.

The same question — *did this change make the corpus more truthful?* — can be answered
in **five seconds**. This guide is how.

---

## The principle

> Run the decision, not the pipeline. Ask the smallest population that can answer the
> question. Spend the expensive run on acceptance, never on iteration.

Everything below follows from that.

---

## Four tiers, cheapest first

Pick the cheapest tier that can actually answer your question, and only move up when
the answer demands it.

| tier | what it runs | population | cost | answers |
| --- | --- | --- | --- | --- |
| 1. Golden set | the decision functions, in-process | ~40 chosen episodes | **~5 s** | did I fix the case, and break a known-good one? |
| 2. Full replay | the decision functions, in-process | all 4,543 stored episodes | **~10 min** | what else moved, anywhere in the corpus? |
| 3. Targeted pipeline | the real CLI, one stage | 1–2 episodes | ~5–20 min | does it hold end-to-end, through the LLM and the artifacts? |
| 4. Harness | the real CLI, every stage | 292 episodes | **~8.5 h** | acceptance, for the PR |

### Tier 1 — the golden set (seconds)

Load each episode's **stored artifacts** (segments, diagnostics, metadata) from the
read-only snapshot and call the decision function directly. No CLI startup, no cleaning,
no GI/KG, no GPU.

What makes it work is that stored artifacts are a *complete input record*. The
diarization is already on disk, so a naming change can be evaluated against it without
re-deriving it.

The speaker-attribution one lives at `scripts/measure/2075_speaker_golden_set.py`:

```bash
.venv/bin/python scripts/measure/2075_speaker_golden_set.py --snapshot /path/to/corpus
```

Each case pins one claim in the language of the transcript, not the code:

```python
dict(id="dwarkesh-authored-asr-name", ep="Grant Sanderson _ AI and the fut",
     kind="absent", name="Drance Anderson",
     why='host says "chatting with Drance Anderson" — the ASR mangling of Grant Sanderson')
```

When it fails it prints *why it matters*, so the next reader can check the claim by
reading the episode rather than by reading the rule.

**Composition matters more than size.** A useful golden set is roughly half defects and
half known-good cases that a plausible fix would break. The known-good half is what
catches an over-broad rule, and it is the half people forget.

**A case that cannot run must say so loudly.** A case whose episode is absent from the
snapshot proves nothing, and silently counting it as "not failing" is how a suite drifts into
reporting green for cases it never executed. Report it as its own verdict (`n/a` here),
separate from pass and fail, and treat a rising `n/a` count as the suite rotting.

### Tier 2 — the full replay (minutes)

The same in-process call, over every stored episode, dumping
`{voice: (name, role, named, source)}` per episode. Diff two runs to get every voice
whose name changed, then **read the changes against the transcripts**.

This is the tier that answers "what else did I move?" — the question that a targeted
test can never answer, and the one that has repeatedly turned a confident fix into a
revert.

**Know what the replay cannot see.** It calls one entry point, so anything the real pipeline
does before or after that call is invisible — and worse, anything the pipeline *filters*
before the call will be present in your inputs and absent in production. Tier 1 here calls
`resolve_speaker_roster` and therefore never reaches `resolve_voices_and_roles`, where the
LLM answer is parsed and guarded; the `llm=True` switch injects **post**-resolution names,
downstream of that. A defect in the guarding logic must be tested where it lives, with a stub
`complete`, not through this harness. Write that limit into the harness itself — the case
list here carries it as a comment, because the next reader will otherwise conclude a working
fix is broken. (It happened. See the pitfall below.)

### Tier 3 — targeted pipeline (one or two episodes)

Some things are invisible to tiers 1–2 by construction: anything the LLM decides,
anything about the written artifacts, anything about stage routing. Run the real CLI on
one or two episodes. Choose the episode that exhibits the defect, not a convenient one.

### Tier 4 — the harness (hours)

The acceptance gate for a PR, not a feedback loop. Run it once, at the end, on the code
you intend to ship.

---

## Worked example A — a name the episode never says

**The defect.** On a Dwarkesh episode titled *Grant Sanderson*, the published transcript
credits 27 turns to **`Drance Anderson`** — the ASR's mangling of the guest's name,
bound to the answering voice by the introduction rule, while the correct
`Grant Sanderson` sat unbound in the metadata. This is the one thing #876 forbids: a
heuristic may identify a voice, never author a name.

**Why the slow loop would have been wrong anyway.** The first classifier compared the
SET of names per episode before and after. That episode also gained `Dwarkesh Patel`, so
the loss was filed as "replaced by another name" — a benign bucket — and the authored
name never surfaced. Aligning **voice by voice** instead (pair transcript lines by
identical text, compare the label on each) showed 643 lines where a voice changed from
one name to a *different* name. The regressions were all in there.

> **Lesson.** Never bucket a corpus regression by episode. An episode-level set
> comparison hides exactly the case you care about: the same voice, a different person.

**The fast loop.** One golden case pins the defect and a second pins the person who
should be there instead:

```text
FAIL dwarkesh-authored-asr-name   Drance Anderson   published on ['SPEAKER_01']
ok   dwarkesh-correct-guest       Grant Sanderson
```

Five seconds, no GPU. Any candidate fix is now measurable before it is committed, and
the accompanying known-good cases (`netflix-not-gergely`, `journal-not-ryan`) fail
immediately if the fix over-reaches.

---

## Worked example B — a stated guest erased from the record

**The defect.** On a show with two feed-stated hosts, the speaker-name cap
(`hosts[:cap] + guests[:cap - len(hosts)]`) leaves an empty guest slice. Both pipeline
lists derive from that one capped value, so the guest reaches neither `detected_guests`
nor `metadata_named` — and the record builder reads only those. The person vanishes from
`content.speakers` entirely, not even as `placed: false`.

`Mackenzie Price` appears in **no record file in the whole control corpus**, on an
episode whose transcript says *"So let's bring in Alpha School cofounder, Mackenzie
Price"* followed by *"Thanks for having me"*.

**Why a targeted population beat the corpus.** A previous attempt removed the cap
outright and was reverted: uncapping pushed wrong names through the arithmetic
("one spare name, one spare voice") path. The decisive population was not 4,543
episodes but the **56 multi-host episodes** that killed that attempt, with the real
uncapped detector output stored on disk. Replaying just those distinguished the
policies in seconds:

| policy | changes | forced names |
| --- | --- | --- |
| full uncap (the reverted one) | 13 | **6 wrong** |
| seat-cap the arithmetic list only, keep the full stated list | **7** | **zero** |

> **Lesson.** When a previous attempt failed, the episodes that failed it ARE the
> regression suite. Keep their inputs on disk; they are worth more than a larger sample.

**And know when the cheap tiers simply cannot answer.** This fix is upstream of everything
the stored artifacts record: detection runs at ingest, so `detected_guests` is frozen in the
diagnostics as the capped list it was. Tiers 1 and 2 replay those artifacts, so neither can
see the change at all — the golden case for it fails identically before and after, and that
is the harness being honest rather than the fix being wrong. A fix upstream of the recorded
inputs is verified by a unit test at the decision itself plus a tier-3 run, and nowhere else.
Say so out loud; a tier that structurally cannot see your change must never be reported as
having cleared it.

---

## Pitfalls, each of which cost real time here

**Never score against the roster.** The corpus's own labels are the thing under repair.
Three changes were built, measured against those labels, shipped or nearly shipped, and
reverted. Score against evidence a person can check: a voice stating its own name, a
host naming a guest, who reads the outro, who is addressed by name.

**A wrong measurement is worse than no measurement.** It produces confident, wrong
conclusions faster. Budget for verifying the instrument, not just the fix.

**Measure how often a candidate rule FIRES before arguing about whether it is right.**
The firing rate usually settles the design on its own, and it is far cheaper to obtain
than a correctness judgement. "A voice greeted by name is not that person" is obviously
true, and the only real question is which textual shape to match. Counted over the
corpus's 6,121 named voices: the start-of-voice shape (`"Hey, Jordan."`) fires on 2
records, both the episode the bug was reported on; the sentence-anywhere shape
(`", Jordan."`) fires on 858 of 4,896 self-introduced voices, because that is not people
being addressed, it is diarization bleed. Same rule, same intuition, two orders of
magnitude apart — and no amount of reasoning about the rule would have told you which.

**A harness that skips a filter will manufacture a defect, and you will fix it.** The worst
case is not a harness that misses a bug — it is one that invents a plausible one. The swap
rule is gated on the episode having exactly two voices. Replaying Ground Truths showed
*three*, because a 9.4-second cluster saying `" Yeah."` counted as a participant; the rule
therefore never fired on the very episode its code comment was written for. That is a
convincing bug. A threshold was written, justified from a corpus-wide distribution, and
measured — all of it rigorous, all of it wasted, because the real pipeline passes
`real_voice_texts`, already filtered by `classify_voices` at `cameo_max_talk_s = 20s`. The
gate was correct the whole time; the harness was not.

> **Lesson.** Before fixing a defect only your harness can see, find the caller in the real
> pipeline and read what it actually passes. "My replay disagrees with production" is a
> statement about the replay until proven otherwise. Rigor downstream of a bad premise is not
> rigor — measuring carefully made the wrong fix *more* convincing, not less.

The residue worth keeping is a comment at the gate saying why the count is already correct,
so the next person replaying raw clusters reaches the answer without repeating the detour.

**Verify the instrument before trusting its verdict.** The replay harness recomputed
feed hosts for nine hand-picked feeds and reused stored values everywhere else, unlike
the real pipeline — a difference on 89 episodes across 3 feeds. It was found by
comparing the harness against the code path it claimed to imitate, not by looking at
results.

**Compare like with like.** Re-running a repair over an already-repaired corpus measures
a repair of a repair. Always take a fresh copy from the read-only snapshot.

**Replaying a model's stored answers tests guards, not prompts.** Stored diagnostics
record the verdicts that survived the guards. Injecting them is the right way to ask
"would my new guard refuse this?" and the wrong way to evaluate a fix that works by
changing what the model is asked — doing so briefly "resurrected" a name the fix had
successfully eliminated.

**A diagnostic can defeat its own watchdog.** A stalled-run detector keyed on log
freshness stopped working the moment the stalled loop began logging why it was stuck.
Detect *progress* (episodes completed), never liveness (bytes written).

**Distinguish "not caused by this change" from "not a problem".** Both deserve to be
written down; only one of them blocks a merge. Establish which by running the *old code*
on the *same inputs*, not by comparing against production data, which was produced by a
different code path entirely.

---

## Adding to the golden set

Add a case whenever you read an episode and form a judgement about it — that reading is
the expensive part, and a case makes it permanent. Also add one for every bug found in
production and every regression caught in review.

A good case names the episode, the claim, and the transcript evidence for it. If you
cannot write the `why` from the transcript, you do not yet understand the case well
enough to pin it.

---

## Related

- [Corpus Reprocessing](CORPUS_REPROCESSING.md) — the reprocess stages these loops exercise
- [DGX Runbook](DGX_RUNBOOK.md) — the GPU host tiers 3 and 4 depend on
- [Agent-Pipeline Feedback Loop Guide](AGENT_PIPELINE_LOOP_GUIDE.md) — structured run artifacts
- [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md) — where these tiers sit beside the test suite
