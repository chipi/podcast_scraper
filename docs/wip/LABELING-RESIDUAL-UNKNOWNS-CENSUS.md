# Residual `unknown` voices after naming-4 — what's left and what's actually fixable

- **Status**: Corpus-wide census + honest tractability verdict.
- **Date**: 2026-07-29
- **Method**: For every substantial (≥120s) still-`unknown` voice across the 100-ep gate corpus, read
  the **pipeline's own recorded `reason`** (not a re-derived guess) + whether a candidate name is
  actually available and actually spoken. Buckets below are from that census.

## The census (66 substantial `unknown` voices)

| Count | Bucket | Name available? | Safely bindable in naming? |
| --- | --- | --- | --- |
| ~30 | **R1 — metadata = production credits, not speakers** | Only credits (engineer/editor/producer) | **No** — binding credits = wrong name |
| ~26 | **R2 — more substantive voices than candidate names** | No (panel/callers, no self-intro, not in metadata) | **No** — nothing to bind |
| ~6  | **R1 — metadata name never spoken at all** | Name exists but never uttered | **No** — no in-episode anchor |
| ~4  | **tape — no name anywhere** | No | Reclassify `unknown`→`unidentified` (cosmetic; adds no name) |

## Why the biggest bucket is NOT a naming bug (the finding that killed my optimism)

I first read "the metadata name IS spoken in-episode" as a **missed host-intro anchor** — 30 voices,
looked tractable. It is **not**. Reading the actual context on NPR/Planet Money 0005:

- metadata = `[Sophia Paliza-Carre, Sierra Juarez, Maggie Luthar, Alex Goldmark, Dan Wang]`
- those names appear as **production credits**: *"The show is engineered by Maggie Luthar"*, *"edited
  by Alex Goldmark"*, *"Dan Wang is a fellow at Stanford"* (a third-person subject).
- the 5 unnamed on-air voices (288/257/222/210/140s) are the **reporters/host**, whose names are
  **not in metadata at all**.

So the closed list (#876) contains the *wrong* names (credits) and lacks the *right* ones (on-air
reporters). Binding anything here paints an editor's name onto a reporter's voice — exactly the
failure #876 exists to prevent. **The declines are correct.**

## Honest bottom line

**naming-4 already extracted essentially all the safely-bindable names.** The residual `unknown`
voices are unbindable *in the naming layer* because the name is one of:
(a) a production credit that must NOT be bound, (b) absent — more voices than candidate names, or
(c) never spoken. Adding names to these requires **new capabilities, not naming tweaks**:

1. **Speaker-name extraction from on-air narrator handoffs** (NPR desks): bind a reporter introduced
   *on air* ("here's Sophia in Beijing" → next voice) even when absent from metadata. This is a
   **guarded #876 relaxation** and needs an intro-vs-credit disambiguator — *"here's Sophia"* binds,
   *"edited by Alex Goldmark"* must not. Real, but its own design + risk; the 0005 metadata is 100%
   credits, so it would fire only on episodes that actually introduce reporters on-air.
2. **Feed-metadata enrichment / speaker-vs-credit signal**: the metadata frequently lists credits, not
   speakers — a data-quality problem upstream of naming.
3. **Re-diarization** (`rediarize_only`): the merged-cluster cases (flightcast Lukas/Axel) need split
   clusters, not more naming heuristics.

The only zero-risk in-layer change is cosmetic: reclassify the ~4 genuinely-nameless tape voices
`unknown`→`unidentified` so they stop counting as naming *defects* (they add no names). Everything
that adds a real name is capability (1)–(3) above, each a separate design decision.
