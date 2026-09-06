# Phase 1 — UXS reconciliation

Part of [#1941](https://github.com/chipi/podcast_scraper/issues/1941) /
[#1948](https://github.com/chipi/podcast_scraper/issues/1948).

Diffing what the UXS docs **claim** against what the app actually **renders**, using the baseline
screenshots (`npm run design:shots`) and the blind critic's findings as the prompt list.

**Baseline score: 5/10** overall — Player 6, Home 5, Browse 5, Profile 5, Search 4, Library 4.

The critic saw only pixels. These docs say what each surface is *for*. Where they disagree is
where the interesting work is.

---

## 1. Confirmed violations — the docs say we broke a rule

### Profile's segmented control is a hand-rolled duplicate

UXS-014 is unambiguous:

> Recurring affordances are **single classes in `style.css`**, not per-element Tailwind hand-rolled
> on each page. Adding a one-off `class="text-muted …"` for one of these is a regression.

There is **no shared segmented-control class**. Two pages solve the same problem differently:

| | Search (`Everything / My listening`) | Profile (`Compact / Full`) |
| --- | --- | --- |
| Semantics | `role="tab"` + `aria-selected` | plain `<button>`, no ARIA |
| Styling | pill-in-track | `:class="… ? 'bg-accent text-accent-foreground' : 'text-muted'"` |

So it is **two** findings, not one: the design-system regression UXS-014 names, and an
accessibility gap — a screen-reader user gets a labelled tab pair on Search and two unrelated
buttons on Profile, for the same interaction.

The critic reached the same place from pixels alone: *"It reads as two mismatched controls glued
together. Rebuild it as one track with a single sliding pill, matching the `Everything / My
listening` control on Search."*

**Action:** filed. Fix is a shared class, not a per-page patch — otherwise it regresses again.

---

## 2. Killed by the docs — intent the redesign must PRESERVE

These read as defects in a screenshot and are deliberate. They are the constraints Phase 3
diverges *within*.

### Library: `Highlights` as an `h2` inside `Saved`

The critic called the two-level hierarchy a weakness. UXS-014 (§117–121) specifies exactly this,
with the history attached:

> **Highlights**, **Queue** and **Recent** are **not tabs** — Highlights is an `h2` section inside
> **Saved**. […] `a9705819` (#1141) deliberately removed the Knowledge tab and merged insights
> back into **Saved**.

A redesign may restyle it. It may not promote Highlights back to a tab without re-litigating
#1141.

> **Revisited (#1962, `520b3489`).** The constraint holds and the section is still an `h2` inside
> Saved. What the critic was actually seeing was narrower than the hierarchy: Highlights was the
> only *unconditional* section, so on an empty account it was the entire tab — one heading naming a
> third of what Saved holds. It is conditional now like Episodes and Insights, and an empty Saved
> shows one empty state naming all three plus a `Find something to listen to →` action. Placement
> unchanged; only gating. Recorded here because the "must PRESERVE" list is what Phase 3 diverges
> within, and a stale entry in it would over-constrain the divergence.

### Home: the purple kicker on the discover hero

The critic flagged "three competing accent families", singling out the purple
`ASK ACROSS EVERY EPISODE` eyebrow against the orange `FOR YOU`. UXS-012 §103 specifies it:

> **Hero (discover):** `surface` panel, **`topic`-toned kicker**, large search input.

The purple is the *topic* token, chosen deliberately to mark that hero as topic-space rather than
accent-space. The aesthetic complaint may still stand — two kicker treatments in one scroll is a
real tension — but it must be argued as a change of intent, not filed as a bug.

> **Revisited (#1964, `520b3489`).** Argued and changed, and the token stayed. The kicker was the
> ONLY `topic`-toned element on Home, so a token meaning "this is a topic" was carrying no meaning:
> with nothing to contrast against, it read as decoration — the same failure mode as the orange
> accent on Browse, where the critic's own line was "when everything is accented, nothing is". The
> fix gave the colour siblings instead of removing it: a row of real topic chips under the search
> field, from `getTrendingTopics()`. Phase 3 inherits the constraint as *the discover hero is
> topic-space* — which is what §103 actually protects — not as *exactly one purple element*.

---

## 3. Unmet spec — the app under-delivers against its own documentation

The most valuable category, and the one a critic alone could never produce.

### The Player artwork zone is decoration; the spec says it must not be

UXS-011 §43:

> **The artwork zone is a live intelligence surface, not decoration** — speaking-now, a grounding
> badge, and the insight surfacing *at this moment* live on the show colour field.

Three things are specified. Today the zone carries **one** (the `INSIGHT NOW` card). There is no
speaking-now and no grounding badge. The rest is roughly 60% empty gradient with a burned-in
wordmark.

The critic, with no access to this document, diagnosed the same emptiness and prescribed the
opposite remedy:

> "The artwork panel is roughly 880px tall and about 60% of it is empty gradient. Cut the panel
> height by ~40%."

**Shrink it, or fill it as specified?** That is a genuine product question and the single most
interesting thing this reconciliation surfaced. It should be answered before Phase 3, because the
two answers lead to completely different directions.

---

## 4. Undocumented — on screen, described by no UXS

The surface-map guard already treats undocumented surfaces as failures, so each of these is a real
gap.

- **Generated show artwork carries the show name as burned-in type.** No UXS covers what the
  generated tile contains. The consequence is visible everywhere: Browse repeats the show name on
  every row (burned into the artwork, then again as the `.lp-kicker` eyebrow beside it), and
  Home's "Your shows" tiles repeat it as a caption directly beneath. ~20+ duplications per Browse
  scroll. The kicker itself is correct per UXS-014 (`.lp-kicker` = "e.g. a show name"); the
  *artwork* carrying the same string is the undocumented half.
- **`Save` vs the bookmark glyph on Search results.** UXS-011 §9 mentions "saved queries" in
  passing; nothing specifies two affordances, and the critic read them as doing the same job.
- **The `Sign out` pill in the masthead.** UXS-011 specifies an "account/sign-in affordance" but
  not that a signed-out action should occupy the top-right of every screen — the position a mobile
  app reserves for its most-used action. The critic's strongest structural note.

---

## What this changes for Phase 3

1. **Two constraints are now hard:** Highlights stays inside Saved; the discover kicker is
   topic-toned by intent (change it deliberately or not at all).
2. **One product question is open and blocking:** shrink the artwork zone, or make it the live
   intelligence surface UXS-011 already promises?
3. **Three documentation gaps to close**, each of which is also a design simplification: stop
   painting the show name twice, resolve Save-vs-bookmark, and decide whether `Sign out` has
   earned the masthead.

The exercise did what it was supposed to: it found **functional** gaps (an empty reach pill
shipped to production — [#1957](https://github.com/chipi/podcast_scraper/issues/1957), now fixed)
and **documentation** gaps, not only aesthetic ones.
