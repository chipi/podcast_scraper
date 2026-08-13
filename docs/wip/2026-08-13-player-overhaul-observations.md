# Player overhaul — running observations

Things noticed while working the #1596 (UX overhaul) and #1605 (spec drift) epics that are **not
covered by an existing issue**, or that deserve reassessment beyond what the issue says.

Kept as a running log so they aren't lost in commit messages. Started 2026-08-13.

---

## A. Deliberate deferrals — known, chosen, not forgotten

| # | Thing | Why deferred | Cost of leaving it |
|---|---|---|---|
| A1 | **User-empty states render forever** | The "show until first success" fix needs a per-user preference flag; not built | Someone who deliberately follows nothing sees the "Your shows" and "Your Week" prompts permanently. Recorded in UXS-012's state contract. |
| A2 | **#1595's 30-second first-listen moment** | A new surface, not a correction — wants design input | The differentiator is still not self-evident to a first-time visitor |
| A3 | **"Because you follow ‹storyline›" provenance** | Same — new surface. Data already exists unused in `graph_refs` (`YourWeekCard.vue:24`) | Your Week looks like opaque taste-matching, which is Spotify's weakness, not our strength |
| A4 | **Insight favourites left readable** | Non-destructive by choice; a migration of users' saved data shouldn't run on an inferred preference | Two lists exist in Library until users clear the old one |

## B. Things I found but did NOT file as issues

- **B1. `EpisodeCard`'s stretched-link pattern is load-bearing and fragile.** Every new interactive
  element inside the card must be `relative z-30` or it silently click-throughs to the player. This
  is now documented in the component, but the *pattern* invites the bug — a card whose whole surface
  is a link cannot safely host controls. Worth reconsidering whole-card links.
- **B2. `useSectionState` is not used by the search/library/catalog views**, which have their own
  ad-hoc loading and error handling. The contract in UXS-012 is written as a *Home* contract; it
  probably wants to be an app-wide one.
- **B3. The consumer app has one locale (`en`) but full i18n plumbing.** Several components had
  hardcoded English precisely because no one would notice. If a second locale is never coming, the
  plumbing is cost without benefit; if it is, there should be a check that no user-facing string
  bypasses `t()`.
- **B4. `catalog` is titled "All episodes" and is the nav's "Browse".** With shows now meaningfully
  separate from episodes (follow, Your shows, "See all"), there is no shows index — "See all" had to
  point at Library. A `/shows` route is implied by several changes and doesn't exist.
- **B5. Two `INTERESTS_DISMISSED` constants** with the same value in `HomeView.vue` (one localStorage,
  one USERPREFS). Harmless, but the duplication invites them to drift apart.

## C. Process observations

- **C1. A codemod that needs line-by-line verification is not saving work.** My regex conversion of
  five components produced a duplicate closing tag that only the build caught. The remaining four
  were faster by hand.
- **C2. Fixtures that are unrepresentative of production are worse than no fixtures**, because they
  invite confident wrong decisions. I sized the insights panel against 85-char fixture bullets when
  production is 207 — and the fixture's shape came from a regex, not the pipeline. See #1586.
- **C3. Test-fixture gaps present as missing elements, not errors.** `YourWeek.test.ts` lacked a
  `catalog` route, so a new `RouterLink` threw during setup and silently removed an entire block.
  The symptom looked like a `v-if` bug.
- **C4. The surface-map guard paid for itself within a day** — it caught the retired `trend-*`
  selectors during #1589 automatically. The equivalent prose claims (copy, behaviour) are still
  unguarded and did drift in the same session (I corrected the Home shows row after the fact).

## D. Worth reassessing

- **D1. Home is still long.** #1589 removed one section, capping helped, but a signed-in user still
  scrolls through hero → search → interests → Your Week → What's New → browse chips → Trending →
  Storylines → Momentum → Trending shows → Recommended → Your shows. The IA question (#1594) is
  bigger than the bottom-nav change it's filed as.
- **D2. Three "trending" surfaces remain** — Trending topics, Momentum ("Trending now"), Trending
  shows. #1589 merged the *views*, not the *modules*. To a user these are still three answers to
  "what's hot".
- **D3. The `.lp-fav` heart is now used in fewer places** after #1593. Worth checking whether the
  favourites concept still earns its own store and Library tab, or whether everything should be
  captures/highlights.
- **D4. `PRD-038` FR3.3/FR3.4 are marked _(superseded)_ but still present**, as is a "Retrospective
  (shipped #1091)" block that is now two revisions stale. Superseded requirements accumulating
  inside a live PRD is the same rot the drift audit found in the UXS specs.
