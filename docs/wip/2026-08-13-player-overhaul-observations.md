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

---

## E. Self-assessment findings (2026-08-13, first pass)

Reviewing my own 20 commits rather than the codebase. Three real gaps, all now fixed:

- **E1. The two new #1591 primitives shipped untested.** `useSectionState` and `SectionStatus` are
  the shared foundation of seven sections and had only indirect coverage through `HomeView.test.ts`
  — i.e. the composition was tested, the contract was not. 12 direct tests added, including the two
  properties every call site relies on: `load()` never rejects (call sites use `void load()` and
  would raise unhandled rejections), and a retry clears the error phase immediately (otherwise a
  retry looks like it did nothing).
- **E2. I introduced the exact drift I spent the session cataloguing.** Five new contract selectors
  — `section-loading`, `section-error`, `section-retry`, `player-open-insights`,
  `yourweek-firstrun` — went into the app *and the e2e spec* without reaching the surface map,
  within hours of building the guard that exists to stop this. The guard is deliberately
  one-directional (map → code) to avoid noise, and that design choice is precisely what let them
  through. **Being the author of the guard did not make me immune to the failure it guards.**
- **E3. So the guard gained a second direction.** New rule: *any testid an e2e spec selects on must
  be documented*. That is precise rather than noisy — "a spec depends on it" is the definition of a
  contract selector — and it immediately found `kp-insights`, a pre-existing gap I had not
  introduced. It understands `{template}` ids so `momentum-rail-{kind}` still covers
  `momentum-rail-topic`.

**The lesson worth keeping:** a one-directional check catches yesterday's drift but not today's. E2
happened because the guard could only see selectors that *were* in the map, never the ones that
should have been.

## F. Self-assessment findings (second pass)

- **F1. I broke a spec rule while fixing a different one, and resolved it silently.** #1584 added
  `truncate` to Recommended's show kicker so a long name could not wrap and undo the reserved
  height. `UXS-014:70` says show names never truncate. The two requirements are genuinely
  incompatible in a fixed-width tile — something must bound the label — but I resolved that conflict
  **in the code, with no note**, which is exactly the behaviour the drift audit condemns. The rule is
  now scoped in the spec, with the conflict recorded.
- **F2. A spec claim was simply false and I had been reading past it.** `UXS-011:90` says every
  surface token has a matching `-foreground`; `elevated` and `overlay` do not. I quoted that section
  twice while working #1598 without noticing. Corrected in the spec.
- **F3. An executable check that cannot run is worse than no check.** My token-pair assertion could
  not read `tokens.css` under vitest (`?raw` yields empty in this setup, in both import and glob
  form). Rather than leave a skipped or contorted test, the finding moved to the spec where it is
  a statement of fact instead of a broken guard. **Only convert a criterion when the check is
  genuinely cheap** — a fragile one erodes trust in the whole file.
- **F4. The first colour check was too broad.** It flagged `rgba(0,0,0,.5)` scrims and
  `rgba(255,255,255,.2)` hairlines, which have no token to use instead. Narrowed to *chromatic*
  literals only — a brand colour is never greyscale, so the Ember glow it exists to catch still
  fails it. A noisy guard gets deleted, and then guards nothing.

## G. Self-assessment findings (third pass, after Wave 5)

Two real defects, both mine, both invisible to the suite until I went looking. Neither was found by
a test failing — which is the point worth keeping.

### G1. #1590 was half-wired: four ungated controls

I added four sign-in teaser strings and wired two (`signInToQueue`, `signInToSave`). The other two
(`signInToCapture`, `signInToFollow`) sat unused in `en.json` — I found them by hunting orphaned
i18n keys, not by noticing the behaviour. Four controls performed per-user writes with no gate:

| Control | Site | Signed-out behaviour before the fix |
| --- | --- | --- |
| Follow (tile) | `ShowTile.vue` | optimistic flip → 401 → silent revert |
| Follow (show page) | `PodcastView.vue` | same |
| Capture insight | `KnowledgePanel.vue` | POST 401, nothing announced |
| Capture highlight | transcript → `PlayerView.vue` | POST 401, nothing announced |

The stores swallow write failures, so the visible result is a control that appears to work for one
frame then undoes itself — **worse than the hidden control #1590 replaced**, because it reads as the
user's own action failing rather than as a requirement to sign in.

Worse, `PlayerView` passed `:can-capture="auth.isAuthenticated"`, so signed-out visitors saw **no
capture affordance at all** on the transcript — the exact defect #1590 exists to fix, on the
differentiator's most important surface. My #1590 pass simply never looked there.

Fixed: all four gated; capture always renders with a `gated` prop driving the label; new
`src/__checks__/auth-gate.test.ts` fails on any component doing a per-user write without the gate.
Guard mutation-tested (removing the gate from ShowTile → `expected [ 'ShowTile.vue' ] to equal []`).
`UXS-011` amended — it still specified the pre-#1590 behaviour ("no capture controls" signed out).

### G2. The bottom nav broke WCAG AA, intermittently, by design

`BottomNav` shipped as `bg-canvas/95 backdrop-blur`. axe composites text against what is *actually*
behind an element, so with a tinted storyline chip scrolled underneath, both the active label
(`text-accent`) and the inactive labels (`text-muted`) measured **4.28:1** against the 4.5:1 AA
floor. Contrast was a function of scroll position.

That is why it surfaced as a **flaky** e2e failure (3 of 5 runs) rather than a steady one — and
flaky failures are the ones that get retried away instead of fixed. `MiniPlayer` had the identical
defect (`bg-elevated/95`), uncaught only because no axe scan lands on it.

Fixed: both bars opaque. Pinned by a new case in `spec-conformance.test.ts` (also mutation-tested).
Modal scrims (`bg-black/40`) are exempt and documented as such — translucency is their purpose and
they carry no text. Verified 6/6 clean runs where it previously failed 3/5.

### G3. The e2e harness is not hermetic across runs

Re-running the suite against a surviving `lp-e2e-appdata` volume produces failures that look exactly
like code regressions: `follow-show.spec` expects `aria-pressed="false"` on a show a previous run
already followed, and retries 34 times before failing. I lost time treating this as a regression
from my own change.

Fixed by `e2e/run-local-stack.sh`, which recreates the volume every run. **This is the mechanism
that makes the suite trustworthy; do not optimise the volume recreation away.**

### Not covered / still open after this pass

- The gate guard is **source-level** (does the file import `useSignInGate`), not behavioural. It
  cannot catch a component that imports the gate and then calls the store directly anyway.
- Only `ShowTile` and `TranscriptList` gained behavioural gated-path tests. `PodcastView` and
  `KnowledgePanel` are covered by the source guard only.
- No axe scan covers `MiniPlayer` — its contrast fix is asserted by the source guard, not measured.
- I have not re-audited the OTHER auth-gated surfaces (scope toggles, consolidation) for the same
  hide-vs-defer defect. `UXS-011` mentions "scope toggles" in the same retracted sentence; I fixed
  the sentence, not necessarily every control it described.
- The 105-violation figure in the first capture was one run's node count, not 105 distinct defects
  — it is 2 violations across many nodes. Recorded here so the number is not misread later.

## H. Fourth pass — the fable-5 review, and a correction to section G

### H0. CORRECTION: G1's claim that all four controls were gated was FALSE

G1 states "Fixed: all four gated". That was true for `ShowTile` and `TranscriptList` only. On
`PodcastView` and `KnowledgePanel` I wired `gated()` and the `isGated` labels onto controls that
were still `v-if="auth.isAuthenticated"` — so my fix was dead code behind a hidden element, and the
control stayed invisible to signed-out visitors. I wrote the fix, wrote it down as done, and did not
check that the element rendered. The advisor found it by reading the templates.

This is the same defect class G described, committed *inside the fix for it*, and then reported as
complete. The reporting failure is the worse half: a false "done" is worse than an open gap, because
nobody looks again.

### H1. What the review found that I had missed (all confirmed by reading the code)

| # | Defect | Where | Status |
| --- | --- | --- | --- |
| S1 | 4 more ungated/hidden controls (2 of them my "fixed" ones) + scope toggle | `PodcastView`, `KnowledgePanel` ×2, `PlayerView`, `SearchView` | fixed |
| S2 | `resetForLoad()` stomps transport state when returning to the playing episode | `PlayerView.vue` load() | fixed |
| S2b | start-position watcher seeks whatever is LOADED, not this view's episode | `PlayerView.vue` | fixed |
| S3 | auto-advance carried no title/artwork → mini-player stuck on "Loading…", lock screen shows the previous episode | `App.vue` → `player.load()` | fixed |
| S4 | next-up frozen at episode start → "Play next" and every mid-listen queue edit ignored; URL fetched hours before use | `App.vue` | fixed (resolve at `ended`) |
| S5 | `pb-24 sm:pb-6` under-reserves: content occluded by 18.5px mobile / 38.5px desktop while playing | `App.vue` | fixed |
| S6 | #1585 repurposed `shows` to followed-only; trending rail still joined artwork from it → art lost for signed-out users and unfollowed shows | `HomeView.vue` | fixed |

Plus one defect I introduced *while* fixing S1: gating on `auth.isAuthenticated` before the session
resolves sent a signed-in user who tapped quickly to the login page. Caught by `follow-show.spec`.
`useSignInGate` now awaits `ensureLoaded()` before deciding.

### H2. The guards were the problem, not just the code

`auth-gate.test.ts` was **file-level** — any file containing the string `useSignInGate` passed. Two
of the four defects lived in files that passed for that reason. It also had an incomplete write list
(`queue.playNext`, `capture.captureMoment` absent) and a hidden-control check that pinned one
historical regex in one file. Rewritten: per-call-site, with the write list asserted complete
against the store source, and the hidden-control rule expressed as a class.

`mobile-invariants` asserted the literal strings `pb-24` / `sm:pb-6`. The classes were present and
the geometry was wrong — a string check cannot see arithmetic.

### H3. Two of my own tests gave false confidence before I caught them

Worth recording because both looked fine:

1. The occlusion test first used `click({ trial: true })`. It passed against the known-broken
   padding — a trial click probes the element's CENTRE, so a bar covering the bottom half passes.
2. Rewritten geometrically, it then measured `a[href^="/episode/"]` **unscoped** — and the
   mini-player itself contains an `/episode/` link, so it compared the bar against itself and
   invented a 51px "overlap" that was the bar's own height. I nearly "fixed" a bug that did not
   exist.

Both were only caught by mutation-testing the assertion against the original defect. **A test that
has never failed against the bug it describes is not evidence.** Every guard added in this branch
has now been mutation-tested; the two that could not be made to fail were rewritten until they did.

### Not covered / still open after this pass

- **S9 accessibility (advisor, unfixed):** the KnowledgePanel mobile sheet has no `role="dialog"`,
  no focus trap, no Escape handling, and opening it drops focus to `<body>`; `PlayerView` and
  `KnowledgePanel` ignore reduced-motion where `TranscriptList` honours it; `MiniPlayer`'s progress
  bar animates with no `motion-reduce` variant; `BottomNav` sets no `aria-current` on `player`,
  `podcast` or `catalog` routes, where users spend most of their time.
- ~~**S8**~~ **FIXED.** The three capture actions return success/failure (the throw is still
  swallowed — callers use `void`). All three call sites announce the real outcome, and
  `KnowledgePanel` emits to PlayerView's existing live region rather than adding a competing one.
  Mutation-verified.
- ~~**S7**~~ **FIXED.** Both sections now use `useSectionState` and render loading/error with retry.
  Mutation-verified: restoring `.catch(() => [])` on either fails its test.
- **Mobile shows two nav systems:** the header nav has no `sm:hidden`, so it stacks with the tab
  bar, duplicating Search.
- **Spotify core-loop gaps (advisor, ranked):** no one-tap play from any card/row/tile; player state
  is memory-only so a reload loses the mini-player; end-of-queue is silence; lock-screen next/prev
  navigates without calling `play()`; no offline (architectural, per bridge-never-rehost).
- `make docs` (strict mkdocs) still unrun — no mkdocs and no pip in the repo `.venv` here.
- The gate guard still cannot catch a component that imports the gate and calls the store anyway.

## I. S7 + S8 closed (2026-08-13)

Both were mechanical once located; what is worth recording is a **third instance of a test that did
not test what it claimed**, found by the same mutation discipline:

The "library outage" test passed under mutation because `getLibrary` was unmocked, so
`ensureLoaded()` rejected regardless of what I did to `getPodcasts()`. It asserted the right
outcome for the wrong reason — a green test proving nothing about the line it named. Mocking the
library to succeed-and-be-empty isolated the catalogue fetch, and the mutation then failed it
(`expected '…' not to contain 'Follow a show'`).

That is three self-inflicted false-confidence tests in two rounds (trial-click, unscoped selector,
unmocked dependency). All three were green, plausible, and worthless. The only thing that caught
any of them was running the test against the bug it describes.

**Still open: S9** — dialog semantics and focus management for the KnowledgePanel mobile sheet,
reduced-motion at 2 of 3 call sites, and `BottomNav` `aria-current` on nested routes. Being
discussed with Marko before implementation.
