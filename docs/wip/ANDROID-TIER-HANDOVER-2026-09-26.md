# Android device tier — handover, 2026-09-26

Branch `fix/ui-followups-2026-09-18`. Supersedes `ANDROID-TIER-HANDOVER-2026-09-25.md`.

**iOS is green: `make test-ios` → 27 passed, 0 failed, `TEST_IOS_EXIT=0`.**
**Android: 10 of 12 suites green. Two open, both diagnosed.**

## Read this first

Every claim below is followed by the measurement behind it. Where something is unverified it
says so. The dominant lesson of this session, on both platforms: **a failure message names a
symptom, not a cause, and nearly every one of them was wrong about why.**

- "the seed coloured too few items" → the seed worked; the filter buttons had no accessible name
- "no storyline row was tappable" → the rail was rendering; `By.desc` cannot see WebView content
- "Home tab did not open" → a topic card was covering the tab bar
- "interests card still shows its empty state" → the test had switched its own interests off
- "no dictation control" → the mic is not rendered at all; `voiceEnabled` never took

## Status, measured

| Suite | Result |
| --- | --- |
| HarnessSmokeTests | OK (2) |
| DownloadThroughUITests | OK (1) |
| OfflineAutoAdvanceTests | OK (1) |
| OfflinePlaybackTests | OK (1) |
| OfflineCacheTests | OK (1) |
| ConfigOfflineToggleTests | OK (1) |
| **AppJourneyTests** | **OK (8)** — first pass ever |
| **PersonalisationTests** | **OK (2)** — verified twice consecutively |
| **ServerDegradedTests** | **DEGRADED_EXIT=0** — first run ever |
| **NativeOnlySurfacesTests** | **OK (4)** — first run ever |
| AccessibleNameAuditTests | **1 finding** (pre-existing) |
| StackDepthProbeTests | **FAILS** — see below |
| NativeCapabilityTests | **FAILS** — see below |

A full `make test-android` has NOT been run since these fixes.

## Open 1 — StackDepthProbeTests

    no topic control in the insights panel, after expanding and scrolling twice

Four theories tried and all wrong: below-the-fold (scroll added), `By.desc` blindness (fixed
in `tapTopmost`, real but not this), contents-open-below-header (second scroll added),
expanded-section-collapsed-by-the-guard (re-tap added). It still fails, with the panel dump
showing the top of the panel and no Topics section.

`AppJourneyTests.test11` does the SAME sequence and PASSES. Diff those two paths first — that
is the cheapest next step and I did not get to it.

## Open 2 — NativeCapabilityTests

    no dictation control on the note field after enabling Voice input

NOT a naming bug. "Your notes" renders and the mic does not, so `canDictate` is false:

    canDictate = voiceEnabled && (isNative || !!WebSR)     // NoteComposer.vue:73, useDictation.ts:79

`isNative` is true under Capacitor, so `voiceEnabled` is false — the test's "enable Voice input"
step is not taking effect. Check that step, not the mic.

(The mic DID also lack an accessible name; that is fixed, and was a real TalkBack defect, but it
was never why this test failed.)

## Seven shipped accessibility defects fixed

`aria-label` on a control whose subtree has no text node is DROPPED by Android System WebView:
the control is announced as an unnamed "Button" and is unfindable by name. Found:

| # | Control | Found by |
| --- | --- | --- |
| 1 | `SavedColorControl` trigger | device run 2026-09-24 (already fixed) |
| 2 | the five colour swatches | `AppJourneyTests.test07` |
| 3 | Saved colour-filter swatches | `test07` |
| 4 | "Any colour" reset | **static guard** |
| 5 | muted-only toggle | **static guard** |
| 6 | dictation mic | `NativeCapabilityTests` |
| 7 | transcript capture button | **static guard** |

**Three of seven were found by a static check, not by any test**, and two of those are on
surfaces no suite reaches. `AccessibleNameAuditTests` — the suite whose entire job this is —
reported ONE finding throughout, because it walks static screens and never opens a popover or a
note composer. That blind spot is now written down; it is not fixed.

The guard is `web/learning-player/src/__checks__/accessible-names.test.ts`. It requires a text
node (visible text OR `sr-only`) inside any `aria-label`led button, for the components listed in
`SR_ONLY_REQUIRED`. Add components as device runs find them. Mutation-tested.

**Text must match `aria-label` EXACTLY.** Android reads `getText()` before
`getContentDescription()`, so a shorter `sr-only` string shadows the label and the two drift. My
first swatch fix used the bare colour name and silently broke the colour-seeding loop, which
addresses swatches as "Set colour: Rose".

## The port was written against iOS accessibility semantics

Seven instances of one root cause. **Android's tree contains only ON-SCREEN nodes; iOS keeps
off-screen ones with negative coordinates.** So on Android "X is missing" usually means "X is
below the fold":

- insights Topics & People section (AppJourney, StackDepthProbe)
- storyline rail on Home
- masthead Queue control after the page scrolled (`NativeOnlySurfaces` a1)
- Profile interests chips

Related divergences, also measured:

- `By.desc` DOES NOT MATCH WEBVIEW CONTENT. `Journey.find` and `waitForField` were converted to
  enumeration on 2026-09-25; `tapTopmost` was missed and nothing ran it until now. Fixed.
- Android FLATTENS a kind prefix into the label node with no separator — `THEMEShow Themes` —
  where iOS keeps separate elements. Exact matching cannot see these.
- `aria-pressed` arrives as a ToggleButton CLASS with NO state: `checked`, `selected` and
  `checkable` are all false on a chosen chip. Measured. Any "is it selected?" read must come
  from what the app renders, not from the node.

## Two tests destroyed the precondition they then asserted on

Worth calling out as a class, because both hid behind plausible messages:

- `PersonalisationTests.test10` toggled its own interests OFF, because the guard that was meant
  to skip already-chosen chips read `isChecked()`, which is always false. It alternated pass/fail
  across runs. `pm clear` masked it: the DEVICE resets, the ACCOUNT does not — interests live
  server-side.
- `StackDepthProbeTests` collapses the accordion it wants, because it decides whether to expand
  by asking `find` (on-screen only) whether the contents are visible.

**Account state is shared across runs and nothing resets it.** Same cause as the iOS `simtest`
queue reaching (6). The tier's isolation stops at the device boundary.

## Also open

- The one audit finding: `[Discover] <NO NAME> ToggleButton Rect(267, 674 - 354, 761) near=[]`.
  Pre-existing. I could not identify the control from source; screenshot Discover and tap around
  that rect.
- Sign-in intermittently leaves a Capacitor `BrowserControllerActivity` in front of the app, so
  the app's own WebView is empty and the failure reads "<nothing labelled>". Seen twice.
  NOTE: `AppSession.relaunch()` already polls for content up to 30s — the earlier handover's
  "fixed sleep" framing (and mine) was wrong; that was fixed on 2026-09-25.
- `02391c3e0`'s a1 rewrite rests on a premise iOS falsified ("there is no add-to-queue on the
  player"; iOS a1 queues from the player and passes). a1 passes on Android via the Home list row,
  so this is a deliberate parity decision, not a bug. Raised, not changed.
- iOS should copy Android's `pm clear` every run: a persisted FAILED download survives reinstall
  and poisoned two iOS runs.

## Where to start

1. Diff `StackDepthProbeTests`' insights sequence against `AppJourneyTests.test11`, which passes.
2. Find why `voiceEnabled` is false in `NativeCapabilityTests`.
3. Then a full `make test-android`.
