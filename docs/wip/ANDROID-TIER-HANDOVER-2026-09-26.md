# Device tiers (iOS + Android) — handover, 2026-09-26

Branch `fix/ui-followups-2026-09-18`. Supersedes `ANDROID-TIER-HANDOVER-2026-09-25.md`.

**Living document — being updated as the session runs. Status lines are timestamped.**

## Status

| Tier | State |
| --- | --- |
| iOS | `make test-ios` RUNNING (started 13:31). Phases 1–3 green; phase 4/6 in progress, all green so far. |
| Android | All suites measured green individually. Full `make test-android` NOT yet run this session. |

**Nothing here claims a full-tier pass.** Both full runs are the gate and only one is in flight.

## Issues opened this session

| Issue | What it owns | Why it is not being done now |
| --- | --- | --- |
| [#2157](https://github.com/chipi/podcast_scraper/issues/2157) | Native push is not wired end to end: Android crashed on enable, and no device-token sender exists server-side | Needs a Firebase project + an FCM sender in the delivery worker. Next arc, operator's call. |
| [#2156](https://github.com/chipi/podcast_scraper/issues/2156) | 37 icon-only buttons unnamed on Android | Deliberately out of this arc. Widening the audit to find them turns the tier red. |

`#2156` carries a long comment added 2026-09-26 recording the audit's blind spot, the iOS gap, and a
SECOND failure mode — read it before touching accessible names.

## Commits this session

| Commit | What |
| --- | --- |
| `18c498cc7` | Push plugin guarded on Android — enabling push killed the process |
| `9669fcc7f` | `testN1` tapped the note field and the keyboard hid the mic |
| `a41cbf146` | Tile buttons out of the link; `Journey.shot` for Android |
| `39e6d7a3f` | A wrapper span cost the tile's heart its accessible name |

## Open

1. **Fragilities** — not currently red; being adjudicated by the two full tier runs rather than
   hardened speculatively. Shared ACCOUNT state across runs (the tier's isolation stops at the
   device boundary — `pm clear` resets the device, not the server), iOS has no `pm clear` equivalent,
   a Capacitor `BrowserControllerActivity` sign-in overlay seen twice, `test07SavedColourPicker`
   ~210s.
2. **Five unnamed controls + guard coverage** — agreed for AFTER both tiers. `aria-label` with no
   text node: `CollectionsView` (`collections.remove`, `collections.removeItem`) and
   `ResurfacingInbox` (`revisit.dismiss`, `revisit.retire`, `revisit.remove`). Then extend
   `src/__checks__/accessible-names.test.ts` to resolve paths from `src/` rather than
   `src/components/`, and add both views. Closes 5 of #2156's 37.

## Parity gap found 2026-09-26 — NOT closed, deliberately not written

`NativeCapabilityTests` has FOUR tests on iOS and THREE on Android:

| | iOS | Android |
| --- | --- | --- |
| N1 dictation affordance | yes | yes |
| N2 native share sheet | yes | yes |
| N3 push permission on enable | yes | yes |
| **N4 avatar upload and crop** | **yes** | **MISSING** |

Avatar upload/crop is a native capability — camera and photo-picker plumbing that is entirely
different on Android — so it is currently unverified on that platform. Not written here because the
standing instruction for this arc is to FINISH the suite as it exists, not to add tests during
stabilisation. Decide whether to port it in the next arc.

## Closed, with the reasoning, so it is not reopened

- **The a1 product question — RETRACTED.** There is no add-to-queue on the player and there never
  was (operator, 2026-09-26), so `02391c3e0`'s premise was correct. The "iOS falsifies it" claim
  came from misreading iOS `NativeOnlySurfacesTests` line 63's error string, "neither queue control
  was reachable on the player". That test calls `AppSession.openEpisode` and taps "Add to queue" on
  the EPISODE surface, where `EpisodeActions` renders `QueueButton`. Nothing to decide.
- **The audit blind spot** — parked into #2156 rather than fixed, because widening the audit's walk
  IS the #2156 work and would re-block the tier.

## Traps that cost real time today — read before debugging either tier

**A failure message names a symptom and is usually wrong about the cause.** Every one below was
diagnosed in the wrong place first.

- **"no dictation control after enabling Voice input"** → the mic was rendered. The test TAPPED the
  note field first, which raises the soft keyboard, and the keyboard covers the button row beneath
  it. Android's tree holds only ON-SCREEN nodes, so the mic left the tree exactly when drawn. The
  tell was in the dump the whole time: the note field followed by NO buttons, not even the
  always-rendered Add. Two missing buttons means something covered the row.
- **`<NO NAME>` on a control that has both `aria-label` and `sr-only`** → a wrapper `<span>` around
  the button, there only for styling. See #2156's comment; bisect table included.
- **A rect in an audit finding cannot identify a control.** Matching one against a screenshot you
  navigated to yourself is guesswork — the surface need not be at the same scroll position. Doing
  that produced a confident wrong identification and two pointless component changes. `Journey.shot`
  now exists on Android and the audit photographs any surface that produces a finding.
- **A control whose state you cannot READ must never be driven blind.** The Settings switches report
  `checkable=false checked=false` through the Chromium bridge whatever their real state, so
  `isChecked()` can only answer false. `testN1` therefore clicked unconditionally and, when the
  opt-in happened to start ON, DISABLED dictation and then failed on the mic it had just removed.
  Same class as `test10` and `StackDepthProbeTests`. The pattern that works is
  `Journey.setOfflineMode`'s: observe what the APP does, one interaction per round.
- **Check the origin is alive before believing a suite result.** A `ctx_shell` job that starts
  `ios-origin-up` in the background takes the origin down with it when it ends; a whole suite run
  then tests against a dead server and reports app bugs. Probe `/api/health` AND `/api/app/me` —
  503 means "cannot authenticate anyone" (the degraded drill's leftover), 401 means healthy.

## Tooling added

- **`Journey.shot(name)` on Android** (`Journey.java`). iOS has had this from the start; its absence
  is why most of a day went into inference. Writes to the app's external files dir —
  `/sdcard` directly fails ENOENT under scoped storage, and `mkdirs()` reports that by returning
  false rather than throwing. Pull with
  `adb pull /sdcard/Android/data/app.closelistening.player/files/lp-shots/<name>.png`.
- **`make android-suite SUITE=<Class> TEST=<method>`** runs a single test. Use it. Running the whole
  tier to check one test wastes ~10 minutes per iteration.

## Where to start

1. Finish `make test-ios`, then `make test-android`, then stabilise.
2. Then the five controls + guard coverage (item 2 above).
3. Then push and update the open PR.
