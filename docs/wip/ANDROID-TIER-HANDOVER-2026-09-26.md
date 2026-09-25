# Android device tier — handover, 2026-09-26

Branch `fix/ui-followups-2026-09-18`. Supersedes `ANDROID-TIER-HANDOVER-2026-09-25.md`, whose
conclusions about *why* suites were failing were wrong — see "What the previous handover got wrong".

**iOS is green: `make test-ios` → 27 passed, 0 failed, `TEST_IOS_EXIT=0`.**
**Android is red and needs its own session.**

## Read this first

Every claim below is followed by the measurement that produced it. Where something is
unverified it says so. Three times in the session that produced this document I reached a
confident conclusion before measuring and was wrong each time — twice in opposite directions
about the same suite. Treat an unattributed claim here as a bug in the document.

## Android status, measured 2026-09-25 22:45–23:39 (warm emulator)

| Phase | Suite | Result |
| --- | --- | --- |
| 1/7 | HarnessSmokeTests | OK (2 tests) |
| 2/7 | DownloadThroughUITests | OK (1 test) |
| 3/7 | OfflineAutoAdvanceTests | OK (1 test) |
| 4/7 | OfflinePlaybackTests | OK (1 test) |
| 4/7 | OfflineCacheTests | OK (1 test) |
| 4/7 | ConfigOfflineToggleTests | OK (1 test) |
| 5/7 | **AppJourneyTests** | **8 tests, 3 failures** — first execution ever |
| 5/7 | PersonalisationTests | **2 tests, 2 failures** (run separately) |
| 5/7 | NativeCapabilityTests | **NOT RUN** — phase stops on first failure |
| 5/7 | StackDepthProbeTests | **NOT RUN** |
| 5/7 | AccessibleNameAuditTests | **NOT RUN** |
| 6/7 | ServerDegradedTests | **NOT RUN** — the drill is rewritten but has never executed |
| 7/7 | NativeOnlySurfacesTests | **NOT RUN** |

`TEST_ANDROID_EXIT=2`.

### The three AppJourneyTests failures (new — the suite had never run)

    Home tab did not open
    no topic control in the insights panel
    colour popover rendered no colour choices

Undiagnosed. The previous handover's table listed this suite as NEVER RUN, so these are not
regressions; they are the first results it has ever produced.

### The two PersonalisationTests failures

    test09: Stats still shows its never-listened empty state after playing two episodes   (line 99)
    test10: none of the interests just chosen render on the Profile Topics tab            (line 226)

**The iOS-F1 fix DID carry to Android.** `test10`'s assertions run in this order: `:197` interests
card empty state, `:211` Home stops showing "Choose interests", `:226` interests render on Profile
Topics. The failure is at **226**, so **211 passed** — Home no longer prompts after interests are
chosen, on both platforms. iOS `test10` passes outright.

Both remaining failures are Android-only: iOS passes the equivalent assertions. They are genuinely
new findings, invisible while the suite was parked.

### A cold-start flake at phase 2

On the FIRST run (cold emulator, fresh install) phase 2 failed:

    sign-in did not complete as simtest.
    On screen: <nothing labelled; foreground window = app.closelistening.player>

It does not reproduce warm — verified three ways: the suite alone after `pm clear` (`OK (1 test)`),
phase 1 → phase 2 in sequence with no clear between (`P1_EXIT=0`, `P2_EXIT=0`), and the full tier on
a warm emulator. The only differing conditions were a freshly-booted emulator and a freshly-
installed APK; phase 2 ran ~3 minutes after boot.

`AppSession.relaunch()` ends with a fixed `Journey.sleep(5_000)`. A fixed settle is the wrong shape
— it passes warm and fails cold, which is backwards for a tier meant to run on CI. The fix is to
wait for CONTENT (poll until the tree has labelled nodes) rather than for a duration. **Not done.**

## What changed this session

- **Both parked suites UNPARKED.** `_UNWIRED_BY_DESIGN` is now empty and all 7 wiring-guard tests
  pass. The reasons the old parking gave were both wrong (next section).
- **`test-android-server-degraded` restored and rewritten.** It had been deleted by `9aeca42bb`
  when the suite was parked. It now restarts the api with **no** secret on the **same** volumes and
  PROVES the state before asserting — it aborts unless `/api/app/me` answers 503.
- **`_app-e2e-api-restart`** replaces a third copy of the container-run block. Round trip measured
  against the live api: `me=401 → 503 → 401`, corpus intact. The old restore went through
  `app-e2e-api-up`, which deletes both volumes — including the user store every later Android phase
  is still signed in as.
- **`android-suite` gained `TEST=<method>`** for single-method runs (`Class#method`). The drill needs
  it. **It has never executed** — the tier stopped before phase 6.
- Phases renumbered 6 → 7.

## What the previous handover got wrong

It parked both suites on the reasoning that iOS failed the same assertions, so the cause must be
shared app behaviour rather than an Android defect. The observation was right; both conclusions
were wrong, for different reasons.

- **PersonalisationTests** — a REAL product bug, now fixed. `InterestsPicker.save()` PUT the
  interests and updated no store, so Home kept prompting after they were chosen from Profile.
  Present since #1111 (2026-06-28). The iOS lines the old note cited (39, 71) were a different
  failure entirely: `test-ios` destroyed its own api at phase 2 and never restored it, so the app
  was simply signed out.

- **ServerDegradedTests** — a TEST defect, not the app. The drill rotated `APP_SESSION_SECRET`
  instead of removing it. Measured: secret present → `/api/app/me` 401; secret absent → 503.
  `services/api.ts` keys degraded state on 503 alone, because 401 means the caller's credential is
  bad and must sign them out. Worse, the rotation never happened — `APP_SESSION_SECRET=… $(MAKE)
  app-e2e-api-up` sets a variable that recipe ignores, and that recipe deletes the state volume, so
  the drill actually tested "the user record was deleted". With the real incident reproduced, the
  iOS suite passes: the app degrades exactly as designed.

## Also still open (recorded, not fixed)

- **`02391c3e0`'s a1 rewrite rests on a false premise.** It changed Android's a1 to queue from a Home
  list row because "there is no add-to-queue on the player, and that is correct product behaviour".
  iOS a1 queues from the player and passes. The premise is false; the change is unverified.
- **A persisted FAILED download survives reinstall** and poisons `DownloadThroughUITests`
  permanently, with a symptom that points at the UI. Cost two iOS runs. Android's tier does
  `pm clear` every run and is immune — **iOS should copy that.**
- `AppJourneyTests.test07SavedColourPicker` (iOS) takes ~210s against 26–88s for its siblings,
  reproducibly (208.4s, 212.1s).
- Shared-account pollution: the `simtest` queue reached (6) across a day of iOS runs.
- `Backend target DEV, tap to switch` appears in every Android inventory. The previous handover
  reported this stuck on PROD. Never investigated on either platform.

## Where to start

1. `make test-android` on a warm emulator to reproduce the current state.
2. `AppJourneyTests`' three failures — most likely Android port issues rather than app bugs, but
   nobody has looked. Screenshot the emulator at failure; on iOS that answered in one look what
   three rounds of log-reading could not.
3. Then run phases 6 and 7, which have never executed.
