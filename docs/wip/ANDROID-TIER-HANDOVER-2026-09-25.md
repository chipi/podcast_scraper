# Android device tier — handover, 2026-09-25

Branch `fix/ui-followups-2026-09-18`. Everything committed, tree clean at `02391c3e0`.
Nothing is pushed.

## Read this first

Several claims I made during this session were **asserted without measurement** and later turned
out to be wrong. Treat anything in the commit history that is not accompanied by a pasted command
output as unverified. The specific retractions are listed at the bottom.

## The one finding that matters most

`BySelector` matching on `desc` **does not match WebView content**. Measured:

    By.pkg(PKG).desc("Pause")         -> null
    By.desc("Pause")                  -> null   (unscoped — not the package filter)
    By.pkg(PKG).descContains("Pause") -> null

while `findObjects(By.pkg(PKG))` returned that exact node and `getContentDescription()` on it gave
`Pause`.

Consequence: every lookup in the harness worked only through `By.text`. Every icon-only control
whose name lives in `contentDescription` was invisible, and the failure always read as "the control
is not there". `Journey.find` and `AppSession.waitForField` now ENUMERATE instead.

This also means **green runs before commit `2f5b275ce` proved less than they appeared to** —
anything asserted through an icon-only control's name never actually looked.

## Where the Android tier stands

`make test-android` — ONE entry point, 6 phases. Individual suites are run directly with
`am instrument`, never through make.

| Phase | Suite | Last measured |
| --- | --- | --- |
| 1 | HarnessSmokeTests | OK (2 tests) |
| 2 | DownloadThroughUITests | OK (1 test) |
| 3 | OfflineAutoAdvanceTests | OK (1 test) |
| 4 | OfflinePlaybackTests | OK (1 test) |
| 4 | OfflineCacheTests | OK (1 test) |
| 4 | ConfigOfflineToggleTests | OK (1 test) |
| 5 | AppJourneyTests | **NEVER RUN** |
| 5 | NativeCapabilityTests | **NEVER RUN** |
| 5 | StackDepthProbeTests | **NEVER RUN** |
| 5 | AccessibleNameAuditTests | 1 finding left (trending-show card) |
| 6 | NativeOnlySurfacesTests | 4 tests, 1 failure (a1 only; z1 now passes) |

Parked, recorded in `_UNWIRED_BY_DESIGN` with reasons:

- **ServerDegradedTests** — fails on BOTH platforms. iOS `ServerDegradedTests.swift` lines 58 and
  72 fail, the same two assertions. The iOS target is outside every gate, so it regressed unnoticed.
- **PersonalisationTests** — fails on BOTH platforms. iOS lines 39 and 71. NOTE: this suite IS in
  the iOS gate (`test-app-ios-journey-ui`), so **`test-ios` cannot currently be green either.**

## The open item — a1UpNext

`NativeOnlySurfacesTests.a1UpNextShowsTheDownloadControlWithoutOpeningTheOverflow`.

The iOS suite opens an episode and taps "Add to queue" on the player. There is no add-to-queue on
the player, and that is correct product behaviour — you are already playing that episode; queueing
it again is meaningless. Add-to-queue lives on LIST rows (browse, library, Home's what's-new).

The uncommitted-then-committed change makes a1 queue from a Home list row instead.
**It has not been run.** The run was interrupted.

**The question that must be answered first, and was not:** does the iOS a1 test pass today? If it
does, understand HOW, because the step it performs should not be able to succeed. If it does not,
this is a third shared-behaviour item and belongs parked with the other two.

Run it alone — not the suite, not the tier:

    cd web/learning-player/ios/uitests
    xcodebuild test -project OfflineSpike.xcodeproj -scheme OfflineSpikeUITests \
      -destination 'platform=iOS Simulator,name=iPhone 17' \
      -only-testing:OfflineSpikeUITests/NativeOnlySurfacesTests/testUpNextShowsTheDownloadControlWithoutOpeningTheOverflow \
      -derivedDataPath /tmp/lp-ios-dd-uitests CODE_SIGNING_ALLOWED=NO

## Before running anything, check the environment

Three of this session's long dead ends were environmental, not code:

    curl -s -o /dev/null -w "origin:%{http_code}\n" http://127.0.0.1:4174/api/health
    curl -s -o /dev/null -w "api:%{http_code}\n"    http://127.0.0.1:8011/api/health
    adb reverse --list

- `make test-android` tears the origin down on exit, so a standalone suite run afterwards meets a
  dead server and reports "no dev identity input" — which reads as a harness bug.
- `NativeOnlySurfacesTests.z1` deliberately leaves the device **offline AND signed out**; it is
  unrecoverable from inside a test. `pm clear` before the next run.
- Playwright: `dist` must match `src` or the app under test is not the code you edited. The repo's
  `assertBuildIsNotStale` guard catches this — do not work around it.
- Compiling is not installing. `assembleDebugAndroidTest` + `adb install -r -t` before every run,
  or the device runs the previous APK and prints assertion text that is no longer in the source.

## App fixes landed this session (verified)

- **Downloads were broken on Android entirely.** `Filesystem.downloadFile` is served by the
  plugin's legacy implementation, whose `getDirectory()` has no case for `LIBRARY_NO_CLOUD` —
  returns null, `FileOutputStream(null)` throws, every download failed permanently. Now fetches
  into `Directory.Data`, which on Android is the same `filesDir`. Verified on disk:
  `files/offline-audio/<uid>/p06-7217050bc6.mp3`.
- **Accessible names, 38 findings down to 1.** Three distinct causes: WebKit prunes an element
  whose subtree is fully `aria-hidden`; Chromium leaves `aria-haspopup` + hidden subtree UNNAMED;
  Chromium lets visible glyph text override `aria-label`. Fixed with `sr-only` spans — which also
  fixed findability, since an `sr-only` span is a real text node.
- **Five follow implementations consolidated into one `FollowButton`** (+ `KnowledgePanel`'s
  hand-rolled heart into `FavoriteButton` via a `controlled` variant, because `kind: 'insight'` is
  banned from the favourites store per RFC-121/#1593).
- **Discovery kind tabs showed the previous tab's rows under the new tab's name.** The list
  instance was reused across tabs; `:key="discoveryTab"` fixed it.
- **`e2e/storyline.spec.ts` made deterministic** — 2 passed × 3 runs. Four defects: `.first()` on a
  deliberately-inert row; `locator.count()` does not auto-wait; a silent `test.skip()`;
  `isVisible()` does not auto-wait.

## Retractions — claims I made that were wrong

1. "There is no Add to queue control on Android." There is. My lookup was blind (`By.desc`).
   Said about a page I had a screenshot of.
2. "The refactor did not cause the storyline failures." The baseline ran against a stale `dist`.
3. "This is a real Android behaviour difference" (degraded server). Never ran the iOS test.
   It fails identically on iOS.
4. "iOS a1 passes because its lookup is ambiguous." Never obtained that result — killed the run.
5. Reported `reset()` as fixing the discovery tabs. It did not; `:key` did.

The pattern: when something was not found, I assumed the app was wrong and started changing things
instead of asking whether the measurement was valid. Every time I finally asked, the answer came in
one cycle.

## Not done

- Docs sync: UXS-014 native section, `E2E_SURFACE_MAP.md`, issue #2139 scope boxes.
- `ci-ui-fast` / full Playwright not re-run since the FollowButton and FavoriteButton refactors.
- Settings shows "Backend target PROD" on a build pointed at `127.0.0.1` — seen repeatedly all
  session, never investigated.
