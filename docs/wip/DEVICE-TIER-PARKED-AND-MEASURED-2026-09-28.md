# Device tiers — what is parked, what is measured, 2026-09-28

Branch `october-fixed`. Companion to `BRANCH-HANDOVER-ui-followups-2026-09-27.md` and
`DEVICE-TIERS-HANDOVER-2026-09-27.md`; this file covers only the a11y-audit and
test-independence work of 2026-09-28, and exists so the PARKED half is not lost.

---

## PARKED — blocked, with the reason

### P1. Two Settings toggles have no accessible name, and no markup fix reaches the bridge

**Status: real defect, cause unidentified, do NOT guess a third fix.**

`AccessibleNameAuditTests`, widened to walk Profile/Settings/composer, reports exactly two
findings in the whole app:

```
[Settings] <NO NAME> CheckBox near=[Voice input for notes / …]
           platform=… name='' via=<none>
           [text='' desc=null labeledBy=null hint='' state=null checkable=false range=false]
[Settings] <NO NAME> CheckBox near=[Offline mode / …]
           platform=… (identical)
```

The `platform=` half comes from `A11yProbe`, which reads `AccessibilityNodeInfo` directly —
**all five name-bearing fields are empty**, so this is NOT the `UiObject2` blind spot that
made ten of the original thirty-eight findings false positives. TalkBack has nothing to
announce for either switch.

### RETRACTED — the "two failed fixes" were NEVER ON THE DEVICE

The original version of this section said two fixes had been tried and both failed, and told the
next reader not to re-apply either. **That was wrong, and the way it was wrong is the useful part.**

I tried `:aria-label` on the input, then explicit `for`/`id` + `aria-label`, re-ran the audit after
each, saw no change, and concluded neither reached the Android bridge. I "verified" each by
confirming the rebuilt chunk was being served by the origin on :4174.

That verified a channel the WebView never reads for markup. `capacitor.config.ts` sets
`webDir: 'dist'` and sets `server.url` **only** when `CAP_DEV_SERVER` is set, which the tier does
not. `CAP_ANDROID_TEST_ORIGIN=1` only enables `allowMixedContent`, and the config's own comment
says why: `<audio src>` is blocked as mixed content. The origin proxies **`/api` and `/audio`**.
The app's HTML and JS are served from the **APK assets**.

Markup reaches the device only through
`npm run build && cap sync android && gradlew assembleDebug && adb install`. My iteration loop was
`android-suite`, which rebuilds the **instrumentation** APK and never the app APK. I ran
`npm run build` twice and none of the other three steps.

Proven by pulling the installed APK:

```
assets/public/assets/SettingsView-CijsYOFq.js
  aria-label occurrences: 0
```

The chunks I built were `SettingsView-CjywRybp.js` and `SettingsView-B7jV4oGW.js`. Neither is in
the APK. The device ran a build predating both fixes for the whole investigation.

**The fix list is REOPENED, including the two I banned.** The likeliest answer is the first thing I
tried. What survives is only the MEASUREMENT of the defect — the probe's five empty fields — and it
was taken against a build containing no fix.

**The lesson is worth more than the fix:** I verified the wrong channel, got a negative, and
promoted it to a documented dead end in a commit message and this file. A negative result is a claim
about the experiment before it is a claim about the world, and "I verified it reached the device"
was the sentence that deserved the scrutiny.

**Before re-testing anything here:** do the full build/sync/assemble/install, then confirm delivery
with `unzip -p <pulled apk> 'assets/public/assets/SettingsView-*.js' | grep -c aria-label`
BEFORE reading the audit's verdict.

**What is known:**

- `labeledBy=null` — the wrapping `<label>` produces no label relation in the Android bridge.
- `checkable=false` on a node the bridge classes `android.widget.CheckBox`, which is why
  `Journey.nearestCheckable` matches it via `By.clazz` and not `By.checkable(true)`.
- `lp-check` is `appearance: none` + `display: inline-grid` + a `::before` tick
  (`style.css:402-434`). The same class is used in `ProfileView` and `PodcastView`.

**Why it matters beyond a11y:** `Journey.setOfflineMode` cannot address these by name, so it
finds the row text and walks to `nearestCheckable`. `Journey.java:600-604` records that
heuristic driving the **Voice input switch instead of Offline mode, three times, reporting
success each round**. The nameless control is the root of a bug the harness already papers over.

**Next step, when picked up:** find out why a styled checkbox is exposed without a name at
all — suspects are `appearance:none`, the `inline-grid` display, and the WebView version. A
`role="switch"` + `aria-label` on a wrapping element is untried. Decide it on the device with
`A11yProbe`, not from source.

### P2. The widened audit is uncommitted because it is red on P1

`AccessibleNameAuditTests.java` (widened 5 → 10 surfaces) and `A11yProbe.java` are written,
build clean, and run — but the suite fails on P1's two findings. Landing it red would knowingly
break the tier. It lands the moment P1 is fixed, or behind an explicit decision to exclude them.

### P3. iOS accessible-name audit

Not started. iOS has no equivalent of `AccessibleNameAuditTests`, so every defect in this class
so far was found by Android or the static guard. The two engines fail differently, so Android's
green says nothing about VoiceOver.

---

## MEASURED — facts established this session, with the evidence

### M1. The `aria-haspopup` class is closed, and "~32 remaining" was never a list

`#2156` claimed 37 unnamed controls, ~32 remaining. Re-derived:

- The measured Android failure is **`aria-haspopup` + nothing readable inside**, not "no text
  node". `OverflowMenu.vue:90-95` records the general version being disproved on the same page.
- There are **6** `aria-haspopup` buttons in `web/learning-player/src`, and **all 6** already
  carry an `sr-only` name.
- Controls the old theory predicted were broken are named on BOTH engines, proven by passing
  tests: `app.buttons["Play"]`/`["Pause"]` in four iOS suites; `"Skip forward 30 seconds"` and
  `"Change photo"` tapped by name on both tiers.
- Widening the audit to Profile / Profile▸Topics / Profile▸Stats / Settings / player▸Insights /
  player▸notes found **2** findings, both P1. Every other new surface opened and was clean.

**19 speculative fixes were started against the old theory and fully reverted.** The wider rule
would have added an absolutely-positioned `sr-only` span to ~28 already-named controls, and an
unanchored one is what dragged the masthead link's a11y frame off the display.
`testN4AvatarUploadAndCrop` asserts *"'Change photo' was present but not tappable"* as its own
failure branch — that is what spraying spans across named controls causes.

### M2. RESOLVED — the contradiction, settled on the device

Two measurements disagreed about when Android leaves a control nameless:

- `OverflowMenu.vue` (2026-09-24): the ⋯ trigger was unnamed, and the same page had `Play`,
  `Skip back 15 seconds`, `Mark this moment` and `Playback speed` all icon-only and all NAMED. It
  concluded the cause was `aria-haspopup` PLUS a hidden subtree, and that "neither alone does it".
- `SavedColorControl.vue`: five colour swatches measured `<UNLABELLED>[ToggleButton]` on
  2026-09-26 — with **no** `aria-haspopup`.

Settled 2026-09-28 by A/B on device rather than by argument. Removing the swatches' `sr-only`
spans (full build → cap sync → assembleDebug → install) made
`AppJourneyTests#test07SavedColourPicker` fail with exactly five `<UNLABELLED>[ToggleButton]` in
the inventory; restoring them made it pass. **Pass → fail → pass, one variable.**

So "neither alone does it" is WRONG as a general rule. The shape that fits every measurement:

```
aria-haspopup + no text node  ->  UNNAMED   (the ⋯ trigger)
aria-pressed  + no text node  ->  UNNAMED   (the colour swatches)
plain button  + no text node  ->  named     (Play, Skip back 15 seconds)
```

Both attributes remap the node's ROLE — PopUpButton and ToggleButton — and the computed name is
lost in that remapping. A plain Button keeps it.

**Consequence:** the static guard was drawn at the intersection and was too narrow. It now covers
`aria-haspopup` OR `aria-pressed`, which is **32 controls instead of 6**, and all 32 already carry
a name, so the widening landed green. What previously needed a device run is now caught by
`vitest`.

Note what this says about the earlier reasoning: the morning's conclusion that the "no text node"
theory was disproved was itself too strong. The theory was wrong for PLAIN buttons and right for
role-changed ones, and the disproof only ever tested plain ones.

### M3. `Profile ▸ Topics` churns while being walked — cause NOT known, and my diagnosis was wrong

Walking it reports `15 of 29 nodes went stale mid-read` on roughly half of runs, and always exactly
15 of 29 when it fires — so a fixed subset of controls is destroyed and recreated during the read.
That is worth fixing on its own terms: a control recreated after load drops focus, moves a screen
reader's position, and can swallow a tap already in flight.

**RETRACTED — the mechanism this section originally claimed.** It blamed the dynamic tag at
`ProfileView.vue:641` (`:is="i.openId ? 'button' : 'span'"`) flipping when `getStorylines()`
resolved, which would force Vue to destroy and recreate each chip. The code contradicts it:
`load()` assigns `interests`, `clusters` and `storylines` together out of a single `Promise.all`
(`ProfileView.vue:226-247`), so there is no window in which `openId` resolves late. **Do not
re-derive that theory.**

That was the second mechanism I published today without checking it against the code — the first
being the "two failed fixes" above. Both had the same shape: a plausible story, written down with
the confidence of a finding, on evidence that only looked like it fit.

Untested candidates: `onActivated` firing a second `load()` while the walk is in progress
(`ProfileView.vue:359-360`), and the Topics tab's own content re-rendering.

**The surface is EXCLUDED from the audit walk** until this is settled, stated in the test with its
reason rather than silently dropped — a silent exclusion is how the audit came to walk five
surfaces while an issue claimed it covered the app. Note the consequence: Profile ▸ Topics has
still never been audited, so whatever accessible names it holds remain unknown.

### M4. Server-side accounts were immortal across runs

`lp-e2e-api` measured **up 19 hours** still holding two accounts from earlier runs.
`ios-origin-up` reuses a healthy container unless the image inputs are newer, so `simtest`
accumulated without bound. Fixed by `app-e2e-users-reset` (below).

### M5. Sign-in was flaky at ~22%, and the cause is a cross-process step

"sign-in did not complete as simtest" failed **2 of 9** runs while iterating. The UI path is six
sequential races ending in a consent screen owned by `com.android.chrome`. `Makefile:2327-2344`
already disables Android's cached-app freezer for that step, with the trace recorded. Addressed
by the callback sign-in path (below) rather than by retrying.

---

### M6. RESOLVED — the UI sign-in fallback does work, and is no longer unexercised

`ensureSignedIn` tries the callback path and falls back to the real UI flow. Across every run the
callback path succeeded, so the fallback had executed ZERO times — insurance nobody had tested.

Exercised deliberately on 2026-09-28 by pointing the mint at a dead port while leaving the app's own
port intact (`make android-suite ... IOS_ORIGIN_PORT=9999`), which breaks the mint and nothing else:

```
OK (1 test)
Custom Tab / customtabs logcat lines: 51      (normal callback runs: 0)
```

The test still passed and the Custom Tab really opened, so the fallback lands and signs in. Note the
evidence is the logcat count rather than the harness markers — instrumentation `System.out` does not
reliably reach `am instrument` stdout, which is the same reason an earlier hierarchy dump vanished.

### M7. ASSESSED — aggregate state is real, and the per-run wipe changed its starting point

`APP_MOMENTUM_MIN_TOTAL=1` (Makefile) means a single listen event moves trending/momentum surfaces
for EVERY account, so per-suite and per-test identities do nothing for assertions on those rails.

Which suites actually depend on them — the question that was never asked:

| Tier | Site |
| --- | --- |
| Android | `AppJourneyTests:207,217` (finds and taps a storyline row by "momentum") |
| Android | `StackDepthProbeTests:101` (storyline row tappable) |
| Android | `NativeCapabilityTests:673,679` (person row found by "momentum") |
| iOS | `AppJourneyTests:122,142,437` |
| iOS | `NativeCapabilityTests:83` |

Five suites across both tiers. The direction of risk is a FALSE GREEN — content exists because some
other account listened — and it is order-dependent.

**And `app-e2e-users-reset` interacts with it.** Wiping accounts at tier start changes these rails'
starting point from "everything ever accumulated" to "only what this run has produced by the time
the suite runs". That could make them MORE fragile, not less. The suites sit in phase 5, after
phases 1-4 have downloaded and played, so there should be listening data by then — but "should" is
doing work in that sentence and the full tier had not been run since the reset landed.

NOT FIXED. Nothing here changes the setting or the suites; this is the assessment that was missing.

### M8. The session path: three bugs that hid behind each other

The Android harness answered "is this app signed in, and as whom" six overlapping ways, each
built in a separate slice. Untangling it (advisor-designed, 4 steps) found three defects that were
each concealed by one of the others.

**1. "Signed in" was inferred from the ABSENCE of a control that exists on two pages.**
`hasAnySession()` was "the 'Sign in' link is absent" — but `/login` carries its own "Sign in"
SUBMIT button in the page body. So on the one route you are always on right after signing out, a
signed-IN app reads as signed OUT. Four call sites, and the worst is `signOut()`, which could
report success having signed out nothing.

The same conflation, in a postcondition I wrote that morning, is what broke the tier: the callback
sign-in worked, the masthead showed `simtest`, and the check called it a failure because a "Sign in"
control was still on screen. Fixed by reading a POSITIVE signal — the notifications bell renders
only under `auth.hasSession` (`App.vue:622`), the profile link carries the account name (`:630`).

**2. The UI fallback made failures worse, not safer.** `ensureSignedIn` fell back to the UI flow
when the callback failed, defended as insurance. When the callback actually failed, the fallback ran
`signIn` against an already-signed-in app, `waitForField` grabbed the first `EditText` on screen —
Home's search box — and the suite failed four steps later as "sign-in did not complete" with the
identity typed into search. It converted a precise failure into a confusing one. Deleted; the real
UI flow now has one dedicated test, pinned to a single caller by a guard.

**3. The blank-WebView recovery could never have worked.** Removing the fallback made that failure
loud, and it took ninety seconds to surface:

```
INSTRUMENTATION_RESULT: shortMsg=Process crashed.
=====RELAUNCH webview blank; force-stop + retry 1/2
```

`am force-stop <pkg>` from inside instrumentation kills the app AND the test issuing it —
instrumentation runs in the target's process. The handover recorded this path as "never fired in
~40 clean sign-ins, untested recovery". It was not untested so much as impossible, and its own
comment argued for it ("the only clean recovery is to end it and start again") without noticing the
option does not exist in-process. The same defect had previously cost two full tier runs and was
filed as a SIGN-IN problem, because the first assertion to notice a blank app is always about some
control that was never going to be there.

**Measured**, valid comparisons only (switch confirmed from server state — two accounts under
`/app/state/users` — not from a marker):

| suite | before | after |
| --- | --- | --- |
| `HarnessSmokeTests#signsIn…` | 314.6s | 178.3s |
| `DownloadThroughUITests` (account switch) | 267.6s | 186.7s |

A third of the tier's per-suite cost was navigating to Profile and sleeping, to learn something the
masthead already displayed.

**A FOURTH, found by getting the refactor partly wrong.** The `sleep(6_000)`s were replaced with
"two consecutive agreeing reads", which proves STABILITY but not DURATION — two reads a second
apart both land inside the revalidation window, where boot paints a session and `/me` then refuses
the token. The old path took ~40s and outlasted that window BY ACCIDENT, so deleting it exposed a
race the slack had been hiding. The full tier caught it:

```
=====CALLBACK signed in as appjourneytests in 1906ms, no Custom Tab=====
AssertionError: profile tab 'Topics' not tappable. On screen: … Create your free account … Sign in …
```

Sign-in succeeded, `startClean` believed it, the app signed itself out a few steps later, and the
failure named a Profile TAB rather than the session. `settles` now requires the answer to hold
CONTINUOUSLY for 6s — the sleep's guarantee without the navigation, which was the only part that
was ever waste. `AppJourneyTests` 836.3s/1 failure → 844.7s/OK (8 tests): eight seconds for
correctness.

The transferable bit: **removing a wait is only safe when you know what the wait was for, and "it
looks like padding" is not knowing.** The commit that removed it claimed the guarantee was "kept",
reasoned from the code rather than from a run.

**NOT PROVEN:** that removing the fallback is safe — it needs a run where the callback genuinely
fails, and there has not been one since the net came off. Nor is the relaunch fix verified; it only
proves itself the next time a WebView comes up blank.

### M9. The degraded drill certified an empty api and called it the incident

Tier run 3 reached phase 6 — further than any run before it — and died on `the app's WebView never
painted after two launches`. It had not. The api underneath it had no corpus, no accounts, and an
`/app/state` the non-root app user could not write.

Earlier the same day `APP_E2E_VOL`/`STATE`/`CT` became worktree-scoped. A container from before
that rename was still serving `:8011` against the OLD volumes, and `ios-origin-up` reuses a running
api on health alone — which says nothing about WHICH volumes it has open. So phases 1-5 passed
against volumes the Makefile no longer names. Then `_app-e2e-api-restart` ran
`docker run -v <new-name>:…`, Docker created both volumes empty because that is what it does, and
the drill asserted incident behaviour against an api with nothing in it.

Measured after the fact:

```
lp-e2e-corpus                            07:49:44Z   4.5M   feeds/ enrichments/ search/
lp-e2e-corpus-podcast_scraper-FUTURE     16:36:11Z   8.0K   (empty)
/app/state  root:root  — `touch` as `podcast` -> Permission denied
```

The 503 the drill printed as `✓ api is UP and cannot authenticate anyone — the incident, reproduced`
was `{"detail":"Storage temporarily unavailable (permission denied)."}`, not
`{"detail":"Auth is not configured."}`. **It verified the status code and not the fault** — the same
shape as the bug its own comment describes in the version before it ("really tested 'the user record
was deleted'"). A drill can pass its own proof while reproducing a different incident.

Three guards landed, each where the truth was still available: the restart refuses a missing or
unseeded volume instead of letting Docker invent one; the 503 proof asserts the body; the restore no
longer hides its exit code. Reuse now also requires the container to mount the volumes currently
named.

The transferable bit: **an auto-creating default turns a rename into a failure an hour downstream
and three layers away.** `docker run -v` inventing an empty volume is the same class as
`Journey.originPort()` refusing to default — that argument was already made in this repo, and simply
not applied here until it cost a tier run.

**Caution for the next reader:** the first version of the "is it seeded" guard counted entries and
passed a corpus volume holding one stray `.viewer` directory. It now looks for `feeds/`. A presence
check is not a content check.

**STILL OPEN:** whether `relaunch()`'s blank-WebView check is *also* wrong. It decides from
`labelledInventory`, whose own docstring says it "is allowed to return less than the whole truth"
(it swallows stale-node throws and renders the result as `<nothing labelled…>`), and during the
failure it read non-empty twice and empty once inside ~7 seconds. Run 3 cannot answer it, because
the environment was broken underneath it.

## LANDED this session

| Change | Evidence it works |
| --- | --- |
| Static a11y guard rewritten from a 6-file list to a repo-wide `aria-haspopup` rule | `vitest` 5 passed; mutation-checked (removing `OverflowMenu`'s span turns it red naming that file) |
| `android-suite` rejects `OK (0 tests)` | old check passes that string, new guard rejects it — both demonstrated |
| `app-e2e-users-reset` — per-run server-side account wipe, both tiers | `2 -> 0`; directory keeps `podcast podcast` ownership; no-container case exits 0 |
| `ios-origin-check` between iOS phases 4 and 5 | passes live (`/api/app/me -> 401`), fails correctly against a dead port |
| Audit: stale node no longer blinds a surface | one opaque throw became `15 of 29 nodes went stale` |
| `A11yProbe` + its self-check | self-check asserts ≥5 named controls on Library, so its clearances are not vacuous |
| `AppSession.signInViaCallback` — mint over HTTP, deliver via `appUrlOpen` | see the reliability loop; emits `=====CALLBACK signed in … no Custom Tab=====` |
| `_app-e2e-api-restart` refuses a missing or unseeded volume | missing name and unseeded volume both fail with the reason; a seeded one proceeds — all three run against the live container |
| The degraded drill's 503 proof asserts the body, not the code | the fault it actually got (`Storage temporarily unavailable`) is now rejected by name |
| The drill's restore reports failure instead of `\|\| true` | it had already failed silently once, leaving the api unable to authenticate |
| Reuse requires the running api to mount the volumes we currently name | live container accepted; a foreign name flagged; the `lp-e2e-corpus` prefix correctly rejected |

---

## CONFIRMED but NOT fixed — advisor findings verified by reading

- **iOS `Journey.openProfile` hardcodes `["Your profile", "simtest", "uitest"]`**
  (`Journey.swift:372,379`). Only five iOS suites override to `simtest`; every other suite uses a
  per-class identity that is **not in that list**. Once the masthead resolves to the account name,
  `openProfile` matches only the generic fallback, burns a 25s timeout, and — the sharp part —
  `startClean`'s forced-offline normalisation silently fails, which is the whole guarantee it
  exists to give. Android takes the labels as a parameter; iOS never got that fix.
- **`ios-contact-sheet` photographs an unseeded account.** `ScreenshotTourTests.swift:24`
  overrides to `simtest`, while the seeding suites at `Makefile:2662-2663` use per-class
  identities. Regressed when the per-suite-identity work landed; nothing asserts on it.
- **iOS `startClean` order is inverted** relative to Android's, which documents why the iOS order
  is wrong (`UITestCase.java:76-91` vs `UITestCase.swift:79-84`).
- **`continueAfterFailure = true`** (`UITestCase.swift:68`) — tests keep running, and keep
  mutating server state, after their first failure.
- **The iOS queue step asserts nothing** — `DownloadThroughUITests.swift:264-265` is
  `if exists { tap() }`, which is why `…AndQueuesThem` makes zero `POST /api/app/queue/items`.

None of these were run, because both tiers share `ios-origin-up`'s ports and an iOS run reaps
the Android origin. That port collision is itself worth fixing — it is what stops the two tiers
running concurrently at all.

### M10. iOS: sign-in detection FIXED; sign-out is broken and the runs were not comparable

**Fixed and verified.** `isSignedIn` answered by driving `openProfile` and scrolling Profile for
"Sign out", so a question about the SESSION depended on reaching a page and on a control that is
deliberately the last item on it. `NativeOnlySurfacesTests` spent 500+ seconds in a poll loop
without reaching an assertion: it signed in, could not see that it had, and retried — **five
complete `auth/login` -> `auth/callback` pairs in the api log for one test**. `openProfile`'s own
diagnostic cleared the control twelve times (`PROFILE_CTL link 'simtest' frame=(349.0, 62.0, 48.0,
18.0) hittable=true`), which is what makes it a DETECTION bug, not a navigation one.

Both overloads now read the masthead and navigate nowhere, matching Android. Sign-in resolves at
**t=16s on the first attempt**, and **every run since has shown 0 SETTLE markers** — the session was
always valid and the old check simply could not see it.

**Still broken: `signOut`.** Three attempts, each "no 'Sign out' on Profile", over an inventory
showing generic `Your profile` and no bell — the signed-out render (`v-if="auth.isAuthenticated"`,
`ProfileView.vue:804`). Fails as "signed in as another account and could not sign out".

Why it was invisible before: **two bugs were cancelling.** The broken detection made `signOut`'s own
guard conclude "already signed out" and return success without signing anything out. Fixing
detection is what made this reachable.

**THE METHODOLOGICAL FAILURE, which cost more than the bug.** I recorded "test passed in 142.4s" as
a baseline and spent four changes trying to restore it. It was ONE observation. That run followed a
full tier plus `app-e2e-users-reset`, so the app was signed in as the right account and `signOut`
never ran. Every later run inherited the previous run's account, so `signOut` DID run, and failed.

The test never regressed — it started taking a path the lucky first run skipped. I reverted four
changes on a false premise. (The revert was still correct: two of the four rested on causes I had
inferred rather than read — an Account-tab theory disproved by `Change photo` being in the inventory
all along, and a keyboard/caret theory built from a mid-flow screenshot.)

**Every iOS run in this session measured a different starting device state, and I compared them as
if they were comparable.** That is the test-independence problem raised at the START of the session,
and it is not a side concern — it is why an hour went into chasing a moving target. Fix it BEFORE
the next iOS fix: no iOS result means anything until each run starts from a known account state.

**NOT VERIFIED:** whether `signOut` works from a clean state; the four other iOS defects found by
reading; iOS phases 5-6 end to end. The Account-tab guard landed on BOTH platforms and is unfired on
each — it is real (KEEP_ALIVE_TABS is real) but it fixed nothing observed.
