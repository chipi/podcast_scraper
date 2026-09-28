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
