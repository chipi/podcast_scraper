# Handover — player offline / profile-missing / "can't reach server" (2026-09-15)

**For:** the agent picking up the fix. **From:** static analysis, then a live iOS-simulator
validation pass (2026-09-16). No app source was changed. See "SIMULATOR VALIDATION" below for
what was actually reproduced — that section supersedes the suspicions in the original analysis.

## TL;DR

The three symptoms you reported — (1) offline behaviour breaks normal operation, (2) all profile
screens missing, (3) phone can't reach the server — are the SAME failure mode: on a native device
the app decides it is offline and/or signed-out when it is neither, and then hides/short-circuits
everything that depends on being online + signed-in.

Two commits **from tonight (Sep 16, 00:23 and 00:32)** already target exactly this:

- `5c21ec20a` — *"stop reads failing on the auto-offline signal + popover/data fixes"*
- `701f0fec7` — *"top-voices grid + Your Week dedup at the digest source"*

Their own commit message says, verbatim: **"NOT verified on device: a provider/prod outage is in
progress, so the on-device pass is still pending."** So the state you tested may be *before* this
fix, or the fix may be incomplete. This handover maps each symptom to the exact code, states what
the fix changed, and lists what still needs checking.

**I have NOT verified the fix works, and I have NOT reproduced the still-broken state.** Do not
treat any line below as "fixed" until it is confirmed on a real device.

---

## Symptom → governing code map

### Symptom 2 — "all profile screens are missing"

Profile is no longer a bottom-nav tab (moved to the masthead avatar, operator 2026-09-09,
`components/BottomNav.vue:65`). The **only** entry to every profile screen is the masthead avatar:

- `App.vue:566-575` — the profile `RouterLink` is gated `v-if="auth.isAuthenticated"`.
- `App.vue:562` — the notification bell: `v-if="auth.isAuthenticated"`.
- `App.vue:547-558` — the Library nav icon: `v-if="auth.isAuthenticated"`.
- When `auth.isAuthenticated` is false the masthead shows a **"Sign in"** link instead
  (`App.vue:580-586`).

`auth.isAuthenticated` is `user !== null` (`stores/auth.ts` getters). So the moment `auth.user`
is null, **the avatar, the bell, and the Library icon all vanish and profile becomes unreachable
by tapping** — exactly "all profile screens missing".

`/profile` also has `meta: { requiresAuth: true }` (`router/index.ts`), but the router comment
says that flag is a legacy no-op now that the login-first guard denies by default.

### Symptom 3 — "phone cannot reach server" / "a bit works then a bit not"

Root cause per the `5c21ec20a` RCA: commit `#2034` (2026-09-11) added `if (isOffline()) throw` to
**every read** in `services/api.ts`. On device the offline signal false-negatives (WKWebView
`navigator.onLine`, cellular/VPN handoff, slow cold-start network report), so a **connected** app
had its whole read layer short-circuited → blank profile, empty collections, "couldn't reach the
server", and the tell-tale "a bit works then a bit not".

### Symptom 1 — "offline messes up normal operation"

Same cause as #3: the auto-detected offline signal was gating reads. The fix is to stop trusting
that signal for reads.

---

## What tonight's fix changed (verified by reading the committed diff)

1. **Read gate split by method** (`services/api.ts`, `apiFetch` + `getJSON`):
   - GET reads now fast-fail **only** on `isForcedOffline()` (the explicit Config testing switch),
     **never** on the auto-detected signal.
   - Writes still gate on the full `isOffline()` so mutations route to the outbox.
   - Confirmed: `isOffline` is referenced in only 2 non-test source files — `useOnline.ts` (the
     definition) and `api.ts` (writes only) + `App.vue:411` (a reconnect watch on `forcedOffline`).
     No read path still gates on the auto signal. **The read-gate fix is complete in source.**

2. **`useOnline.ts` connectivity source** — native now derives connectivity from
   `@capacitor/network`, seeded optimistically online; `navigator.onLine` is ignored on native.
   New `isForcedOffline()` is what reads gate on.

3. **Auth boot ordering** (`main.ts` + `services/native.ts`):
   - `main.ts` awaits `Promise.allSettled([rehydrateNativeToken(), initGateCookie()])` **before**
     `app.mount('#app')`. So the native Bearer token is in memory before the router's first
     navigation. This closes the earlier race where the first `GET /me` was anonymous, 401'd, and
     wiped the device snapshot. **This ordering is correct in the committed code.**

4. **Router guard token fallback** (`router/index.ts`): `signedIn = auth.isAuthenticated ||
   (isNative() && !!getAuthToken())` — a native device with a stored token is treated as signed-in
   for routing even if the cold-start `getMe` transport-failed.

5. **Auth store** (`stores/auth.ts`): only a 401/403 destroys cached auth state; a transport error
   keeps the device snapshot. `ensureLoaded()` paints the snapshot instantly and revalidates in the
   background (never blocks first paint).

---

## Build / deploy state (verified)

- `web/learning-player/dist/` was built Sep 16 00:36; bundle build sha = `701f0fec76c2`
  (matches the latest commit).
- `web/learning-player/ios/App/App/public/` bundle sha = `701f0fec76c2` too — `cap copy`/`sync`
  ran after the fix, so the iOS web assets on disk carry the fix.
- **Uncommitted native change:** `android/app/capacitor.build.gradle` and
  `android/capacitor.settings.gradle` add `capacitor-push-notifications` (from `cap sync` after the
  `be58472b0` native-push work). These are in the working tree, NOT committed. The Android project
  is in a dirty state — commit these or the next clean checkout's Android build won't match.

---

## VERIFIED (by static reading, this session)

- The read-gate fix is present and complete in source (no read path gates on the auto signal).
- The native token rehydration is awaited before mount in `main.ts`.
- The router guard has the native-token fallback.
- `dist/` and the iOS-bundled `public/` both carry sha `701f0fec76c2`.

## NOT VERIFIED / open suspects (equal weight — read this section first)

1. **NOTHING has been confirmed on a real device.** The fix commit itself says the on-device pass
   is pending because a prod/provider outage was in progress. **First action for the fixing agent:
   run it on device and read `window.__buildInfo.sha` in the WebView console to confirm the device
   is actually running `701f0fec7` and not a stale build.** "Still broken" may simply be a stale
   install of the pre-fix binary.

2. **Masthead vs guard use two DIFFERENT definitions of "signed in" — likely-remaining bug.**
   - Guard (`router/index.ts`): `isAuthenticated || (isNative() && !!getAuthToken())`.
   - Masthead avatar/bell/library (`App.vue:547,562,567`): `auth.isAuthenticated` only
     (i.e. `user !== null`).
   - Consequence on native: if the token is valid but `auth.user` never populates (no device
     snapshot yet AND cold-start `/me` transport-fails), the guard lets you REACH `/profile` by URL,
     but the masthead shows **"Sign in"** and there is **no tappable way into any profile screen** —
     which reads exactly as "all profile screens missing" while the token is fine. This is a real
     source-level inconsistency. **I have NOT proven it is the cause of your report** — it needs to
     be reproduced on device (kill the network at cold start, or point at a down `/me`). If
     confirmed, the fix is to make the masthead gates agree with the guard's signed-in definition
     (or to guarantee `auth.user` is populated from the token before first paint).

3. **`@capacitor/network` cold-start report on device is unverified.** The fix assumes seeding
   optimistically-online + letting the plugin correct is enough. If `Network.getStatus()` returns
   `connected:false` transiently at cold start, writes (not reads) would still route to the outbox.
   Verify the actual plugin behaviour on the target device/OS.

4. **Android push plugin is only half-wired** (uncommitted gradle, see build state). If you build
   Android, confirm push doesn't crash boot; native iOS push was the committed path (`be58472b0`).

5. **The "provider/prod outage" mentioned in the commit** — confirm it's over before concluding
   anything about "can't reach server". A real backend outage would reproduce symptom #3
   independent of any client bug.

---

## How to reproduce / verify on device (suggested order)

1. Rebuild + reinstall the app from `701f0fec7` (or newer). Confirm `window.__buildInfo.sha` in the
   WebView console.
2. Cold-start online, signed in: does the masthead avatar appear? Can you open every profile tab
   (Account / Topics / Stats)? Do collections/library load?
3. Cold-start with the network killed at launch, then restored: does the app strand on the landing?
   Does the avatar stay hidden after reconnect? (This exercises suspect #2.)
4. Toggle the Config forced-offline switch: reads should fast-fail, writes should queue. Untoggle:
   `resumeAfterReconnect()` (App.vue:411) should replay.
5. If suspect #2 reproduces, align `App.vue` masthead gates with the guard's signed-in definition.

---

# SIMULATOR VALIDATION (2026-09-16) — what was actually reproduced

Setup: `make ios-origin-up` (fixture api + mock media + vite-preview origin on :4174), web build
with `VITE_API_BASE_URL=http://127.0.0.1:4174/api/app`, `npx cap sync ios`, `xcodebuild` →
`** BUILD SUCCEEDED **`, installed on the "iPhone 17" simulator (iOS 26.3). A real native session
token was obtained by driving the mock OAuth flow with `?platform=native`, which returns the
genuine `closelistening://auth#token=…` deep link.

Gotcha for the next agent: `NODE_OPTIONS` in this shell carries a cmux preload shim
(`restore-node-options.cjs`) that does not exist, so `npx vite preview` dies instantly with
`MODULE_NOT_FOUND`. Run node/npm/npx steps as `env -u NODE_OPTIONS …`.

## CONFIRMED BUG — dead native token + no snapshot = signed-out UI on a route the guard admitted

Reproduced by seeding `CapacitorStorage.lp_native_token` with a well-formed but invalid-HMAC token
and deleting `CapacitorStorage.auth.me`, with the server UP. Observed:

- The router guard **admitted the user to Home**. Under login-first an unauthenticated visitor must
  be redirected to `/welcome`; that did not happen, because the guard's `signedIn` is
  `auth.isAuthenticated || (isNative() && !!getAuthToken())` and the stored token satisfied it.
- The masthead simultaneously rendered the **signed-out** state — "Sign in", **no profile avatar,
  no notification bell, no Library icon** — because `App.vue:547,562,567` gate on
  `auth.isAuthenticated` alone. **Profile is therefore unreachable by tapping: this is the
  "all profile screens are missing" report.**
- Server log for that launch: `GET /api/app/me 401`, `/api/app/trending 401` (×2),
  `/api/app/podcasts 401`, while `/api/app/corpus/trending-topics 200` and `/api/app/discover 200`.
  Public endpoints render, authed ones fail → **this is the "a bit works, a bit doesn't" report**,
  and the UI shows "Couldn't load this right now. Try again."
- **No self-heal.** `onUnauthorized` → `markSignedOut()` is gated on `auth.isAuthenticated`, which
  is false here, so the 401 interceptor no-ops and the dead token is never cleared. A full
  terminate + relaunch reproduced the identical broken screen with the token still stored.
  (Accuracy: a "Sign in" affordance IS visible, so the state is recoverable by signing in again —
  it is not a hard lock. But profile/library are gone and the app reads as broken until the user
  works that out. I did NOT test whether tapping Sign in recovers it.)

The guard's own comment asserts the opposite of what happens: *"a token that is genuinely dead
surfaces as a 401 on the first authed call and the interceptor … then routes to signed-out
cleanly."* It does not, whenever no user was painted first.

### Isolation (so the fix targets the right thing)

| State | Result |
| --- | --- |
| Valid token + snapshot + server up | Healthy — avatar, bell, Your Week all render |
| **Valid token + NO snapshot + server up** | **Healthy** — avatar returns; missing snapshot alone is harmless |
| **Invalid token + NO snapshot + server up** | **BROKEN** — guard admits, masthead signed-out, no profile, no self-heal |

So the trigger is specifically an **invalid/expired token with no device snapshot**, not a missing
snapshot. Suggested fix direction (not applied): make the masthead's signed-in test agree with the
guard's, AND make the 401 interceptor clear a native token even when `auth.isAuthenticated` is
false — otherwise the app can sit on a dead credential indefinitely.

How that state is reached in the field: `refresh()` on a 401 removes the snapshot and clears the
content cache but does **not** clear the native token; if `auth.isAuthenticated` was false at that
moment, `onUnauthorized` no-ops too. The next launch is then token-without-snapshot.

## ALSO CONFIRMED — "reads are open" is false, and three comments rely on it

Anonymous probes against the fixture api: `/api/app/me` **401**, `/api/app/episodes` **401**,
`/api/app/podcasts` **401**; only `/api/app/corpus/trending-topics` (and `/discover`) returned 200.
Server-side, those routes depend on `get_current_user` (raises 401), not `get_optional_user`
(`server/routes/app_auth.py:75-102`). Yet:

- `services/api.ts` header: "reads are open, per-user writes require auth"
- `stores/auth.ts` header: "Reads are open, so the app works signed-out"
- `router/index.ts` guard: "Reads are open so the app still populates"

That last one is load-bearing — it is the stated justification for admitting a native token holder
without a confirmed session. Doc-vs-code divergence; decide which side is wrong and fix it.

## VERIFIED GREEN in the simulator

- Build: `xcodebuild … build` → `** BUILD SUCCEEDED **`.
- Cold start signed-out → correctly renders the `/welcome` lure landing.
- Valid token + server up → full signed-in Home: avatar, bell, Your Week, trending, search,
  "What's new"; every `/api/app/*` call in that launch returned 200 and **zero 401s**.

## NOT VERIFIED / still open (read with equal weight)

1. **The original offline false-negative was NOT reproduced, and cannot be on this setup.** The
   field bug is `@capacitor/network`/`navigator.onLine` reporting *disconnected while the server is
   reachable*. In the simulator the plugin reports connected, so both the pre-fix and post-fix code
   take the same branch. Reproducing it needs the Mac's Wi-Fi off while the app talks to
   `127.0.0.1` — a change to the operator's machine I did not make without asking. **So the
   read-gate fix (`5c21ec20a`) remains verified by source reading and its unit tests only, not by
   observed device behaviour.**
2. **Nothing was tested on real hardware.** WKWebView cold-start connectivity, cellular/VPN
   handoff, and background/foreground transitions are all simulator-exempt.
3. **Forced-offline switch, outbox replay, and `resumeAfterReconnect()` were not exercised.**
4. **Whether tapping "Sign in" recovers the stuck state** — not tested.
5. **Android** was not built or run at all; the `capacitor-push-notifications` gradle edits remain
   uncommitted in the working tree.
6. The simulator run used the **fixture corpus + mock OAuth provider**, not prod. A prod-only
   auth/gate difference would not show up here.

## Reproduction recipe (copy-paste)

```
env -u NODE_OPTIONS make ios-origin-up
cd web/learning-player && env -u NODE_OPTIONS VITE_API_BASE_URL=http://127.0.0.1:4174/api/app npm run build
env -u NODE_OPTIONS npx cap sync ios
cd ios/App && xcodebuild -workspace App.xcworkspace -scheme App -configuration Debug \
  -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17' \
  -derivedDataPath /tmp/lp-ios-dd CODE_SIGNING_ALLOWED=NO build
xcrun simctl boot "iPhone 17"; xcrun simctl install booted /tmp/lp-ios-dd/Build/Products/Debug-iphonesimulator/App.app
# real native token via the mock provider:
#   GET /api/app/auth/login?as=simtest&platform=native  -> follow redirects -> closelistening://auth#token=...
# seed state (app must be terminated first; write via the daemon, read the container plist):
xcrun simctl spawn booted defaults write app.closelistening.player CapacitorStorage.lp_native_token -string "<token>"
xcrun simctl spawn booted defaults delete app.closelistening.player CapacitorStorage.auth.me
xcrun simctl launch booted app.closelistening.player
```

Inspection gotcha that cost me a false alarm: `simctl spawn defaults read` and the container plist
(`…/Data/Application/<UUID>/Library/Preferences/app.closelistening.player.plist`) are two lazily
converging views of the same domain. Mid-session the daemon showed `lp_native_token` but not
`auth.me`, and the file showed `auth.me` but not `lp_native_token`. Neither was missing. The
Makefile documents this: write through the daemon, read the file — and do not conclude a key was
deleted from one view alone.

---

# THE PROD INCIDENT: "services up, nothing working" (2026-09-16)

The operator's report: after a hardware reboot prod lost some secrets. Services were reachable but
non-functional, and the app showed "a weird mix of cached things and broken things" without ever
detecting that the server was down.

**That is the same defect reproduced above, and the mechanism is now identified.**

## Root cause: a server fault is reported with a client-fault status code

`server/routes/app_auth.py`:

```python
def get_current_user(request: Request) -> User:
    secret = _secret(request)
    data_dir = _data_dir(request)
    if not secret or data_dir is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")   # <-- server fault, 401
```

`session_secret` comes from `APP_SESSION_SECRET` (`server/app.py:117`). Lose it on reboot and
`_secret()` returns `""`, so **every authed endpoint returns 401** — the status that means *your
credential is bad*, for a condition that is entirely the server's fault and identical for every
user on the platform.

**The same file already disagrees with itself.** For the *same* condition, the login route returns
503:

```python
@router.get("/auth/login")
...
    if provider is None or not secret:
        raise HTTPException(status_code=503, detail="Auth is not configured.")  # <-- 503 here
```

So "auth is not configured" is a 503 on one route and a 401 on the others. The client can only act
on what it is told, and it was told every user's credential had gone bad simultaneously.

## Why `/api/health` did not save it

`server/routes/health.py` reports code version, corpus version, and feature flags. It does **not**
check `session_secret`, the user store, or any auth dependency — so during the incident it kept
answering **200 OK** while every authed read 401'd. A client health-check would have been reassured
by it. Health is reporting *liveness*, and what the app needs is *readiness*.

## What the client then did with that 401 — two bad outcomes, both observed

- **With a painted user** (`auth.isAuthenticated` true): `refresh()` maps 401 → `getMe()` returns
  null → it deletes the device snapshot **and calls `clearCached(CACHE_KEYS)`**. So a server-side
  secret loss **destroys the user's cached content** — the very thing offline mode depends on.
  Then `onUnauthorized` → `markSignedOut()` → token cleared → bounced to the landing.
- **Without a painted user**: `onUnauthorized` no-ops, nothing is cleaned up, the dead token
  survives, and the app sits in the half-broken state reproduced and screenshotted above — signed-out
  masthead, no profile, public endpoints rendering and authed ones failing. **This is the operator's
  "weird mix" exactly.**

Neither is the desired behaviour. The desired behaviour is: *notice the server is not trustworthy,
go offline, keep the cache, and say so.*

## VERIFIED 2026-09-16: the app does NOT go offline when the server is unavailable

Direct reproduction of the operator's scenario — server stopped, **network still up**, cache warm,
valid token, cold launch (`/tmp/lp-shot-09-serverdown.png`):

- **No offline banner at all.** Forced-offline raises "Offline — showing saved" (screenshot `08-d`);
  with the server actually down, that banner is **absent**. The app still believes it is online.
- Cached content renders — "Continue listening · Resume 0:12", "Jump back in" — alongside a card
  reading "Showing what you had last time — we couldn't reach the server. [Try again]".
- **That is the "weird hybrid" precisely:** cached/public things render, live things show error
  cards, and nothing tells the user the app is in a degraded mode.

Mechanism: `isOffline()` is `forced || navOnline === false`, and `navOnline` is driven **only** by
`@capacitor/network`, which reports the DEVICE's link state. A dead, misconfigured, or
secret-less server is completely invisible to it. The app has no concept of "server unreachable =
offline".

Consequence beyond the cosmetics: because `isOffline()` stays false, **writes do not route to the
outbox** — a mutation made while the server is down fails instead of queueing for replay, which is
the whole point of having an outbox.

This confirms the operator's ask is the correct fix: server-unreachable should enter offline mode
the same way no-network does, with its own reason and copy.

## Proposed hardening

Server (small, and removes the ambiguity at source):

1. **Return 503, not 401, when auth is unconfigured** — make `get_current_user` agree with
   `/auth/login`. A client can then distinguish "my token is bad" (401) from "this server cannot
   authenticate anyone" (503) with no heuristics at all. This single change is the highest-value fix.
2. **Add readiness to `/api/health`**: `auth_ready` (signing secret present), `user_store_ready`
   (data dir readable). A degraded server should say so rather than answering 200.
3. **Expose a non-secret `auth_epoch`** — a fingerprint of the signing key. When it changes, every
   token is invalid *for a server-side reason*; the client can then re-authenticate deliberately
   instead of inferring mass credential death.

Client:

4. **Never destroy cached content on a 401 alone.** Deleting the offline library is unrecoverable
   from the user's side and is the wrong response to a server fault. Require corroboration — a 503,
   a changed `auth_epoch`, or repeated 401s across a healthy server.
5. **Replace the binary offline flag with a REASON**: `online` / `offline:network` /
   `offline:server` / `offline:forced`. `offline:server` is the new state the incident needed; it
   keeps the cache, suppresses writes to the outbox, and drives its own copy.
6. **Fix the masthead/guard disagreement** (see the confirmed bug above) so a degraded server can
   never produce a signed-out masthead on a route the guard admitted.

## How to TEST it — the incident is directly reproducible

The fixture api takes its secret from the environment, so the reboot can be simulated exactly:

```
# 1. sign in and browse so the cache is warm
make ios-journey-signin && make test-app-ios-journey-ui      # or browse by hand
# 2. simulate the reboot: same container, different signing secret
docker rm -f lp-e2e-api
APP_SESSION_SECRET=<different-value> make app-e2e-api-up
# 3. relaunch the app and assert the DESIRED behaviour
```

Assertions that should hold and currently do not:

- cached content still renders (no `clearCached` wipe)
- an offline/degraded banner naming a SERVER problem, not a credential problem
- no silent bounce to the landing, no stuck signed-out masthead
- a deliberate "sign in again" prompt only if the server says the credential is bad (401 from a
  *healthy* server)

This belongs in the journey suite as `test11ServerSecretRotation`. **I have NOT written or run it**
— it is specified here, not verified.

---

# OFFLINE COPY — what to say instead

With the cache warm, forced-offline Home renders correctly (verified, screenshot `08-d`): banner
"Offline — showing saved", topics rail with real data, Your Week, mini-player. The defect is one
card:

> **Showing what you had last time — we couldn't reach the server.**  [Try again]

Two things are wrong with it in forced-offline: "we couldn't reach the server" is false (the app
chose not to ask, and the server is up), and "Try again" offers a retry that the read gate will
refuse — a button that cannot work.

Copy should be chosen by the REASON (item 5 above), and should distinguish "we have something
cached" from "we have nothing cached for this":

| State | Has cache | Suggested copy | Action |
| --- | --- | --- | --- |
| `offline:forced` | yes | "Offline mode is on — showing what you saved." | "Turn off offline mode" |
| `offline:forced` | no | "Offline mode is on. You haven't opened this yet, so there's nothing saved to show." | "Turn off offline mode" |
| `offline:network` | yes | "You're offline — showing what you saved." | "Retry" (legitimate) |
| `offline:network` | no | "You're offline and this hasn't been saved to your device yet." | "Retry" |
| `offline:server` | yes | "We can't reach your account right now — showing what you saved." | "Retry" |
| `offline:server` | no | "We can't reach your account right now. Try again shortly." | "Retry" |

The principles: never claim a failure the app didn't actually attempt; never offer an action that
cannot succeed; and when nothing is cached, say *why* it isn't (you haven't opened it yet) rather
than reporting a load error — the app knows which surfaces it caches.

---

# THE NATIVE JOURNEY SUITE (new, 2026-09-16)

Files, all under `web/learning-player/ios/uitests/Sources/UITests/`:

| File | Covers |
| --- | --- |
| `Journey.swift` | shared helpers: screenshots, compact element inventory, multi-type lookup, scroll-to, nav, `setOfflineMode` |
| `ConfigOfflineToggleTests.swift` | Settings → Config → Offline mode (the only way to set the flag; it lives in `localStorage`, which the host cannot reach) |
| `AppJourneyTests.swift` | 01 profile tabs · 02 episode → insights · 03 topic → storyline · 04 person · 05 collections · 06 share · 07 saved colour picker |
| `PersonalisationTests.swift` | 09 play → stats · 10 interests → chips → Home |
| `OfflineCacheTests.swift` | 08 browse-then-offline cache contract |

Run:

```
make ios-origin-up            # fixture api + media + single origin (NODE_OPTIONS must be clear)
make ios-journey-signin       # seed a real native session — the surfaces are auth-gated
make test-app-ios-journey-ui
make ios-journey-shots        # extract screenshots from the .xcresult
```

## Suite status — VERIFIED PASSING

01 profile tabs · 02 episode+insights · 03 topic+storyline · 04 person · 05 collections ·
06 share · 07 colour picker · 09 play→stats · 10 interests. Each attaches screenshots.

## Suite status — NOT DONE / WEAK (equal weight)

- **08 offline-cache test has never passed.** It asserts no "Try again" in forced-offline, which is
  the defect above — it is currently a RED test describing desired behaviour, and it is the only
  test in the suite written to fail on purpose. Decide whether to keep it red or mark it pending.
- **10 interests is weak**: only 1 of 4 interest chips was tappable (3 `TAP_MISS`), and it does NOT
  assert that Home's recommendations changed — it only screenshots Home afterwards. The operator
  specifically asked that interests be shown to fuel Home; that assertion is missing.
- **Two false passes were found and fixed** — worth knowing the shape: 04 matched the person's NAME
  in transcript text and "passed" while never leaving the player; 03 asserted against the storyline
  sheet's own "STORYLINES" kicker. Both now assert on controls unique to the destination. Assume
  other label-based assertions carry the same risk.
- **`test11ServerSecretRotation` is specified above but NOT written.**
- No coverage yet: search, queue, downloads through these surfaces, notifications/bell, sign-out,
  Discover tab, Revisit tab.
- Suite runs **only** on the simulator against the **fixture** corpus and **mock** OAuth. It cannot
  reproduce WKWebView cold-start connectivity, cellular/VPN handoff, or prod-only auth/gate config.

## UI defects noticed while screenshotting (not investigated)

- A glyph renders as a "?" box in at least three places: the profile avatar's ✎ edit badge, the
  Stats "day streak" tile, and the person page's action row. Looks like a missing-font fallback for
  a non-emoji symbol character on iOS.
- Fixture episode artwork shows "?" placeholders on the logged-out landing — probably fixture data,
  not a bug.

---

# IMPLEMENTED 2026-09-16 (working tree, uncommitted, NOT pushed)

## Server

- **`get_current_user` now returns 503, not 401, when auth is intended but impossible.**
  `server/routes/app_auth.py`. Scoped deliberately: 503 only when an OAuth provider IS configured
  (the deployment authenticates) yet the secret / user store is missing. With no provider at all —
  the tailnet and operator modes — it still returns 401, because nothing is broken there and the
  login-first route matrix rightly asserts a deny. My first attempt was unscoped and broke 13
  integration tests; the scoped version fixes all 13 **without editing any of them**.
- **`/api/health` reports readiness**: new `auth_ready`, and `status` becomes `"degraded"` when a
  deployment that authenticates cannot. Also scoped to provider-configured deployments so the
  no-auth modes aren't cried wolf over.

## Client

- **Offline REASON model** (`composables/useOnline.ts`): `'forced' | 'network' | 'server' | null`,
  with `forced` > `network` > `server` precedence. New `reportServerReachable(ok)` and
  `offlineReason()`.
- **`services/api.ts` probes the server on every request** — a transport failure or a 503 counts
  against it; any success clears it instantly. Debounced at 2 consecutive failures so one slow
  request cannot flip the UI. **Reads are still gated ONLY on `forced`** — gating them on `server`
  would mean a degraded app could never discover the server had recovered.
- **`OfflineBanner`** names the cause, and now appears at all for a server fault — the case where it
  previously stayed silent.
- **`StaleNotice`** picks copy by reason and **hides "Try again" in forced offline**, where the read
  gate guarantees the retry cannot succeed.
- **One definition of "signed in"**: `auth.hasSession`, read by BOTH the router guard and the
  masthead, so they can never disagree again. Two subtleties found while doing it:
  - the native token had to become a `ref`, because a Pinia getter is a Vue computed and a plain
    module variable is not a reactive dependency — the masthead would have cached its first answer;
  - the getter must read the token **before** the `isNative()` short-circuit, or on web (and on the
    first native evaluation) the ref is never read, nothing is tracked, and the computed caches
    forever. This is what the guard test caught.
- **`main.ts` 401 interceptor gates on `hasSession`**, so a dead native token with no painted user
  is now discarded and routed — closing the no-self-heal stuck state.
- **`refresh()` will not clear the content cache while the server is degraded** — belt to the 503's
  braces, so an outage can never again destroy a user's offline library.

## `auth_epoch` — telling "my session expired" from "everyone was invalidated"

A 401 is ambiguous: it is emitted both when one user's token ages out AND when the server rotates or
loses its signing key and kills every token at once. The correct client response differs — the first
should sign out and clear that account's cached content; the second must keep it, because the user
did nothing wrong and their library is still theirs. Conflating them is what made the incident
destructive rather than merely annoying.

- **Server** (`routes/health.py`): `auth_epoch` = `HMAC-SHA256(secret, "auth-epoch")`, truncated.
  Deliberately an HMAC **keyed by** the secret, not a digest **of** it: this is served on an
  unauthenticated endpoint, and a bare digest of a low-entropy secret is brute-forceable offline.
  `None` when auth is unconfigured.
- **Client** (`services/authEpoch.ts`): remembers the last epoch seen. A *first* sighting is
  explicitly **not** a rotation (a fresh install has no prior key), and a `null` is ignored rather
  than stored (so turning auth off and on again does not read as a rotation). Recorded on the health
  call `useAppUpdate` already makes, and cleared on a deliberate sign-out.
- **Decision point** (`stores/auth.ts`): on a 401, the cache is preserved if ANY of three
  independent signals says the fault is the server's — connectivity already considers it degraded,
  health reports `auth_ready: false`, or the epoch changed.

## Glyph "?" boxes — fixed at the cause

Three tofu boxes in the 2026-09-16 screenshots were all non-ASCII symbol characters with no font
fallback behind them:

- `＋` U+FF0B (**fullwidth** CJK plus) in the follow / add-to-collection affordances — 8 occurrences
  across components and copy. Replaced with an ASCII `+`; a fullwidth CJK plus in a Latin UI is
  simply the wrong character, and no Latin font fallback reliably carries it.
- `✎` U+270E on the profile avatar's edit badge → replaced with an inline SVG pencil, which is the
  convention every other icon in this app already follows. It was the outlier.
- `🔥` U+1F525 on the Stats day-streak tile → a genuine emoji, covered by the fallback below.
- `theme/directions.css`: every UI and display stack now carries
  `'Apple Color Emoji', 'Apple Symbols', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji'`,
  inserted **before** the trailing generic family — entries after a generic may never be consulted,
  since the generic resolves to a concrete font. Mono stacks are untouched: they render code and
  version strings where widening the stack would break column alignment.

## Tests added

- `composables/useOnline.reason.test.ts` — 5 tests: threshold, instant recovery, precedence, and
  that reads are never gated on `server`.
- `tests/integration/server/test_app_auth.py` — 503-not-401, **401 still returned for a bad
  credential on a healthy server** (so "always 503" cannot pass), and health readiness both ways.
- `ios/uitests/.../ServerDegradedTests.swift` + `make test-app-ios-server-degraded` — reproduces the
  incident by restarting the api with a different `APP_SESSION_SECRET`: warm cache while healthy,
  rotate, then assert the app notices, stays honest, and keeps its cache.
- `PersonalisationTests` test10 strengthened: it now asserts Home stops prompting for interests and
  that a chosen interest actually surfaces — previously it only took a screenshot.
- `router/guard.test.ts` updated to drive the real `setAuthToken` instead of stubbing the accessor.
  The assertion is unchanged; a stubbed function cannot be a reactive dependency.

## Gates run

- `vue-tsc --noEmit` → clean.
- `vitest run` → **144 files / 1466 tests passed**.
- `pytest tests/integration/server/` → **1327 passed**, 7 skipped, 2 xfailed.
- `pytest tests/unit/podcast_scraper/server/` → **1031 passed**.
- `xcodebuild build-for-testing` → `** TEST BUILD SUCCEEDED **`.

## NOT VERIFIED — read with equal weight

- **None of the client changes have been run on the simulator or a device.** Everything above is
  unit-tested and type-checked only. The whole point of the work is device behaviour, so this is
  the significant gap: `make test-app-ios-server-degraded` has **never been executed**, and the
  rebuilt app has not been installed since the changes.
- `test08` (offline copy) has not been re-run against the new `StaleNotice`; it should now pass, but
  that is a prediction, not a result.
- The `auth_epoch` / signing-key-fingerprint idea from the hardening list is **not implemented** —
  only the 503 and the health readiness were.
- No web-tier (Playwright) run, no `make ci-fast`, no `make ci-ui-fast`.
- The "?" glyph-fallback rendering bug is untouched.
- Nothing is committed and nothing is pushed.

---

# OPEN QUEUE — operator visual review, 2026-09-16 (device build, prod data)

Raised while reviewing the live build. Nothing here is implemented yet unless marked DONE.

## Functional (ahead of the cosmetics)

- **Tapping an episode from a topic or person sheet lands on HOME instead of the player.** Opening a
  SHOW from the same sheet works. Shape strongly suggests the `/episode/:slug` push failing to match
  and falling through to the catch-all `{ path: '/:pathMatch(.*)*', redirect: { name: 'home' } }` —
  which is exactly "goes to the home page". Two candidate causes, NOT yet distinguished:
  a missing/blank `slug` on the entity card's episode rows in PROD data (the fixture corpus DOES
  carry `slug`, so this cannot be reproduced locally without prod-shaped data), or a slug that does
  not survive as a single path segment. `useModalSheet`'s dismissal was inspected and looks correct
  — the `closedByNavigation` guard already prevents the extra `router.back()`.
  **Needs reproduction on device before any fix.** Worth a defensive change either way: an episode
  row with no usable slug should not render as a link, and the catch-all silently swallowing a
  malformed app route is what turned this into "it goes home" instead of a visible error.

## Layout / sizing

- **"Your Week" rails**: artwork is not square and row heights differ once expanded via *Show more*.
  Should be square, like *Jump back in*. Rectangular is CORRECT on *Continue listening* and on the
  featured pick in *What's new* — do not change those.
- **Topic sheet ordering**: move *Top voices* up to sit directly under *Strongest shows on this topic*.

## Entity sheet stacking

- **Storyline opened from a topic** covers too much: it should sit lower so the topic beneath still
  shows its kicker + title, and it currently hides the action bar.
- **Person opened from a topic** goes fully to the top and hides the topic entirely — same fix.

## Remove / add

- **Remove "Open in page ›"** from BOTH the topic sheet and the person sheet (no value).
- **Add a queue control** next to the ▶ play affordance on the *What's new* rows.

## DONE in this session, pending VISUAL confirmation on the contact sheet

- On-artwork action buttons: per-button scrim (`overlay` variant on `EpisodeActions`).
- Saved row: heart → ⋯, colour control takes its slot (no more second row).
- Board covers: lazy backfill on read (server).
- Glyph tofu: `＋` → ASCII, `✎` → inline SVG, emoji/symbol font fallbacks.

# NATIVE CAPABILITY COVERAGE (approved 2026-09-16)

The Capacitor shell enables ten plugins plus two local ones, and almost none are exercised on
device. `cap sync` reports: speech-recognition, app, browser, filesystem, network, preferences,
push-notifications, share, splash-screen, sentry — plus local `BackgroundAudio` (Android) and
`AuthSession` (iOS ASWebAuthenticationSession).

`NativeCapabilityTests`, first three by value — all user-visible, newly wired, and unverified:

1. **Speech recognition** — mic → dictated notes, including the permission prompt path.
2. **Share sheet** — the NATIVE sheet, as opposed to the web popover the journey suite captures.
3. **Push notifications** — registration + permission prompt (wired in #2072, never tested on device).

Already covered elsewhere: network (offline/degraded work), preferences (token/snapshot), filesystem
+ download → offline playback (`DownloadThroughUITests` / `OfflineAutoAdvanceTests`).
Still uncovered after the first three: deep links via `App.getLaunchUrl` (cold-start shared link),
ASWebAuthenticationSession (tests currently seed a token and bypass it), Android background audio,
splash, native crash reporting.

## Key files

- `web/learning-player/src/services/api.ts` — read/write gate.
- `web/learning-player/src/composables/useOnline.ts` — connectivity signal (`isOffline` vs
  `isForcedOffline`).
- `web/learning-player/src/stores/auth.ts` — snapshot / refresh / ensureLoaded.
- `web/learning-player/src/services/native.ts` — token rehydration + native OAuth.
- `web/learning-player/src/main.ts` — boot ordering.
- `web/learning-player/src/router/index.ts` — login-first guard + signed-in definition.
- `web/learning-player/src/App.vue:547-590` — masthead gates (the profile entry point).
</content>
</invoke>
