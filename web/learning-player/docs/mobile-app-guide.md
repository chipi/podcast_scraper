# Mobile app guide — learning-player → native iOS / Android / TV

How to take this Vue SPA to a **good** native app, what to think about, and how to
keep it fast. Distilled from the Orrery mobile build (a sibling Vite/PWA app that
shipped iOS + Android + Google-TV via Capacitor) and grounded in what
`learning-player` actually is today.

The companion **[capacitor-build-runbook.md](./capacitor-build-runbook.md)** is the
step-by-step "type these commands" doc. This one is the *why* and the *design bar*.

---

## 0. The model — the web app IS the product; native is a thin shell

Capacitor wraps your existing static `dist/` in a native `WKWebView` (iOS) /
`WebView` (Android). You are **not** rewriting anything. 95% of the app is the Vue
SPA you already have; the native layer adds the handful of things a browser can't
do. Keep that split clean:

- **Web owns:** UI, routing, state, transcript sync, rendering.
- **Native shell owns:** background audio, lock-screen/headphone controls, deep
  links, share sheet, file save, splash, safe-area insets, OAuth via in-app
  browser, store packaging.

If you find yourself forking large chunks of app logic per-platform, stop — that's
the wrong layer. Branch on `Capacitor.isNativePlatform()` for the *shell* concerns
only.

---

## 1. For an audio player, the native value IS background audio + controls

This is the crux and it's where `learning-player` has the biggest gap today. A PWA
can't do these well; a native shell can, and they are the reason to ship native at
all:

| Native must-have | Today in the player | Why it matters |
| --- | --- | --- |
| **Background playback** | ❌ none | Audio stops the instant the user leaves the app. Unusable as a podcast app. |
| **Lock-screen / notification controls** (`MediaSession`) | ❌ absent (grep: zero `navigator.mediaSession`) | No artwork, no play/pause/seek from lock screen or headphones. |
| **Headphone / BT transport** | ❌ | AirPods double-tap, car controls do nothing. |
| **Interruption handling** (calls, other audio) | ❌ | A phone call pauses iOS audio but the UI still shows "Playing" — state desyncs. |
| **Offline downloads** | ❌ (SW caches shell+API, never audio) | No commute-without-signal listening. |

**Design bar:** before you call the native app "done," a user must be able to lock
the phone, control playback from the lock screen + headphones, take a call and have
it resume, and (eventually) download an episode for the subway. The runbook has the
exact wiring; the point here is *these are P0, not polish*.

Two things make this cheap in the player:

- The audio engine is a **single `<audio>` element** with clean `toggle()`,
  `skip()`, `seek()`, `cycleRate()` functions (`PlayerView.vue`). `MediaSession`
  action handlers wire straight onto those — an afternoon of work.
- Playback state currently lives as **local refs in `PlayerView.vue`**. Before you
  add `MediaSession` + background audio, **lift it into a Pinia `player` store.**
  The lock screen, a notification, and the UI must all reflect *one* source of
  truth; component-local refs can't back a native control that outlives the view.

---

## 2. The WebView is not the browser — know the differences

Your app runs in `WKWebView` (iOS) / Android `WebView`, not Safari/Chrome. The
gotchas that will bite:

- **iOS `WKWebView` has NO service worker.** Your `vite-plugin-pwa` offline shell +
  runtime API caching **silently no-op on iOS.** Anything you want offline on iOS
  must be native (Capacitor Preferences / Filesystem), not the SW. (Android WebView
  SW support is partial/unreliable — treat both as "no SW.") So: the PWA is great
  for the *web* deploy, but **plan offline as a native feature**, not a SW feature.
- **Origin is `capacitor://localhost` (iOS) / `http://localhost` (Android)**, not
  your server. The player builds API URLs as `new URL('/api/app'+path,
  window.location.origin)` (`services/api.ts:46`) → in the shell that resolves to
  `capacitor://localhost/api/app/...` and **every request fails.** You must
  introduce an absolute API base (`VITE_API_BASE_URL`). This is the #1 thing that
  breaks; fix it first (runbook §3).
- **Full-page redirects break OAuth.** `window.location.assign(loginUrl)`
  (`auth.ts:32`) → Google OAuth inside the WebView is blocked/broken. Use
  `@capacitor/browser` (SFSafariViewController / Chrome Custom Tabs) + a deep-link
  callback.
- **`<a download>` doesn't save files** in `WKWebView` — it navigates. The
  highlights export (`HighlightsView.vue:108`) needs `@capacitor/filesystem` +
  `@capacitor/share`.
- **History routing needs the local server.** `createWebHistory` works because
  Capacitor serves the bundle from `capacitor://localhost/` with a fallback; keep
  it, but know that a naked `file://` load would 404 deep routes.

---

## 3. Mobile-first UX — the design details that separate good from janky

- **`dvh`, never `vh`.** `min-h-screen` (`App.vue:35`) and `max-h-[85vh]` sheets
  (`EntityCard.vue:65`, `InterestsPicker.vue:119`) hit the iOS `100vh` bug (chrome
  over-reports height → clipped footers/sheets). Switch to `min-h-dvh` / `max-h-[85dvh]`.
- **Safe-area insets.** Notch / Dynamic Island / home indicator / rounded corners.
  Pad with `env(safe-area-inset-*)`. In Capacitor iOS `env()` sometimes returns 0 —
  Orrery shipped a tiny native `SafeAreaViewController` shim that injects the real
  insets as CSS vars; the simpler path is `@capacitor-community/safe-area`. Nav and
  the player bar are the two things that *will* render under the Island if you skip
  this.
- **No hover-only affordances.** The player already handles this (the episode
  summary panel uses `[@media(hover:none)]` to stay visible on touch — good). Audit
  the rest: anything that only appears on `:hover` is invisible on a phone.
- **Touch targets ≥ 44pt**, thumb-reachable transport controls (bottom, not top).
- **Splash → shell, no white flash.** Set the splash + a dark background color in
  the Capacitor config so the first frame isn't a white flash before Vue mounts.

---

## 4. Fast — the app is already small; keep it that way

The player's `dist/` is ~560 KB (biggest chunks: `api` 91 KB, Vue runtime 83 KB,
shell 55 KB, `PlayerView` 36 KB, Tailwind CSS 31 KB). That's healthy — Orrery's
naive build was ~2 GB and needed a whole stream-heavy pruning pipeline; you don't.
Keep the discipline:

- **Bundle the shell on-device; stream media from origin.** Audio already streams
  from a bridged origin URL (never rehosted) — correct. Never bundle audio.
  `preload="metadata"` (not `auto`) is already set — correct; don't preload full
  files on cellular.
- **Lazy-load routes.** Route-level `import()` so the player view + transcript code
  isn't in the first paint. (Check the router — if views are statically imported,
  split them.)
- **Content-address artwork + cache it.** The SW already CacheFirsts `/api/app/artwork`
  for 30 days — but remember that's web-only. For native, cache artwork via the
  native layer or accept re-fetch; either way keep artwork URLs content-addressed so
  they're safe to cache forever.
- **Transcript sync is already efficient** — event-driven `timeupdate` (~4 Hz) + an
  O(log n) binary search for the active segment (`transcriptSync.ts:14`). No rAF
  loop, no heavy work per tick. Don't regress this: keep per-`timeupdate` work to
  setting one ref + pure computeds. If you add a karaoke-style word highlighter,
  drive it off the same event, not a 60 Hz loop.
- **First interaction fast.** Precache the shell; defer non-critical stores
  (favorites, interests) until after first paint.

---

## 5. Telemetry — port Orrery's env-ladder verbatim (you're 90% there)

The player already uses `@sentry/vue` + Umami with a **tailnet-only dev DSN
fallback** (`main.ts` — `homelab:8090/8` for Sentry, `homelab:3001` for Umami) —
the *exact* pattern Orrery uses. So the two lessons transfer directly:

- **Env ladder (Orrery ADR-082):** one isolated GlitchTip project + Umami site per
  tier (dev / staging / prod), selected by baked `VITE_*` env vars, dev via the
  tailnet host. You already have `component: 'player'` tagging — add a `platform`
  tag (`web`/`ios`/`android`) so a shell-only regression is separable from web, and
  make `environment` the **tier** (dev/staging/prod), uniform across web + native.
  **Don't** overload `environment` with the platform.
- **Runtime target switcher for internal builds (Orrery ADR-083):** internal /
  TestFlight / simulator builds ship a staging↔prod toggle that repoints assets +
  telemetry together; **release is prod-locked** (toggle tree-shaken out via a
  `__MOBILE_INTERNAL__` build flag). This is how you test staging safely on a real
  device without a rebuild per flip. Two gotchas Orrery learned the hard way:
  (1) native `@sentry/capacitor` sends via the *native* transport — invisible to the
  WebView network inspector, so verify tier via the injected Umami `data-website-id`
  - asset origin, not by watching Sentry traffic; (2) a prerendered/baked asset
  origin won't follow a runtime switch unless you re-resolve it on mount — the player
  is a pure SPA (no prerender), so this bites you *less*, but any build-time-baked
  origin constant has the same trap.

---

## 6. TV is a real fork — be honest about it

"TV" is not one thing:

- **Android TV / Google TV** — this is the **same Android app** with a
  `leanback` manifest entry + **D-pad focus navigation** (every interactive element
  must be focusable and reachable with up/down/left/right; visible focus ring). The
  current layout tops out at `max-w-6xl` two-column and has **no D-pad support** —
  that's the work. A podcast player on TV is legitimate (10-foot "now playing" +
  queue). Orrery shipped a dedicated TV layer (10-foot UI, roving focus).
- **Apple TV / tvOS** — **Capacitor does not target tvOS.** There's no
  `WKWebView`-app story there like iOS. If you want Apple TV, it's a separate native
  effort (or skip it). Don't promise tvOS off the Capacitor build.

So plan: iOS (iPhone/iPad) + Android (phone/tablet) + **Android TV** from the
Capacitor build; tvOS is out of scope unless separately funded.

---

## 7. How to actually iterate + verify (don't guess)

- **Browser-first for anything visual.** Run `vite dev`, resize to a phone
  viewport, screenshot, read it. Seconds per turn. Docker/e2e/device builds are the
  *final* gate, never the iteration loop. (Orrery burned ~6h once by rebuilding a
  container per CSS tweak — don't.)
- **Debug the real WebView on device/simulator** when you need to (native audio,
  MediaSession, safe-area): you can attach a full JS console to the running app.
  - **iOS:** `ios-webkit-debug-proxy` (`brew install`) → it exposes the WKWebView's
    WebKit inspector; drive `Runtime.evaluate` / read the DOM / capture network over
    a WebSocket. (This is how the switcher was verified on Orrery — read the injected
    Umami id + asset origin live, and even clicked the in-app toggle from the
    inspector.)
  - **Android:** `adb forward` the `webview_devtools_remote_<pid>` socket →
    standard Chrome DevTools Protocol → same JS eval / network capture.
- **localStorage flush nuance** (found on Orrery): iOS `simctl terminate` flushes
  WebView storage cleanly; Android `am force-stop` kills before flush and **loses**
  the last write — background the app first (`HOME` → `onPause` → flush) then
  relaunch. Matters when you test "set a pref, relaunch, confirm it persisted."

---

## 8. The gotcha checklist (learned on Orrery, apply here)

- **Lockfile / Linux optionals:** an incremental `npm i <pkg>` on macOS can strip
  `@rollup/rollup-linux-*` / `@esbuild/*` optionals from the lockfile → CI `npm ci`
  fails on Linux. After adding a dep, check the lock diff for stripped `linux`
  optionals; regenerate on Linux if needed.
- **iOS `scrollEnabled`** config isn't reliably applied — if scrolling dies on
  content pages, force `webView.scrollView.isScrollEnabled = true` in the native
  shim (the same shim that injects safe-area).
- **Splash flashes by unseen** without `@capacitor/splash-screen` + a
  `launchShowDuration`.
- **`capacitor-assets generate`** errors on the PWA step — scope it with `--ios` /
  `--android`; it still generates icons/splash. It also drops a stray `icons/` at
  repo root — delete it.
- **CI can't build native** (no Xcode/Android SDK on the runners) — native builds
  are **local**. Version lives in `package.json` + iOS `MARKETING_VERSION` + Android
  `versionName`; keep them in sync.

---

## TL;DR priority order

1. **Absolute API base** (`VITE_API_BASE_URL`) — nothing works in the shell without it.
2. **Lift playback into a Pinia `player` store** — prerequisite for controls.
3. **`MediaSession`** metadata + action handlers + `positionState`.
4. **Background audio** (Info.plist `UIBackgroundModes: audio` + audio-session plugin + `pause`-event state sync).
5. **OAuth via `@capacitor/browser` + deep link**; `<a download>` → filesystem/share.
6. **`dvh` + safe-area** pass.
7. **Telemetry env-ladder + platform tag** (port ADR-082/083).
8. Then: offline downloads (native), Android-TV D-pad layer.

*learning-player · docs/mobile-app-guide.md · adapted from the Orrery mobile build*
