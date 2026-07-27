# Capacitor build runbook — learning-player → iOS / Android / (Android) TV

Step-by-step to wrap the Vue SPA in a native shell and run it. Commands are the
ones actually used to build + verify the sibling Orrery app on device this week.
Read **[mobile-app-guide.md](./mobile-app-guide.md)** first for the *why*; this is
the *how*.

> **Scope:** iOS (iPhone/iPad) + Android (phone/tablet) + **Android TV**. **tvOS
> (Apple TV) is NOT supported by Capacitor** — out of scope here.

---

## 0. Prerequisites (one-time, local machine)

- **Node 20+**, the repo installed (`npm ci` in `web/learning-player`).
- **iOS:** macOS + **Xcode** (`xcodebuild -version`) + **CocoaPods** (`pod --version`,
  `brew install cocoapods` if missing). Simulator builds need **no** Apple ID / no
  code signing.
- **Android:** **JDK 21** (`java -version` → 21; Capacitor 8 requires it),
  **Android Studio** + SDK, `adb` on PATH (`$ANDROID_HOME/platform-tools`), and a
  working **system image** (`sdkmanager --list_installed | grep system-images`; if
  the `.img` files are missing the emulator errors *"No initial system image"* —
  `sdkmanager --install "system-images;android-36;google_apis;arm64-v8a"`).
- CI does **not** build native binaries — this is a **local** workflow.

---

## 1. Install + init Capacitor

```sh
cd web/learning-player
npm i @capacitor/core @capacitor/cli
npx cap init "Learning Player" com.<org>.learningplayer --web-dir dist
```

This writes `capacitor.config.ts`. Start it like this (mirrors Orrery's proven config):

```ts
import type { CapacitorConfig } from '@capacitor/cli';
// Local-dev live-reload origin: set CAP_DEV_SERVER to your vite dev URL to load
// the whole app from it (device debugging). UNSET in every release build.
const devServer = process.env.CAP_DEV_SERVER;
const config: CapacitorConfig = {
  appId: 'com.<org>.learningplayer',
  appName: 'Learning Player',
  webDir: 'dist',
  server: {
    androidScheme: 'https', // stable WebView origin for history routing
    ...(devServer && process.env.NODE_ENV !== 'production'
      ? { url: devServer, cleartext: true }
      : {}),
  },
  ios: {
    contentInset: 'always',
    backgroundColor: '#0b0b0f',      // your dark canvas — no white flash
    scrollEnabled: true,             // see gotcha §11
    limitsNavigationsToAppBoundDomains: true, // App Store req; external links via @capacitor/browser
    allowsLinkPreview: false,
  },
  android: { backgroundColor: '#0b0b0f', captureInput: true },
  plugins: {
    SplashScreen: { launchShowDuration: 1200, launchAutoHide: true, backgroundColor: '#0b0b0f' },
  },
};
export default config;
```

## 2. Add the platforms

```sh
npm i @capacitor/ios @capacitor/android
npx cap add ios
npx cap add android
```

Creates `ios/` + `android/` native projects (commit them). Add the plugins you'll
need now so the first sync links them:

```sh
npm i @capacitor/splash-screen @capacitor/browser @capacitor/app \
      @capacitor/share @capacitor/filesystem @capacitor/preferences \
      @capacitor-community/safe-area
```

---

## 3. Web-app changes REQUIRED before the shell works

Do these **before** the first device run — without §3.1 the app can't reach its API
at all. Each is small; all are gated to native where relevant.

### 3.1 Absolute API base (P0 — nothing works without it)

`services/api.ts:46` builds URLs against `window.location.origin`, which in the
shell is `capacitor://localhost` / `http://localhost`. Introduce an absolute base:

```ts
// services/api.ts
const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api/app'; // absolute in native builds
function url(path: string) {
  return API_BASE.startsWith('http')
    ? new URL(path.replace(/^\/api\/app/, ''), API_BASE).toString()
    : new URL(API_BASE + path, window.location.origin).toString();
}
```

Set `VITE_API_BASE_URL=https://api.<your-domain>` for native builds. Ensure the API
sends CORS for `capacitor://localhost` + `http://localhost` (or configure Capacitor
`server.hostname` / a proxy). **Verify first with a single `GET /me` on device.**

### 3.2 Lift playback into a Pinia `player` store

Move `audioEl`/`playing`/`currentTime`/`duration`/`rate` out of `PlayerView.vue`
local refs into `stores/player.ts`. The lock screen + notification + UI must share
one source of truth. Keep the `<audio>` element owned by the store (or a singleton
service) so it outlives the view.

### 3.3 MediaSession (lock-screen / headphone controls)

```ts
// call when an episode loads / metadata changes
navigator.mediaSession.metadata = new MediaMetadata({
  title: episode.title, artist: podcast.title, album: '', artwork: [{ src: artworkUrl, sizes: '512x512', type: 'image/png' }],
});
navigator.mediaSession.setActionHandler('play', () => player.toggle());
navigator.mediaSession.setActionHandler('pause', () => player.toggle());
navigator.mediaSession.setActionHandler('seekbackward', () => player.skip(-15));
navigator.mediaSession.setActionHandler('seekforward', () => player.skip(30));
navigator.mediaSession.setActionHandler('previoustrack', () => queue.prev());
navigator.mediaSession.setActionHandler('nexttrack', () => queue.next());
// on timeupdate:
navigator.mediaSession.setPositionState({ duration, playbackRate: rate, position: currentTime });
```

Wire onto the existing `toggle()`/`skip()`/`seek()`/`cycleRate()`.

### 3.4 Background audio

- **iOS:** in Xcode → the App target → Signing & Capabilities → **Background Modes →
  Audio**, which adds `UIBackgroundModes: [audio]` to `Info.plist`. Set the audio
  session category to `playback` (a plugin like `capacitor-native-audio` or a small
  native shim, or `@capacitor-community/background-mode`). Add a `pause` **event
  listener on the `<audio>` element** to sync the store when iOS interrupts (call
  → auto-pause) so the UI doesn't lie.
- **Android:** foreground service for playback (a media plugin) so the OS doesn't
  kill audio when backgrounded; the `MediaSession` above drives the notification.

### 3.5 OAuth + downloads + viewport

- **OAuth:** replace `window.location.assign(loginUrl)` (`auth.ts:32`) with
  `import { Browser } from '@capacitor/browser'; await Browser.open({ url: loginUrl })`,
  and register the callback as a **deep link** (§8) so control returns to the app.
- **Highlights export** (`HighlightsView.vue:108`): `<a download>` doesn't save in
  WKWebView — write with `@capacitor/filesystem` then `@capacitor/share`.
- **Viewport:** `min-h-screen` → `min-h-dvh` (`App.vue:35`); `max-h-[85vh]` sheets →
  `max-h-[85dvh]`. Add safe-area padding (`@capacitor-community/safe-area` sets the
  CSS vars; then `padding-top: env(safe-area-inset-top)` on nav + player bar).

---

## 4. Build the web bundle + sync into native

```sh
npm run build            # vue-tsc + vite build → dist/
npx cap sync             # copies dist/ into ios/ + android/, runs pod install + gradle sync
```

Re-run `npx cap sync` after **every** web build or plugin add. (The player's `dist`
is ~560 KB — no stream-heavy pruning needed, unlike Orrery.)

---

## 5. Run on iOS (simulator — no signing)

**GUI path:** `npx cap open ios` → Xcode → pick a simulator → Run.

**Headless path** (what CI-less automation uses):

```sh
# build for the simulator (no code signing needed)
xcodebuild -workspace ios/App/App.xcworkspace -scheme App -configuration Debug \
  -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17' \
  -derivedDataPath /tmp/lp-ios CODE_SIGNING_ALLOWED=NO build
# boot + install + launch
xcrun simctl boot "iPhone 17"        # or: xcrun simctl bootstatus <UDID> -b
xcrun simctl install booted /tmp/lp-ios/Build/Products/Debug-iphonesimulator/App.app
xcrun simctl launch booted com.<org>.learningplayer
xcrun simctl io booted screenshot /tmp/lp.png    # eyeball it
```

## 6. Run on Android (emulator)

```sh
export JAVA_HOME="$(/usr/libexec/java_home -v 21)"
# build the debug APK
(cd android && ./gradlew assembleDebug)
# boot an emulator (needs a valid AVD; create one on an installed image if none)
$ANDROID_HOME/emulator/emulator -avd <avd> -no-snapshot -no-audio &
adb wait-for-device
until [ "$(adb shell getprop sys.boot_completed | tr -d '\r')" = 1 ]; do sleep 2; done
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.<org>.learningplayer/.MainActivity   # explicit start is more reliable than monkey
adb exec-out screencap -p > /tmp/lp-android.png
```

---

## 7. Debug the real WebView on device (attach a JS console)

Essential for native audio / MediaSession / safe-area work — you get live
`Runtime.evaluate`, DOM reads, and network capture inside the running app.

**iOS (WebKit):**

```sh
brew install ios-webkit-debug-proxy
SOCK=$(lsof -U | grep -i webinspectord_sim | awk '{print $NF}' | head -1)
ios_webkit_debug_proxy -F -s "unix:$SOCK"          # listens on :9221 (list), :9222 (pages)
curl -s localhost:9222/json                         # → webSocketDebuggerUrl for capacitor://localhost
```

The WebKit protocol multiplexes through a **`Target`** domain (wrap messages in
`Target.sendMessageToTarget`, pick the `type:"page"` target) — plain CDP
`Runtime.evaluate` returns *"'Runtime' domain not found"* until you do.

**Android (Chrome DevTools Protocol — standard):**

```sh
PID=$(adb shell pidof com.<org>.learningplayer | tr -d '\r')
SOCK=$(adb shell cat /proc/net/unix | grep -a webview_devtools_remote | awk '{print $NF}' | head -1)
adb forward tcp:9223 "localabstract:${SOCK#@}"
curl -s localhost:9223/json                         # → ws url; drive with plain CDP (Runtime.evaluate directly)
```

Requires the WebView to be inspectable — Capacitor **debug** builds enable it. Node
22+ has a built-in `WebSocket` client, so a ~30-line script drives either endpoint.

---

## 8. Deep links + share

- **Deep links:** register a scheme (`learningplayer://…`) — iOS `Info.plist`
  `CFBundleURLTypes`, Android `AndroidManifest.xml` `<intent-filter>` — and handle
  in JS via `@capacitor/app`'s `appUrlOpen` → route with vue-router. Needed for the
  OAuth callback (§3.5) and for share-to-episode.
- **Share:** the OS share sheet via `@capacitor/share` (share a public
  `https://<web-domain>/episode/<slug>` URL, not the internal `capacitor://` one).

---

## 9. Android TV

Same Android app, plus:

- `AndroidManifest.xml`: `<uses-feature android:name="android.software.leanback"
  android:required="false"/>` + a `LEANBACK_LAUNCHER` category on the launcher
  activity so it appears on the TV home.
- **D-pad focus:** every control must be focusable + reachable with up/down/left/
  right and show a visible focus ring. This is app-code work (roving focus / a TV
  layout), not a Capacitor toggle — plan it as a real layer (a 10-foot "now
  playing" + queue is the sensible TV surface).
- Test on the `Orrery_TV`-style AVD (a `google-tv` / Android-TV system image) or a
  real Chromecast-with-Google-TV.

---

## 10. Ship

- **iOS TestFlight:** `npx cap open ios` → set your Apple Developer **Team**
  (Signing & Capabilities, needs your Apple ID) → Product → Archive → Distribute →
  App Store Connect. Create the app record (bundle id) → it appears in TestFlight.
- **Android internal testing:** `npx cap open android` → Build → Generate Signed
  App Bundle → upload the `.aab` to Play Console → Internal Testing track.
- **Versioning:** `package.json` version + iOS `MARKETING_VERSION` + Android
  `versionName` — keep in sync.
- **Telemetry per build:** don't hand-export the build-args — copy
  `.env.mobile.example` → `.env.mobile` (gitignored), fill it, and use
  `make mobile-build-internal` / `make mobile-build-release`. Both source
  `.env.mobile` so real internal URLs never touch git; `mobile-build-release`
  additionally **fails if `VITE_SENTRY_DSN_PLAYER` is empty** (no silently-o11y-less
  release). Release = prod-locked (`MOBILE_RELEASE=1`); an internal build carries the
  dev↔prod runtime switcher (guide §5).
- **Dev-tier o11y on a real device needs https.** The committed dev defaults
  (`homelab:8090` GlitchTip, `homelab:3001` Umami) are plain http. That works in
  desktop `vite dev` (http origin) but a native WebView is an https origin, so the
  http calls are blocked (iOS App Transport Security + Android mixed-content). Expose
  the tailnet services over https with `tailscale serve`, then inject the https URLs
  via `VITE_SENTRY_DSN_PLAYER_DEV` / `VITE_UMAMI_SRC_DEV` in `.env.mobile`. Because the
  ACL caps homelab ports and `:443` already serves Umami (`/umami`), mount GlitchTip on
  a **path** on 443 (GlitchTip must know its base path), e.g. on the homelab host:

  ```sh
  tailscale serve --bg --https=443 --set-path=/glitchtip http://127.0.0.1:8090
  tailscale serve status   # confirm the mapping + that the cert issued
  ```

  then in `.env.mobile`:

  ```sh
  VITE_SENTRY_DSN_PLAYER_DEV=https://<key>@homelab.<tailnet>.ts.net/glitchtip/8
  VITE_UMAMI_SRC_DEV=https://homelab.<tailnet>.ts.net/umami/script.js
  ```

  This is the cause-fix (ADR-126 already does exactly this for prod Umami); no ATS
  exception or Android cleartext hack is shipped. Verify the ingest actually lands
  in GlitchTip after the first on-device dev-tier run — subpath routing depends on
  your `tailscale serve` path-strip + GlitchTip base-URL config.

---

## 11. Gotchas (from the Orrery build)

- **`am force-stop` loses the last `localStorage` write** on Android (kills before
  flush) — background first (`adb shell input keyevent KEYCODE_HOME`) then relaunch
  when testing persistence. iOS `simctl terminate` flushes cleanly.
- **iOS `scrollEnabled` config unreliable** — if content pages won't scroll, force
  `webView.scrollView.isScrollEnabled = true` in a native shim.
- **`env(safe-area-inset-*)` can return 0** in Capacitor iOS — use the safe-area
  plugin (or a native shim that injects the real insets as CSS vars); don't trust
  `env()` alone. Instrument the computed value on-screen when debugging.
- **macOS `npm i` strips Linux-only lock optionals** → CI `npm ci` red on Linux;
  check the lock diff for `@rollup/rollup-linux-*` / `@esbuild/*`, regenerate on
  Linux if stripped.
- **No service worker in the shell** (iOS especially) — your `vite-plugin-pwa`
  offline + update toast silently no-op; plan offline as native.
- **`capacitor-assets generate`** — scope with `--ios`/`--android`; delete the stray
  repo-root `icons/` it drops.

*learning-player · docs/capacitor-build-runbook.md · commands proven on the Orrery iOS+Android build*
