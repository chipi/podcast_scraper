# Reference: how the Orrery Capacitor project is wired + the steps that worked

The sibling **Orrery** app (SvelteKit → Capacitor iOS + Android) is a known-good
Capacitor setup that was built + run on a simulator/emulator this week. If you got
stuck "opening Xcode / bringing the Cap project into Xcode," this is what the
working project looks like and the exact sequence that worked — copy the shape.

> **The single most common Xcode stumble → read this first.** `npx cap add ios`
> generates **two** files side by side: `App.xcodeproj` **and** `App.xcworkspace`.
> You must **open `App.xcworkspace`** (the CocoaPods workspace). Opening
> `App.xcodeproj` builds without the Pods → "module 'Capacitor' not found" / linker
> errors. Same for the headless build: pass `-workspace App.xcworkspace`, **not**
> `-project App.xcodeproj`.

---

## What the Orrery project looks like (mirror this shape)

```text
orrery/
  capacitor.config.ts          # appId, appName, webDir: 'build'  (yours → 'dist')
  ios/
    App/
      App.xcworkspace          # ← OPEN THIS (has App + all Pod schemes)
      App.xcodeproj            #   generated, do NOT open directly
      Podfile / Podfile.lock   # cap-managed pod list (Capacitor + each plugin)
      Pods/                    # installed by `pod install` (run by `cap sync`)
      App/                     # the native shell:
        AppDelegate.swift
        Info.plist
        Assets.xcassets        # icons/splash
        capacitor.config.json  # copied from capacitor.config.ts at sync
        config.xml
        public/                # ← your web `dist/` gets copied here at sync
        Plugins/
      App-Bridging-Header.h
  android/
    settings.gradle / build.gradle / gradlew
    capacitor.settings.gradle  # includes each plugin module (cap-managed)
    app/
      build.gradle
      capacitor.build.gradle   # cap-managed plugin deps
      src/main/                # AndroidManifest.xml, MainActivity, assets/public/ (your dist/)
```

Key facts:

- **Scheme is `App`.** (The workspace also lists `Capacitor`, `CapacitorApp`,
  `SentryCapacitor`, … — those are the Pods; you build/run **`App`**.)
- **`webDir`** in `capacitor.config.ts` points at the built SPA. Orrery = `build`
  (SvelteKit). **learning-player = `dist`** (Vite). Set it once at `cap init`.
- **`cap sync` does the heavy lifting:** copies the web build into
  `ios/App/App/public` + `android/app/src/main/assets/public`, **runs `pod install`**
  (iOS) and the gradle plugin sync (Android). You rarely touch `pod install`
  directly — run `npx cap sync`.
- **Plugins are wired by installing the npm package + `cap sync`.** The Orrery
  Podfile lists `Capacitor`, `CapacitorApp`, `CapacitorBrowser`, `CapacitorHaptics`,
  `CapacitorShare`, `CapacitorSplashScreen`, `CapacitorCommunitySafeArea`,
  `SentryCapacitor` — each is just `npm i @scope/pkg` then `cap sync`. Nothing is
  hand-edited into the Podfile.

---

## The exact sequence that worked (verified this week)

### 0. Prereqs

Xcode (`xcodebuild -version`), CocoaPods (`pod --version`), Node 20+. Simulator
builds need **no** Apple ID / no signing.

### 1. Build the web bundle, then sync

```sh
npm run build            # → dist/ (yours) | build/ (Orrery)
npx cap sync             # copies web into native + pod install + gradle sync
```

Do this **every time** the web build or plugin set changes. If the app shows old
content, you forgot `cap sync`.

### 2a. Open in Xcode (GUI)

```sh
npx cap open ios         # opens App.xcworkspace in Xcode (the RIGHT file)
```

In Xcode: select the **`App`** scheme + a simulator (top bar) → **Run**. For a real
device or Archive you must set a **Team** under the App target → Signing &
Capabilities (needs your Apple ID). For a **simulator** you don't.

### 2b. Or build + run headless (no Xcode GUI — what the automation used)

```sh
# build for the simulator, no signing
xcodebuild -workspace ios/App/App.xcworkspace -scheme App -configuration Debug \
  -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17' \
  -derivedDataPath /tmp/app-ios CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO build
# → look for "** BUILD SUCCEEDED **" and /tmp/app-ios/Build/Products/Debug-iphonesimulator/App.app

xcrun simctl bootstatus "iPhone 17" -b            # boot (blocks until ready)
xcrun simctl install booted /tmp/app-ios/Build/Products/Debug-iphonesimulator/App.app
xcrun simctl launch booted io.github.chipi.orrery # ← your appId
xcrun simctl io booted screenshot /tmp/app.png    # eyeball it
```

`-workspace … -scheme App` + `CODE_SIGNING_ALLOWED=NO` is the whole trick — no
Team, no provisioning profile needed for the simulator.

### 3. Android (for completeness)

```sh
export JAVA_HOME="$(/usr/libexec/java_home -v 21)"      # Capacitor 8 needs JDK 21
(cd android && ./gradlew assembleDebug)                 # → app/build/outputs/apk/debug/app-debug.apk
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n io.github.chipi.orrery/.MainActivity   # explicit start > monkey
```

(Android needs a valid emulator AVD + a *complete* system image — if the emulator
errors "No initial system image", the `.img` files are missing; reinstall the image
with `sdkmanager --install "system-images;android-36;google_apis;arm64-v8a"`.)

---

## The failure modes that "get you stuck opening Xcode" (and the fix)

| Symptom | Cause | Fix |
| --- | --- | --- |
| "module 'Capacitor' not found" / linker errors in Xcode | Opened **`App.xcodeproj`** | Open **`App.xcworkspace`** (or `npx cap open ios`) |
| Blank / old web content in the app | Forgot `cap sync` after building | `npm run build && npx cap sync` |
| `Podfile`/Pods errors, "sandbox not in sync" | `pod install` didn't run / stale | `npx cap sync` (runs it); or `cd ios/App && pod install` |
| Xcode wants a signing Team you don't have | Building for a **device** or Archive | For simulator: `CODE_SIGNING_ALLOWED=NO`; for device: set a Team |
| Nothing to run / no scheme | Wrong scheme selected | Select **`App`** |
| App can't reach its API on device | Web uses origin-relative API base | (learning-player) make the API base **absolute** — see the runbook §3.1 |

---

## Map to learning-player (the deltas from Orrery)

- **`webDir: 'dist'`** (Vite), not `'build'`.
- **`appId`** = your `com.<org>.learningplayer`, used in every `simctl launch` /
  `am start`.
- **No stream-heavy pruning.** Orrery has a `build:mobile` that prunes a ~2 GB
  build to ~65 MB; your `dist` is ~560 KB, so plain `npm run build` is your
  "build:mobile". Just `npm run build && npx cap sync`.
- **Same debugging trick works** (attach a live JS console to the running WebView):
  `ios-webkit-debug-proxy` for iOS, `adb forward` + Chrome DevTools Protocol for
  Android — see the runbook §7.

*learning-player · docs/orrery-capacitor-reference.md · from the working Orrery iOS+Android build*
