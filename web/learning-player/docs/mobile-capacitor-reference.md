# Capacitor iOS + Android: how the project is wired, and the steps that worked

Two parts:

- **Part 1 — iOS (and the project shape).** Written from the sibling **Orrery** app (SvelteKit →
  Capacitor), the known-good setup this player was built from. If you got stuck "opening Xcode /
  bringing the Cap project into Xcode", this is what the working project looks like and the exact
  sequence that worked.
- **Part 2 — Android, from an empty `android/` to testers' phones.** Written back the other way:
  what this repo (Close Listening, `web/learning-player`) learned taking its Android app to Google
  Play internal testing with push, 2026-07 to 2026-10. Generic enough for any Capacitor app; each
  item names the file in this repo that does it, so an agent can read the working version.

> **The single most common Xcode stumble → read this first.** `npx cap add ios`
> generates **two** files side by side: `App.xcodeproj` **and** `App.xcworkspace`.
> You must **open `App.xcworkspace`** (the CocoaPods workspace). Opening
> `App.xcodeproj` builds without the Pods → "module 'Capacitor' not found" / linker
> errors. Same for the headless build: pass `-workspace App.xcworkspace`, **not**
> `-project App.xcodeproj`.

---

## Part 1 — iOS, and the shape of the project

### What the Orrery project looks like (mirror this shape)

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

### The exact sequence that worked (verified this week)

#### 0. Prereqs

Xcode (`xcodebuild -version`), CocoaPods (`pod --version`), Node 20+. Simulator
builds need **no** Apple ID / no signing.

#### 1. Build the web bundle, then sync

```sh
npm run build            # → dist/ (yours) | build/ (Orrery)
npx cap sync             # copies web into native + pod install + gradle sync
```

Do this **every time** the web build or plugin set changes. If the app shows old
content, you forgot `cap sync`.

#### 2a. Open in Xcode (GUI)

```sh
npx cap open ios         # opens App.xcworkspace in Xcode (the RIGHT file)
```

In Xcode: select the **`App`** scheme + a simulator (top bar) → **Run**. For a real
device or Archive you must set a **Team** under the App target → Signing &
Capabilities (needs your Apple ID). For a **simulator** you don't.

#### 2b. Or build + run headless (no Xcode GUI — what the automation used)

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

#### 3. Android

The short version (debug APK on an emulator) is in Part 2, §A–§B; everything after it is Part 2.

---

### The failure modes that "get you stuck opening Xcode" (and the fix)

| Symptom | Cause | Fix |
| --- | --- | --- |
| "module 'Capacitor' not found" / linker errors in Xcode | Opened **`App.xcodeproj`** | Open **`App.xcworkspace`** (or `npx cap open ios`) |
| Blank / old web content in the app | Forgot `cap sync` after building | `npm run build && npx cap sync` |
| `Podfile`/Pods errors, "sandbox not in sync" | `pod install` didn't run / stale | `npx cap sync` (runs it); or `cd ios/App && pod install` |
| Xcode wants a signing Team you don't have | Building for a **device** or Archive | For simulator: `CODE_SIGNING_ALLOWED=NO`; for device: set a Team |
| Nothing to run / no scheme | Wrong scheme selected | Select **`App`** |
| App can't reach its API on device | Web uses origin-relative API base | (learning-player) make the API base **absolute** — see the runbook §3.1 |

---

### Map to learning-player (the deltas from Orrery)

- **`webDir: 'dist'`** (Vite), not `'build'`.
- **`appId`** = your `com.<org>.learningplayer`, used in every `simctl launch` /
  `am start`.
- **No stream-heavy pruning.** Orrery has a `build:mobile` that prunes a ~2 GB
  build to ~65 MB; your `dist` is ~560 KB, so plain `npm run build` is your
  "build:mobile". Just `npm run build && npx cap sync`.
- **Same debugging trick works** (attach a live JS console to the running WebView):
  `ios-webkit-debug-proxy` for iOS, `adb forward` + Chrome DevTools Protocol for
  Android — see the runbook §7.

---

## Part 2 — Android, from an empty `android/` to testers' phones

What it took to get this player from a Capacitor scaffold to a signed build on Google Play's
**internal testing** track, installed by testers, with push delivered to a real handset. Every
item below cost at least one debugging cycle. Paths are in this repo (`web/learning-player/…`):
read the working version before writing your own.

**Not covered, because this project has not done it:** a public Play Store release (store listing,
data safety form, content rating, production track). Everything here stops at internal testing.

### A. Toolchain on the build Mac

| Piece | What works | The trap |
| --- | --- | --- |
| JDK | **21**, exactly | AGP's own floor is 17, but a Capacitor plugin pins toolchain 21. With 17, Gradle resolves every dependency and *then* fails in `:capacitor-filesystem:compileDebugJavaWithJavac` with `Cannot find a Java installation … matching {languageVersion=21}`, which reads like a missing dependency. |
| Finding JDK 21 | `/usr/libexec/java_home -F -v 21` | Without `-F`, a Mac with no JDK 21 registered returns **JDK 1.8**, not an error. |
| Where `JAVA_HOME` comes from | pinned in the Makefile (`ANDROID_JAVA_HOME`) | A build that only works for whoever exported `JAVA_HOME` fails confusingly for everyone else. |
| Installing the JDK | Temurin 21 `.pkg` (arm64), or the tarball unpacked anywhere (Intel) | Homebrew no longer builds Intel bottles (2026-09). |
| SDK | `~/Library/Android/sdk`; `platforms;android-36`, `build-tools;36.0.0` | Match `android/variables.gradle` (here: minSdk 24, compile/target 36). |
| Emulator image | `system-images;android-36;google_apis;arm64-v8a`, AVD `Pixel_8` | "No initial system image" means the `.img` files are missing: reinstall the image with `sdkmanager --install`. |

Makefile targets that wrap all of this: `android-build`, `android-emulator-up`,
`android-emulator-down`, `android-device-install` (repo-root `Makefile`).

### B. Debug build, emulator, device

```sh
npm run build && npx cap sync android
export JAVA_HOME="$(/usr/libexec/java_home -F -v 21)"
(cd android && ./gradlew assembleDebug)       # → android/app/build/outputs/apk/debug/app-debug.apk
adb install -r -d android/app/build/outputs/apk/debug/app-debug.apk   # -d: see §F, versionCode
adb shell am start -n <appId>/.MainActivity    # explicit start is more reliable than monkey
```

- **Check the artifact, not the exit code.** `make android-build` fails unless the APK contains web
  JS chunks (`unzip -l … 'assets/public/assets/*.js'`: proves `cap sync` ran) and an `https://…/api`
  base (proves the app will reach a server). Gradle succeeding proves neither.
- **Emulator boot:** wait for `getprop sys.boot_completed` = `1`, not just `adb wait-for-device`.
  A crashed emulator leaves `hardware-qemu.ini.lock` and `multiinstance.lock` in
  `~/.android/avd/<avd>.avd/`, and the next launch then fails for a *different* reason; delete them
  before launching. `adb emu kill` is asynchronous: wait until the device is gone before reporting
  it stopped.
- Right after boot the emulator can show "System UI isn't responding" over the app; dismiss it
  before trusting a screenshot.
- **Real phone:** USB debugging on, accept the RSA prompt, then `adb install -r`. A debug build and
  the Play build cannot both be installed if they share an applicationId (§F).

### C. Where Android's WebView differs from iOS's — each one needed native or config work

1. **No lock-screen controls from `navigator.mediaSession`.** iOS's WKWebView turns the page's
   MediaSession into Now Playing by itself; Android System WebView does not. The result: no
   lock-screen controls and no output switcher, and a backgrounded app's audio gets suspended. The
   fix is a native media session plus a foreground service of type `mediaPlayback`, fed by the
   player store and relaying button presses back under the web MediaSession's action names:
   - `android/app/src/main/java/…/NowPlaying.java`: the `android.media.session.MediaSession`;
   - `…/PlaybackService.java`: the media notification and the foreground keep-alive;
   - `…/BackgroundAudioPlugin.java`: the Capacitor bridge (`start` / `pause` / `stop` / `update`,
     `action` events);
   - `src/services/native.ts`: mirrors the store into it.

   These use framework classes only, so no new dependency. The manifest needs the `<service
   android:foregroundServiceType="mediaPlayback">` plus `FOREGROUND_SERVICE`,
   `FOREGROUND_SERVICE_MEDIA_PLAYBACK` and `POST_NOTIFICATIONS`.
2. **Register local plugins before the bridge starts:** `registerPlugin(…)` in
   `MainActivity.onCreate`, *before* `super.onCreate` (`MainActivity.java`).
3. **Cross-origin API calls: turn on `CapacitorHttp`.** With `androidScheme: 'https'` the app's
   origin is `https://localhost`, and the API lives elsewhere, so browser `fetch` hits CORS, and an
   `Authorization` header triggers a preflight. `plugins.CapacitorHttp.enabled: true` sends
   fetch/XHR through the native stack, which has neither (`capacitor.config.ts`). **But `<audio
   src>` is still loaded by the WebView**, so an `http:` media URL on the `https://localhost` page
   is blocked as mixed content.
4. **Cleartext is blocked since targetSdk 28.** To test against a local `http://127.0.0.1` server:
   - add a network-security config **in the `src/debug/` source set only**
     (`app/src/debug/AndroidManifest.xml` + `res/xml/network_security_config_debug.xml`), scoped to
     `127.0.0.1`, `10.0.2.2` and `localhost`. A release build cannot include a debug source set
     whatever a flag says, so it cannot ship plaintext;
   - reach the host with `adb reverse tcp:<port> tcp:<port>`, so the Android and iOS builds use the
     same base URL;
   - for `<audio>` from that origin, also set `android.allowMixedContent`, gated to the test build
     (`CAP_ANDROID_TEST_ORIGIN` in `capacitor.config.ts`).
5. **Status-bar icon colour.** For a dark app, set `plugins.SystemBars.style: 'DARK'` (light icons).
   The default follows the phone's theme, so light mode drew dark icons on a dark masthead; a theme
   attribute cannot override it.
6. **The name on the home screen comes from `res/values/strings.xml`** (`app_name`,
   `title_activity_main`), not from `capacitor.config.ts`. `appName` there only seeds `cap init`.
   This app was renamed on the web and iOS and showed the old name on Android for weeks. Have the
   playback notification read `getString(R.string.app_name)` rather than repeat the literal.
7. **Deep links:**
   - **OAuth callback:** a custom-scheme `<intent-filter>` (`<data android:scheme="yourapp"/>`)
     with `launchMode="singleTask"`.
   - **App Links:** an `https` `<intent-filter android:autoVerify="true">` limited to *content*
     paths (`/episode/`, …). Keep the sign-in email's `/api/…/auth/…` link out of it, or the app
     swallows it and email sign-in breaks.
   - **`/.well-known/assetlinks.json` must carry the Play *app-signing* certificate's SHA-256**:
     Play Console → App integrity → App signing key certificate. Not the upload key, and not a
     placeholder. Here it shipped with the placeholder, so every shared link opened in the browser.
8. **Speech recognition** (`@capacitor-community/speech-recognition`):
   - **Manifest:** `RECORD_AUDIO`, plus a `<queries>` entry for `android.speech.RecognitionService`.
     Android 11+ package visibility hides the recogniser without it.
   - **Audio focus:** the recogniser takes it, so the WebView pauses your audio. Resume it yourself
     when dictation ends.
   - **The last word:** the final result arrives *after* the engine reports "stopped". Keep the
     listeners for about 3 s after a stop tap, or the last word is lost.
   - **A dead recogniser does not reject `start()`:** with partial results on, `start()` has already
     resolved. Poll `isListening()` to turn the mic off.
9. **Persistence tests:** `am force-stop` kills the app before the last `localStorage` write is
   flushed. Press HOME first (`adb shell input keyevent KEYCODE_HOME`), then kill.
10. **Google's robots run your build.** Play's pre-launch report runs every new build on Firebase
    Test Lab devices, which showed up as 12 "users" in analytics within minutes. Detect them with
    `Settings.System.getString(resolver, "firebase.test.lab")` = `"true"` (Firebase's documented
    check) and keep analytics off (`AppProcessPlugin.java`, `src/services/analytics.ts`).
11. **Why the app restarted:** Android keeps `ApplicationExitInfo` (API 30+), and Capacitor's
    `WebViewListener` reports a WebView renderer death. `AppProcessPlugin.java` records both and the
    web layer forwards them on the next launch.

### D. Push (FCM)

- **`google-services.json` is gitignored and keyed by package name.** A build for a different
  applicationId needs its own client added in the Firebase console first.
- **A build without it crashes natively, and JS cannot catch it.** Enabling push throws
  `IllegalStateException: Default FirebaseApp is not initialized` on a native handler thread.
  `app/build.gradle` warns when the file is missing and **fails** under `-PandroidPushRequired=true`,
  which the release target passes.
- **Tag tokens with their kind.** The client first registered FCM tokens as `kind: 'apns'`; the
  server routed them to Apple, Apple rejected them, and the server read that as a dead token and
  switched push off. Check what is stored server-side (`fcm://…` vs `apns://…`).
- **Tokens rotate.** Re-register at every launch once signed in, not only from the settings toggle.
  Store the new token **before** removing the one it replaced, so a failed registration keeps the
  old one (`src/composables/usePushSubscription.ts`).

### E. Release signing and the AAB

1. **Create the upload keystore once:**

   ```sh
   keytool -genkeypair -v -keystore upload-keystore.jks -alias upload \
     -keyalg RSA -keysize 2048 -validity 10000
   ```

   **Back it up.** Losing it means losing the ability to update the listing.
2. **Credentials:** `android/keystore.properties` (gitignored; `storeFile` is an **absolute** path),
   or the four `ANDROID_KEYSTORE_FILE` / `ANDROID_KEYSTORE_PASSWORD` / `ANDROID_KEY_ALIAS` /
   `ANDROID_KEY_PASSWORD` env vars, which win.
3. **Never let a release build sign with the debug key, or come out unsigned.** `app/build.gradle`
   creates the release signing config only when all four values are present and the file exists,
   and **fails** `bundleRelease` / `assembleRelease` otherwise. Play rejects such an AAB, but only
   after the whole upload.
4. **Play App Signing:** Play re-signs what you upload with its own key. So you have two
   certificates: the *upload* key (yours) and the *app-signing* key (Play's). App Links (§C.7) and
   any "is this the Play build" check use the app-signing one.
5. **Build a release web bundle for the store, not an internal one.** Here an env prefix bound to
   the wrong command (`MOBILE_RELEASE=1 npm install && npm run build`; the flag reached
   `npm install` only), so every store build carried the dev tier switch and the build host's
   private hostname. The release target now asserts their *absence on the built artifact*; see
   `docs/guides/MOBILE_STORE_RELEASE_RUNBOOK.md` → "What a store build must not contain".
6. **`versionName` comes from `package.json`, and `versionCode` is never typed by hand** (§F).

### F. versionCode — the thing that broke the most

- **Every upload needs a new, higher `versionCode`, and you cannot go back.** Play rejects a
  duplicate or lower code, and only *after* the upload finishes. Once a code is used it is gone:
  one accidental `115305320` would push every future code above 115 million.
- **The git commit count is not monotonic across squash merges.** A branch builds a higher count
  than the single squashed commit that lands on main. A local install then fails with
  `INSTALL_FAILED_VERSION_DOWNGRADE` (install with `adb install -d`), and a Play upload is rejected.
- **The fix: ask Play.** The `next_version_code` lane reads the internal track's codes and prints
  `NEXT_VERSION_CODE=<n>`. `make android-play` passes `<n>` to Gradle as `ANDROID_VERSION_CODE`.
- **Parse the labelled line, not fastlane's last line.** Taking the last stdout line and stripping
  it to digits read fastlane's timestamped summary and produced `115305320`. It was caught before
  upload.
- **The git count is only the offline fallback** for local builds.

### G. Google Play Console and API access

1. **Create the app in Play Console. The first upload must be by hand.** The API refuses a package
   it has never seen (an opaque 404), so upload the first AAB in the Console once, to the
   **Internal testing** track.
2. **A dedicated service account for uploads.** Not the Firebase admin account: that one sends push
   and has broad rights.
   - **Google Cloud console:** enable **Google Play Android Developer API**; create a service
     account (e.g. `play-release`) with **no** project roles; add a JSON key. It downloads once;
     keep it outside the repo, `chmod 600`.
   - **Play Console**, at the developer-account level → **Users and permissions**. There is no
     "Setup → API access" page any more and no Cloud project to link. "Invite new users" is inside
     the **"Manage users ▾"** dropdown above the user table.
   - **Permissions:** add the service account's email with app permissions *Release to testing
     tracks* and *View app information (read-only)* only. No account permissions, never production.
3. **fastlane** (`android/fastlane/Fastfile`, credentials in `android/fastlane/.env` →
   `SUPPLY_JSON_KEY=<path to the JSON>`). Lanes:

   | Lane / make target | What it does |
   | --- | --- |
   | `preflight` / `make android-play-preflight` | credentials + app record, no build; run first on a new machine |
   | `next_version_code` | §F |
   | `internal` / `make android-play` | release web build → signed AAB → upload to `internal` |
   | `status` / `make android-play-status` | read-only: what the track holds, whether Play has processed a build |
   | `universal_apk` / `make android-play-apk` | download Play's signed universal APK (below) |

4. **Testers:**
   - **Adding them:** an email list ticked on the internal track. Each tester opens the opt-in link
     (Play Console → Testing → Internal testing → Testers; the API does not expose it), taps
     **Accept**, then **Download test app**, with the same Google account in the browser and in
     the Play Store.
   - **Updates arrive by themselves** after each upload, minutes to hours later.
   - **"Uploaded" is not "installable":** a build showed as published and was not installable for
     ~30 min.
   - **Never tell a tester to search the Play Store.** Internal-only apps are not listed.
   - **"Item not found" right after accepting** clears with time. One tester also had to update
     their phone's OS first.
   - **Fallback when Play will not serve a tester:** `make android-play-apk` downloads the
     universal APK Play generated from the upload, **signed with Play's app-signing key**, so Play
     updates it in place later. Internal app sharing is not an alternative: the API refuses it for
     a never-published app (`NOT_PUBLISHED`).
   - **Verifying that APK:** use `apksigner verify --print-certs --min-sdk-version 24
     --max-sdk-version 36` (without the max it fails on Play's newer signature scheme) and compare
     the SHA-256 with the app-signing certificate.

### H. One phone, two builds

- **What iOS does:** the debug build has its own bundle id (`….dev`), name ("CL Dev") and muted icon,
  so it sits beside the TestFlight app.
- **What Android does here:** debug and release still share one applicationId. They are signed by
  different keys, so installing one over the other fails with
  `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, and the only way through is uninstalling the Play build,
  which deletes its data.
- **Do it from day one instead:**
  1. Add an Android app for `<appId>.dev` in the Firebase console, then add its client to
     `google-services.json`.
  2. Then set `applicationIdSuffix ".dev"` on the debug build type.
  3. Ship a muted icon in the `debug` source set (`scripts/tools/make_internal_icon.py` makes one).

  The order matters: the Firebase client comes first, because the file is keyed by package name.
  Set the suffix first and every Android build breaks.

### I. UI tests on the device (UiAutomator)

- **UiAutomator, not Espresso.** It reads the accessibility tree, which is what TalkBack uses.
  `Espresso.onWebView()` runs against the DOM and misses what screen readers cannot reach
  (`android/variables.gradle` explains the bug that decided it).
- **WebView content is unusual in that tree:**
  - `By.descContains(...)` **never matches** WebView content; find nodes by text or class;
  - a checked chip reports `checkable=false checked=false`, so read selection from what the app
    shows, not from the node;
  - screens behind a sheet stay in the tree, so a "find any clickable node" search returns the
    page behind it.
- **`UiObject2.click()` taps the bounds it last read.** After a scroll the view is still settling,
  so re-find the element until two reads ~300 ms apart agree, then tap.
- **A test that kills the app** (cold-launch restore) cannot run as instrumentation, which lives in
  the app's own process. Use a shell script instead (`android/scripts/cold-launch-restore.sh`).
- **No speech on the emulator** when it runs with `-no-audio`; cover speech in unit tests.
- **The full device tier** (`make test-android`) took 2 h 38 min for 14 suites here. Run single
  suites while iterating.

### J. Debugging the WebView on a device

Chrome DevTools Protocol through `adb forward` (debug builds only):

```sh
SOCK=$(adb shell cat /proc/net/unix | grep -a webview_devtools_remote | awk '{print $NF}' | head -1)
adb forward tcp:9223 "localabstract:${SOCK#@}"
curl -s localhost:9223/json        # → webSocketDebuggerUrl; plain CDP (Runtime.evaluate) works
```

`adb logcat` shows the native side: the FCM token, the plugin errors, the foreground service.

### K. Checklist for a new Capacitor app going to Android

1. JDK 21 pinned in the build (not the shell), SDK 36, an arm64 AVD (§A).
2. `CapacitorHttp` on; cleartext only in a `src/debug/` source set (§C.3–4).
3. Native media session + `mediaPlayback` foreground service, if the app plays audio (§C.1).
4. `strings.xml` names, a real launcher icon, `SystemBars` style (§C.5–6).
5. Custom-scheme callback + App Links with the Play **app-signing** fingerprint (§C.7).
6. A `….dev` applicationId for debug, with its Firebase client created first (§H).
7. Upload keystore made **and backed up**; Gradle fails a release without signing (§E).
8. versionCode from Play, parsed from a labelled line (§F).
9. First AAB uploaded by hand; a service account with testing-track rights only; `preflight`
   green (§G).
10. `google-services.json` present for release, and push tokens tagged `fcm` (§D).
11. Analytics ignore Firebase Test Lab devices (§C.10).
12. Assertions on the built artifact: web assets present, https API base, no dev-only code (§B,
    §E.5).

Detail: [capacitor-build-runbook.md](./capacitor-build-runbook.md) (both platforms, step by
step), `docs/guides/MOBILE_STORE_RELEASE_RUNBOOK.md` (TestFlight + Play, credentials, moving release
machines), `docs/guides/MOBILE_E2E_TESTING.md` (device tiers).

*learning-player · docs/mobile-capacitor-reference.md · Part 1 from the Orrery build; Part 2 from this repo's Android app on Play internal testing (2026-10-16)*
