# Mobile store release runbook (TestFlight + Play internal testing)

How the Learning Player reaches a tester's phone on either platform, what each build type is
allowed to contain, and the traps that have already cost time. Companion to
[MOBILE_E2E_TESTING.md](MOBILE_E2E_TESTING.md), which covers testing rather than shipping.

Scope: the **test channels** — TestFlight and Play internal testing. Public release on either
store is out of scope. Tracked under #2193.

## The two build types, and why the distinction matters

Everything here follows from one split.

| | Internal / dev | Release / store |
| --- | --- | --- |
| Made by | `make mobile-build-internal`, `fastlane device`, `assembleDebug` | `make mobile-build-release`, `fastlane beta`, `bundleRelease` |
| Goes to | the operator's own phone | TestFlight / Play internal testing |
| Dev↔prod tier switch | present, in Settings › About | **absent** |
| Dev API base (tailnet host) | baked in | **absent** |
| App icon | muted | full colour |
| `__MOBILE_INTERNAL__` | `true` | `false` |

A store build is a **release** build. This has not always been true, and the gap is recorded below
because the symptom was invisible.

## iOS → TestFlight

```bash
make ios-fastlane-install      # once per machine
make ios-testflight-preflight  # creds + app record + signing; builds nothing, fails in seconds
make ios-testflight            # release web build -> cap sync -> archive -> upload
```

`preflight` exists so a credential problem costs seconds instead of a ten-minute archive. Run it
first on any machine that has not shipped before.

**Credentials** — `web/learning-player/ios/fastlane/.env` (gitignored; see `.env.example`):

| Variable | What |
| --- | --- |
| `ASC_KEY_ID` | App Store Connect key id |
| `ASC_ISSUER_ID` | the issuer UUID above the key list — one per team, not per key |
| `ASC_KEY_PATH` | absolute path to `AuthKey_<KEY_ID>.p8` |
| `APPLE_TEAM_ID` | 10-char Developer Team ID |

The `.p8` downloads from Apple **exactly once**. Keep it outside the repo.

**The `.p8` trap.** An App Store Connect key and an **APNs** key are both `.p8` files holding an
EC P-256 private key with a 10-character id. They are not interchangeable, and using the ASC key
where APNs is expected produces `403 InvalidProviderToken` — an error that reads like a signing
bug. See "Push" below.

## Android → Play internal testing

```bash
make android-fastlane-install  # once per machine
make android-play-preflight    # creds + app record; builds nothing
make android-play              # release web build -> signed AAB -> upload to `internal`
```

`make android-bundle` alone produces the signed AAB without uploading.

**The first upload cannot be automated.** Play refuses an API upload for a package it has never
seen, so the first artifact for `app.closelistening.player` goes up by hand in the Play Console,
once. `android-play-preflight` detects this and says so, rather than letting the first upload fail
with an opaque 404.

**Credentials** — `web/learning-player/android/fastlane/.env` (gitignored):

| Variable | What |
| --- | --- |
| `SUPPLY_JSON_KEY` | absolute path to the Play service-account JSON |

Grant that service account **"Release to testing tracks"** and nothing more. It exists to push
internal builds; its blast radius should say so.

**Signing** — `android/keystore.properties` (gitignored), or the four `ANDROID_KEYSTORE_*` /
`ANDROID_KEY_*` environment variables, which take precedence and are what CI would use.

```properties
storeFile=/absolute/path/to/upload-keystore.jks
storePassword=…
keyAlias=upload
keyPassword=…
```

When nothing is configured the release signing config is **not created**, and `android-bundle`
refuses before building. That is deliberate: an AAB silently self-signed with the debug key is
worse than one that fails, because Play rejects it anyway and only after the upload completes.

> **Back the keystore up.** Once an app is published signed with a key, losing that key means
> losing the ability to update that listing, permanently.

**Versioning** is mechanical, never hand-edited: `versionCode` is `ANDROID_VERSION_CODE` when set —
the hook for "ask Play for the last one and add one", the trick the iOS lane plays with
`latest_testflight_build_number` — and otherwise the git commit count, which is monotonic and needs
no network. `versionName` comes from `package.json`. Play rejects a duplicate `versionCode` *after*
the upload finishes, which is why neither is typed by hand.

## The Android toolchain on this build Mac

Three things that are not obvious and each cost a debugging cycle.

**Homebrew cannot supply any of it.** Homebrew dropped Intel x86_64 support in September 2026 and
no longer builds bottles for it; `/usr/local/Cellar` is also not writable by the build user. The
working route is a Temurin tarball unpacked anywhere readable.

**It needs JDK 21, not 17.** AGP 8.13's own floor is 17, but a Capacitor plugin pins a toolchain 21
requirement. With 17, Gradle resolves every dependency and *then* dies in
`:capacitor-filesystem:compileDebugJavaWithJavac` with `Cannot find a Java installation … matching
{languageVersion=21}` — an error that reads like a missing dependency rather than a wrong JDK.

**`JAVA_HOME` is pinned in the Makefile** (`ANDROID_JAVA_HOME`), not inherited from the caller's
shell. A build that works only for whoever exported it is a build that fails confusingly for
everyone else, CI included.

| Component | Where |
| --- | --- |
| Temurin JDK 21 | `~/tools/jdk-21.*/Contents/Home` (override: `ANDROID_JAVA_HOME`) |
| Android SDK | `~/Library/Android/sdk` (override: `ANDROID_SDK_DIR`) |
| Platform / build-tools | `platforms;android-36`, `build-tools;36.0.0` — match `variables.gradle` |

## What a store build must not contain

`mobile-build-release` asserts three things **on the built artifact**, because each was at some
point assumed and each assumption was wrong:

```text
OK: preview gate credential absent from the release bundle
OK: tier switch absent from the release bundle
OK: no private tailnet hostname in the release bundle
```

The reasoning is in the original credential check and applies to all three: a build-time
substitution reaches the bundle, and it only disappears if the bundler happens to fold the branch.
That is an optimisation, not a guarantee — so verify it at the one moment it matters.

### The bug these were written to catch

`mobile-build-release` ran:

```make
MOBILE_RELEASE=1 npm install && npm run build
```

A variable prefix binds to **one** command. `npm install` does not read `MOBILE_RELEASE`;
`npm run build` — the only command that does — ran without it. So `__MOBILE_INTERNAL__` was `true`
in every release build ever produced, and the prod-locking the target advertises never happened.
Two things shipped because of it: the dev↔prod tier switch, and the build host's **private tailnet
hostname** (`resolveDevApiBase()` derives it from the build machine when `VITE_DEV_API_BASE` is
unset). Fixed with `export`, and the assertions above now stand where the assumption was.

A related trap: the tier switch would not have been removed even with the flag right. A static
import plus a runtime `v-if` inside a component puts that component in the bundle unconditionally
and gates only its *rendering*. It is now imported dynamically behind the raw `__MOBILE_INTERNAL__`
constant — not `isInternalBuild()`, which computes the same answer through a cross-module call the
bundler cannot see through.

## Telling the two apps apart on one phone

Internal builds carry a **muted** app icon — same artwork, drained saturation. Regenerate with:

```bash
python scripts/tools/make_internal_icon.py
```

Android picks it up from the `debug` build type's resources; iOS names it in the `device` lane via
`ASSETCATALOG_COMPILER_APPICON_NAME`, because both iOS lanes build the *Release* configuration and
the Xcode configuration therefore cannot be the discriminator.

## Push

Native push is **unverified end to end on both platforms**. Do not read a shipped build as
evidence it works.

**iOS** has a complete path — client, entitlement, and an APNs transport in the homelab delivery
worker — and it fails at authentication: three real sends, three `403 InvalidProviderToken`, zero
successes. `InvalidProviderToken` rejects the provider JWT, not the device token. Check the `.p8`
is an **APNs** key rather than the App Store Connect one, and that the key id and team id match it.

**Android** has an FCM transport in the worker but no Firebase project, so there is no
`google-services.json` and push is switched off in the client
(`ANDROID_PUSH_NATIVE_READY = false`). A release build without that file only **warns**; set
`-PandroidPushRequired=true` to make it fatal, in the same change that flips the client guard.

Confirm the `aps-environment` entitlement is `production` for Release, or push silently fails on a
TestFlight build regardless of everything else.

## Beta testers need accounts

RFC-120 made the app login-first, and signup defaults to an allowlist
(`APP_SIGNUP_MODE`, fed from the `PLAYER_ALLOWED_EMAILS` repo variable). A tester who is not on it
cannot create an account, and a build nobody can sign into is not a beta. See #2190.
