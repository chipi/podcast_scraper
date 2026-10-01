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

Android picks it up from the `debug` build type's resources. iOS sets it in the **Debug** build
configuration (`ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon-Internal`), alongside that build's own
bundle id and display name — see below.

## Push

**Working on both platforms since 2026-09-30** — a real notification has been delivered to a real
handset on each. What is NOT proven is the nudge path: every send so far called the transport
directly with the outbox empty, so `resurface-nudge.v1` -> outbox -> worker -> APNs/FCM has never
run.

**iOS.** The long-running failure was the credential, not the code: `apns_key_id` held an **App
Store Connect API key**, not an APNs key. Both are ES256 `.p8` files with a 10-character id and
they are indistinguishable by inspection — the ASC key even returns HTTP 201 to App Store Connect
while returning `403 InvalidProviderToken` to APNs. Three real sends failed that way before anyone
looked. The two live in different halves of Apple's site:

| | Where | Used for |
| --- | --- | --- |
| APNs key | developer.apple.com -> **Keys** | sending notifications |
| ASC API key | App Store Connect -> Users and Access -> **Integrations** | uploading builds |

To tell a key apart without a device, send to APNs with an all-zeros token: `403
InvalidProviderToken` means the key is wrong, `400 BadDeviceToken` means the key is **right** and
only the dummy token was rejected.

`apns_sandbox: false` is correct for TestFlight/App Store builds — their tokens are production
tokens. Dev-signed builds are sandbox and route through the `podcast-dev` tenant, whose
`apns_bundle_id` must be the **`.dev`** id (see below), because `apns-topic` has to equal the
bundle id the token belongs to.

**Android.** FCM, via a service account in the delivery worker. `google-services.json` is
gitignored and keyed by package name, so a build for a different applicationId needs its own client
added in the Firebase console. A release build without the file is fatal under
`-PandroidPushRequired=true` (which `make android-bundle` sets).

Confirm the `aps-environment` entitlement is `production` for Release, or push silently fails on a
TestFlight build regardless of everything else.

## The debug build has its own identity

iOS identifies an app by **bundle id alone**, so a local build sharing the shipped id replaces the
TestFlight app on the home screen rather than sitting beside it. The Debug configuration therefore
ships:

| | Release | Debug |
| --- | --- | --- |
| bundle id | `app.closelistening.player` | `app.closelistening.player.dev` |
| display name | Close Listening | CL Dev |
| icon | `AppIcon` | `AppIcon-Internal` (muted) |
| APNs environment | production | development |

Consequences worth knowing: the two apps have **separate storage**, so the dev build starts signed
out with no downloads; `IOS_BUNDLE_ID` and the UI-test suite default to the `.dev` id (override with
`LP_UITEST_BUNDLE_ID`); and the `podcast-dev` delivery tenant's `apns_bundle_id` must match it.

**Android has no equivalent split yet (#2208)** — debug and release share
`app.closelistening.player`, and because they are signed by different keys `adb install -r` fails
with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`. The only way forward is uninstalling the Play build,
which takes its data with it. Fixing it needs an `applicationIdSuffix ".dev"` plus a matching
Firebase client, since `google-services.json` is keyed by package name — so the console step has
to come first or every Android build breaks.

## Beta testers need accounts

RFC-120 made the app login-first and the allowlist gates **every sign-in**, not just account
creation — so a tester who is not on it cannot get in, and a build nobody can sign into is not a
beta.

Add one **without a deploy** (#2190):

```bash
curl -X PUT https://closelistening.app/api/app/admin/access-policy \
  -H 'content-type: application/json' -b "$SESSION_COOKIE" \
  -d '{"mode":"allowlist","allowed_emails":["you@example.com","tester@example.com"]}'
```

It is a **full replacement**, so include every address that should keep working — including your
own. The endpoint refuses a policy that would lock the caller out, and warns when it excludes
another bootstrap admin. `PLAYER_ALLOWED_EMAILS` remains the bootstrap seed, used only when no
policy file exists.

Removing an address stops the **next** sign-in; it does not end a live session (30-day cookie). To
cut someone off immediately, `PATCH /api/app/admin/users/{id}` with `disabled: true`.
