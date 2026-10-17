---
name: mobile-release
description: Ship the Close Listening app to testers and run its device tiers in the right order — preflight, dry run, build and upload for TestFlight and Play internal testing; the iOS and Android device suites with the emulator brought up and torn down; a build on the operator's own phone. Each upload needs the operator's go. Use when asked to "ship to TestFlight", "push to Play internal", "send a build to testers", "run the device suite / test-ios / test-android", or "put it on my phone".
---

# mobile-release

The procedure lives in the Makefile; this skill is the order and the gates. Background and
credentials: `docs/guides/MOBILE_STORE_RELEASE_RUNBOOK.md`; building Android from nothing:
`web/learning-player/docs/mobile-capacitor-reference.md` Part 2; device tiers:
`docs/guides/MOBILE_E2E_TESTING.md`.

## Ship to testers

Each upload is outward-facing and spends a build number: **the operator's go for each one.**
"Push everything when done" covers git, not a store upload.

**iOS → TestFlight** (`ios-testflight` does not run its own preflight):

```bash
make ios-testflight-preflight; echo "MAKE_EXIT=$?"   # credentials + app record, seconds, builds nothing
make ios-testflight-validate; echo "MAKE_EXIT=$?"    # the whole release path, stops before upload
make ios-testflight; echo "MAKE_EXIT=$?"             # with the go: build, archive, upload
```

**Android → Play internal testing** (`android-play` does not run its own preflight):

```bash
make android-play-preflight; echo "MAKE_EXIT=$?"     # credentials + app record + the track's codes
make android-play; echo "MAKE_EXIT=$?"               # with the go: next versionCode FROM PLAY, signed AAB, upload
make android-play-status; echo "MAKE_EXIT=$?"        # read-only: has Play processed it
```

- Before `android-play` uploads, read its `Play says the next free versionCode is …` line: it must
  be one more than the track's current code. Play never lets a code go down.
- "Uploaded" is not "installable": allow ~30 min. The proof is a tester's Play Store offering the
  update.
- A tester Play will not serve yet ("Item not found"): `make android-play-apk` → Play's signed
  universal APK to send directly. Never tell anyone to search the Play Store: internal-only apps
  are not listed.
- Both store builds are RELEASE builds. `mobile-build-release` asserts on the artifact that the
  preview credential, the tier switch and the tailnet hostname are absent. A failing assertion is
  a stop, not something to work around.

## Device tiers

```bash
make test-ios; echo "MAKE_EXIT=$?"          # the iOS entry point: every suite, in dependency order
make test-android; echo "MAKE_EXIT=$?"      # the Android entry point (~2.5 h for 14 suites)
```

- Long runs: watch them live and open the first failure while the rest runs (Monitor on the log,
  grep for the suite's failure markers). Do not sleep through them.
- One suite while iterating: the `test-app-ios-*` / `android-suite` step targets. They are the
  internals of the entry points, for debugging, not a gate.
- The emulator: `make android-emulator-up` / `android-emulator-down`. They act on THIS repo's AVD
  by name; down waits until the device is really gone. Never `pkill emulator`: other worktrees run
  their own.
- After a stack-test or Playwright run, `make stack-test-reap` (CLAUDE.md).

## On the operator's phone

```bash
make ios-device-install; echo "MAKE_EXIT=$?"       # internal build, the .dev identity (CL Dev)
make android-device-install; echo "MAKE_EXIT=$?"   # internal debug APK over USB
```

On Android the debug build and the Play build share an applicationId and different signing keys:
installing one over the other fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`. Uninstalling the
Play build deletes its data, so ask before doing it.

## Not covered

- Public store release (listing, data safety, content rating): never done in this project.
- Which suites a change needs: decide from what changed; the entry points run everything.
