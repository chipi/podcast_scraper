fastlane documentation
----

# Installation

Make sure you have the latest version of the Xcode command line tools installed:

```sh
xcode-select --install
```

For _fastlane_ installation instructions, see [Installing _fastlane_](https://docs.fastlane.tools/#installing-fastlane)

# Available Actions

## iOS

### ios preflight

```sh
[bundle exec] fastlane ios preflight
```

Verify credentials and signing WITHOUT building or uploading anything.

Run this first on a new machine — it fails in seconds instead of after a 10-minute build.

### ios beta

```sh
[bundle exec] fastlane ios beta
```

Build a signed release archive and upload it to TestFlight.

----

This README.md is auto-generated and will be re-generated every time [_fastlane_](https://fastlane.tools) is run.

More information about _fastlane_ can be found on [fastlane.tools](https://fastlane.tools).

The documentation of _fastlane_ can be found on [docs.fastlane.tools](https://docs.fastlane.tools).
