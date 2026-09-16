# Mobile end-to-end testing — which servers, in which order, and why

Running the native suites means standing up the right backends *before* building the app. Almost
every failure in this tier is a precondition that was not met, and none of them say so: they surface
as a missing control, a blank screen, or an assertion about the wrong thing entirely.

This guide is the map. Read the failure table at the bottom first if something is already broken.

---

## The one thing that causes the most lost time

**The app bakes its API base at BUILD time.** `VITE_API_BASE_URL` is read by `npm run build` and
compiled into the bundle; nothing at runtime can repoint it (the dev/prod switch only chooses
between the baked value and the live site — see `src/services/tier.ts`).

So a build and a backend are a matched pair. Rebuilding against the wrong port, or forgetting to
rebuild at all, produces an app that looks completely normal and silently talks to the wrong place.

Two consequences worth internalising:

- **Running the suites does not rebuild the app.** `make ios-contact-sheet` and
  `test-app-ios-journey-ui` run tests against whatever is installed on the simulator. A whole
  contact sheet was once reviewed showing sheet geometry from a build two fixes old.
- **`npm run build` with no `VITE_API_BASE_URL` falls back to production.** The bundle then points at
  `https://closelistening.app/api/app` and every fixture assertion fails against the live site.

`make ios-app-install` exists so you never do this by hand. It builds against the single origin,
installs, and then **verifies the installed bundle** and that the origin serves audio, failing loudly
if either is wrong.

---

## The servers

| Port | What | Started by | Serves |
| --- | --- | --- | --- |
| `8011` | Fixture API container (`lp-e2e-api`) | `make app-e2e-api-up` | `/api/**` only |
| `18765` | Fixture media host | `make ios-origin-up` | the episode `.mp3` files |
| `4174` | **Single origin** (vite preview) | `make ios-origin-up` | `/api` → 8011, `/audio` → 18765 |

### Always build against `:4174`, not `:8011`

The corpus stores `content.media_url` as a **relative** `/audio/<id>.mp3`, and `resolveMediaUrl()`
absolutises it against the API base. Point the app at `:8011` and every audio URL becomes
`http://127.0.0.1:8011/audio/<id>.mp3`, which does not exist.

The episode page then renders **"Couldn't load the audio from the source. Try again later."** with no
transport at all — so a test looking for a Play control reports *"no Play control"*, which reads like
a UI regression and is really a missing media host.

`:4174` fronts both, which is also the shape production has (one origin for API and media).

---

## Order of operations

```bash
make ios-origin-up        # 8011 + 18765 + 4174   (app-e2e-api-up is implied)
make ios-app-install      # build against :4174, install, VERIFY the bundle
make ios-journey-signin   # mint a native session — AFTER the install, see below
make test-app-ios-journey-ui     # the assertion suites
make ios-contact-sheet    # install + signin + seed + tour + grid (does all of this itself)
```

**The order of those middle two matters.** `ios-app-install` pulls in `app-e2e-api-up`, which
recreates the fixture api container — so an account minted BEFORE the install is destroyed moments
later, and the app is left holding a token for a user that no longer exists. Every auth-gated
surface then returns empty and the suites fail as "element not found", which reads like a UI
regression rather than a wiped backend. `ios-contact-sheet` now sequences this for you.

Tear down with `make ios-origin-down && make app-e2e-api-down`.

### Installing the app WIPES the backend — always re-seed after it

`ios-app-install` depends on `ios-origin-up` → `app-e2e-api-up`, and that target **tears the fixture
api container down and re-seeds it from the corpus**. Every board, favourite, colour, chosen
interest and minute of listening history created by a previous run is gone.

So "the data is already seeded, I'll just re-run the tour to save ten minutes" is wrong, and it
fails in a way that looks like a UI regression: the tour shoots Home and Discover fine, then misses
the Library sub-tabs, the profile tabs and every entity surface, because there is nothing in them
to tap. Re-running `ios-contact-sheet` end to end is the cheap path; the seeding suites regenerate
everything as a side effect of asserting on it.

The session token survives (the session secret is a fixed env var); only the DATA does not.

### Sign-in is a precondition, not part of the suites

Every consumer surface is auth-gated — the server 401s anonymous reads — and the journey suites do
not sign in. `make ios-journey-signin` mints a session through the mock provider's native flow and
writes it into the app's durable store. Skip it and the suites fail on empty surfaces.

Write through the preferences **daemon**, never the container plist directly; the plist lags and the
two views converge lazily. The target already does this.

---

## Suites and their preconditions

These differ per suite, and the bundle-wide `-only-testing:OfflineSpikeUITests` runs them all —
including ones whose harness you have not started. Use the make targets.

| Suite | Needs | Target |
| --- | --- | --- |
| `AppJourneyTests`, `PersonalisationTests`, `OfflineCacheTests` | signed in, network up | `test-app-ios-journey-ui` |
| `NativeCapabilityTests` | signed in + host permissions (mic, photo) | (journey tier) |
| `DownloadThroughUITests` | single origin, network up | `test-app-ios-sim-download` |
| `OfflinePlaybackTests`, `OfflineAutoAdvanceTests` | episodes ALREADY downloaded, then everything **down** | `test-app-ios-journey` |
| `ScreenshotTourTests` | a POPULATED account — runs after the journey suites | `ios-contact-sheet` |

Suites also inherit each other's leftovers today; see issue #2091 for the per-suite setup/teardown
plan. Until that lands, prefer the make targets over ad-hoc `-only-testing` selections.

---

## Failure → cause

| Symptom | Actual cause |
| --- | --- |
| "no Play control" / "Couldn't load the audio from the source" | App built against `:8011`; no `/audio`. Rebuild via `ios-app-install`. |
| Every assertion fails at once, app looks fine | Bundle points at prod — `VITE_API_BASE_URL` was missing at build time. |
| **Empty** accessibility inventory; app appears blank | iOS's *"Open in 'Close Listening'?"* SpringBoard prompt is up (it re-appears after a reinstall). While it is, the app is not frontmost and every query returns empty. `AppSession.openEpisode` taps it. |
| "neither Sign in nor Sign out present" | The profile entry point is labelled with the **display name**, not "Your profile". Use `Journey.openProfile`. |
| Frames missing from the middle of the contact sheet | A sheet was left open and swallowed every later tap. Look for `=====SHEETS_STUCK=====`. A card's own ✕ can scroll out of the viewport (seen at y = −619). |
| Screenshots don't show a fix you just made | The app was not rebuilt. Suites do not rebuild; `ios-app-install` does. |
| Tour shoots Home/Discover then misses Library sub-tabs, profile and entity surfaces | The backend was re-seeded (any `ios-app-install`) and the seeding suites were not re-run. The app is empty, so there is nothing to tap. |
| A card cannot be dismissed; `SHEETS_STUCK` | An entity card labels its dismiss control **Back**, not Close, when `dismissAtRoot` is false. A Close-only search finds nothing. |
| `MODULE_NOT_FOUND` from any node step | The cmux `NODE_OPTIONS` shim. Prefix with `env -u NODE_OPTIONS`. |
| A `make ios-*` target never returns, though its work plainly finished | It was piped (`\| grep`, `\| tail`). `ios-origin-up` leaves a backgrounded vite preview holding the pipe open, so the reader never sees EOF. **Redirect to a file** (`> /tmp/x.log 2>&1`) and read that; never pipe these targets. |
| Suite passes but shot fewer screens than expected | Best-effort frames. The tour now asserts its full list; older runs did not. |

---

## Reading results

`xcodebuild` output is filtered by the make targets. Diagnostics are printed as `=====TAG=====`
lines (`INVENTORY`, `TAP_MISS`, `SHEETS_STUCK`, `SHOT`), so any filter must keep `=====` or you lose
the only explanation of a failure.

For a failed run, the assertion text lives in the result bundle:

```bash
RES=$(ls -td /tmp/lp-ios-dd-uitests/Logs/Test/*.xcresult | head -1)
xcrun xcresulttool get test-results tests --path "$RES"
```

A bundle with no `Info.plist` is still being written — take the next one down.

---

## Contact sheet

`make ios-contact-sheet` seeds by running the journey + personalisation suites (they create boards,
favourites, colours, interests and listening history as a side effect of asserting on them), tours
every surface, and stitches the frames into one labelled grid.

At native resolution the grid lands around 82 MP, which macOS Preview will not open, so
`contact_sheet.py` also emits row-aligned `-partN.png` files. Cuts fall **between** rows — splitting
on a fixed pixel height slices screenshots in half.
