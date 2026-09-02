# Handover — offline listening (#1905) + offline mode (#1906)

**Date:** 2026-09-02 (revised after review)
**Branch:** `feat/player-offline-downloads` (20 commits, **not pushed**, no PR)
**Issues:** [#1905](https://github.com/chipi/podcast_scraper/issues/1905) · [#1906](https://github.com/chipi/podcast_scraper/issues/1906) · [#1904](https://github.com/chipi/podcast_scraper/issues/1904) (unrelated dependency conflict, filed in passing)

Two sibling features on one branch. #1905 downloads episodes to the device; #1906 makes the app
usable with no network. They ship together because **a downloaded episode you cannot navigate to
is not a feature** — before #1906's Phase 0, an offline launch did not merely fail to sign you
in, it aborted boot.

> **Zero device verification.** Everything native sits behind `isNative()`, which Playwright
> cannot reach. All of it is unit-tested against a mocked Capacitor layer only. Nothing in this
> arc has run on a phone or a simulator. Treat every native claim below as "tested in isolation",
> not "seen working".

---

## 1. Where it stands

| item | state |
|---|---|
| `npm run test:coverage` | green — 89 files / 755 tests, `TEST_EXIT=0` |
| `npm run build` (`vue-tsc -b && vite build`) | green, `BUILD_EXIT=0` |
| Playwright e2e (containerised API, `--workers=4`) | green — 102 passed, 6 skipped, 0 failed |
| Device / simulator | **never run** |
| `app-lighthouse` CI gate | **never run** |
| Python side of the repo | untouched, never run |

The e2e suite needs the containerised API (`make e2e-api-image`) and `--workers=4`; the committed
config's full parallelism saturates a single uvicorn container and produces spurious failures.
See §6.

## 2. The finding that shaped everything

**On native there is no service worker, so there was no caching of any kind.** This is the repo's
own documented position (`docs/mobile-app-guide.md:67-71` — "plan offline as a native feature"),
and two further mechanisms make it moot even if one registered: `capacitor.config.ts:35` enables
`CapacitorHttp`, so requests never reach a SW `fetch` handler, and the native API base is a
different origin entirely.

Consequence: the existing PWA/service-worker work is correct **for the web deploy only**. Nothing
in `vite.config.ts` or `e2e/offline.spec.ts` was changed, and nothing should be.

## 3. What landed

Oldest first.

| commit | what |
|---|---|
| `8c46ec29` | device-local registry + `services/deviceStore` |
| `9495b4ca` | origin→device transfer |
| `25314502` | review fixes; entry schema made offline-sufficient |
| `69f32222` | **#1906 Phase 0** — persisted identity, boot survives offline |
| `36ff2795` | downloaded episodes play from disk |
| `1a474bea` | Wi-Fi policy + queue drain (adds `@capacitor/network`) |
| `94804453` | transcript cached with the episode |
| `6d95de8e` | device-local playback positions |
| `fa78937b` | a failed request no longer wipes painted content |
| `c942cec5` `14ffdf94` | queue writes stop lying; repair of what that broke |
| `5f619b4d` | the mark-for-offline control |
| `e7bcd14b` | the Downloaded surface in Library |
| `9a8f7eb3` `daa62cf0` `9f3cae5e` | settings → moved into the profile's Device section |
| `7a703e57` | registry + files namespaced per account |
| `27fe68ab` | reclaim finished episodes, refuse rather than evict |
| `cf1feeca` | offline position flush stops clobbering another device |
| `eb839bfa` | preferences sync survives one offline blip |

Principle 4 (*bridge, never rehost*, PRD-035) holds throughout: bytes travel origin host → device,
and no audio is stored on, proxied by, or served from our infrastructure.

## 4. Decisions taken (operator: Marko)

Full rationale, including what each one costs, is recorded on the issues.

1. **Device-side download is not rehosting** (#1905). Bytes go origin → device; our servers are
   never in the path. Recorded with its limits: the app *automates* the fetch, podcast-client
   precedent is not a licence grant, feed terms are not parsed, and no lawyer reviewed it.
2. **Native mobile only.** The web PWA is out of scope, because storing audio there would
   contradict the tested invariant in `e2e/offline.spec.ts`.
3. **L1, not L2.** Downloads progress only while the app is running. Flag it, close the app, and
   nothing happens until you reopen on an allowed connection. L2 needs iOS background
   `URLSession` / Android `WorkManager`, which `@capacitor/filesystem` does not expose.
4. **Eviction: reclaim finished, then refuse.** Only episodes played to completion are removed
   automatically; when that is not enough the next download is refused rather than evicting
   something unplayed.
5. **Registry namespaced per account; device settings shared.** Opposite rules on purpose — the
   download list is listening history, but whoever holds the phone decides how it uses their data
   plan.
6. **Positions: newest write wins where the stamps allow it.** See §5.

## 5. Corrections made during the arc

Both were mine, and both are recorded on the issues so nobody re-derives them:

- I claimed **"the server carries no position timestamp."** Wrong — `app_user_state.py:133-153`
  stores `updated_at`. What it lacks is a *write* time: `put_playback` stamps `int(time.time())`,
  i.e. **arrival**. The limitation stood; the stated reason did not. Reading the arrival stamp
  turned out to buy more than the approved rule: a **rewind made offline now survives**, which
  "push only if ahead" would have silently discarded.
- I landed **two commits on red gates** (`c942cec5`, `9a8f7eb3`) because the check piped `build`
  into `grep`, so grep's exit status masked tsc's, and a later chain ran `git commit`
  unconditionally. Both fixed in follow-ups; gates are now asserted on real exit codes.

## 5b. Review outcome (2026-09-02) — read this before trusting §3

An adversarial review of the whole arc found **ten confirmed defects**, one of which broke the
arc's core promise: **an offline cold start could not play a downloaded episode at all.**
`PlayerView`'s critical path awaited `getEpisode` with no `.catch()`, so any transport failure
aborted the load and `player.load()` — and therefore the injected source resolver — was never
reached. The registry carried title/duration/artwork precisely so that path could work; nothing
used it. Fixed in `2f6e6d34`.

All ten are fixed, plus three items missing from both issues entirely: offline auto-advance
(playback stopped at the end of every episode even with the whole queue downloaded), per-user
store reset on account switch, and a queued state mislabelled "Waiting for Wi-Fi" for users who
had allowed cellular.

**Two things §1 and §3 of this document originally overstated**, now corrected above:

- "A downloaded episode plays from disk" was true only *within a single session*.
- "Positions pushed on reconnect" fired only on an in-app network transition, so the common case —
  listen offline, kill the app, relaunch online — never flushed. Positions were also device-global
  while the registry was per-account, which crossed accounts on a shared phone.

**Three lessons worth carrying past this arc:**

1. The persist-before-hydrate defect was fixed in the downloads registry and then **reintroduced
   verbatim in `playbackPositions`**. When a storage module earns a correctness rule, check its
   siblings the same day.
2. A test asserted that `logout()` *rejects* — it had enshrined a bug as desired behaviour. A green
   suite is not evidence that the assertions are right.
3. The suite can report "755 passed" while emitting 18 unhandled mount errors. **The exit code, not
   the pass line, is the gate.** Two commits landed on red gates in this arc because a `grep`
   masked a non-zero exit.

## 6. Open threads — where to resume

**Blocking anything shipping:**

- **Device spike, not yet done.** `Capacitor.convertFileSrc` playback from `LibraryNoCloud`,
  **seeking** (byte-range through the custom scheme handler), backgrounding mid-transfer, and
  resolving the URI from `path` rather than a persisted absolute URI. Slices 3–7 are built on
  assumptions this spike would confirm or destroy.
- **No PR, nothing pushed.** 20 commits sit local on `feat/player-offline-downloads`.

**#1906 Phase 2, not started** — the general stale-content cache. Until it lands, only the
Downloaded list works offline; every other Library surface is still API-driven.

**Known gaps, deliberately left:**

- `DOWNLOAD_CAP_BYTES` is a hardcoded 4 GiB starting figure, not a device setting.
- A client-supplied timestamp on `PUT /playback` would make position conflict resolution exact
  rather than approximate. Small backend change; the real fix if conflicts ever show up.
- No download **resume** — `Filesystem.downloadFile` cannot, so an interrupted transfer restarts
  from zero. The registry demotes an interrupted `downloading` back to `queued` at launch.
- A general **write outbox** is out of scope. The queue is a whole-list `PUT`, so replaying stale
  offline writes is a conflict-resolution problem, not a retry problem.
- Partial files from failed transfers are unaccounted for; a folder-vs-registry reconciliation
  sweep is still owed.
- The "these actions cannot 401" reasoning behind skipping the auth gate **expires** when
  #1063/#1066 land and the episode routes become auth-gated. Noted in `services/downloads.ts`.

**Local harness (not committed):** `web/learning-player/playwright.docker.config.ts` is untracked
and is what makes the e2e suite runnable here — `[search]` cannot install on Intel macOS
(`torch>=2.11`, `lancedb>=0.33` have no x86-64 wheels), so the API must run in Docker. Decide
whether to formalise it with a Makefile target or delete it.
