# Device tiers — handover, 2026-09-27

> Branch-level context for all 110 commits: `BRANCH-HANDOVER-ui-followups-2026-09-27.md`.
> This file covers the device-tier arc (2026-09-25 → 09-27) only.

Branch `fix/ui-followups-2026-09-18`, **pushed**. Supersedes
`ANDROID-TIER-HANDOVER-2026-09-26.md`, which covered only the Android half.

## Where things stand

| Gate | Result |
| --- | --- |
| Rebase onto `origin/main` | done — 105 commits replayed, 3 conflicts resolved |
| `make ci-fast` | `CI_FAST_EXIT=0` |
| `make ci-ui-fast` | `CI_UI_FAST_EXIT=0` |
| `make test-android` | `TEST_ANDROID_EXIT=0` — 14 suites, 0 failures |
| `make test-ios` | `TEST_IOS_EXIT=0` — 25 suites, 0 failures |
| PR #2127 | pushed; CodeQL cleared; long jobs were still running at handover |

**Check the PR before assuming green.** `test-e2e-fast` (~43 min), `viewer-e2e`
and `coverage-unified` had not reported when this was written.

## Issues

**Closing on merge (6):** #2091, #2120, #2139, #1588, #1593, #1603.

**Deliberately left open — do not close these:**

| Issue | Why it stays open |
| --- | --- |
| #2156 | 37 unnamed icon-only controls; **5 fixed here**, ~32 remain. Its comment thread carries the audit's blind spot and a SECOND failure mode (a wrapper element strips a name even when a text node exists). Read it before touching accessible names. |
| #2157 | Native push. Android is **guarded so it cannot crash**, but FCM is not wired: no Firebase project, no sender, and the client labels Android tokens `apns`. Four legs of work listed in the issue. |
| #1595 | Five-part packaging issue. **Closer to done than it looks** — see below. |
| #1978 | Standing backlog — "work item by item" by its own instruction. |

### #1595, part by part (checked 2026-09-27)

Worth a deliberate pass rather than leaving it to drift; 3 of 5 are already done.

| Part | State | Evidence |
| --- | --- | --- |
| 1. Knowledge Panel hides behind a `💡 {count}` chip | **done** | Labelled Ask/Insights toolbar (`PlayerView.vue:1365`, `:1399`); device inventories show a button named "✦ Insights" |
| 2. Three names for one idea (Theme / Similar / Storylines) | **done** | The #1603 rename arc, A1–A4 + B1–B2. The issue asks for exactly the word that landed |
| 3. "×" explained only in `title` (invisible on touch) | **done** | `DiscoveryList.vue:197` renders it on screen; measured in a device dump: `↑2× = twice as often as in recent weeks` |
| 4. "Grounded" is never introduced | **OPEN** | The underline + aria-label exist; no first-contact explanation found |
| 5. Search outranks synthesis on the entity card | **partial** | Perspectives moved up under Top voices, conversation arc beside the sparkline; could not confirm the "Search every episode for X" CTA was demoted |

The issue also carries a "Definition of done" with unit / e2e / Tier-3 test
requirements that have NOT been checked against.

### #1978 is a backlog, not a task

It tracks the compositional remainder of a blind-critic review, and instructs:
*"The findings are input, not orders … re-verify it still reproduces, and re-ask
whether the proposed remedy is actually the right call."* Several critic claims
have already been measured and rejected. There is no completion criterion — it
closes when the list is judged exhausted, not when a given change lands. One
finding was addressed on this branch (an insight covering the artwork before
playback).

## THE pattern worth carrying forward

**Five of this session's failures had one shape: something persisted that nothing
reset, and a suite was green because of history rather than behaviour.**

1. A stale `lance_index` in `tests/fixtures/app-validation-corpus/v3/` was copied
   into the e2e corpus, so it indexed 36 episodes instead of 40 and `ci-ui-fast`
   failed on a search spec. Gitignored, regenerable, invisible in `git status`.
2. Server-side **accounts** persist across runs. `pm clear` / `simctl uninstall`
   reset the DEVICE only. `PersonalisationTests.test09` passed for days on an
   account that already had listening data — masking a real product bug.
3. Simulator **app state** survives reinstall, including a FAILED download.
   `DownloadThroughUITests` failed at 103.6s, passed at 103.7s after nothing but
   `simctl uninstall`.
4. The **queue** is server-side. `NativeOnlySurfacesTests` queued from a surface
   with no queue control and "passed" for months on a queue left by earlier
   sessions.
5. `AppJourneyTests.test03` assumed a short Home; once the account had history,
   Home grew and pushed the Trends rail out of the accessibility tree.

Both tiers now reset the device (`pm clear`, `simctl uninstall`). **Nothing resets
the server-side account.** That is the biggest remaining source of false greens.

## Known-broken, NOT fixed

- **iOS phase 1 does not queue.**
  `testDownloadsTwoEpisodesThroughTheUIAndQueuesThem` makes **zero**
  `POST /api/app/queue/items` — measured over a container's whole lifetime. It
  passes; the "AndQueuesThem" half of its name is untrue. Left deliberately:
  fixing it changes the state every later suite sees, which is not a change to
  make at the end of a long chase.
- **The blank-WebView sign-in flake is quiet, not proven fixed.** It took out
  three tier runs (`NativeOnlySurfaces`, `PersonalisationTests`,
  `StackDepthProbeTests`), each time as `sign-in did not complete … <nothing
  labelled>` with the backend verified healthy. `AppSession.relaunch` now
  force-stops and relaunches up to twice — but **that path never fired** in the
  ~40 clean sign-ins since. Untested recovery.
- **The accessible-name audit walks 5 surfaces** (Home, Discover, Library,
  player, player ⋯) and never opens Profile, Settings, a note composer or any
  popover — where 3 of 7 defects lived. Parked into #2156.
- **iOS has no accessible-name audit at all.** Every defect in that class was
  found by Android or the static guard.

## Traps that cost hours — read before debugging either tier

- **A failure message names a symptom and is usually wrong about the cause.**
  "no dictation control" meant a keyboard was covering it. "no storyline row"
  meant the rail was off-screen. "no Play control offline" meant the test had
  opened the wrong episode. In every case the dump attached to the failure held
  the answer and I theorised instead of reading it.
- **`scrollTo` is the wrong tool for anything that loads over the network.** Both
  platforms' versions give up once two swipes leave the page signature unchanged
  — about four seconds. Use probe-and-swipe loops (`find` with a deadline, then a
  swipe, repeated) for content that arrives from an API.
- **Android's tree holds only ON-SCREEN nodes; iOS keeps off-screen ones.** On
  Android "X is missing" usually means "X is below the fold". On iOS an element
  can be found and still not be tappable.
- **Check the origin before believing a suite result.** A `ctx_shell` job that
  starts `ios-origin-up` in the background takes it down when the job ends; a
  whole run then tests against nothing. Probe `/api/health` AND `/api/app/me` —
  503 means "cannot authenticate anyone" (the degraded drill's leftover), 401
  means healthy.
- **A renamed file re-issues its CodeQL dismissals.** `theme_clusters.py` →
  `storylines.py` turned the PR red with "1 new high severity". Inline
  `# codeql[...]` pragmas move with the code and do **not** suppress. See the note
  appended to `docs/ci/CODEQL_DISMISSALS.md`; `gh api` caps dismissal comments at
  280 characters.
- **Git hooks do not inherit the Makefile's `unexport NODE_OPTIONS`.** A commit
  touching markdown fails lint with `Cannot find module …restore-node-options.cjs`
  under cmux. Use `env -u NODE_OPTIONS git commit`, not `--no-verify`.

## Neither tier runs in CI

No workflow references `test-ios`, `test-android`, `android-suite`, `xcodebuild`,
`simctl` or `emulator`. CI runs two Python guards that only read files
(`test_{ios,android}_uitest_suites_are_wired.py`), which catch "a suite stopped
being run" but never "a suite runs and fails".

**Consequence:** all four product bugs fixed this session were invisible to CI and
would have shipped. A green PR says nothing about the device tiers.

Discussed but not decided (worth an ADR): Android *could* run on Linux runners via
`reactivecircus/android-emulator-runner`; iOS needs macOS runners at a 10×
multiplier. Recommendation was path-filtered + nightly + a separate mobile release
pipeline, **after** the determinism work above — putting a flaky tier on required
checks just relocates the pain.

## Where to start

1. Confirm #2127's remaining checks went green; fix anything that did not.
2. Decide on the server-side account reset — the single highest-value fix for
   tier trustworthiness.
3. iOS phase 1's queueing step.
4. #2156's remaining ~32 controls, and whether iOS gets a name audit.
