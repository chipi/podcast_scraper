# Branch handover — `fix/ui-followups-2026-09-18` (PR #2127)

**110 commits, 2026-09-18 → 2026-09-27, six arcs.** Pushed. This is the
branch-level view; the last three days have their own detailed handover at
`DEVICE-TIERS-HANDOVER-2026-09-27.md`, which this does not repeat.

## Gates, measured

| Gate | Result |
| --- | --- |
| Rebase onto `origin/main` | 105 commits replayed, 3 conflicts resolved |
| `make ci-fast` | `0` — 12,108 unit · 1,863 integration · 122 e2e |
| `make ci-ui-fast` | `0` — 512 Playwright · 386 vitest files |
| `make test-android` | `0` — 14 suites |
| `make test-ios` | `0` — 25 suites |
| PR #2127 | CodeQL cleared; long jobs still running at handover — **verify before trusting** |

## Issues

**Closing on merge (6):** #2091, #2120, #2139, #1588, #1593, #1603.
**Held open deliberately (4):** #2156, #2157, #1595, #1978 — reasons in the PR
body and the device-tier handover.

## The six arcs

### 1. Review-round closure (2026-09-18, 11 commits)

Closing the drift #2118 left, plus two adversarial review rounds. The one that
mattered: **the three revisit outcomes (reviewed / retire / delete) removed the
card BEFORE awaiting the write, with no rollback** — a failed write lost the item
from the UI while it still existed on the server. Also: a dead auth server was
indistinguishable from a missing corpus path in the viewer; a Show-rail theme chip
routed on an id that is not a node; and integration coverage for three shipped
export routes that had none.

### 2. Device-feedback rounds (2026-09-19, 17 commits)

Two rounds of operator feedback on a real device, plus an episode dossier. Dead
controls found: **Collection and Share did nothing on any sheet**; storyline rows
in Trends were dead on tap (the client joined trending against an endpoint that
floors at 4 members and returns top-N by size, so misses were routine); `all ›` on
Discover navigated to the page you were already on. Also a board drag lost if you
left the screen immediately, an insight covering the artwork before playback, and
the avatar cropper being a keyboard trap with a pointer-only crop.

### 3. The Storyline / Theme rename (2026-09-20, 25 commits)

The largest single arc, and the one most likely to surprise someone reading the
diff. It closes **#1603**: the consumer vocabulary said the opposite of what
UXS-013 specifies. Executed in six staged pieces — **A1** (storyline half of the
backend), **A2** (theme half), **A3** (clients, player + viewer), **A4** (docs),
**B1** (API fields), **B2** (route paths), plus a follow-up for a wire field B1
missed. `theme_clusters.py` → `storylines.py`,
`corpus_theme_clusters.py` → `corpus_storylines.py`.

**This rename is why CodeQL went red on the PR** — see the note now in
`docs/ci/CODEQL_DISMISSALS.md`. Moving a file re-issues its dismissed
`py/path-injection` alerts under new ids.

Same day: Search folded into Discovery, one nav IA for both widths (**#1588**),
an anyio CVE floor, and **#2120** fixed — the download UITest was looking where
the control used to be, before it moved into an overflow menu that teleports
outside `<main>`.

### 4. Played state, queue rework, auth (2026-09-24, 18 commits)

`played` now means played however the listener got there, not just hand-marked.
Two auth fixes worth knowing: **a dead credential now takes the bearer token with
it**, and **a token without an identity no longer blocks the login page** — that
second one could lock a user out of signing in. The tier badge now names where
traffic GOES rather than what the switch says.

### 5. The Android tier, stood up (2026-09-24 → 09-25, ~30 commits)

Closes **#2139**: Android had no UI test coverage of any kind — the only file
under `androidTest/` was Capacitor's generated scaffold, asserting a package name
that was wrong and which nothing ran.

Suites ported from iOS one at a time. The recurring discovery, worth carrying:
**the port was written against iOS accessibility semantics and Android's differ.**
`By.desc` never matches WebView content, so the harness was blind to every
icon-only control; Android's tree holds only ON-SCREEN nodes; a kind prefix is
flattened into the label with no separator.

This arc also produced the **accessible-name audit** — 38 controls a screen reader
could not announce, brought to 1, and the audit made honest about what it cannot
judge (it excludes EditText and SeekBar, because `UiObject2` exposes neither a
hint nor slider metadata; ten of the original 38 were that false positive).

### 6. Both tiers green (2026-09-25 → 09-27)

Detailed in `DEVICE-TIERS-HANDOVER-2026-09-27.md`. In brief: **four product bugs
CI cannot see** (push enable killed the app on Android; the favourite heart had no
accessible name; Profile never refreshed so stats froze until restart; five more
unnamed controls), **seven test defects that were passing for the wrong reason**,
and the environment fixes. `testN4` avatar upload ported so Android matches iOS
N1–N4.

## What is NOT done

- **#2156** — ~32 unnamed icon-only controls remain. Its comment thread carries
  the audit's blind spot and a second failure mode; read it first.
- **#2157** — native push is guarded so it cannot crash, but FCM is not wired.
- **#1595** — 3 of 5 parts verified done; part 4 ("Grounded" is never introduced)
  is open, part 5 partial. Its Definition-of-done test list is unchecked.
- **#1978** — standing compositional backlog, by its own instruction.
- **iOS phase 1 does not queue** despite being named `…AndQueuesThem` — zero
  `POST /api/app/queue/items` measured over a container's lifetime.
- **Nothing resets server-side accounts** between runs. Both tiers reset the
  device only. This is the biggest remaining source of false greens.
- **Neither device tier runs in CI**, so the PR's checks say nothing about them.
- The e2e coverage floor is **deliberately unchanged at 37.0** — raise it once CI
  reports a real full-run figure rather than guessing.

## Where to start

1. Confirm #2127's remaining checks; fix anything red.
2. Server-side account reset — highest value for tier trustworthiness.
3. iOS phase 1's queueing step.
4. #1595 parts 4 and 5 — it is close enough to finish deliberately.
