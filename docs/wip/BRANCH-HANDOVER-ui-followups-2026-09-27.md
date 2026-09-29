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

---

# Session 2 — `october-fixed`, 2026-09-27 evening

**PR #2127 merged** at 14:20Z. This branch is `origin/main` + 9 docs-hygiene commits + 21 from this
session. **Nothing is pushed.** Everything below was driven by the operator testing real builds on an
iPhone 15 Pro; each fix was installed and re-checked on the device the same evening.

## Done

| Commit | What |
| --- | --- |
| `859cf98e6` | Queue control out of the transport row, beside the heart, with in-queue state. `QueuePanel` **deleted** — both its openers were gone, so nothing could reach it; its 6 behaviours ported onto `RecentlyPlayedList` rather than lost. |
| `ec7c5e696` | Saved colour strip fits. Measured off the screenshot: 284pt of swatches in 263.6pt, so violet was clipped exactly in half behind a suppressed scrollbar. |
| `f7679eb7b` | **Person photos.** The edge answered every `/api/app/persons/*/photo` with coming-soon HTML and a **200**. 712 hosted portraits, never once fetchable. Third time this defect shipped. Deployed that night; verified live at 31,392 bytes. |
| `c6a87e3b3`, `e82b1c591` | PDF chip opens the notes in-app, then **on top of** the panel. The second fixes my own regression: I teleported the viewer to `<body>` while the panel is `showModal()`'d into the top layer. |
| `1c1e04cae` | Nav tooltips no longer latch. iOS applies `:hover` on tap and never ends it, so the label survived the navigation. |
| `91ecba387`, `0c14e6173`, `7db7b88fa` | **One glyph per concept** (UXS-014): heart = favourite, bookmark = highlight, 2×2 board = collection. The third fixes a collision *I* created in the second. |
| `09b03359d` | Conversation arc announces loading instead of appearing from nowhere; bars fill the box when a topic has few weeks. |
| `5f30fb16d` | **Collection names** — not a colour bug; the row was clipped to nothing. See below. |
| `979061738` | Ask → **Search**. It renders `hit.text`; there is no synthesis endpoint. Duplicates collapsed, keyboard drops on submit. |
| `aa86ee277`, `4df86aab6` | Person roles (host/guest/mentioned) on the show page and the person card, ordered by role. Reuses `_role_of` / `_aggregate_role` / `_ROLE_RANK` rather than a second implementation. |
| `c99948f33`, `31686d1a3` | **Person photos, instances 4 and 5.** `getKeyVoices` and `getTopicPerspectives` never absolutised, so relative URLs resolved against `capacitor://localhost`. Guard added. |
| `2b11f1d2a` | Mini-player progress uses `transform: scaleX`, not `width` — it was re-laying-out a fixed bar ~4×/sec for the length of every episode. |
| `9373fc1ca`, `3d12c1d81` | **#2164** — 96 episodes render zero topics because GI wrote insight sentences into the `Topic` nodes. Measured, time-clustered, handed off. |

## NOT done — equal weight, read this half

- **Nothing is pushed.** 21 commits exist only on this machine.
- **The scroll artifact is undiagnosed.** Fixed chrome paints mid-list during momentum scroll; the
  operator could not reproduce it on demand. The advisor killed the containing-block theory with
  evidence and proposed a main-thread-commit race. `2b11f1d2a` removes the most obvious commit
  generator but **is not claimed as the fix**. The remaining lever — `will-change: transform` on the
  two bars — is **deliberately unshipped**: speculation against an unreproducible symptom, with no
  way to tell whether it worked. The zero-code A/B (nothing playing / paused / playing) is still the
  cheapest next step.
- **Ask's missing timestamps and duplicate hits are index-side and unfixed.** `hitStartSeconds` finds
  nothing in `lifted.quote`, `supporting_quotes` or `metadata`, so jump-to-moment is dead on those
  hits. A 44,924-char chunk cannot carry a meaningful timestamp either, so this is plausibly one root
  cause with #2159's chunking bug. The client dedupe is **presentation, not a fix**.
- **#2164 is recorded, not repaired.** The fix is a corpus re-derive — other thread, deploy-gated.
  **Re-run ONE episode before any batch**; that is also the cheapest test of whether it still ships.
- **Person roles are unverified against real data.** If chips render unbadged, the KGs do not
  populate `role` — that is data, not the wiring.
- **The absolutiser guard polices the CLIENT only.** A new server builder attaching a photo to a
  payload no existing fetcher handles would still slip through.
- **No unit test for the co-appearance role collection loop** (`4df86aab6`). The aggregation reuses
  tested code; "every shared episode's role reaches the list" is only exercised via integration.
- The original four remain: **#2156, #2157, #1595 parts 4–5, #1978**.

## The pattern — the most useful thing here

**Nine defects this session were invisible to 1,714 passing tests.** Every one was found by the
operator on the device. jsdom models neither the top layer, nor iOS hover latching, nor real layout —
and the browser tier **structurally cannot** see the relative-URL bug, because on the web the origins
match and it does not exist.

Four of the nine were mine:

- **Three wrong fixes for the collection names**, each reasoning from a Chromium repro I trusted over
  the phone. A diagnostic build — tint the element, screenshot it — settled it in two minutes and
  should have been the FIRST move. The elimination table is in
  `DEVICE-ONLY-DEFECTS-2026-09-27.md`; do not re-run those ten dead ends.
- **A glyph collision created while fixing one**, hours after writing the rule against it.
- **A regression three of my own tests passed over** (the top layer).
- **Three near-vacuous tests**, one nearly recorded as mutation-checked when the mutation had
  silently failed to apply. Every guard added this session is mutation-checked because of that.

The recurring shape is *a check that looks like it is checking something*: a test splitting on a tag
the file no longer contains; a sort-key test rebuilding the expression instead of calling it; a grep
against the wrong chunk. Assume it of your own work.

## Where to start (session 2)

1. **Push, or decide not to.** Nothing has left the machine.
2. The scroll A/B — zero code, discriminates the one live theory.
3. #2164 step 2: re-run GI on one affected episode.
4. Then the original four.
