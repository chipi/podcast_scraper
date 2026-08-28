# Handover — `fix/v2.7-bug-fixing` PR + what to pick up next

**Date:** 2026-08-28
**Branch:** `fix/v2.7-bug-fixing` (rebased on `origin/main`, **not pushed** at time of writing)
**Author of pass:** Claude (opus) session, continued across compaction
**Purpose:** After this PR merges the session is being cleared. This is the durable record of what
landed, what did **not**, and the recommended order for the next PR.

---

## 1. What this PR contains (14 issues resolved)

A bundled **bug + spec-drift cleanup** branch. Every fix has a mutation-verified test (a test that
was confirmed to FAIL without the fix). All changed-area Python tests green (`453 passed`), viewer
`npm run build` (vue-tsc strict) + `614` vitest green, `make docs` (mkdocs strict) clean.

| Issue | Fix | Commit |
| --- | --- | --- |
| #1483 | 503 (not uncaught 500) on appdata file-lock `PermissionError` | `e9d82b5` |
| #1570 / #1564 | join worker threads before provider cleanup on the exception path (thread storm) | `ec7bc7a` |
| #1865 | `index_corpus(rebuild=True)` must delete the fingerprint sidecar, not just the index | `872f4ab` |
| #1862 | GI+KG bare-name scoping written both-or-neither (def 1) + durability locks (def 2) | `2127d8f`, `5033400` |
| #1864 | missing index at `index --stats` is a WARNING, not an ERROR | `bdb8a90` |
| #1454 | validate RSS URL at `run_pipeline` entry, not deep in the run | `ecd3604` |
| #1852 | bound each chat call with a per-request transport timeout | `e24740c` |
| #1855 | orphaned (never-matched) repair ids are a WARNING, not an ERROR | `21caa12` |
| #1854 | name the culprit feed(s) in the multi-feed failure summary | `1bd88ed` |
| #1546 | guard `/index/timeseries` against the LanceDB read error too | `e8f4ae3` |
| #1600 | UXS-009 timeline is oldest-first (doc) | `b9dd9a1` |
| #1599 | UXS-014 Library tabs match shipped code (Following·Saved·Collections·Revisit) | `d0cc52a` |
| #1598 | **built** per-show adaptive accent (artwork extraction + WCAG ≥4.5:1 clamp + wiring) | `627/8ea` |

Plus closed with rationale (no code): **#1556** (working-as-designed — reprocess, not a bug) and
**#255** (pre-release checklist already exists: `RELEASE_PLAYBOOK.md` + ADR-031 + `make pre-release`
/ `make bump`).

WIP triage records committed under `docs/wip/`: `ISSUE-CLEANUP-2026-08-27.md`,
`V2.7-BUG-TRIAGE-2026-08-27.md`, `TRIAGE-FLEET-HANDOVER-2026-08-27.md`,
`OPERATOR-SMOKE-TEST-PLAN.md`.

---

## 2. What is NOT in this PR — next-PR candidates

### 2a. The UX overhaul epic — #1596 (the big arc)

**13 approved items, 2 closed (#1585, #1591), 11 open.** This is the main body of next work. The
epic issue carries `file:line` evidence and a phased sequence; reproduced here so it survives the
session clear:

**The one that outranks the rest — #1587** "playback stops when you navigate." The only `<audio>`
is `PlayerView.vue:530`; leaving an episode unmounts it. The learning features are most valuable
*mid-listen*, and every such journey currently costs the audio. Groundwork done: state/transport/
MediaSession are already centralised in `stores/player.ts` (its `:18-19` comment names promoting the
element to a store-owned detached `Audio()` as the next step). **High regression risk** (player,
MediaSession, Capacitor shell, Android background audio, test-enforced invariants) — land it **alone,
on its own branch, e2e green.**

Epic's suggested order:

- **Phase 0 — answers, not builds (gate later work)**
  - #1586 — audit whether episode insight bullets are distilled insights or raw transcript turns
    (gates the always-on-insights option in #1583)
  - #1585 — *(CLOSED)* "Your shows" identity question — resolved; was gating #1584
- **Phase 1 — cheap sweep (small, independent, most of the visible quality jump)**
  - #1584 — shared `ShowTile` + fix four unclamped-label sites
  - #1583 — episode card: expand-in-place, delete the whole-card summary overlay
  - #1588 — search in the primary navigation (currently one entry point)
  - #1592 — capture loop: visible on touch, with a destination *(see §2b — I was asked to explain
    this one; it is the "save/collect an insight" flow that is invisible on touch)*
- **Phase 2 — medium, independent**
  - #1589 — merge the two trending modules, delete the shipped A/B switcher
  - #1590 — sign-in teasers instead of hidden controls (signed-out users see no evidence the
    differentiators exist)
  - #1593 — unify the two parallel save systems on the same row (capture vs favourite)
  - #1595 — make the learning differentiator legible in 30 seconds
- **Phase 3 — structural, on its own branch**
  - #1594 — bottom tab bar + IA cleanup
  - #1587 — global audio host + persistent mini-player *(the top-priority item above)*

**Recommended entry point for the next PR:** do **Phase 0 first** (#1586 is an answer, not a build,
and it unblocks #1583), then the **Phase 1 sweep** as one themed branch. Keep #1587 and #1594
(Phase 3) each on their own branch — they are structural and risky.

### 2b. #1603 — cluster vocabulary "Storyline collapse" (deferred from THIS PR)

Code ships **Theme** (co-occurrence) + **Similar** (semantic); UXS-013 spec'd Theme = semantic
(deliberately swapped in the #1678 pass). Operator chose the **"Storyline" collapse** — adopt one
word across all cluster concepts — then deferred it to the next PR because it is a cross-surface
vocabulary redesign (viewer copy + specs + tests), not a doc-only fix. **Do it as its own change**,
not folded into the UX sweep.

### 2c. #1605 — spec-drift audit epic + #1604 (the process fix)

`docs/uxs/` vs code drift was found in every spec area. Three of its code/doc children are now
closed by THIS PR (#1598 accent, #1599 Library tabs, #1600 code-side bugs); its sibling #1603
remains (above). The load-bearing open item is **#1604 — "make specs and tests fail
together"**: the reason drift recurs is that tests encode the code's version and in two cases
*enforce* the drift (`LibraryView.test.ts:83`, `NodeDetail.test.ts:206-213`), so a spec that
contradicts a passing test is already dead. #1604 is a process/CI change (a spec-referenced test
must amend the spec in the same PR). Worth doing early in the next arc so the UX work above does not
re-introduce drift.

### 2d. Parked decisions (need an operator ruling before I can execute)

- **#1849** — cold-storage upload produces truncated / byte-inconsistent copies (~5.7 MB size
  drift). Advisor review suspects it is **misdiagnosed** (expected: dynamic-ad re-encode / legitimate
  size variance, not corruption). Needs a **~5-min read-only prod diagnostic** to decide
  *reclassify-as-not-a-bug (doc)* vs *harden the upload (code)*. Operator asked to keep it aside for
  now.
- **#1395** — transcript cleaner over-strips (36k→9k chars, 26.4%). A precision/recall **threshold
  tuning** decision (current 0.20) — needs a ruling on the tradeoff before changing.

### 2e. Kept open, no action

- **#1345** — single unsymbolicated iOS crash (SIGABRT), no repro, one occurrence. Left open to
  accumulate more samples before deciding.

### 2f. Deep native crashes (separate, uncertain, each its own effort)

- **#1323** — GI embedding evidence backend loads all-MiniLM on **MPS → flaky SIGSEGV** (3rd
  occurrence). Native crash; see memory note on pyannote/MPS. Not reproducible on arm64 vs x86 in the
  usual way — budget real investigation time.
- **#1346** — fatal native crash in `android::Looper::pollOnce` on the Android dev build.

### 2g. #627 — manual viewer QA with screenshots (parked)

A **manual** pass on the viewer (feeds API, operator config API, pipeline jobs UI + health flags)
with screenshots attached, plus a holistic code/test/doc gaps backlog. Cannot be executed by an
autonomous agent (needs a live viewer + human-captured screenshots). Left in the v2.7 milestone as a
**manual session** task — pick a hands-on sitting to walk the surfaces and attach screenshots.

---

## 3. Suggested plan for the NEXT PR

1. **#1604 first** (process): make specs + tests fail together, so the UX work below can't
   re-introduce drift. Small, high-leverage.
2. **#1596 Phase 0** — #1586 (insight-bullet audit; an answer, not a build). Unblocks #1583.
3. **#1596 Phase 1 sweep** as one themed branch — #1584, #1583, #1588, #1592. This is where most of
   the visible quality jump is, and the items are small + independent.
4. **#1603 Storyline collapse** — its own change (cross-surface vocab), ideally after #1604 lands so
   the spec + tests move together.
5. **#1596 Phase 2** — #1589, #1590, #1593, #1595.
6. **#1587 and #1594 each on their own branch** (Phase 3, structural, high regression risk — e2e
   green, land alone).
7. When you have bandwidth for hard native work: **#1323**, **#1346**.
8. Decisions to make (then I can execute): **#1849** (run the read-only prod diagnostic),
   **#1395** (threshold).

---

## 4. Mechanical notes for whoever resumes

- **Playwright/e2e prerequisite:** the full suite needs the search index, which needs the `[search]`
  extra installed. (Noted in #1596's testing-prerequisite section.)
- **Local `NODE_OPTIONS` gotcha:** a stale `--require=.../restore-node-options.cjs` preload crashes
  node/npm/vue-tsc/pre-commit-markdownlint. Run `unset NODE_OPTIONS; export NODE_OPTIONS=""` in the
  same shell as any node/git-commit invocation.
- **Local `ci-fast` caveat:** cannot be green under the ML `.venv` (ML breaks the no-ML dedupe
  guards). Use `PYTHON=.venv-dev` (`.[dev,llm]` + docs); ML e2e skips via `@requires`. See memory
  `reference_ci_fast_local_no_ml_recipe.md`; tier-split tracked in #1805.
- **Mutation verification:** `git stash` is BROKEN for this (no-ops on committed clean files, and
  collides with a pre-existing stash). Use edit-in-place + `git checkout HEAD -- <file>` (committed)
  or temp-backup `cp` (uncommitted).
- This branch is **held for a rebase**: the operator is landing commits elsewhere and will ask to
  rebase `fix/v2.7-bug-fixing` on top of them, then build the PR. Do NOT push until told.
