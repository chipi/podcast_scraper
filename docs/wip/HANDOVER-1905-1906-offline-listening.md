# Handover — the offline arc and what grew out of it

**Date:** 2026-09-03
**Branch:** `feat/player-offline-downloads` — **61 commits, not pushed, no PR**
**Supersedes:** the 2026-09-02 revision of this file, which described 20 commits and asserted a
green Playwright run. That assertion is no longer verified (see §4) and was left standing for 40+
commits — the reason this rewrite exists rather than an edit.

Companion documents:

- `docs/wip/OFFLINE-ARC-GAP-INVENTORY.md` — everything NOT done, and what must close before the
  arc closes. Read that one first if you are deciding whether to merge.
- `docs/wip/1923-collaborative-signal-audit.md` — the collaborative-filtering signal audit.

---

## 1. What this branch is

It started as two sibling features — download episodes to the device (#1905) and make the app
usable with no network (#1906) — and grew into the whole personal-data layer around them, because
each closed gap exposed the next one.

**Closed on this branch:** #1905, #1906, #1908, #1909, #1910, #1911, #1912, #1913, #1914, #1924,
#1925.
**Advanced, still open:** #1916 (Phase 0 only), #1923 (audit + groundwork), #1917 (its Phase 0).
**Untouched:** #1922 (Android), #1915 (OAuth).

### The rule the whole arc enforces

> **Only a 401/403 may destroy cached state. A transport error never may.**

A failed refresh used to sign the user out; a failed GET used to blank a populated view. Both
turned a moment of no signal into data loss on screen. Every store, the auth boot path and the
outbox now follow this.

---

## 2. The subsystems, and where to look

| Area | Entry point | Note |
|---|---|---|
| Download registry | `stores/downloads.ts` | per-account, demotes interrupted transfers to `queued` (no resume) |
| Transfer | `services/downloads.ts` | in-flight map, epoch tokens, namespace guards, `settleDownloadedFile` |
| Scheduler | `services/downloadScheduler.ts` | L1: foreground only; `unknown` connection never starts |
| Offline writes | `services/outbox.ts` | follow, favourite, queue item, capture — all idempotent |
| Content cache | `services/contentCache.ts` | library / favourites / queue only |
| Positions | `services/playbackPositions.ts` | per-account, skew-aware `shouldPush` |
| Listen log | `services/listenLog.ts` | queued offline with the moment it happened |
| Deep links | `services/deepLinks.ts` | closed allow-list; `?t=` carries a moment |
| Recaps | `server/app_recap.py` | pure aggregation; coverage fields are load-bearing |
| Listening time | `app_user_state.accrue_listening` | pure; clamped `[0, 30s]` per save |
| Topic exposure | `app_user_state.append_topic_exposure` | recorded, not derived |
| MCP scope | `mcp/auth.py` | `require_scope`; stdio is local-trust |

### Four production bugs found by the device tier that no web tier could see

1. Native media URLs resolved against `capacitor://localhost` — every image broken on native.
2. `refreshLocalUris()` repaired only whoever was signed in at boot, so a fresh sign-in left
   downloads pointing at a dead container path.
3. **`Filesystem.downloadFile` ignores `directory` on iOS** — bytes landed in `Documents` (which
   iOS backs up to iCloud) while everything else read `LibraryNoCloud`. Every download failed AND
   the audio leaked into user backups.
4. A succeeded download kept a stale `errorKind`, which the drain's retry sweep reads.

---

## 3. Decisions taken, with their reasons

| Decision | Outcome |
|---|---|
| Queue → item-level operations | `POST/DELETE /queue/items`; `move` stays whole-list and still refuses offline |
| Capture idempotency | client-minted `client_id`; replay returns the stored row, 200 not 201 |
| No resume | accepted for L1; revisit with `URLSession`/`WorkManager` |
| Download through the UI on device | done; it found bug 3 above |
| Recap day boundaries | the LISTENER's local day, offset sent per save (right for DST and travel) |
| Recap honesty | show the real number WITH its coverage, never hide it |
| No sharing in v1 | operator-confirmed; rights question on verbatim third-party quotes unreviewed |
| CF: no model | zero measured interactions; topic-level first at ~200 users |
| MCP `mcp:write` | enforced but **not mintable** — corpus writes unreachable over HTTP |

---

## 4. Verification status — read this before trusting anything

| Gate | Last run | Result |
|---|---|---|
| `npm run test:coverage` | current | 100 files / 871 tests, exit 0 |
| `npm run build` | current | exit 0 |
| Python unit + integration (server, mcp) | current | 2227 passed |
| `make lint` | current | 0 / 0 |
| mypy (CI-matching invocation) | current | no issues, 1861 files |
| `make format-check` | current | passes — **was failing for most of the branch** |
| `make lint-markdown`, `check-doc-structure`, `docs` | current | pass — markdown **was failing** (59 errors) |
| `make check-test-policy`, `check-unit-imports`, `docstrings`, `security-bandit` | current | pass |
| **Playwright (`make test-app-e2e-docker`)** | **NEVER on this branch** | the previous revision of this file claimed green; that was 40+ commits ago |
| **Full Python suite (`make test`)** | **never** | only server + mcp directories were run |
| **`make coverage-enforce`** | **never** | branch adds ~7,000 lines |
| Device tier (`make test-app-ios-journey`) | `315fe181` | passed then; ~17 commits have landed since |
| `security-audit`, `complexity`, `deadcode` | **never** | |

**An attempt to run Playwright on 2026-09-03 was abandoned:** the machine was saturated by another
worktree (12 cores, load 16, the docker VM at ~490% CPU) and the containerised API died mid-run,
failing every spec. That is the failure mode `AGENTS.md` warns about — a saturated machine looks
like broken code. Re-run when the box is quiet; do not read the abandoned run as a result.

---

## 5. Running it

```bash
make test-app-e2e-docker        # browser tier (needs docker + a quiet machine)
make test-app-ios-journey       # device: download through the UI, then the offline journey
make ios-origin-up              # one origin serving /api and /audio, for the simulator
make stack-test-reap            # ALWAYS after stack work — reap what you start
```

The device tier needs Xcode, a simulator runtime, CocoaPods and xcodegen. It never runs in CI.

---

## 6. If you are picking this up cold

1. Read `OFFLINE-ARC-GAP-INVENTORY.md` §5 — the list of what must close.
2. The branch has never been rebased and other work is expected on `main` soon.
3. Two product questions are the operator's and block nothing else: is per-user *write* an MCP
   goal (#1916), and where does collaborative signal belong in the product (#1923).
