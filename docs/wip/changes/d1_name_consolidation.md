---
id: d1_name_consolidation
class: full
status: open
issue: none (RFC-131; plan in podcast-scraper-eval-data docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md)
touches:
  - src/podcast_scraper/gi/pipeline.py
  - src/podcast_scraper/providers/ml/diarization/roster.py
  - src/podcast_scraper/speaker_detectors/*.py
  - src/podcast_scraper/upgrade/cli_handlers.py
---

# d1_name_consolidation — the name-consolidation arc's first deploy: one home for identity, raw labels never shown, guards and the one-word rule inside the join

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Covers the code of
RFC-131 steps S1b (one home, the identity ledger), S3 (raw labels never shown) and S4 (guards
inside the join, the one-word rule), on branch `feat/name-consolidation`. Migration N11 (raw labels
out of the files) has its own record. Person names are kept out of this public record; the pairs
and names are in the eval-data repo under `data/eval/name-consolidation/`, cited by path.

## 1. Intake

- RFC-131 (person identity, one home), ADR-166 (the ledger; no joins without it), ADR-167 (guards
  inside the join): the design this change implements.
- The arc plan V3 (eval-data `docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md`): steps, measures
  R1–R7, the operator's decisions.
- #852 (the cross-episode map), #904 (its predicate), #2056 (one-token drift), #1978 (no regex on
  placeholder ids, superseded for two shapes by S3): the code this replaces or extends.
- #2352 (open: near-identical-name merge, among three non-English defects): the class S4 refuses.
- #2403 (Phase 5: the roster's same-person rules): out of this change, next.
- #2197 (Person nodes that are not one person), #2219 (entity registry by evidence): related, not
  addressed here.
- #2333 (cross-episode voice identity): not used; no voice embeddings in this arc (plan V3).
- #2069 (reversible role migrations, a ledger): the ledger here is for identity joins only.
- Measured before this record: the S2 truth set (192 pairs, two blind readers, controls 10/10),
  G0 (the deployed map scored on it), S3's prod report runs, S4's corpus-wide export and replay.

## 2. Outcome and baseline

Own numbers (definitions in `quality/name_report.py` and eval-data `s4/s4_score.py`):

- **R1** raw labels on shown surfaces. Prod, image sha-48e3945, corpus token 1793330465.0899494:
  insights 66 in 21 episodes, transcript segments 222,461 in 732 episodes (report c57479f).
- **R3b** wrong merges on the truth set: S1 train 1 (P159), S3 train 1 (P013, residual). Held-out
  0. (G0, eval-data `g0/g0_score_a89e53f.json`.)
- **R3a** truth-set splits: S1 held-out 12, S1 train 10 (G0 ids, S4 export).
- **Corpus-wide person joins:** 760 (export `s4/s4_export_a89e53f.json.gz`, sha256
  31f0e745…a54a18, prod run 2026-10-30 21:32Z, same corpus token).
- **Scoreboard (S1, S1n, S1c, S2, S3, S10a/b, S12, A1): not yet measured for this change.** It is
  taken before and after on the step-7 snapshot clone. Until then this section has no scoreboard
  baseline.

## 3. Diagnosis by loss

- Raw labels shown: R1 above, one cause (no predicate on the display path), counted per surface by
  the report.
- Wrong merges: the corpus-wide export's 760 joins replayed and read join by join (eval-data
  `s4/RESULT-2026-10-30.md`). Causes by count of wrong joins: a chain through a bare surname or a
  title-stripped one-word name (14 of the 18 joins the no-bridge rule removes; 3 false splits, 1
  read as one person by the operator), a fuzzy one-word vs one-word join (16 of 23 by my reading;
  the operator read one of those as one person), two people in one episode with no voice (guard c:
  3 right, 5 false splits), one person's two spellings on two voices (guard a: 1 false split, by
  the operator's reading).

## 4. Prediction

MISSED: no prediction was written before the code. S1b, S3 and S4 were built and measured (S3's
prod report, S4's replay) before this record existed; the method's hook and pre-commit check did not
run on this worktree (the hook points at a main checkout without `scripts/hooks/`, and no git hook
is installed). A prediction written now would describe known numbers, not predict them.

Operator accepted: 2026-10-31 steps 4–5 missed for S1b, S3 and S4; steps 7–8 in full on a fresh
snapshot clone.

What is known before step 7 (measured, not predicted), and what must not move:

- R1 shown surfaces 0 (prod report c71876a).
- Person joins 760 → 720; 40 removed, 11 moved, 0 added (S4 replay, gate PASS).
- Truth set: S1 train wrong 1 → 0; S1 held-out joined 21 → 19, train 35 → 34; S3 residual 1.
- Must not move: everything outside person joins and raw-label display (orgs' joins, insights'
  visibility except the 66 raw-label insights becoming unattributed, quotes, transcripts' text).

## 5. Acceptance

MISSED: see section 4. Operator accepted: 2026-10-31.

## 6. Build

Commits on `feat/name-consolidation` (S1b … S4): 042c8d6af and the S1b series, 1802035eb (S3),
a89e53fe2 (report), 855a12da6 (guards), a41530cbb (one-word rule, no-bridge linkage), 774de1a28
(non-Latin names, report reads the ledger), c5bc10c77 (ledger undo), and the ledger step (below).
- **The ledger step on every corpus-writing path** (plan V3: "the last stage of every
  corpus-writing path"), added 2026-10-31 after the D1 inventory found it unwired: only the
  `identity-ledger` command wrote the ledger, so episodes ingested after the deploy would have had
  no cross-episode joins until a hand-run build. `identity.ledger.refresh_ledger` runs after a
  successful `enrich` run (the post-pipeline chain, #1653, queues one coalesced run after every
  ingest and reprocess batch, so one build per batch, not per feed), after an `upgrade run` that
  applied a migration, and after an `enrich-edges` pass that rewrote artifacts. A failed build
  leaves the previous ledger and fails the command with exit 4 (never silent). `enrich
  --no-identity-ledger` opts out. Tests: `test_ledger_refresh.py` 11 passed; 6 mutants killed (each
  hook off, the upgrade hook on a dry run, enrich-edges always rebuilding, a swallowed failure).
  It writes only `identity_ledger.json` and its change log and previous copy; no content file.

## 7. Measure

Write rehearsal, 2026-10-31, on a copy of **snapshot-prod-20261030** (release 11:48Z, archive
sha256 7e366e70…7035f; corpus token mtime 1793330465, newest_kg_count 3069, equal to prod's at the
S2–S4 runs), outside `/tmp` (`/Users/claude/snapshots/`, 105 GB free before). The copy is the
0038 change gate's: 0038 applied, then `enrich --only insight_density,insight_sentiment`
(24 episodes, 48 sidecars), the after state of D1's corpus steps. Code: the branch at 22762f1c2
(ledger step) for the ledger and warm-up runs, b0d1a34bb–476c0002d after (docs and two gate fixes).

| Check | Result |
|---|---|
| ledger build (`identity-ledger --dry-run`) | 93.4 s; 949 joins: 720 person, 229 org; 9 blocked joins |
| person joins equal the S4 replay's final map | yes: sha 1f48d22a02b5 both (so 0038 changes no join) |
| raw `person:speaker-*` ids in any join | 0 |
| server's served map (`cached_entity_id_map`) equals the ledger | yes |
| corrupted ledger | state `unreadable`, 0 joins served, a warning logged (ADR-166) |
| undo | second write kept `identity_ledger.previous.json`; after corrupting the current one, `restore_previous` → state ok, 949 joins |
| api cache warm-up, branch | 118.4 s (entity map 0.0 s from the ledger; reel_clips 101.3 s) |
| api cache warm-up after a ledger write | 123.5 s (one rebuild of the projections) |
| api cache warm-up, main 5d2b3d6b5, same copy | 144.5 s (entity map 28.3 s built in memory) |

Outcome rows measured on the copy so far (`name_report`, eval-data
`data/eval/name-consolidation/d1/name_report_after_clone.json.gz`), against prod's report c71876a
(same corpus token, before 0038 and before the ledger):

| Row | Before (prod, c71876a) | After (copy) |
|---|---|---|
| R1 shown surfaces | 0 (S3) | 0 |
| R1 data: raw-label quotes / bridge identities | 107 in 23 / 28 in 24 | 0 / 0 |
| R2 roster pairs a+b, groups | 8+98, 71 | 8+98, 71 |
| R2 voice pairs a+b, groups | 6+86, 58 | 6+86, 58 |
| R4 identical names across shows (joined), variants (joined) | 95 (86), 55 (3) | 95 (86), 55 (3) |
| person joins | 760 (in-memory map) | 720 (ledger) |
| R3 truth set, S1 train wrong merges | 1 | 0 (S4 replay) |

**Whole-D1 scoreboard** (eval-data `d1/d1_measure.sh`): main 5d2b3d6b5 on snapshot-prod-20261030
(before) against the branch on the copy with 0038 applied (after).

| # | Before | After | Δ | Cause, by identity |
|---|---|---|---|---|
| S1 | 17,811 | 17,860 | +49 | 0038's N4 `unknown` set (accepted, record 0038) |
| S2 | 31,421 | 31,474 | +53 | 0038's N4 set (accepted) |
| S3 | 9,217 | 9,219 | +2 | 0038, episode 987ebd8e (accepted) |
| S10a | 191 | 194 | +3 | 0038: 631e269e (accepted). **S3's read rule: 2 episodes** whose every visible insight was credited only to a raw label, now unattributed: art19 `41402031` run 349d0b7e "0010 - Why the neoclassical philosophy", omny `c9fdce2d` run fb659a6a "0001 - AI Agents Are Already Changing" (their panel fallback still shows them: S10b unchanged). Operator accepted: 2026-10-31 |
| S1n, S1c, S10b, S12, A1 | 7, 0, 2, 24, 396/297 | the same | 0 | |

**Gold (S7, `scripts/measure/gold_insights.py`, gold in `.test_outputs/naming_gold`):** correct and
wrong-person unchanged on every set (val_v2 correct 3,707, wrong 439, unnamed 1,109; dev_v1 and
val_v1 likewise). Three insights outside the scored classes move from shown to hidden
(quote not placeable 2, scored-no-name 1). **S6 (naming gate)** is not run: D1 renames and unnames
no voice (0038 writes no segment, transcript or roster), so it would read "no effect".

**Unit suite** (`make test-unit`, branch 07133f738): 17,226 passed, 62 skipped, 1 failed: the
landing test, refusing 0038 until its step 8 and outcome exist (expected at this point).
Cheap gates (format-check, lint, type, docstrings, spelling, check-unit-imports,
check-test-policy) PASS after 476c0002d; `make docs-check` PASS.

**Added after the first step-8 read (NO-GO, 2026-10-31), each item it found missing:**

- *Gold moves by identity* (eval-data `d1/gold_changed.json`): the 3 insights moving shown →
  hidden are dev_v1 `e8457907…` `insight:2203f596755b967a`, val_v1 `1_2721666`
  `insight:8f5207da04583268`, val_v2 `https://iono.fm/e/1726016` `insight:1e582c4abc31bf37`; all
  three are in 0038's accepted set of 53 (record 0038, N4). None is a correct or wrong attribution.
- *Organisation joins unchanged:* main's in-memory map on the snapshot and the ledger on the copy
  both hold 229 org joins, sha `bfb00faa2eec` (eval-data `d1/orgs.py`).
- *The 9 blocked joins, listed* (ids carry names, so in eval-data `d1/ledger_blocked_joins.json`):
  the same 9 the S4 replay recorded (record S4 result: 8 mention co-occurrence, 1 co-voice).
- *The two S10a episodes, stored:* `d1/s10a_zero_visible_main.json` (191) and
  `d1/s10a_zero_visible_branch_before_0038.json` (193); the difference is exactly the two named
  above (eval-data `d1/zero_vis.py`).
- *A second ledger build changes nothing:* `identity-ledger --dry-run` on the copy after the write:
  949 joins, 0 added, 0 removed, 0 moved, previous state ok.
- *The person-join changes by identity:* 40 removed and 11 moved, each with a verdict, in eval-data
  `s4/s4_reviewed_changes.json` (the S4 gate requires the list to equal the map's changes).
- *Playwright* (phone size, `npx playwright test transcript-speaker-labels entity-pages
  episode-notes-people entity-and-rails-invariants`, branch 30303f269 tree, its config writes the
  ledger before the api starts): **43 passed (2.1 m)**.

**The code that ships:** aaf08c6e8, the last code commit (a test fix for S3's segment contract,
found by ci-fast); the migration code is 30303f269's, plus 0038's outcome file (085594053). Later
commits are records and docs only.

**Gates on aaf08c6e8**, run in a fresh worktree (`/Users/claude/snapshots/branchsrc-fresh`, no
leftover generated files, like CI's checkout):

| Gate | Result |
|---|---|
| `make ci-fast` | **PASS** (`MAKE_EXIT=0`; unit 17,168 passed, integration-fast 2,185 passed, e2e 244 + 40 passed; 0 failed) |
| `make test-integration` | 4,604 passed, **1 failed: `test_player_nginx_syntax_ok`**, which runs `docker` on the production socket (closed to development work here); the branch does not change `web/learning-player/nginx.conf` (0 diff lines); CI runs it |
| player Playwright, full suite | **702 passed**, 22 skipped, 0 failed |
| player Playwright, D1 specs at phone size | 43 passed (above) |
| viewer Playwright, full suite | 322 passed, **1 failed, 1 flaky**: `library-pagination-ranking-keyboard` "Load more appends the next page" (a row intercepts the click until timeout) and `artifact-list-load-graph` (browser launch failed, passed on retry). Alone on the branch the pagination spec passes (4 passed, 1 flaky); on main 5d2b3d6b5 the same spec fails too (1 failed, 1 flaky, 3 passed; another of its tests, the same pointer-intercept timeout and 2 ms launch failures). The branch changes no viewer code (only `playwright.config.ts`, the ledger before the api) |

Two earlier runs are not counted, and why:

- the first ci-fast (working worktree) failed one test, `test_segments_endpoint_returns_contract`,
  which pinned the pre-S3 segment shape: fixed in aaf08c6e8;
- two test-integration runs failed the tier-3 search tests (`test_cluster_hits_…`,
  `test_consensus_pairs_…`) on a fixture index whose unseeded IVF build reorders near-ties
  (measured: main 5d2b3d6b5 fails on the same index 2/2; a rebuilt index passes); not this
  change; the fix ships in D2 (operator, 2026-10-31). A third run failed 22 tests because I
  moved the tracked `search/topic_clusters.json` aside with the index (my error); restored, the
  run above is the next one.

**Builder's verdict: GO WITH CONDITIONS.** Deciding numbers: wrong merges refused corpus-wide 25
by my reading (none confirmed by the operator from the cards; 1 disagreement); the truth set's
reachable wrong merge P159 fixed; R1 0 on every surface including the files; every scoreboard move
listed by identity and accepted. Conditions: `make ci-fast` and `make test-integration` green on
the shipping tree; the viewer and player suites once; the prod plan diff at step 10 (22 episodes
ingested since the snapshot).

## 8. Verdict

**First read (2026-10-31, fresh agent): NO-GO**, on missing items: the builder's verdict, the
not-covered list, the 3 gold moves by identity, the org joins unchanged, the 9 blocked joins
listed, the S10a artefacts, ledger idempotence, the shipping sha, and the gates not run. Each is
answered in section 7 ("Added after the first step-8 read") and the gates table.

**Second read (2026-10-31, another fresh agent; not the builder, not the first reader), condensed:**

```text
REVIEW d1_name_consolidation 2026-10-31 commit 2b1bbfbb3 (shipping code aaf08c6e8; no src diff after it) reviewer Sonnet (fresh) minutes ~6
PREDICTION vs GATE: no prediction (MISSED, accepted); the scoreboard diff shows exactly the moves listed (S1 +49, S2 +53, S3 +2, S10a +3), each by identity with an acceptance; S1n, S1c, S10b, S12, A1 unchanged; gold 3 moves in 0038's N4, correct/wrong unchanged
REQUIRED ITEMS: refusals listed by identity; code clean aaf08c6e8; ledger undo and second build recorded; census: the ledger writes only its own files (0038 owns the file census); not-covered present (13 items); ci-fast MAKE_EXIT=0; no per-verdict deciding segment for the 51 joins
AFTER-STATE READ: from the read list and the operator's reading, not from served-text excerpts: at least 15 false splits accepted against 25-35 wrong merges refused, none confirmed by the operator from the cards; the direction matches "a wrong name is worse than a hidden one"
OPEN CALLS: titles (Mayor, Senate President) deferred; guard (c) 3 right vs 5 false splits kept on principle; held-out shows only cost; guard (b) unexercised; 22 post-snapshot episodes; S6 not run; nginx test acceptable (CI must be green; its docker guard failed instead of skipping, unexplained); viewer spec acceptable (fails on main too; green or no worse in CI)
DOUBTED NUMBER: none
VERDICT: GO WITH CONDITIONS
```

Verdict: GO WITH CONDITIONS

- **Condition 1:** CI green on push for `test_player_nginx_syntax_ok` and the viewer suite (or no worse than main).
- **Condition 2:** the operator's explicit acceptance, as a set, of the at-least-15 false splits and of guard (c)'s trade (3 wrong merges refused, 5 false splits).
- **Condition 3:** step 10's prod plan diff covers the 22 episodes ingested after the snapshot.

Operator accepted: 2026-10-31 condition 2, as a set: the at-least-15 false splits and guard (c)'s
trade (3 wrong merges refused, 5 false splits). The operator also asked that they be tracked and
monitored on prod: every changed join of `s4/s4_reviewed_changes.json` (40 removed, 11 moved: the
refusals and the false splits) is read as served on prod after the deploy (step 11), and the
same list is re-read after each later ledger build until D2.

Operator decided: 2026-10-31 `NAMING_ARC_MARKER` stays in D1 (removing it would change 0038's
code digest); it is removed in D2.

Open, not a condition: the nginx test's `_docker_is_usable` guard returned true here although `docker run` then hit the closed production socket; why `docker version --format {{.Server.Version}}` succeeded is not diagnosed.

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

- The method's gates did not run on this worktree, and I did not check the method when S3 and S4
  started. The change: until the hooks run on this worktree, `quality.change_gate.check_edit` is
  run by hand before each content-code edit and each commit (it refused 0038's paths until the
  record existed, and allowed them after).
- Missed once more, 2026-10-31: `upgrade/cli_handlers.py` (the ledger hook) was edited before this
  record listed it; the hand check run before the commit refused it, and it was added under
  `touches` before the commit.

## Not covered

- **The reading of the 51 changed joins is mostly mine.** The operator read 10 cards: 2 agree, 1
  disagrees (Yushin / Yushan, read as one person: a false split by the one-word mode), 4 can't tell
  from the corpus evidence shown (2 of them I judged from who the names are), 3 of my 5 unclear read
  as one person (false splits). No refusal was confirmed by the operator from the cards alone.
- **At least 15 false splits are accepted** (one person shown under two names): 11 by my reading
  plus 4 by the operator's. Titles missing from the rule's list (Mayor, Senate President) cause 2;
  adding them changes mode `cluster` everywhere and is not done.
- **The held-out truth set shows only cost** (S1 held-out joined 21 → 19; it held no wrong merge).
  The gain is corpus-wide and rests on the reading above.
- **Guard (b), publisher list, is not exercised by prod data** (switching it off changes nothing);
  unit tests only.
- **The ledger step on prod's real chain is not rehearsed:** the hooks are tested in unit tests and
  the enrich hook ran on the copy (24 episodes); the nightly's post-pipeline chain with the new image
  is first exercised on prod.
- **Cross-show joining (R4) does not move;** that is S8.
- **The prod corpus moved after the snapshot:** 22 new served episodes (3,069 → 3,091, 2026-10-31
  nightly, written by the deployed pre-S3 code): their joins and any raw-label Persons are not in
  this measurement; prod's plan diff at step 10 covers 0038, and the ledger build on prod covers
  the joins.
- **R3a / R3b / R6 as report rows:** the report does not compute them yet; the S4 replay and the
  read list stand in.
- **`test_player_nginx_syntax_ok` is not run here** (production Docker socket); the viewer's
  `library-pagination-ranking-keyboard` spec is unstable on this machine on main's code too.
- **The 5 episodes failing 0025's verify** are corpus state on main's code too; they are S6's.
