---
id: c2_episode_rerun
class: light
status: open
issue: none (ingestion arc, tier C: C2)
touches:
  - src/podcast_scraper/upgrade/corpus_selection.py
  - src/podcast_scraper/upgrade/outcome_hook.py
  - src/podcast_scraper/upgrade/runner.py
  - src/podcast_scraper/upgrade/cli_handlers.py
  - tests/unit/podcast_scraper/upgrade/
---

# C2 — `upgrade rerun`: an applied migration again, for named episodes, under the outcome hook

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Arc list
`docs/wip/INGESTION-ARC-2026-11-03.md`, tier C, item C2. Ids only in this record.

## 0. Why light

The deploy that carries this writes no stored file: nothing runs it. The default path (`upgrade
run`, `upgrade verify`, every reader of `select_served_artifacts`) is unchanged when no scope is
set. When an operator runs `upgrade rerun`, what it writes is the named migration's own output,
for the named episodes only, measured by the outcome hook as `upgrade run` measures it. Every
such run is a corpus change of its own and needs its own record and go.

## 1. Intake

- **The gap**: no way to re-run an applied migration
  for named episodes. Today that means calling the class from Python in the api container. That
  path has no plan check, no outcome hook, no lock and no receipt of the run.
- **When it is needed:** a reprocessed or re-enriched episode gets new artifacts from the
  pipeline, and a migration already applied (0029, 0034, 0043 and others) has to be applied to
  it again. `upgrade run` refuses an applied migration, and the episode is not in that
  migration's measured plan any more.
- **Machinery read:** `upgrade/runner.UpgradeRunner.run` and `_apply_measured` (the lock, the
  regime), `upgrade/outcome_hook.prepare`, `check_accept`, `measure`, `episode_plan`,
  `score_by_episode`, `upgrade/corpus_selection.select_served_artifacts` (used by 0020–0025,
  0027, 0029–0031, 0034–0036, 0038–0042, and 0043 through 0034's class), `upgrade/cli_handlers`.

## 2. Outcome and baseline

- **Outcome:** `podcast_scraper upgrade rerun --migration <id> --episodes <file>` applies an
  applied migration to the named episodes only, with the same safeguards as `upgrade run`: the
  corpus lock; the installed code digest equal to the outcome's; the per-episode plan compared
  with the outcome's rows for those episodes; any difference stopping the run until an accept
  file gives each a cause and the operator's go; the per-episode scoreboard measured before and
  after; a receipt per rerun, not over the apply's receipt.
- **Refusals:** an unapplied migration (`upgrade run` is the path); a migration without
  `dry_run_report`; a plan that names an episode outside the scope (the migration does not
  select through `select_served_artifacts`, so a scope cannot hold it).
- **Baseline:** no such command; `upgrade run` / `verify` behaviour as on `69802a6d`.

## 3. Diagnosis by loss

Not a ranking: one missing capability (section 1).

## 4. Prediction

Dated: 2026-11-04

- **Default path, no scope set:** `select_served_artifacts` returns the same lists; every
  existing upgrade test passes unchanged; 0043's plan on the gate copy and on the snapshot is the
  same (`0 episodes` after, 101 before).
- **Scoped, on snapshot-prod-20261103 with 0043's outcome:** a dry rerun of 0043 for 3 of its
  101 planned episodes plans exactly those 3, with the outcome's rows for them (plan diff empty).
  For 2 planned episodes plus 1 not in its plan, it plans the 2.
- **On the gate copy (0043 applied):** a dry rerun of 0043 for any named episodes plans 0.
- **Refused:** 0043 named while unapplied in the ledger; a migration with no `dry_run_report`;
  a plan row outside the scope (unit test with a migration that ignores the scope).
- **Must not move:** the outcome hook's `upgrade run` path (its tests), the ledger (a rerun does
  not record the migration again), the apply's receipt `.podcast_scraper/upgrade-outcomes/<id>.json`.

## 5. Acceptance

Prediction read 2026-11-04 by a fresh agent (reviewer, Sonnet; not the builder), 5 minutes,
before any code. Verdict: **rows added**. Its rows, each taken into the build:

1. **Static checks for a rerun:** the code digest equals the outcome's; the outcome exists; the
   migration is in the ledger; every migration in the outcome's `applied_before` is in the
   ledger. Otherwise refused. `check_static` for `upgrade run` does not change.
2. **The real use stops at the plan diff:** a reprocessed episode's row differs from the outcome's
   (or is absent from it), so the rerun stops and writes the scoped plan diff until an accept file
   gives each entry a cause and the operator's go. Such an episode is in `differing`, where
   `compare` only refuses a rise.
3. **The original apply's bookkeeping does not move:** its backups, its receipts lines, its
   outcome receipt `<id>.json` and `<id>.plan-diff.json`. A rerun writes its backups under its own
   tag (`<tag>.rerun-<stamp>`), appends its receipts with a `rerun` field, and writes its outcome
   receipt as `<id>.rerun-<stamp>.json`. Its undo is the per-file backups under its own tag.
4. **Writes checked at apply time:** every episode the apply reports in `details["episodes"]`
   must be in scope, and the per-episode scoreboard outside the scope must be equal before and
   after; otherwise `HookOutcomeFailed`. Unit test: a migration whose plan honours the scope and
   whose apply does not. A named episode that is only a superseded copy, or an orphan artifact
   (no metadata sibling), is out of scope.
5. **Scope file hygiene:** refused when empty, when it holds a duplicate line, or when a line is
   not a served episode's metadata path. An empty scope never means "no scope".
6. **Empty scoped plan:** a real rerun whose plan is empty writes nothing (`apply_empty`, R2).
7. **Safeguards of `upgrade run`:** the corpus lock for a real rerun, none for a dry one; a
   rerun names one migration; no corpus snapshot (`snapshot: none` in its receipt), the per-file
   backups are its rollback, as the undo hint says.
8. **Derived surfaces:** a rerun prints the named episodes' derived steps it leaves to the
   operator: the ledger refresh (run after it, as after `upgrade run`), the incremental index
   for those episodes, their enrichment.
9. **Scoreboard scope:** before and after are scored over the whole corpus; every episode outside
   the scope must be unchanged (row 4).
10. **A 0043 rerun** uses 0043's `dry_run_report`, its `BACKUP_TAG` and `RECEIPTS_FILE`
    (through 0034's class).

Step-8 spot checks added by the reader: whole-tree sha manifest outside the scope unchanged after a
real rerun on a copy; ledger byte-identical; the apply's outcome receipt and plan diff
byte-identical; the apply's backups and receipt lines unchanged; 2 planned + 1 unplanned named →
2 written; each hygiene refusal writes nothing; a superseded copy unchanged; one rerun undone
leaves the apply's undo intact; scoreboard outside the scope unchanged; index and bridge rows
listed; lock held → dry run proceeds, real run refused.

## 6. Build

- `upgrade/corpus_selection.episode_scope(root, metas)`: within it `select_served_artifacts`
  returns only the named episodes' served artifacts; an empty scope raises.
- `upgrade/runner._checked_scope` (rows 4, 5) and `UpgradeRunner.rerun`: regime only; the
  migration must carry `_backup_tag` and `_receipts_file` (its copy gets `<tag>.rerun-<stamp>`
  and `<stem>.rerun-<stamp>.jsonl`); the lock for a real rerun; the plan inside the scope, the
  scoring outside it; an empty scoped plan returns without an apply; the ledger is not written.
- `upgrade/outcome_hook.prepare_rerun` (rows 1, 2): outcome, applied, earlier migrations,
  digest; a plan row outside the scope refused; the outcome's rows for the episodes the rerun
  writes; a named episode the outcome planned and that needs nothing now is not a difference
  (found by the tests: it read `clone_only` and stopped a rerun that writes nothing to it).
  `_accepted` is the diff step `prepare` and `prepare_rerun` share; the rerun's diff file is
  `<id>.rerun-<stamp>.plan-diff.json`. `measure` writes `<id>.rerun-<stamp>.json`, records the
  scope, and fails on `_rerun_leaks`: a file in the rerun's receipts that belongs to no named
  episode, or a scoreboard row outside the scope that changed. The undo hint names the rerun's
  receipts and tag.
- `upgrade/cli_handlers`: `upgrade rerun --migration --episodes [--dry-run]
  [--accept-plan-diff] [--json]`; prints the derived steps it leaves to the operator (row 8).
- Tests: `tests/unit/podcast_scraper/upgrade/test_upgrade_rerun.py`, 14, on a corpus on disk
  (the scope, each refusal with an unchanged tree, a superseded copy, dry rerun, a rerun's tree
  diff keeping the apply's backups, receipts and ledger, the stop at a changed episode, an apply
  that writes outside the scope, an empty scoped plan, a plan that ignores the scope, the CLI).
  Mutation: 11 of 11 killed (scratchpad `mut_c2.sh`).
- `MISSED:` the corpus lock leaves its own empty `.podcast_scraper.lock` at the root, as `upgrade
  run` does; a tree manifest that counts it sees one more file after a real rerun.

## 7. Measure

On the gate copy of 0043 (snapshot-prod-20261103 with 0043 applied by the clone gate, code
`c91191da`, outcome `584ff3c0…`), scratchpad `c2_measure.py`: 0043 recorded in the copy's
ledger; the first 2 episodes of 0043's plan put back to the snapshot's bytes (a reprocess
stand-in); scope = those 2 plus the 3rd planned episode, which 0043 had already fixed; a
sha256 manifest of all 145,558 files before and after.

| Row (4 / 5) | Predicted | Measured |
| --- | --- | --- |
| dry rerun, 2 planned + 1 fixed | plans the 2, diff empty | **`rerun plan: 2 of 3 named episode(s), 0 differ from the outcome`**, exit 0, nothing written |
| real rerun | PASS | **exit 0, receipt `0043_…rerun-20261104T163920Z.json` verdict PASS, failures []**, `snapshot: none` |
| files changed outside the 2 episodes | only the rerun's bookkeeping | **0** besides the rerun's own: 52 changed = 25 in the 2 episodes + 27 under `0043.rerun-<stamp>` backups, its receipts file and its outcome receipt |
| the 2 episodes after the rerun | as the gate's apply left them | **31 of 31 files byte-identical** to the gate apply's bytes |
| ledger, other migrations' receipts and outcome receipts | unchanged | **unchanged** (in the manifest, `other []`) |
| default path | unchanged | 0043's re-gate on `c91191da`: plan sha256 `d167d230…` equal to every gate before C2 |

- **First run, before the fix in `9d37253e`:** the same rerun failed closed (exit 5) with the
  episodes correctly written: 0043 records some files as `run_x/../run_x/transcripts/…`, and the
  leak check compared them unnormalised. Fixed in `files_outside` with a test from that shape.
- Not measured: undoing one rerun on the copy (the per-file undo is 0029's tested
  `undo_from_receipts` with the rerun's receipts and tag); a real rerun with the lock held by
  another process; the index and bridge rows of the 2 episodes (their files are byte-identical
  to the apply's, so they are what the apply's derived steps would see).
- The "apply receipt untouched" check is empty on the gate copy: the clone gate applies 0043
  directly, so no `0043_….json` receipt exists there to compare.

**Builder: GO.** Deciding numbers: a correct rerun writes the named episodes byte-identically to
the measured apply and nothing else; the default path's plan is unchanged.

**After the step-8 review (2026-11-04), its conditions:**

- **Undo, measured** on the same copy (scratchpad `c2_undo.py`): undoing the rerun with its own
  receipts and tag restored 25 files, refused 0, and all 25 equal the snapshot's bytes; undoing
  0043's apply afterwards with its own receipts restored 1,155 files, refused 0, and all 1,155
  equal the snapshot's. A rerun leaves the apply's undo whole.
- **Lock held, unit-tested:** `test_a_held_lock_refuses_a_real_rerun_and_not_a_dry_one`: the dry
  rerun plans, the real one raises and the tree is unchanged. 16 tests (the count above was 14
  when written; the normalisation test made 15). **Then measured** (2026-11-04, clone-0043,
  scratchpad `c2_lock.py`, the lock held by another process): the dry rerun exit 0 ("rerun plan:
  0 of 1 named episode(s)"), the real rerun exit 1 ("the corpus is in use: Corpus directory is
  locked by PID …"), 0 files changed in a whole-tree sha256 manifest.
- **Prediction misses, stated as misses:** section 4's snapshot rows ("3 of 101", "2 planned + 1
  not planned") cannot happen as written, because 0043 is unapplied on the snapshot, which is
  that same section's refusal row; they were measured on the gate copy instead. The "gate copy
  plans 0" row was not run as written: 2 episodes were put back first.
- **Rows 4 and 8, as built** (amended to what is built rather than built further, since either
  build edits `outcome_hook` or the runner and re-gates 0043): row 4 checks the rerun's own
  receipts against the scope and every scoreboard row outside it, after the apply; it does not
  read the apply's `details["episodes"]`, and it detects, it does not prevent: a write outside
  the scope is found after it happened, and the rerun's undo is the recovery. Row 8 prints one
  line naming the steps left for the named episodes; it does not list their index or bridge rows.
- The 27 bookkeeping files, counted on the copy: 25 backups under `0043.rerun-<stamp>`, 1
  receipts file, 1 outcome receipt. The manifest skipped symlinks only. `.podcast_scraper.lock`
  comes with the snapshot itself (prod's, 2026-10-25 02:00:09, 0 bytes) and did not change.

## 8. Verdict

Verdict read 2026-11-04 by a fresh agent (reviewer, Sonnet; not the builder, not the step-5
reader), inputs as file paths, ran only the unit tests.

Review budget: minutes 5, tokens 60000

```text
REVIEW c2_episode_rerun 2026-11-04 commit bc03d093 reviewer Sonnet (fresh) minutes 5
PREDICTION vs MEASURE: 3 rows equal; the snapshot rows unmeasurable as written (0043 unapplied there), the gate-copy "plans 0" row not run as written, test count 14 vs 15
REQUIRED ITEMS: 1, 2, 3, 5, 6, 9, 10 done; 4 partial (receipts and scoreboard, not details["episodes"], post-hoc); 7 lock-held not measured; 8 partial (one generic line)
AFTER-STATE READ: one real rerun on one gate copy: PASS, 31 of 31 byte-identical, nothing outside the scope; undo, lock, a non-0043 migration, a real reprocess not shown; ContextVar scope not inherited by threads or processes
OPEN CALLS: amend rows 4 and 8 or build them; measure undo and lock-held; fill Not measured; correct the test count; name the thread risk
DOUBTED NUMBER: 52 = 25 + 27, the 27 not itemised, the lock file not accounted for
VERDICT: GO WITH CONDITIONS (a) rows 4 and 8 amended or built; (b) undo, lock held and the snapshot rows measured or listed; (c) Not measured filled; (d) every prod rerun its own record and go
```

Conditions, each answered in section 7 ("After the step-8 review"): (a) amended to what is
built; (b) undo measured, lock held unit-tested, the snapshot rows stated as misses; (c) Not
covered below; (d) section 0 already says it, and the deploy inventory carries it.

Verdict: GO WITH CONDITIONS

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

- **Each prod rerun is its own corpus change:** its own record, plan diff, accept file with
  causes, and the operator's go. This change ships the command only; no prod rerun is approved by
  this record.
- The scope is a `ContextVar`: a migration that applies in worker threads or child processes
  would not see it and could select outside it. None of 0029–0043 does; the receipts and
  scoreboard checks would find it after the fact.
- The leak check detects after the apply; it cannot prevent a write. A write a migration does not
  receipt and the scoreboard does not see (a sidecar outside `score_episode`) is not detected.
- Measured on one migration (0043) and one copy; a below-regime migration is refused, not run.
  A real reprocess (new pipeline bytes) was stood in for by the snapshot's bytes, so the stop at a
  changed episode is unit-tested only.
- The rerun does not refresh the identity ledger, the search index or enrichment for the named
  episodes; it prints that they are left.
