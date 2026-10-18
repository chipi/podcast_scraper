---
id: migration_outcome_hook
class: full
status: open
issue: 2380
touches:
  - src/podcast_scraper/upgrade/runner.py
  - src/podcast_scraper/upgrade/cli_handlers.py
  - src/podcast_scraper/upgrade/outcome_hook.py
---

# The migration outcome hook (method step 0, D5 and D15, before 0031's step 10)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md). Operator 2026-10-18:
0031 waits for S7/S8 and this hook (step 0, option B).

## 1. Intake

- Method D5: "a migration cannot apply on prod without a receipt holding S1/S2 before and after
  from its clone run". D15: "before apply, prod produces the per-episode plan (every counter) and it
  is diffed against the clone's ... the migration hook carries this diff". Step 10: every
  difference listed with its cause, or the run stops. Step 11: re-measure on prod, the apply's
  counts checked against the restated prediction.
- What exists: `upgrade/cli_handlers.py` `upgrade run` (confirm, a pre-upgrade snapshot under one
  lock, `UpgradeRunner.run`, ledger), `upgrade verify`; `--dry-run` prints each migration's plan line.
  Per-episode plans exist only where a migration has `dry_run_report` (0029, 0031, 0032).
- The change gate (`docs/wip/changes/`): each measured migration has `<id>.gate.json` (the clone
  run: scoreboard before/after, files changed by suffix, verify, code sha) and a per-episode plan
  file kept in scratch (`plan_0031_clone30.jsonl`), not in the repo, not in the image.
- The corpus-migration skill (homelab) refuses a handover without its outcome table; the operator
  and prod run `upgrade run` inside the container with the image's code.
- `quality/corpus_scoreboard.py`: `score_corpus` over 2,685 episodes takes about a minute.

## 2. Outcome and baseline

What the hook must make impossible, each a check that can fail: an apply with no clone outcome
shipped with the code; an apply whose prod plan differs from the clone's without each difference
listed; an apply whose prod scoreboard after moves a gating number in a direction its clone run did
not (a rise of S1n, S1c, S12, or of S1/S2 beyond the clone's identity-listed moves); an outcome that
nobody can read back (no receipt). Baseline: none of these is checked today (0031's run steps are
prose in a handover).

## 3. Diagnosis by loss

The failures it answers: 0031 (then numbered so) handed over with mechanics only (D5); 0030's plan
assumed equal to prod's (D15, 13 episodes differed); "run the migrations" as one blanket
`upgrade run` (the corpus-migration skill's `expect_pending` is the only guard).

## 4. Prediction

Dated: 2026-10-18T09:49

The design. A migration of class full ships its clone outcome with the code:
`upgrade/outcomes/<id>.json` (package data): the snapshot id, the scoreboard before and after, the
per-episode plan (meta path, every counter, the input files' sha256), the verdicts file's sha when
it reads one, and the record's step-8 verdict. `upgrade run` (not `--dry-run`), for each pending
migration of class full:

1. refuses when the outcome file is missing, unreadable, or its verdict is not GO / GO WITH
   CONDITIONS (named error, nothing applied);
2. computes prod's per-episode plan read-only and diffs it against the shipped one; any difference
   (an episode in one and not the other, a counter that differs) stops the run, prints every
   difference, and writes them to `.podcast_scraper/upgrade-outcomes/<id>.plan-diff.json`; a
   rerun with `--accept-plan-diff <that file>` applies only if prod's plan still produces exactly
   that diff (the operator's step-10 decision, recorded);
3. takes the scoreboard before, applies, takes it after, and writes
   `.podcast_scraper/upgrade-outcomes/<id>.json` (before, after, the clone's, the diff);
4. exits non-zero and names the undo when, after the apply, S1n, S1c or S12 rose, or S1 or S2 moved
   the other way from the clone's move (the apply stays; the ledger records it as applied; the
   operator decides on undo).

A migration of class light (or without an outcome file and not in the regime, below 0031) runs as
before. `--dry-run` also prints the plan diff.

Known answers, on fresh copies of `lost/clone30` with 0031 B.3 and its outcome file built from the
gate run:
- the outcome file present and the copy untouched: plan diff 0 episodes, applied, outcome receipt
  written, exit 0, scoreboard after = the gate's after (S1 14,841, S1n 7, S2 26,351, S3 7,710);
- the outcome file removed: refused, 0 files written, exit non-zero;
- one plan episode's segments edited before the run (a reprocess stand-in): plan diff 1 episode,
  run stopped, 0 files written; rerun with `--accept-plan-diff`: applied, 1 difference recorded;
- an outcome whose verdict is NO-GO: refused;
- after an apply, a gating rise injected (an insight hidden again on a shown voice) before the
  after-scoreboard: exit non-zero, the undo named, S1n reported +1.
- 0026-0030 (below the regime) unaffected: their dry runs and applies behave byte-for-byte as before.

Dated: 2026-10-18T09:54

**Restated before code, after the step-5 review (09:52).** Accepted from the review: the binding
plan is computed under the lock just before each regime migration, which must be the first pending
one with prod's applied set equal to the outcome's (B3); the after-check is exact per episode on
every `gating_metrics` metric, not "the other way" (D2/D3); three non-zero exit codes (D4); a
content digest of the installed modules binds the code, and the record's step-8 line names the
outcome's sha256 (A2/A3); plan rows carry their input files' sha256 (B1, so 0031's
`dry_run_report` changes); the plan-diff file is the one write a stopped run makes, and a dry run
writes nothing (C1/F5). Correction to the review's concern 3: prod's ledger does record 0030
(applied there by `upgrade run` on 2026-10-17); it is the local clone's ledger that lacks it,
because the gate script applies directly, so the outcome records the effective applied set and the
gate script records the ledger it leaves. Known answers as in the criteria E1-E6 (E1 after =
S1 14,841, S1n 7, S2 26,351, S3 7,710, S10a 178, S12 24, S1c 0, A1 357; deltas S1 −1,370,
S1n −1,779, S2 −1,896, S3 −526, S10a −12, others 0).

Dated: 2026-10-18T09:58

**Restated during the build, before any measurement:** the first build refused every `upgrade run`
that is not prod's: the fixture corpora, a fresh corpus and the drill run 0001 to 0031 in one run,
and on them the plan cannot equal the clone's (4 existing tests went red: `test_cli_handlers`
`test_run_yes_json_on_empty_corpus`, `test_end_to_end` x3, `test_pinned_fixture_upgrade`). A corpus
carries no identity to tell prod apart (`corpus_manifest.json` has none). The rule instead: the hook
guards WRITES. A regime migration whose plan on this corpus, computed before the run, has 0
episodes writes nothing, so it runs without measurement and the run says so ("plan empty: nothing
to measure"), alone or with others; with 1 or more episodes every check of the criteria applies,
including "runs alone". A dry run never refuses: it prints what the hook would decide. Known
answers added: an empty corpus and the pinned fixture corpus: `upgrade run --yes` exits 0 as
before, every migration recorded, 0 files written by 0031, the message printed; a copy of clone30
with 0031 pending among others: refused (exit 3), nothing written.

## 5. Acceptance

Dated: 2026-10-18T09:59

Amendment 1 (the empty-plan rule), appended to the acceptance file.

Dated: 2026-10-18T09:52

The fresh reviewer's criteria, written before code: [migration_outcome_hook.acceptance.md](migration_outcome_hook.acceptance.md).

## 6. Build

Committed in 28a76ce01 (local), plus the one-line print fix below.
- `src/podcast_scraper/upgrade/outcome_hook.py`: load and check the shipped outcome; the code
  digest over the migration's import closure plus its verdicts file; the per-episode plan from
  `dry_run_report`, its diff and the accept file; scoreboards before and after with the exact
  per-episode compare; the receipt; exits 3 / 4 / 5; the empty-plan rule with its write guard.
- `upgrade/runner.py`: the apply of a regime migration goes through `measure`. An unprepared one
  recomputes its plan at the apply: non-empty is refused, empty goes through `apply_empty`.
- `upgrade/cli_handlers.py`: `--accept-plan-diff`; `_prepare_regime` inside the lock before the
  snapshot; a non-empty regime migration runs alone; each hook exit is returned as the CLI's code.
  After the E runs, the advisory "plan empty" line prints only on a dry run, because the runner
  prints it at the apply (it printed twice).
- `upgrade/read_recorder.py` (in 0031 B.4): the inputs each plan row hashes.
- `scripts/measure/corpus_change_gate.py` (`--snapshot`; refuses a clone whose ledger lacks an
  earlier migration; records the binding fields) and `scripts/measure/build_outcome.py`.
- `change_gate.py`: landing requires the shipped outcome, bound by sha to the record's step 8.
- `pyproject.toml`: `upgrade/outcomes/*.json` is package data.
- Tests: `test_outcome_hook.py`, 41 collected (24 functions, parametrised), among them the run-alone refusal, S1c left after, the
  accept file in the receipt, a row on one side only, and the built-in metric list equal to the
  gate config.
- Mutations (`scratchpad/lost/hookmut.py`, `hookmut_b.py`): 22 of 22 real mutants killed. The
  first list held a 21st, "auto undo", that only added an import and changed no behaviour; it was
  dropped as no mutant, not counted as killed.
- The corpus-migration skill (homelab repo, uncommitted): `apply --accept-plan-diff`, the
  `plan-diff` command, the meaning of exits 3 / 4 / 5 in `watch`, and a SKILL.md section;
  `python3 -m unittest test_migrate` gives 4 tests OK.

Note for every later hook change: `outcome_hook.py` is in 0031's import closure, so it is in the
code digest. Changing it after 0031's outcome is built means re-measuring 0031 and rebuilding the
outcome. `cli_handlers.py` and `runner.py` are not in the closure.

## 7. Measure

All on copies of `lost/clone30L`: `lost/clone30` with 0030 recorded in its ledger. 0031's outcome
was built from the clean commit 28a76ce01 (sha256 b1546e27…). Each run is the real CLI
(`scratchpad/lost/e_run.py`; scripts `e1.sh`, `e25.sh`, `e6.sh`, `r6.sh`, `r6b.sh`).
- **E1 untouched:** the dry run gives "plan diff 0 episode(s)", exit 0, and writes nothing.
  `run --yes` takes the snapshot, applies 54 episodes and exits 0 in 192 s. The receipt is PASS
  with plan diff 0. Totals after: S1 14,841, S1n 7, S2 26,351, S3 7,710, S10a 178, S12 24,
  S1c 0, A1 357, each as predicted. The tree equals the gate's after-state (`diff -rq`) except
  the ledger (the gate does not record 0031) and the receipt directory.
- **E2 outcome removed:** exit 3, "no shipped outcome". The tree is identical and no snapshot
  was taken.
- **E3 one segments byte edited** (episode 0008's `.adfree.segments.json`): exit 4. The diff
  names that episode, kind `differs`, field `inputs`. It writes only the plan-diff file and takes
  no snapshot. With a different accept file (one invented extra entry): exit 4, "not the accepted
  one". With the matching accept file: applied, exit 0, receipt PASS, carrying the accept file's
  sha and the 1 difference. For the reviewer: the code exits 4 (stopped) on a wrong accept file,
  not 3; criterion C2 says "refused".
- **E4 NO-GO outcome:** exit 3, "outcome verdict is 'NO-GO', not GO". The tree is identical and
  no snapshot was taken.
- **E5 a rise through a test seam** (`e_run_rise.py`: S1n +1 on one episode in the
  after-scoreboard; the seam is in the scratch script, not in the product): exit 5. The receipt is
  FAIL and names "S1n_named_but_hidden changed +1, the clone's +0". The undo is printed, the
  ledger records 0031, and nothing is reverted (receipts file 404 lines). **F4:** the printed
  undo (`m0031.undo`) then restores 403 / 0; `feeds/` is byte-identical to clone30L; the ledger
  no longer lists 0031 (re-runnable), and its `version` stays 2.7.25, deliberately
  (`state.record_reverted`, #2069). The backups, the receipts file and the hook's receipt stay as
  the audit trail.
- **E6 below the regime:** the pinned fixture corpus (`tests/fixtures/upgrade/
  corpus_at_last_prod_release`) through `upgrade run --yes`, on f1817141e (before the hook) and on
  28a76ce01. The trees differ only in the ledger's `at` times. The output differs only by the
  line "0031_named_voices_shown: plan empty: nothing to measure", once after the print fix.
- **R6:**
  - An empty corpus: exit 0, the line printed. Run alone after 0001–0030, 0031 changes only the
    ledger. The other new files in the full run (`search/`, two lock files) come from 0002 and
    the corpus lock.
  - A clone30 copy with 0030 and 0031 pending: exit 3, "runs alone: pending in this run
    ['0030_adfree_cut_resynced', '0031_named_voices_shown']". `diff -rq` against clone30 exits 0,
    and no snapshot was taken.
- **D6 cost:** a plan takes 12.2 s and a scoreboard pass 31.8 s on clone30L (2,685 served
  metadata files), so the hook adds about 76 s per run. That is below 15 minutes; the skill's
  guard and the drill timeout stand. It was measured while other copies were being made, so it
  is on the high side.
- **G2:**
  - `make upgrade-corpus` is `upgrade run --yes`, the command above.
  - The drill (`drill-corpus-upgrade.yml`) restores the newest `snapshot-prod-*` and runs every
    pending migration in one `upgrade run --yes --no-snapshot`. On the 10-16 snapshot (0030 and
    0031 pending) it is refused, exit 3 (R6 above). On a snapshot from 10-17 on, 0031 runs alone
    and any episode that changed since the clone's snapshot stops it, exit 4.
  - There is no bypass, so the drill is red by design until its snapshot is the measured one.
    Stated for the operator, not changed.
- **A5 from a built wheel:** `pip wheel . --no-deps --no-build-isolation` into the scratchpad
  (`scratchpad/wheel`, nothing installed) packages `upgrade/outcomes/0031_named_voices_shown.json`
  and `upgrade/verdicts/0031_named_voices_shown.tsv`. With only the unpacked wheel on the path
  (`scratchpad/wheelload.py`): the outcome loads (sha256 b1546e27…, verdict GO) and the runtime
  digest equals the outcome's (aaf844b1…). The digest does not depend on where it is installed. The build also wrote `build/` and
  `src/podcast_scraper.egg-info/` into the worktree (setuptools), which made `make type` fail on a
  duplicate module; both were removed and ci-fast re-run.
- **G1:** the skill as in step 6. Its new paths are tested against the command it builds, not
  against a live container.

## 8. Verdict

_TODO_

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

- The prod plan diff (road step 4) and a live apply: neither has been run. The hook has run only
  on local clone copies.
- The skill's new paths against a real container: they are tested on the parser and the command
  string only.
- The drill run under the hook in GitHub Actions: the behaviour above is reasoned from the
  workflow file plus R6, not from a drill run.
- The accept file has no check that a human wrote the operator's go (criterion I: out of scope).
- A coordinated edit of the outcome, the gate report and the record passes CI (A4's stated limit);
  only step 8 guards it.
