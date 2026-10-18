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

_TODO_

## 7. Measure

_TODO_

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

_TODO_
