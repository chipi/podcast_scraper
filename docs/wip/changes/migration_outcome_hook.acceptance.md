# Migration outcome hook: step-5 acceptance criteria

Written by the fresh step-5 reviewer, 2026-10-18T09:52 UTC, before any code. Condensed here by
criterion (A1 … I); the reviewer judges step 8 against its own full text.

- **A1 outcome content:** migration id; snapshot release id; the effective applied set the clone
  held; a code digest; the verdicts sha (when read); scoreboard before/after, totals and per
  episode, every `gating_metrics` metric; the per-episode plan; `computed_class`; the step-8 verdict
  and conditions. A schema test refuses each missing field by name.
- **A2 code binding:** sha256 over the migration module and every `podcast_scraper` module it
  imports, sorted, computed the same way at runtime from the installed package; a byte changed in
  the migration, in an imported module, or a verdicts row: refused, 0 files written, exit
  "refused". An outcome measured `dirty: true` is refused (0031 must be re-measured from a clean
  commit).
- **A3 verdict binding:** GO / GO WITH CONDITIONS only (NO-GO, empty, unknown, unreadable, missing:
  refused); CI: the outcome's verdict equals the record's step-8 line, and that line names the
  outcome's sha256.
- **A4** the outcome's scoreboards equal the gate report's; the plan sha is recorded by the gate
  script; CI red on one changed integer. Stated limit: a coordinated edit of outcome, gate report
  and record passes CI; only step 8 guards it.
- **A5 packaging:** `upgrade/outcomes/*.json` in package-data; a test loads it from a built wheel.
- **B1 plan contract:** every full-class regime migration has `dry_run_report(root)`; rows sorted
  by relative meta path with every counter, the refusal, and the sha256 of every input file the
  decision reads (an `open()` spy test proves the set); without it: refused at runtime and red at
  landing; no copy-and-apply fallback.
- **B2** the plan is read-only, deterministic (byte-identical twice, and at two paths), and a dry
  run takes no lock.
- **B3** the binding plan is computed under the lock just before that migration's apply; a regime
  migration applies only as the first pending within `--to` and with prod's applied set equal to
  the outcome's; otherwise refused naming what must run first.
- **C1 diff:** episode presence, any counter, any input sha; stops before snapshot and content,
  prints every difference, writes exactly `.podcast_scraper/upgrade-outcomes/<id>.plan-diff.json`.
- **C2 accept file:** migration id, outcome sha, the entries, a non-empty `cause` per entry, the
  operator's go verbatim; applies only on exact set equality with the diff recomputed under the
  lock; refused on a missing cause, another migration/sha, a gained or lost difference, a dropped
  entry; it relaxes nothing else; the receipt stores its sha and contents.
- **C3** no bypass (no flag, env var, or `--yes` / `--no-snapshot` branch skips the hook).
- **D1 order under one lock:** static checks, plan, diff, snapshot, before-score, apply, ledger,
  after-score, receipt; a refused or stopped run takes no snapshot.
- **D2** with an empty diff: every `gating_metrics` delta equals the clone's exactly, per episode;
  episodes outside the plan 0 (on E1: S1 −1,370, S1n −1,779, S2 −1,896, S3 −526, S10a −12, others 0).
- **D3** with an accepted diff: unchanged episodes exact; differing episodes: any rise fails; S1c 0.
- **D4 exit codes:** refused / stopped on plan diff / applied but outcome failed or not measured /
  0; on failure the apply stays, the ledger records it, no auto-undo, the exact undo command is
  printed, no later migration runs; an after-score exception = "not measured", receipt written; a
  before-score exception refuses.
- **D5 receipt** `.podcast_scraper/upgrade-outcomes/<id>.json` on every path past apply (before,
  after, clone's, per-episode deltas, diff, accept sha, outcome sha, code digest, snapshot path,
  PASS / FAIL / NOT MEASURED, failing metrics with episodes).
- **D6 cost** measured in step 7; above 15 minutes added, the skill's guard and the drill timeout
  are revisited in the same change.
- **E1–E6 known answers** on copies of `lost/clone30` with a truthful ledger (0030 recorded) and
  0031's outcome rebuilt from a clean commit: untouched (diff 0, applied, exit 0, after = the gate's);
  outcome removed (refused, tree identical); one segments byte edited (diff 1, only the diff file
  written, stopped; accepted: applied, recorded; a different accept file: refused); NO-GO (refused);
  a rise injected through a test seam (exit outcome-failed, S1n +1 against the clone, undo named,
  ledger records 0031, nothing reverted); below 0031 byte-identical to the old code.
- **F1–F5:** `--to` above-ceiling migrations never checked; snapshot after the plan check, inside the
  lock; lock contention refuses a run before the plan (a dry run still prints the diff); undo after
  E5 restores byte-identical, the ledger after undo stated and tested; dry run writes 0 files.
- **G1** the corpus-migration skill handles the accept path, the three exit codes and ownership of
  `.podcast_scraper/upgrade-outcomes/`, or each is listed Not covered. **G2** the drill workflow and
  `make upgrade-corpus` behaviour under the hook stated and tested; no bypass.
- **H1** 15 listed mutations each red; **H2** scenario tests on real-shaped data.
- **I out of scope:** gold gates on prod; derived surfaces; 0032's `dry_run_report`; direct Python
  calls to `apply` / `undo` and `scripts/audit/*`; proof that a human gave the go.

## Amendment 1 (2026-10-18T09:59, the same reviewer): the empty-plan rule, accepted with conditions

Correction acknowledged: prod's ledger records 0030; only the local clone's lacks it (the E1-E5
setup records 0030 in the copy's ledger).

- **R1** "empty" is decided again at the apply: the pre-run plan is advisory; inside the lock, after
  every earlier migration of the run, the plan is recomputed and decides. Test: empty before the
  run, an earlier fake migration (below 0031) writes an episode into its population: refused before
  the regime migration (named, 0 writes by it, the earlier one applied and recorded, exit refused).
- **R2** empty means apply writes 0 bytes, enforced at runtime: only the ledger entry; no receipts
  file or header, no `upgrade-backups/<tag>`, no `upgrade-outcomes/*`; after apply the hook checks
  the result (0 episodes written) and that the receipts file and backup tag directory did not appear
  or grow; failing: outcome-failed, the undo named. Tree hash on the fixture and an empty corpus
  differs only in `upgrade_ledger.json`; a fake with an empty plan that writes one file: outcome-failed.
- **R3** the plan is complete: on a clone copy the episodes whose files apply changes are a subset
  of the plan's rows (refused episodes are rows; corpus-level writes are rows); dropping one written
  episode from `dry_run_report` turns that test red.
- **R4** an empty plan counts only when `dry_run_report` exists (a full-class regime migration
  without one is refused even on fixtures), returned normally (a raise, a non-iterable, an unreadable
  verdicts file: refused), and ran on the same root and served selection as apply; no flag makes a
  non-empty plan empty; an empty plan does not need the outcome file (parsed and warned when present).
- **R5** prints `plan empty: nothing to measure` with the id; exit 0 when nothing else failed; the
  dry run prints the same and never refuses; the ledger records it applied, and the record states
  that a corpus which later gains that population will not run it again.
- **R6** known answers: empty corpus exit 0, tree differs only in the ledger, 0031 recorded, the
  line printed; the pinned fixture corpus from 0001: exit 0, output and tree as at the branch point
  but the ledger `at` and the `plan empty` line, the 4 red tests green without changing what they
  assert; a clone30 copy with 0031 pending among others: refused before the snapshot, tree identical,
  exit refused, the message names the condition; the drill's expected result stated in step 7.
- **R7** mutations: empty from the pre-run plan only; a `dry_run_report` exception as empty; a
  missing `dry_run_report` as empty; no R2 guard; refused rows counted as empty.
