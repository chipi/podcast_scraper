# The final validation pass for the main scope (written 2026-10-27)

Operator, 2026-10-27: land the fixes as hypotheses, park every re-measure, gate and review, then
validate everything at once and push once ("keep developing and committing, we push at the end").
This is the list of what is landed unvalidated and what the pass must do with it. Nothing below is
pushed. Worktree: `/Users/claude/projects/podcast_scraper-0029`, branch `feat/m0034-list`.

## What is landed (local commits)

| Commit | What | Record |
|---|---|---|
| 2b035c180 | the gate runs the full `upgrade verify` on the clone (`verify_all`, `verify_regressions`); shipped verdicts completeness test; method rules; ad-cut resync step | method docs |
| 9153ca153 | the ledger, its lock and outcome receipts chowned to the corpus owner; undo rewinds `version` when it can | (bookkeeping, no content) |
| 8c062f848 | pipeline: the forced host seat's fourth veto, `named_in_third_person` (#2094 D5) | `changes/known_host_third_person_veto.md` steps 1-6 |
| 4b80a3929 | migration 0037: 6 wrong-person served voices in 4 episodes get their own names (D5 + night 1) | `changes/0037_named_voices_corrected.md` steps 1-6 |
| 1f9c5eaea | D4's 43 candidates sorted: 0 wrong names in 42 read | `D4-CLASSIFICATION-2026-10-27.md` |

Every src change above moves the 0034-0036 code digest: CI's landing test stays red until step 4.

## The pass, in order

1. **Cheap gates first** (black, isort, flake8, mypy, interrogate, codespell on src), so no src
   edit lands after a measurement. Any fix here comes before step 2.
2. **The snapshot:** download and verify snapshot-prod-20261027
   (`scripts/ops/corpus_snapshot/download_and_verify_snapshot.sh --tag snapshot-prod-20261027`)
   under `~/projects`, never `/private/tmp` (check `df` first). It holds 0034-0036 and the 0030
   re-run, and the night-1 episodes. `remeasure-20261027` (snapshot-prod-20261025) is deleted once
   the new one is verified, unless step 3 still needs it.
3. **Re-gate 0034, 0035, 0036** on snapshot-prod-20261025's clone (their outcomes were measured on
   it; the code moved, the behaviour must not): `scripts/measure/corpus_change_gate.py` chain
   (`rm27/chain.sh`), each report compared with the committed one; any difference is read.
4. **Rebuild the three outcomes** from those reports; the landing test turns green.
5. **The veto (`known_host_third_person_veto`), record steps 7-8:**
   - corpus reach: replay `resolve_speaker_roster` with and without the veto over the stored
     segments of every served episode of the snapshot (the trace's `forced_name_vetoes`
     `named_in_third_person`), integers per path;
   - every reached voice read against its own text; above 5 episodes the change stops until each
     has a verdict row (acceptance 6);
   - a fresh reviewer replays D5 with its own code (acceptance 1) and judges every criterion.
6. **0037, record steps 7-8:**
   - a fresh blind reader on D5's two voices (no proposed names shown); if both are not
     confirmed, D5 leaves the list and the record's prediction is restated (4 voices, 3 episodes);
   - the verdicts file (`rv.verdicts_path("0037_named_voices_corrected")`): night-1 rows from both
     readers' agreement, the D5 rows, bound to the list's sha256;
   - register 0037 after 0036 (`to_version` 2.7.31);
   - gate on the clone of snapshot-prod-20261027 (its ledger must hold 0034-0036);
     criteria 1-17 of the record, including the full-clone undo and a second apply writing nothing;
   - a fresh reviewer's verdict.
7. **D4's last candidate** `ce2dc3a9-d06c-43f3-918e-b4ec014a82ff`: read from the new snapshot;
   a wrong name joins 0037 by a record revision (then step 6 again for that voice).
8. **#15** (epic #2097 activities 6-7): re-derive the #2075 measurements (#2093, #2095, the #2075
   PR) on the new snapshot after 0037's clone apply, for the operator's closure decisions.
9. **Gates on the committed tree:** `make ci-fast` and `make test-integration`, both with
   `PYTHONPATH=<worktree>/src` (the venv imports the main checkout otherwise); logs kept.
10. **Push** once: rebase onto origin/main (prod's local d2eaa8d77, the open-items list, may land
    first), secrets scan, push to main, `ci-watch` until GREEN.
11. **Hand to prod:** 0037's apply goes through the corpus change method's step 9-10 (operator's
    go, prod's own plan diff against the clone's) and the `corpus-migration` skill. The veto
    reaches prod with the next deploy; the operator decides when.
12. **Clean up:** delete the clones and the scratch corpora (done 2026-10-27). `close-arc`: not applicable: this repo has no `docs/history`, decisions log or NEXT_STEPS (operator, 2026-10-27: drop close-arc here); the arc's record is the change records, the method doc and the naming handovers.

## Not in this pass

- **0032's 91 uncovered voices** (48 additions, 29 renames, 14 unnamings; operator 2026-10-27):
  their own reviewed list, AFTER this pass and AFTER the evaluation of the LLM naming and the other
  rungs running in shadow on prod. The same machinery as 0037: freeze the 91, two blind readers
  with names hidden, keep only what both confirm, a migration with verdicts, a clone gate and a
  reviewer. The 29 renames first (they can be wrong-name fixes). Estimate 2-3 h, unmeasured.
- **The forced-host-seat veto:** rejected on its reach (52 fired, most real hosts introducing
  themselves); reverted in 97eaf2850. A second attempt starts again at step 3.

- Deploys (the operator plans none now).
- The D4 big unnamed voices (hidden insights, a coverage question for its own record).
- The `d99b4720` cause (a known host forced onto a clip voice that never mentions the host).
- Items 16a-16c, 19, 20 of the open-items list.
