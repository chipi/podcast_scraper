---
id: stale_check_landed_migrations
class: light
status: open
issue: none (the operator opens issues; INGESTION-ARC Stage 1, item 3)
touches:
  - src/podcast_scraper/quality/change_gate.py
  - tests/unit/podcast_scraper/quality/test_change_gate.py
---

# A landed migration's outcome is no longer compared with today's code on main

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan:
[INGESTION-ARC-2026-10-19.md](../INGESTION-ARC-2026-10-19.md), Stage 1, item 3. Operator's
decision, 2026-10-19, verbatim: "A".

## 0. Why light

It writes no corpus file and changes no migration. It changes only which records the CI landing
test (`check_landing`) compares with the tree's code digest.

- **Found:** the landing test (`test_every_registered_migration_under_the_regime_is_allowed_to_land`)
  refused 0031's outcome as stale after any edit inside 0031's import closure (667 modules, 175 of
  them under `server/`; `quality/change_gate.py` itself is one of them). 0031 was applied on prod
  on 2026-10-19, so the comparison could no longer protect an apply, and it blocked unrelated
  work: the prod session's player-api memory watchdog (fd7165609, `server/app.py`,
  `utils/access_log.py`) moved 0031's digest from 5d5477d0… to 00eb66e8….
- **The change:** a record with `status: landed` is checked for the apply's evidence (steps 10
  and 11 done) instead of the digest. Open records are compared exactly as before. Every other
  outcome check (verdict, sha256 in step 8, scoreboards, plan, clean code) still runs for landed
  records.
- **What still guards a landed migration:** the outcome hook (`upgrade/outcome_hook.py`). A landed
  migration applies again only on a corpus that has not run it (a restored older snapshot, a new
  environment). With a non-empty plan the hook refuses installed code whose digest is not the
  measured one ("the installed code is not the code that was measured"). With an empty plan it
  makes no digest check, but the apply then refuses any write (`apply_empty`, exit 5), so nothing
  unmeasured reaches the corpus (fresh reviewer, 2026-10-19).
- **`landed` is a hand-set flag.** Nothing in code ties it to a prod apply; the check only requires
  steps 10 and 11 to be filled. Setting `status: landed` before the apply turns this check off for
  that migration early, and the hook is then the only guard. The method sets `landed` at step 12,
  after the apply; the reviewer of each change checks that.
- **Deploy path:** no step is added to `deploy-all-prod.yml`. The plan said the check would move
  there, but the hook already runs at the apply, after every deploy, and a deploy-time digest
  check on landed migrations would block every deploy after a server edit: the same wall, moved.
- **The closure size (plan item 3):** not changed. With landed records released, the 667-module
  closure only binds a migration between its measurement and its apply, which is what it is for.
- **Tests:** a landed record with steps 10 and 11 done passes with a moved digest; a landed record
  with step 10 or step 11 still `_TODO_` is refused for that, not as stale; the open-record stale test is
  unchanged. Mutations (fresh reviewer): the landed branch disabled, 3 fail; forced on, 4 fail;
  step 11 dropped from `APPLIED_STEPS`, 1 fails; step 10 dropped survived until the step-10 test
  was added.
- **Found by the new check:** 0031's own record still carried the template's `_TODO_` at the end
  of step 10, under complete content. Removed.
- **Dry run and undo:** not applicable; nothing is written. Undo is reverting the commit.
- **Fresh reviewer, 2026-10-19:** GO WITH CONDITIONS: (1) a step-10 test, added; (2) two
  overstated sentences (the hook's empty-plan branch; "applied on prod" as checked by code),
  corrected above and in the method's v0.7 row; (3) the operator to know `landed` is a hand-set
  flag, stated above and in the report to the operator.
