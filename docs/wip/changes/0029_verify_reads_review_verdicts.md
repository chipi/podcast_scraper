---
id: 0029_verify_reads_review_verdicts
class: light
status: open
issue: none (found by prod's upgrade verify, 2026-10-19)
touches:
  - src/podcast_scraper/upgrade/migrations/m0029_naming_arc_voices_repaired.py
---

# 0029's verify takes the reviewer's verdicts into account

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan:
[INGESTION-ARC-2026-10-19.md](../INGESTION-ARC-2026-10-19.md), Stage 1.

## 0. Why light

It writes nothing. Only `verify` changes, which reads the corpus and returns pass or fail. No file a
listener or reader is shown is written. The sequencing cost: `m0029` is in 0031's import closure,
so the edit makes 0031's shipped outcome stale. Either the stale check moves to the deploy path
first (Stage 1, item 3), or 0031 is re-measured and its outcome rebuilt in the same change.

- **Found:** prod's `upgrade verify` after 0031's apply (2026-10-19 08:59Z): `[FAIL]
  0029_naming_arc_voices_repaired: 1 episode(s) still to write, 0 refused`, the Jessica Espinoza
  episode ("Unlocking Gender-Smart Capital", episode 5555e937-4e0a-4e17-abc3-68f5b77e5501).
- **Cause, reproduced on the clone:** 0029's list names that voice "Justin Norman" (host). 0031's
  reviewer judged it `WRONG_PERSON`, and 0031 restored the listed old state (no name, no role).
  0029's verify still expects its own list's name. Before 0031, 0029's verify failed on 2 leftover
  episodes ("Taking down the cultural elite", Migraine); 0031 wrote both, and after 0031 it fails
  only on Espinoza.
- **The change:** 0029's verify reads the reviewer verdicts file (`upgrade/verdicts/
  0031_named_voices_shown.tsv`, the same rows 0031 reads, validated against 0029's list as
  shipped) and takes an entry judged `WRONG_PERSON` as done in either state: 0029's own (the
  listed name, before 0031 runs) or the listed old state 0031 restores. Every other entry is
  verified as before. Plan and apply do not read the verdicts: 0029 writes its list unchanged, so
  the change writes nothing new on any corpus.
- **Sequencing:** `m0029` is in 0031's import closure, so this moves 0031's code digest. It lands
  after `stale_check_landed_migrations` (0031 is landed), which releases that.
- **Tests:** an entry judged `WRONG_PERSON` in its restored state passes, and still passes in 0029's
  own state after an apply, while plan and apply write it as listed; with no verdict the restored
  state is still to write (fails); a `WRONG_PERSON` entry in neither state is refused; the shipped
  file yields exactly one entry, the Espinoza voice (`5555e937-…`, `SPEAKER_00`). Mutation: with
  the restored check disabled, the first test fails.
- **Measured on the clone** (`lost/gateworkB6c`, 10-16 + 0030 + 0031, `scratchpad/v29.py`),
  2026-10-19: before the fix `0029: (False, "1 episode(s) still to write, 0 refused: ['0018 -
  Unlocking Gender-Smart Capital A_…']")`; after it `0029: (True, "every listed voice carries the
  name the naming ladder gives it")`. 0031's verify is unchanged (True, `unnamed_checked=1`), and
  0015's still fails on the David Roblin episode (that is 0033).
- **Fresh reviewer, 2026-10-19:** GO WITH CONDITIONS, both applied: (1) an unusable verdicts
  file made 0029's verify raise, and `upgrade verify` runs every migration in one loop with no
  catch, so every migration after 0029 went unreported; it now returns `(False, "verdicts file
  unusable: …")` as 0031's does, tested; (2) the reset of `_restored` after verify was unguarded
  (the registry holds one instance); a verify-then-apply test on one instance now fails when the
  reset is removed. Not re-derived by the reviewer: the clone lines above.
- **Dry run and undo:** not applicable; it writes nothing. On prod after the deploy: `upgrade
  verify` shows `[OK]` for 0029.
