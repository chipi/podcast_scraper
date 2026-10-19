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
  0031_named_voices_shown.tsv`, the same rows 0031 reads) and expects an entry judged
  `WRONG_PERSON` in its restored state, as 0031's verify does. Every other entry is verified as
  before.
- **Tests:** an entry judged `WRONG_PERSON` and restored passes; the same entry still carrying the
  listed name fails; an entry with no verdict is verified exactly as before.
- **Dry run and undo:** not applicable; it writes nothing. The check is 0029's verify on the
  clone's after-state, then on prod: `[OK]`.
