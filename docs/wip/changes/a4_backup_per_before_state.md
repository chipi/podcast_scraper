---
id: a4_backup_per_before_state
class: light
status: open
issue: none
touches:
  - src/podcast_scraper/upgrade/file_rewrite.py
  - src/podcast_scraper/upgrade/migrations/m0023_transcript_speaker_prefixes_resynced.py
---

# a4_backup_per_before_state — undo restores the state a run replaced, never an older backup

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md), light class.

## 0. Why light

The two backed-up writers (`upgrade/file_rewrite.write_with_backup` and m0023's
`_write_text_with_backup`) kept only the first backup of a file. When a file was rewritten by
something else between two runs of one migration, the second run's receipt says "before: the
in-between state", but undo copied the first backup back, losing the in-between change (naming
handover, section 4). Now a writer whose existing backup does not hold the file's current bytes
keeps one more backup, named `<file>.<sha256 of the state it replaces>`. Undo restores only a
backup whose bytes are the receipt's before-state, and refuses otherwise. A receipt with no
`sha_before` keeps the old behaviour.

What it writes: backup files under `.podcast_scraper/upgrade-backups/<tag>/`, and the files undo
restores. No migration's apply writes different bytes to any content file, so no scoreboard number
can move on an apply. The first backup keeps its path, so the readers of an earlier migration's
backup (m0023 and m0024 read `upgrade-backups/0022/`) see the same file.

Effect on landed outcomes: the code digest of every migration importing either writer moves.
0031–0042 are landed, so no outcome goes stale (method v0.7). Measured by
`tests/unit/podcast_scraper/upgrade/test_file_rewrite_undo.py`: two new tests fail before and pass
after; the four existing ones pass in both.

Revision after Review 1 (2026-11-04): a backup copy cut off midway (a full disk, a killed run) left
a truncated file at the backup path, and every later run took it for the backup, so the file lost
its undo. Both backups are now copied to `<backup>.part` and renamed into place, and a state backup
whose bytes are not the state its name says is copied again. m0023's writer calls the same
`keep_backup`. No content file's bytes change; two more tests in the same file fail before and pass
after, and the 754 other upgrade unit tests pass.
