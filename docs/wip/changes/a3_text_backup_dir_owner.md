---
id: a3_text_backup_dir_owner
class: light
status: open
issue: none
touches:
  - src/podcast_scraper/upgrade/migrations/m0023_transcript_speaker_prefixes_resynced.py
---

# a3_text_backup_dir_owner — the text backup writer gives the directories it creates the corpus owner

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md), light class.

## 0. Why light

`_write_text_with_backup` (m0023; imported by m0024, m0025, m0029, m0030 and
`identity/respell.py`) creates the parent directories of a backup under
`.podcast_scraper/upgrade-backups/<tag>/` and chowned only the file and its backup. Run as root on
prod, the directories it created stayed root-owned: 0041 left 5, chowned by hand on 2026-11-02
(naming handover, item 2). The change adds those directories to the `match_corpus_owner` call, as
`upgrade/file_rewrite.write_with_backup` already does with `created_dirs`.

What it writes: ownership of directories under `.podcast_scraper/upgrade-backups/`, and only when
the process runs as root. No content file (gi, kg, bridge, segments, transcripts, metadata) and no
file's bytes change; the text written, the backups and the receipts are byte-identical to before,
so no scoreboard number can move.

Effect on landed outcomes: the change moves the code digest of every migration that imports this
function. 0031–0042 are all `landed`, and a landed record is no longer compared with today's code
(method v0.7); the prod hook still refuses a stale outcome at any later apply. Measured with the
unit test `test_text_backup_dirs_it_creates_get_the_corpus_owner` (fails before, passes after).
