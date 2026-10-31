---
id: d2_verify_bounded
class: light
status: open
issue: none (D2 item 6; operator requirement relayed by prod, 2026-10-31)
touches:
  - src/podcast_scraper/upgrade/cli_handlers.py
  - src/podcast_scraper/upgrade/runner.py
  - src/podcast_scraper/upgrade/migrations/m0028_transcript_chunk_times.py
---

# d2_verify_bounded — `upgrade verify` runs in bounded memory, can check one migration, and reports as it goes

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4.

## 0. Why light

It writes no corpus file and changes no served value. `upgrade verify` is read-only. This change
changes how it runs: which migrations it checks, when it prints, and how much it holds in memory.
What each check concludes stays the same.

## 1. Intake

- Prod, 2026-10-31: the full `upgrade verify` ran the 15 GB host out of memory. Run 1 was killed at
  anon-rss 4.3 GB. Run 2, capped at 3.5 GB, was steady at about 0.87 GB for 13 minutes, then passed
  3.5 GB in under 30 s, about 23 minutes in. Stdout was empty at both kills: the CLI prints only
  after every check.
- The operator's requirement: stream; `--only`; a measured peak on a 3,091-episode clone,
  target under about 1.5 GB; a test that fails when the peak grows with corpus size.
- Per-migration peaks on the post-0038 snapshot copy, which has no search index: eval-data
  `d2v/verify_each_gate0038.jsonl`.
- Prod's search sidecar `search/metadata.json` is 490 MB, and 0028's verify parses it whole
  (`m0028_transcript_chunk_times.py` `_plan`).

## 2. Outcome and baseline

_TODO_ (the per-migration table, and prod's 0028-only peak)

## 3–5.

_TODO_

## Not covered

_TODO_
