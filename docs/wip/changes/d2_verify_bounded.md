---
id: d2_verify_bounded
class: light
status: open
issue: none (D2 item 6; operator requirement relayed by prod, 2026-10-31)
touches:
  - src/podcast_scraper/upgrade/cli_handlers.py
  - src/podcast_scraper/upgrade/runner.py
  - src/podcast_scraper/upgrade/migrations/m0028_transcript_chunk_times.py
  - src/podcast_scraper/utils/json_stream.py
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

- **Prod, 0028's verify alone** (eval-data `d2v/verify_one.py`, `--memory 3500m`, corpus read-only):
  killed by its cgroup 4 s in, at anon-rss 3.5 GB, before printing. The full verify's ~20 min at
  ~0.87 GB was the 27 migrations before 0028.
- **Snapshot copy, no search index:** every migration measured so far peaks at 224 MB or less
  (`d2v/verify_each_gate0038.jsonl`). That copy has no index, so it does not exercise 0028.

## 3. Diagnosis

- `m0028_transcript_chunk_times._plan` parsed `search/metadata.json` whole (490 MB on prod).
- It also kept the transcript text of every episode it touched, for the whole run.
- Both grow with the corpus.

## 4–6. Change and build

- `utils/json_stream.iter_json_object_items`: reads one JSON object entry by entry with the
  standard library's decoder, keeping only the asked-for keys (the operator chose this over adding
  ijson, or a lazy load). Tests: every entry round-trips at chunk sizes 1, 7, 64 and 1 MiB; a
  broken or non-object file raises; peak memory on a file 4× larger stays within 1.5× (tracemalloc).
- `m0028._plan`: the untimed rows first, then only their sidecar entries, streamed, and one
  transcript held at a time (candidates visited grouped by episode). What the check concludes is
  unchanged: 0028's integration test, 6 passed.
- `upgrade verify --only` and per-check output (786540614).

## 7. Measure

- **The reader** on a synthetic 490 MB sidecar of prod's shape (1,301,074 entries,
  `d2v/synthetic_sidecar.py`, `d2v/measure_reader.py`): 1,000 wanted rows 89 MB peak, 7.6 s;
  10,000 wanted rows 92 MB peak, 7.2 s.
- **Not measured:** 0028's whole verify on prod's real index with this code. It needs a prod run
  with the branch code (read-only, capped), or D2's image.

## Not covered

- 0028's `apply` and `undo` still load the sidecar whole when they rewrite it. 0028 is applied on
  prod, so only an undo would run that path.
- The other tools that read `search/metadata.json` whole (`search/interest_index.py`,
  `search/gil_chunk_offset_verify.py`, `search/cli_handlers.py`) are operator CLIs, not verify.
  They are unchanged.
- `_untimed_ids` returns every untimed row id in one list. That grows with the untimed count, not
  the index size; the prod count is not known.
