# Lessons learned: partial multi-feed success without “blowing up” at the end

**Status:** Draft notes (operator / design). Related GitHub: **#557** (Whisper 25 MB + incident log), **#558**
(FFmpeg/ffprobe subprocess UTF-8 replace + preprocessing fallback incident rows). **#559** (soft-failure taxonomy; default **lenient** with
**`multi_feed_strict: false`**; strict CI uses **`multi_feed_strict: true`** or CLI
**`--multi-feed-strict`**) is **implemented** in code and docs (CONFIGURATION / CLI).

## What went well (resilience that already exists)

- **Per-feed isolation in `service._run_multi_feed`:** One feed’s `run_pipeline` exception
  (e.g. RSS fetch `ValueError`) is caught; other feeds still run. See
  `src/podcast_scraper/service.py` (`try` / `except` around `workflow.run_pipeline(sub_cfg)`).
- **Finalize on partial batches:** `finalize_multi_feed_batch` in
  `src/podcast_scraper/workflow/corpus_operations.py` still writes manifest and
  `corpus_run_summary.json`, and runs parent `index_corpus` where configured; vector index
  failure there is logged as non-fatal in a `try` / `except`.
- **Episode-level transcription errors:** Many paths log `transcription raised an unexpected
  error`, increment metrics, and **continue** the loop unless `fail_fast` / `max_failures`
  triggers (see `src/podcast_scraper/workflow/stages/transcription.py`).

So the long manual run **did** complete batch work: artifacts, corpus summary, and indexing
for feeds that succeeded.

## What felt like “blowing up” at the end

1. **Process exit code:** Operators often want exit **0** when “best effort” corpus work
   finished and artifacts are usable, with failures **only** in structured JSON / logs.
   Today, **any** feed-level error collection in `service._run_multi_feed` sets
   `ServiceResult.success=False` and aggregates `error` text; the CLI / acceptance harness
   then treats the run as **failed** even when most feeds succeeded.
2. **Terminal `Error:` line:** A single concatenated message listing every bad RSS URL is
   easy to read as a fatal crash, even though the session already wrote summaries and ran
   follow-up steps (e.g. acceptance analysis).
3. **Semantic mismatch:** Some failures are **data / network** (404 RSS), some are **bugs**
   (#558), some are **policy / limits** (#557). Lumping them all into one “failed run”
   bucket makes CI and manual runs harder to interpret.

## Design intent to capture

| Layer | Desired behavior |
| --- | --- |
| **Orchestration** | Always finish **finalize** (manifest, summary, optional index) when safe; never lose partial work silently. |
| **Reporting** | Keep `corpus_run_summary.json` `overall_ok` accurate (not all feeds ok). Optionally add **`soft_failures`** or **`hard_failures`** counts for triage. |
| **Exit code** | Separate **“artifacts written”** from **“all feeds green”**. Options: new config flag (e.g. `treat_feed_errors_as_warnings`), or CLI flag `--best-effort-exit-zero`, or distinct exit codes (documented). |
| **Failures** | **Skip** with reason for: RSS 404, optional oversize Whisper (after #557), optional decode (after #558). **Fail** or **non-zero** for: config invalid, lock failure, total loss of corpus root. |

## Follow-up

**#559** exit semantics and optional extra triage fields remain as design notes. **#557** and **#558** behavior is implemented in code (CONFIGURATION.md). Keep this file as narrative until exit-code / reporting follow-ups ship, then trim or link from `docs/guides/` if maintainers want it public.
