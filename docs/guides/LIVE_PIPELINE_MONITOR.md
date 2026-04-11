# Live Pipeline Monitor (RFC-065)

**Audience:** Developers who want a **live** view of pipeline stage, elapsed time, RSS, and CPU%
while a run is in progress.

**Normative design:** [RFC-065: Live Pipeline Monitor](../rfc/RFC-065-live-pipeline-monitor.md)  
**Tracking:** [GitHub #512](https://github.com/chipi/podcast_scraper/issues/512)

---

## What it does

When **`monitor`** is enabled (`--monitor` on the CLI or `monitor: true` in config), the pipeline:

1. Starts a **child process** (multiprocessing **`spawn`**) before the rest of the run.
2. Writes **`<output_dir>/.pipeline_status.json`** on each orchestration stage transition
   (atomic replace — safe to inspect with `cat` while running).
3. The child process polls that file and **`psutil.Process(pipeline_pid)`** (~0.5 s), and renders
   a **`rich.Live`** panel on **stderr** when stderr is a **TTY**. If stderr is **not** a TTY, the
   same snapshots are appended as plain lines to **`<output_dir>/.monitor.log`** (RFC-065 fallback).
4. On pipeline exit, the child prints a **short per-stage summary table**, then the parent
   **joins** the monitor (with **terminate** fallback if it does not exit).

Core dependencies are **`psutil`** and **`rich`** (already required by the package). No extra
install is needed for the MVP dashboard.

---

## Quick start

```bash
python3 -m podcast_scraper.cli https://example.com/feed.xml --output-dir ./out --monitor
```

Config file:

```yaml
monitor: true
```

See also: [CLI.md](../api/CLI.md), [CONFIGURATION.md](../api/CONFIGURATION.md#live-pipeline-monitor-rfc-065-512).

---

## Artifacts

| Path | Purpose |
| ---- | ------- |
| `<output_dir>/.pipeline_status.json` | Current `stage`, `pid`, `started_at`, `stage_started_at`, optional `episode_total` / `episode_idx`. |
| `<output_dir>/.monitor.log` | Plain-text monitor lines when the monitor’s **stderr** is not a TTY. |
| `<output_dir>/debug/flamegraph_<timestamp>.svg` | Optional **py-spy** CPU flamegraph (press **`f`** in the parent TTY when **`py-spy`** is installed; requires **`--monitor`**). |
| `<output_dir>/debug/memray_<timestamp>.bin` | Optional **memray** heap capture when **`--memray`** / `memray: true`. |

---

## Stage strings

Names follow the **RFC-064** stage model where hooks exist in `workflow/orchestration.py`, for
example: `rss_feed_fetch`, `speaker_detection`, `media_download`, `transcript_cleaning` (when
metadata generation runs), `transcription` (when the Whisper concurrent path runs),
`audio_preprocessing`, `summarization` (parallel summarization path only), `vector_indexing`,
`done`.

**Caveat:** **GI** and **KG** work largely inside the concurrent metadata thread; the status file
does not yet advance through distinct **`gi_generation`** / **`kg_extraction`** stages (see RFC
Status / Implementation Plan).

---

## Multi-feed corpus (GitHub #440)

The CLI runs **one pipeline per feed**. Each feed has its own **`output_dir`**; with `--monitor`,
each inner run may spawn its **own** monitor bound to that directory.

---

## Service mode

**`python -m podcast_scraper.service`** has no dedicated **`--monitor`** flag. You can set
**`monitor: true`** in the service YAML for the same behavior. Set **`memray: true`** (and
optionally **`memray_output`**) in YAML to re-exec the service under memray; if **`memray`** is
requested but the **`memray`** executable is missing, **`run_from_config_file`** returns a
**`ServiceResult`** with **`success=False`** and an error message instead of starting the pipeline.

---

## Stderr and logging

The dashboard is drawn on **stderr**. Depending on logging configuration, **log lines may
interleave** with the live panel. For a cleaner interactive view, use **`--log-file`** (or
equivalent config) so human-readable logs are not mixed on the same stream.

---

## Optional profiling extras (`.[monitor]`)

```bash
pip install -e ".[monitor]"
```

Pulls **`py-spy`** and **`memray`** (RFC-065 Phase 3). They are **not** required for **`--monitor`**.

- **`--memray`** / **`memray: true`**: the CLI or service **re-execs** under
  **`memray run -o <path> -f -m podcast_scraper.cli|service …`**. The child sets
  **`PODCAST_SCRAPER_MEMRAY_ACTIVE=1`** so it does not loop. Default capture path: **`debug/memray_*.bin`**
  under the effective output directory (or under **`./debug`** when only the corpus parent / cwd applies).
- **`--memray-output` / `memray_output`**: explicit **`.bin`** path.
- **Press `f`** (parent process, **TTY stdin**): records ~**5 s** of CPU samples with **`py-spy record`**
  into **`debug/flamegraph_<timestamp>.svg`** (Unix-like platforms where **`select`** on stdin works;
  requires **`py-spy`** on **`PATH`**). Only started when **`--monitor`** is on.

---

## Tests

- **`tests/unit/podcast_scraper/monitor/`** — **`test_status`**, **`test_sampler`**, **`test_memray_util`**,
  **`test_py_spy_listener`** (mocked exec and stdin paths).

---

## Related guides

| Guide | Relationship |
| ----- | ------------ |
| [Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md) | **Frozen** release YAML under `data/profiles/` (RFC-064) — complementary, not the same as live monitor. |
| [Performance](PERFORMANCE.md) | Runtime tuning (preprocessing cache, transcription, etc.). |
| [Pipeline and Workflow Guide](PIPELINE_AND_WORKFLOW.md) | End-to-end flow and module roles. |
