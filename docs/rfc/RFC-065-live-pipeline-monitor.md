# RFC-065: Live Pipeline Monitor (macOS Developer Tooling)

## Status

**Completed (v2.6.0)** — core scope delivered; **tmux / automatic terminal split** remains deferred (see below).

**MVP + Phase 3 profiling implemented** (2026-04-11, GitHub #512):

- **MVP:** `--monitor`, `monitor` in YAML, `.pipeline_status.json`,
  `src/podcast_scraper/monitor/` (status, sampler, dashboard, runner subprocess), orchestration
  hooks for RFC-064 stage names (except distinct **gi_generation** / **kg_extraction** signals —
  those run inside the processing thread; the monitor stays on **transcript_cleaning** until
  **summarization** / **vector_indexing**).
- **Non-TTY fallback:** When the monitor subprocess’s **stderr** is not a TTY, plain-text lines
  append to **`<output_dir>/.monitor.log`** (same data as the live dashboard ticks).
- **Phase 3 (optional `.[monitor]` extra):** **`--memray`** / **`memray:`** + **`memray_output`**
  re-exec the CLI or service under **`memray run`** (env **`PODCAST_SCRAPER_MEMRAY_ACTIVE=1`**
  prevents loops). Parent-process **TTY stdin** **`f`** triggers **`py-spy record`** (~5 s) to
  **`debug/flamegraph_<timestamp>.svg`** when **`--monitor`** is on (Unix-like; see guide).
- **Still deferred:** **tmux** / **Terminal.app** automatic split (**`terminal_split`**) — optional
  future UX; developers can use an external split terminal today.

**Documentation:** [Live Pipeline Monitor guide](../guides/LIVE_PIPELINE_MONITOR.md)
(quickstart, artifacts, multi-feed, stderr vs logs, memray/py-spy). API tables:
[CLI.md](../api/CLI.md), [CONFIGURATION.md](../api/CONFIGURATION.md#live-pipeline-monitor-rfc-065-512).

## RFC Number

065

## Authors

Podcast Scraper Team

## Date

2026-04-09 (stub) · 2026-04-10 (expanded)

## Related RFCs

- `docs/rfc/RFC-064-performance-profiling-release-freeze.md` — Parent RFC;
  frozen profiles and release freeze framework
- `docs/rfc/RFC-066-run-compare-performance-tab.md` — Sibling RFC; Streamlit
  performance tab consuming frozen profiles
- `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md` — Quality
  benchmarking framework

## Related ADRs

- ADR-027: Deep Provider Fingerprinting

## Related Issues

- [#512](https://github.com/chipi/podcast_scraper/issues/512) — Tracking issue
- [#510](https://github.com/chipi/podcast_scraper/issues/510) — RFC-064 epic

---

## Abstract

A **live pipeline monitoring dashboard** for `podcast_scraper` — a `--monitor`
CLI flag that spawns a real-time resource dashboard alongside a running
pipeline. The developer sees CPU%, RSS, active pipeline stage, and elapsed time
updating in real time without opening a second terminal manually.

This is developer tooling that complements the frozen profile system in
RFC-064. Where RFC-064 captures a static snapshot at release time, this RFC
provides a live view during development and debugging. The core dashboard
(`psutil` + `rich`) is cross-platform; the terminal split UX (tmux pane split,
`Terminal.app` via `osascript`) is macOS-focused.

Split from RFC-064 to keep the release freeze framework focused on its core
deliverable (frozen profiles + diff tool).

---

## Motivation

During pipeline development and debugging, the developer currently has no
real-time visibility into resource usage per stage. The workflow is:

1. Run the pipeline.
2. Wait for it to finish.
3. Inspect `metrics.json` or run `freeze_profile.py` after the fact.

This is slow feedback. A live monitor lets the developer:

- See which stage is currently running and how long it has been active.
- Watch RSS climb in real time to catch memory leaks early.
- Spot CPU spikes or drops that indicate bottlenecks.
- Get a final summary table without running a separate script.

---

## Design

### Architecture: Separate Process

The monitor runs as a **separate process** that observes the pipeline process
via `psutil` and reads stage status from a shared status file. This design was
chosen over in-process threading because:

- **Isolation**: the monitor's `rich` rendering and `psutil` polling do not
  compete with the pipeline for CPU or GIL time.
- **Crash independence**: if the pipeline crashes, the monitor detects it
  (process gone) and shows a clean exit summary.
- **Simplicity**: no threading coordination, no shared memory, no import-time
  side effects in the pipeline code.

The pipeline process writes a lightweight status file; the monitor process
reads it. Both use the pipeline's PID for coordination.

### Stage Signaling: Status File

The pipeline writes a JSON status file at each stage transition. The monitor
polls this file (same interval as `psutil` sampling, default 0.5s).

**Status file location**: `<output_dir>/.pipeline_status.json`

**Status file schema**:

```json
{
  "pid": 12345,
  "stage": "summarization",
  "episode_idx": 3,
  "episode_total": 10,
  "started_at": 1712764800.0,
  "stage_started_at": 1712764850.0
}
```

**Why a status file (not shared memory or a pipe)**:

- **Easy to integrate**: one `json.dump()` call at each stage boundary in
  `orchestration.py`. No new dependencies, no IPC complexity.
- **Debuggable**: the developer can `cat` the file at any time.
- **Atomic on macOS/Linux**: write to a temp file + `os.rename()` ensures the
  monitor never reads a partial write.
- **Already precedented**: `freeze_profile.py` uses a similar pattern
  (metrics file written by pipeline, read by profiler).

**Pipeline integration points** — the following locations in
`src/podcast_scraper/workflow/orchestration.py` get a
`_write_pipeline_status(output_dir, stage, ...)` call:

| Stage name | Location in orchestration |
| ---------- | ------------------------ |
| `rss_feed_fetch` | Before `_fetch_and_prepare_episodes()` |
| `speaker_detection` | Before `_setup_pipeline_resources()` (host detection) |
| `media_download` | Before episode download loop |
| `audio_preprocessing` | Before preprocessing in episode processing |
| `transcription` | Before transcription stage |
| `transcript_cleaning` | Before cleaning stage |
| `summarization` | Before summarization stage |
| `gi_generation` | Before GI generation |
| `kg_extraction` | Before KG extraction |
| `vector_indexing` | Before vector index build |
| `done` | After pipeline completion |

The `_write_pipeline_status()` helper is a thin function (~15 lines) added to
`orchestration.py`. It is a no-op when `--monitor` is not active (controlled
by a `Config` flag), so there is zero overhead in normal runs. Even when
active, the cost is one small JSON write per stage transition (not per
episode).

### Monitor Process Lifecycle

```text
CLI: podcast_scraper --monitor ...
  │
  ├─ Fork monitor subprocess (target: _monitor_main)
  │    └─ Monitor reads .pipeline_status.json + psutil.Process(pipeline_pid)
  │    └─ Renders rich.Live dashboard every 0.5s
  │    └─ On pipeline exit: prints final summary table, exits
  │
  └─ Run pipeline normally (writes status file at stage boundaries)
       └─ On completion: writes stage="done", exits
```

The monitor subprocess is started before the pipeline begins and is joined
after the pipeline returns. If the user Ctrl-C's, the pipeline's signal
handler cleans up and the monitor detects process exit.

### Terminal Split (Optional UX Enhancement)

When `--monitor` is passed:

1. **tmux detected** (`$TMUX` set): split the current pane horizontally
   (`tmux split-window -h`) and run the monitor in the new pane. The pipeline
   runs in the original pane.
2. **macOS Terminal.app** (no tmux): use `osascript` to open a new
   Terminal.app window running the monitor.
3. **Fallback** (neither tmux nor macOS): run the monitor as a background
   process; output goes to `<output_dir>/.monitor.log`. A message tells the
   developer to `tail -f` it.

The terminal split is a UX convenience. The monitor works identically in all
three modes — only the terminal arrangement differs.

### Rich Live Dashboard

The monitor renders a `rich.Live` display refreshing every 0.5 seconds:

```text
┌─────────────────────────────────────────────────┐
│  podcast_scraper — Live Monitor                 │
│  PID: 12345  ·  Elapsed: 2m 34s                │
├─────────────────────────────────────────────────┤
│  Stage: summarization  (3/10 episodes)          │
│  Stage elapsed: 45s                             │
├─────────────────────────────────────────────────┤
│  RSS:  542 MB  (peak: 617 MB)                   │
│  CPU:  24.8%                                    │
├─────────────────────────────────────────────────┤
│  Stage History:                                 │
│    rss_feed_fetch      1.2s    128 MB   12.3%   │
│    speaker_detection   3.9s    546 MB   24.9%   │
│    transcript_cleaning 87.6s   554 MB    0.2%   │
│    summarization       ...running...            │
└─────────────────────────────────────────────────┘
```

The dashboard shows:

- **Header**: PID, total elapsed time.
- **Current stage**: name, episode progress (if applicable), stage elapsed.
- **Resource gauges**: current RSS, peak RSS, current CPU%.
- **Stage history**: completed stages with wall time, peak RSS, avg CPU%.

### On-Demand Flamegraphs (Optional)

When **`py-spy`** is installed (**`pip install -e ".[monitor]"`** or **`pip install py-spy`**),
**`--monitor`** is on, and the **parent** pipeline process has a **TTY stdin** (Unix-like;
**`select`** on stdin), a daemon thread watches for **`f`** and runs:

`py-spy record -p <parent_pid> -d 5 -o <output_dir>/debug/flamegraph_<timestamp>.svg`

The monitor **child** does not receive reliable stdin under **`spawn`**, so the handler lives in
**`workflow/orchestration`** (via **`py_spy_listener.start_py_spy_stdin_listener`**). On success or
failure, a short message is printed to **stderr**. If **`py-spy`** is missing, a **one-time** hint
is emitted when the listener would have started. On macOS, attaching by PID may require SIP settings
or **`sudo`** (same as upstream **py-spy**).

### Memray Integration (Optional)

**`--memray`** / **`memray: true`** (and optional **`--memray-output`** / **`memray_output`**)
re-execs under **`memray run -o <path> -f -m podcast_scraper.cli|service …`**. Orthogonal to
**`--monitor`** — can be used together or alone. The child environment sets
**`PODCAST_SCRAPER_MEMRAY_ACTIVE=1`** so the new process does not re-exec again.

When both **memray** and **monitor** are active, the dashboard includes a line that memray capture
is active (see **`dashboard.py`**). Analyze the **`.bin`** after the run with **`memray flamegraph`**
(or other memray reporters); that step is not automated by this RFC.

### Final Summary

When the pipeline exits cleanly, the monitor prints a summary table (similar
to `diff_profiles.py` output) before exiting:

```text
Pipeline completed in 3m 12s

Stage                Wall (s)  Peak RSS (MB)  Avg CPU%
─────────────────────────────────────────────────────
rss_feed_fetch           1.2          128      12.3
speaker_detection        3.9          546      24.9
transcript_cleaning     87.6          554       0.2
summarization          106.8          554       0.2
gi_generation           27.8          555       0.2
kg_extraction            5.6          555       0.2
vector_indexing          0.3          555       0.0
─────────────────────────────────────────────────────
Total                  233.2          617       5.4
```

If the pipeline crashes, the monitor shows the last known stage and resource
state, plus *"Pipeline process exited with code N"*.

---

## Module Layout

```text
src/podcast_scraper/monitor/
├── __init__.py          # re-exports start_monitor_subprocess from runner
├── runner.py            # monitor_main(), start_monitor_subprocess() (multiprocessing spawn)
├── dashboard.py         # rich.Live rendering + summary table
├── sampler.py           # Cross-process psutil polling (background thread)
├── status.py            # Atomic read/write .pipeline_status.json
├── memray_util.py       # memray re-exec helpers (CLI + service)
└── py_spy_listener.py   # Optional stdin 'f' → py-spy record (started from orchestration)
```

**Deferred (not in tree):** **`terminal_split.py`** — tmux / **Terminal.app** split helper.

The `sampler.py` module reuses the `ResourceSampler` pattern from
`scripts/eval/freeze_profile.py` (background thread polling `psutil.Process`).
The key difference is that the monitor's sampler runs in the monitor process
targeting the pipeline's PID (cross-process), while `freeze_profile.py`'s
sampler targets its own process.

---

## CLI Integration (implemented)

The main CLI uses **argparse** (not Click).

### Config

- **`monitor`**: `bool`, default **`false`**, field on `Config` in `src/podcast_scraper/config.py`.
- **`memray`**: `bool`, default **`false`** — triggers **`memray run`** re-exec from CLI or
  **`service.run_from_config_file`** (not from bare **`service.run(cfg)`** without going through
  that entrypoint).
- **`memray_output`**: optional **`str`** — explicit capture **`.bin`** path; default under
  **`debug/`** (see [CONFIGURATION.md](../api/CONFIGURATION.md#live-pipeline-monitor-rfc-065-512)).

### CLI flags

- **`--monitor`** — parsed in **`_add_common_arguments()`**; passed through **`_build_config()`**
  into **`Config.monitor`**.
- **`--memray`**, **`--memray-output`** — same; merged from YAML when using **`--config`**.

### Orchestration

- **`workflow/orchestration.py`**: After **`_setup_pipeline_environment()`**, if **`cfg.monitor`**,
  calls **`start_monitor_subprocess(...)`** and **`start_py_spy_stdin_listener(...)`** (optional
  **`f`** → **py-spy**).
- The main body of **`run_pipeline()`** is wrapped in **`try` / `finally`**: on success,
  **`maybe_update_pipeline_status(..., stage="done")`**; **`finally`** **`join`**s the monitor
  (terminate fallback if needed).
- Stage transitions call **`maybe_update_pipeline_status()`** from **`monitor/status.py`**
  (no-op when **`monitor`** is false).

---

## Dependencies

### Core (No New Packages)

`psutil` and `rich` are already project dependencies. The monitor module uses
only these two plus stdlib (`json`, `os`, `time`, `multiprocessing`).

### Optional (`[monitor]` Extra)

```toml
[project.optional-dependencies]
monitor = [
    "py-spy>=0.3.14",
    "memray>=1.5.0",
]
```

`py-spy` and `memray` are optional — the monitor runs without them;
**`f`** flamegraph capture and **memray** re-exec are unavailable (or fail fast with a clear message)
if the binaries are not on **`PATH`**.

### Platform Considerations

- **tmux detection**: checks `$TMUX` environment variable (cross-platform).
- **`osascript`**: macOS-only; used only as fallback when tmux is unavailable.
- **`py-spy`**: requires SIP disabled or `sudo` on macOS for process
  attachment. Works without elevation on Linux.
- **`memray`**: wraps the process; may affect measurements slightly.

---

## Implementation Plan

### Phase 1: Status File + Core Monitor (MVP)

1. Add `_write_pipeline_status()` to `orchestration.py` (no-op when
   `cfg.monitor` is False).
2. Create `src/podcast_scraper/monitor/status.py` — read/write helpers.
3. Create `src/podcast_scraper/monitor/sampler.py` — cross-process
   `ResourceSampler` using `psutil.Process(pid)`.
4. Create `src/podcast_scraper/monitor/dashboard.py` — `rich.Live` rendering.
5. Create `src/podcast_scraper/monitor/__init__.py` — `start_monitor()` that
   spawns the monitor as a `multiprocessing.Process`.
6. Add `--monitor` flag to CLI and `monitor` field to `Config`.
7. Wire `start_monitor()` into `run_pipeline()`.
8. Unit tests for status file read/write, sampler, dashboard rendering.

### Phase 2: Terminal Split + Final Summary

1. **Deferred:** `terminal_split.py` — tmux / osascript auto-split (optional UX).
2. **Done:** Final summary table on pipeline exit (**`rich`** or plain text for **`.monitor.log`**).
3. **Done:** Pipeline crash detection (process gone).

### Phase 3: Flamegraph + Memray (Optional)

1. **Done:** Parent-process stdin **`f`** → **`py-spy record`** (**`py_spy_listener.py`** +
   orchestration wiring).
2. **Done:** **`--memray`** / **`memray`**, **`memray_output`**, **`memray run`** re-exec
   (**`memray_util.py`**, CLI + **`run_from_config_file`**).
3. **Done:** **`[monitor]`** optional dependency group in **`pyproject.toml`**.

---

## Testing Strategy

- **Unit tests** (`tests/unit/podcast_scraper/monitor/`):
  - `test_status.py`: write/read status file, atomic write, missing file.
  - `test_sampler.py`: sampler start/stop, cross-process sampling (mock
    `psutil.Process`).
  - `test_memray_util.py` / `test_py_spy_listener.py`: re-exec and listener edge cases (mocked).
- **Integration test** (`tests/integration/monitor/`):
  - Spawn a dummy long-running process, start monitor, verify it reads status
    and samples resources, verify clean shutdown.
- **Manual QA**: run `podcast_scraper --monitor` with a real pipeline config
  and verify the dashboard renders correctly in tmux and plain terminal.

---

## Open Questions (Resolved)

| # | Question | Resolution |
| - | -------- | ---------- |
| 1 | Separate process or in-process thread? | Separate process — isolation from GIL, crash independence, simpler coordination. |
| 2 | Stage signaling mechanism? | Status file (`.pipeline_status.json`) — easy integration (one `json.dump` per stage), debuggable (`cat`), atomic on POSIX, already precedented by freeze_profile pattern. |
| 3 | Work outside tmux/Terminal.app? | Yes — fallback mode writes to `.monitor.log` with `tail -f` instruction. Core dashboard logic is the same in all modes. |
| 4 | MVP scope? | MVP: stages + dashboard + summary + **`.monitor.log`** non-TTY fallback. Optional **`.[monitor]`**: **py-spy** + **memray**. **Terminal split** still deferred. |

---

## References

- **Parent RFC**: `docs/rfc/RFC-064-performance-profiling-release-freeze.md`
- **Source Code**: `src/podcast_scraper/monitor/` (implemented)
- **Operator guide**: `docs/guides/LIVE_PIPELINE_MONITOR.md`
- **Sampler Pattern**: `scripts/eval/freeze_profile.py` `ResourceSampler`
- **Tracking Issue**:
  [#512](https://github.com/chipi/podcast_scraper/issues/512)
