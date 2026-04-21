# RFC-065: Live Pipeline Monitor (Developer Tooling)

- **Status**: Completed (v2.6.0)
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Developers and operators debugging long pipeline runs
- **Related PRDs**:
  - [PRD-016: Operational Observability & Pipeline Intelligence](../prd/PRD-016-operational-observability-pipeline-intelligence.md) —
    live visibility into runs (with RFC-027 / RFC-064 family)
- **Related ADRs**:
  - [ADR-075: Frozen YAML performance profiles](../adr/ADR-075-frozen-yaml-performance-profiles-for-release-baselines.md) — sibling to this **live** view ([RFC-064](RFC-064-performance-profiling-release-freeze.md))
  - [ADR-014: Codified Comparison Baselines](../adr/ADR-014-codified-comparison-baselines.md)
  - [ADR-027: Unified Provider Metrics Contract](../adr/ADR-027-unified-provider-metrics-contract.md)
  - [ADR-040: Explicit Golden Dataset Versioning](../adr/ADR-040-explicit-golden-dataset-versioning.md)
- **Related RFCs**:
  - [RFC-064: Performance profiling and release freeze](RFC-064-performance-profiling-release-freeze.md) —
    parent split; frozen profiles vs this live view
  - [RFC-066: Run compare — Performance tab](RFC-066-run-compare-performance-tab.md) — consumes frozen
    profiles alongside quality runs
  - [RFC-041: Podcast ML benchmarking framework](RFC-041-podcast-ml-benchmarking-framework.md) — quality
    benchmarking context
- **Related Documents**:
  - [Live Pipeline Monitor guide](../guides/LIVE_PIPELINE_MONITOR.md) — operator quickstart, artifacts,
    multi-feed, stderr vs log, optional **memray** / **py-spy**
  - [CLI.md](../api/CLI.md), [CONFIGURATION.md](../api/CONFIGURATION.md#live-pipeline-monitor-rfc-065-512)
  - [GitHub #512](https://github.com/chipi/podcast_scraper/issues/512) — tracking
  - [GitHub #510](https://github.com/chipi/podcast_scraper/issues/510) — RFC-064 epic (profiles)
- **Updated**: 2026-04-09 (stub), 2026-04-10 (expanded), 2026-04-12 (delivered-scope + style alignment),
  2026-04-11 (terminal-split deferred + implementation record + lifecycle accuracy)

## Abstract

A **live pipeline monitoring dashboard** for `podcast_scraper`: when **`monitor`** is enabled
(`--monitor` or `monitor: true` in YAML), a **child process** polls **`.pipeline_status.json`** and
**`psutil.Process(pipeline_pid)`**, then renders a **`rich.Live`** panel on **stderr** (or appends
plain lines to **`.monitor.log`** when stderr is not a TTY). On exit it prints a **per-stage
summary** table. Core stack uses only **`psutil`**, **`rich`**, and stdlib (**no extra install**).

**Optional** **`pip install -e ".[monitor]"`**: **`memray`** re-exec (**`--memray`** / YAML **`memray:`**,
**`memray_output`**) and parent-TTY **`f`** → **`py-spy record`** flamegraph under **`debug/`** (see
guide for SIP / **`sudo`** notes on macOS).

This complements **RFC-064** (frozen release profiles): RFC-064 is a **static** capture; this RFC is
a **live** view during development. **Not shipped:** automatic **tmux** / **Terminal.app** split
(**`terminal_split.py`** was designed but is **not in the tree** — use an external split terminal or
**`tail -f .monitor.log`**).

## Delivered scope (v2.6.0)

| Area | Shipped |
| ---- | ------- |
| **CLI + config** | **`--monitor`**, **`monitor:`** in YAML; **`--memray`**, **`--memray-output`**, **`memray:`**, **`memray_output`** |
| **Service** | **`monitor: true`** / **`memray:`** via config; **`run_from_config_file`** re-execs under **memray** or returns **`ServiceResult`** error if **memray** missing |
| **Process model** | Child monitor (**`multiprocessing` `spawn`**), parent pipeline; **`join`** in **`finally`** with terminate fallback |
| **Status file** | **`<output_dir>/.pipeline_status.json`** — atomic writes on stage transitions (**`monitor/status.py`**) |
| **Dashboard** | **`rich.Live`** on stderr when TTY; else **`.monitor.log`** one line per tick (**same fields**) |
| **Profiling (optional)** | **`memray_util.py`**: **`PODCAST_SCRAPER_MEMRAY_ACTIVE=1`** guard; **`py_spy_listener`**: stdin **`f`** in **parent** (orchestration), **`debug/flamegraph_*.svg`** |
| **Orchestration hooks** | **`maybe_update_pipeline_status()`** at RFC-064-aligned stage names; **no-op** when **`monitor`** is false |
| **Tests** | **`tests/unit/podcast_scraper/monitor/`** (status, sampler, dashboard pieces, memray, py-spy listener); integration coverage as in guide |
| **Deferred** | **`terminal_split.py`** (tmux / **osascript** auto-window) — **not implemented**; finer-grained **GI** / **KG** stage lines in status file (work largely inside concurrent metadata path — monitor may show **`transcript_cleaning`** through **`summarization`** / **`vector_indexing`**; see [LIVE_PIPELINE_MONITOR.md](../guides/LIVE_PIPELINE_MONITOR.md)) |

## Problem Statement

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
`src/podcast_scraper/workflow/orchestration.py` call
`maybe_update_pipeline_status(cfg, output_dir, stage=...)` (from **`monitor/status.py`**):

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

`maybe_update_pipeline_status()` is a no-op when **`cfg.monitor`** is false, so there is zero
overhead in normal runs. When enabled, cost is one small atomic JSON write per stage transition
(not per episode).

### Monitor Process Lifecycle

```text
CLI: python -m podcast_scraper.cli … --monitor
  │
  ├─ spawn monitor process (start_monitor_subprocess → _monitor_entry)
  │    └─ Reads .pipeline_status.json + psutil.Process(pipeline_pid)
  │    └─ Renders rich.Live every poll interval (or .monitor.log if stderr not a TTY)
  │    └─ On pipeline exit: final summary table, then exit
  │
  └─ Run pipeline (maybe_update_pipeline_status at stage boundaries)
       └─ On completion: stage="done", exit
```

The monitor process is started before the pipeline begins and is joined after the pipeline returns.
On Ctrl-C, the pipeline’s signal handling and process exit are reflected by the monitor via **psutil**
and the status file.

### Deferred: automatic terminal split (not shipped)

An earlier design described **tmux** pane split, **macOS Terminal.app** via **`osascript`**, and a
**fallback** to **`.monitor.log`**. **Only the shipped behavior remains:** the monitor child is
always started via **`start_monitor_subprocess`**; **rich** renders to **stderr** when it is a TTY,
otherwise the same snapshots go to **`.monitor.log`** (see **`runner.py`**). There is **no**
**`terminal_split.py`**, no **`tmux`**, and no **`osascript`** automation in **`src/`**. Users who
want a second pane should split the terminal manually or **`tail -f .monitor.log`**.

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
`scripts/eval/profile/freeze_profile.py` (background thread polling `psutil.Process`).
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

### Platform considerations

- **`py-spy`**: on macOS, attaching by PID may require SIP changes or **`sudo`** (upstream
  **py-spy** behavior); Linux often works without elevation.
- **`memray`**: re-exec wraps the process; can slightly affect timing vs a plain run.
- **Cross-platform core**: **`psutil`** + **`rich`** + status file — no OS-specific code path for the
  dashboard itself.

---

## Implementation record (v2.6.0)

Phases below match what shipped; only **terminal split** remains explicitly out of scope.

| Phase | Scope | Result |
| ----- | ----- | ------ |
| **1** | Status file, child monitor, sampler, dashboard, CLI **`--monitor`** / **`Config.monitor`**, orchestration **`maybe_update_pipeline_status`**, unit tests | **Done** |
| **2** | Final summary table, crash detection, **`.monitor.log`** non-TTY path | **Done** |
| **2b** | Auto **tmux** / **Terminal.app** split (**`terminal_split.py`**) | **Deferred / not in tree** |
| **3** | **`[monitor]`** extra, **memray** re-exec, **py-spy** stdin **`f`**, service YAML **`memray:`** | **Done** |

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
- **Manual QA**: run **`python -m podcast_scraper.cli … --monitor`** with a real config; verify
  **rich** dashboard on a TTY **stderr** and **`.monitor.log`** lines when stderr is not a TTY;
  optionally **`pip install -e ".[monitor]"`** and exercise **memray** / **`f`** flamegraph per
  [LIVE_PIPELINE_MONITOR.md](../guides/LIVE_PIPELINE_MONITOR.md).

---

## Open Questions (Resolved)

| # | Question | Resolution |
| - | -------- | ---------- |
| 1 | Separate process or in-process thread? | Separate process — isolation from GIL, crash independence, simpler coordination. |
| 2 | Stage signaling mechanism? | Status file (`.pipeline_status.json`) — easy integration (one `json.dump` per stage), debuggable (`cat`), atomic on POSIX, already precedented by freeze_profile pattern. |
| 3 | Work outside tmux/Terminal.app? | Yes — when the monitor’s **stderr** is not a TTY, it appends plain lines to **`.monitor.log`** (header line + one line per tick + footer). Operators can **`tail -f`** that file; there is no in-app **tmux**/**osascript** split. |
| 4 | MVP scope? | **Shipped:** stages + dashboard + summary + **`.monitor.log`** non-TTY fallback. Optional **`.[monitor]`**: **py-spy** + **memray**. **Terminal split** not shipped. |

---

## References

- **Parent RFC**: [RFC-064](RFC-064-performance-profiling-release-freeze.md)
- **Source**: `src/podcast_scraper/monitor/`
- **Operator guide**: [LIVE_PIPELINE_MONITOR.md](../guides/LIVE_PIPELINE_MONITOR.md)
- **Sampler pattern**: `scripts/eval/profile/freeze_profile.py` (**ResourceSampler**)
- **Tracking**: [GitHub #512](https://github.com/chipi/podcast_scraper/issues/512)
