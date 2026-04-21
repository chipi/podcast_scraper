# Agent-Pipeline Feedback Loop Guide

**Status:** Reference Guide
**Applies to:** Local development, CI diagnosis, acceptance testing, pipeline
monitoring and post-mortem analysis
**Last updated:** April 2026

---

## Overview

This guide documents the **closed feedback loop between an AI coding agent and the
Python pipeline**. The goal: the agent gets structured, real-time feedback from
pipeline runs instead of you copy-pasting terminal output.

The pipeline and monitor system (RFC-064, RFC-065) produce structured artifacts —
`.pipeline_status.json`, `.monitor.log`, `metrics.json`, and more. The agent reads
these directly as files. During long runs with `--monitor`, the agent can observe
stage progress and resource usage in real time. After a run, the agent performs a
structured post-mortem without you scrolling through logs.

**Companion guide:** [Agent-Browser Closed Loop Guide](AGENT_BROWSER_LOOP_GUIDE.md)
covers the browser-side loops (Playwright E2E, Chrome DevTools MCP as the **default** for interactive repro/validate, Playwright MCP when clearly better for scripted drive, live
co-development). Both guides share the same principle — give the agent direct access
to structured feedback instead of copy-paste.

---

## The problem today

You run an acceptance test or `make ci`. Something goes wrong. You copy-paste the
terminal output into the agent chat. The agent reads it, asks clarifying questions,
you scroll back for more context, paste again. This is slow and lossy — you lose
structured data, timing, and the agent can't inspect files the run produced.

---

## What the agent can read directly

The pipeline and monitor system produce structured artifacts that the agent can read
as files — no copy-paste needed:

| Artifact | When available | What it contains |
| -------- | -------------- | ---------------- |
| `.pipeline_status.json` | During run (live) | Current stage, PID, `started_at`, `stage_started_at`, episode progress |
| `.monitor.log` | During + after run | Plain-text monitor snapshots (RSS, CPU%, stage, timing per tick) |
| `metrics.json` | After run | Per-stage wall time, per-episode timing, device info, token/cost |
| `run.json` | After run | Run metadata, config used, feed info |
| `fingerprint.json` | After run | Model versions, device, git commit, library versions |
| `debug/flamegraph_*.svg` | On demand (press `f`) | CPU flamegraph (py-spy, ~5s sample) |
| `debug/memray_*.bin` | After run (`--memray`) | Heap capture for post-mortem analysis |
| Terminal output | During + after | The agent can read Cursor terminal files directly |

All artifacts live under `<output_dir>/` (the pipeline's output directory). The
status file and monitor log are in the output root; debug artifacts are under
`<output_dir>/debug/`.

---

## During execution: real-time feedback

When you run a pipeline with `--monitor`, the agent can observe the run in progress:

```text
1. You: "Run a single-feed preset with monitor"
   → make test-acceptance CONFIGS="path/to/your_operator.yaml"
     (with monitor: true in the config)

2. During the run, the agent can:
   - Read .pipeline_status.json to see current stage + episode progress
   - Read .monitor.log for RSS/CPU% trend (non-TTY fallback)
   - Read the Cursor terminal file for live CLI output
   - Press 'f' context: remind you to press f for a flamegraph if a stage
     is suspiciously slow

3. Agent observes: "transcription stage has been running for 8 minutes,
   RSS climbed from 400MB to 1.2GB — this looks like a memory leak in
   the Whisper provider. Want me to check the flamegraph after this
   stage completes?"
```

### How the agent watches a long run

The agent backgrounds the command (or you start it yourself), then polls:

1. **Terminal file** — Cursor exposes each terminal as a text file the agent can
   read. The agent checks for new output, errors, and warnings.
2. **`.pipeline_status.json`** — the agent reads this to see which stage is active,
   how many episodes have been processed, and how long the current stage has been
   running.
3. **`.monitor.log`** — when the monitor's stderr is not a TTY (backgrounded
   commands), monitor snapshots are appended here as plain text. The agent reads
   the tail for the latest RSS/CPU% values.

The agent can report proactively: "Stage `transcription` has been running for 12
minutes with RSS at 1.4GB and climbing — this is unusual for a 3-episode feed."

---

## After execution: structured post-mortem

After a run completes (or fails), the agent reads the artifacts directly:

```text
1. You: "The acceptance run finished but something seemed slow — evaluate it"

2. Agent reads:
   - metrics.json → per-stage timing breakdown
   - .monitor.log → RSS/CPU% history across stages
   - run.json → config that was used, feed info
   - fingerprint.json → model versions, device
   - Terminal output → any warnings or errors

3. Agent reports:
   "Summarization took 4m 12s (67% of total wall time). RSS peaked at
    1.8GB during transcription. The Whisper model loaded was large-v3
    on CPU — switching to base.en would cut transcription time ~4x.
    The GI generation stage had 3 retries (visible in terminal output)
    due to rate limiting."
```

### Comparing runs

When you adjust config and re-run, the agent can compare `metrics.json` between
runs:

```text
You: "Summarization is too slow — switch to deepseek and re-run"
    ↓
Agent edits config, re-runs acceptance
    ↓
Agent compares metrics.json (run 1 vs run 2):
  "Summarization dropped from 4m 12s to 1m 03s (75% reduction).
   Total wall time: 8m 41s → 5m 32s. RSS peak unchanged (1.8GB —
   dominated by Whisper, not the summarizer)."
```

For visual comparison across many runs, the Streamlit run comparison tool
(`make run-compare`, RFC-047/066) provides charts and delta tables.

---

## Use cases

### UC-1: Acceptance test loop

The primary workflow. You run an acceptance test, the agent evaluates the results,
you iterate on config or code.

```text
You: "Run the journal acceptance test and evaluate the results"
    ↓
Agent runs: make test-acceptance CONFIGS="path/to/your_operator.yaml"
    ↓
Agent reads terminal output + metrics.json + .monitor.log
    ↓
Agent reports: timing breakdown, any failures, resource usage
    ↓
You: "Summarization is too slow — try with a different provider"
    ↓
Agent adjusts config, re-runs, compares metrics.json between runs
    ↓
Satisfied → commit config change
```

### UC-2: CI failure diagnosis

`make ci` or `make ci-fast` fails. The agent reads the terminal output directly
instead of you copy-pasting.

```text
make ci fails
    ↓
Agent reads terminal output (Cursor terminal file) directly
    ↓
Agent identifies: "test_corpus_search.py::test_semantic_query failed —
  IndexError in search/indexer.py line 142"
    ↓
Agent reads the source, diagnoses, fixes, re-runs make ci-fast
    ↓
Green → done
```

### UC-3: Monitoring a long pipeline run

You start a multi-feed acceptance run that takes 20+ minutes. Instead of watching
the terminal, you let the agent watch and report.

```text
You: "Run the multi-feed acceptance test with monitor, let me know
     if anything looks wrong"
    ↓
Agent backgrounds the command, polls .pipeline_status.json every ~30s
    ↓
Agent: "Feed 1/2 complete (5m 22s). Feed 2 started, currently on
  transcription stage, episode 2/4. RSS at 890MB, CPU 34%."
    ↓
Agent: "Warning — transcription stage on feed 2 has been running for
  9 minutes. RSS climbed to 1.6GB. Previous feed's transcription
  took 3 minutes. Something may be wrong."
    ↓
You: "Press f for a flamegraph"
    ↓
Agent: "Flamegraph saved to debug/flamegraph_20260411_143022.svg.
  Reading it... 78% of CPU time is in whisper.transcribe() —
  this episode's audio is 2h long (vs ~30min for others)."
```

### UC-4: Performance regression detection

You suspect a code change made the pipeline slower. The agent compares before/after.

```text
You: "Run the journal acceptance on this branch and compare with
     the frozen profile from the last release"
    ↓
Agent runs acceptance, reads metrics.json
    ↓
Agent reads data/profiles/v0.42.yaml (frozen release profile)
    ↓
Agent compares:
  "transcript_cleaning: 12.3s → 28.7s (+133%). This stage now
   includes the new hybrid_ml preprocessing (RFC-042). RSS unchanged.
   All other stages within 10% of baseline."
    ↓
You: "Is that expected?"
    ↓
Agent reads the code change: "Yes — hybrid_ml runs an additional
  spaCy pass. The wall time increase is proportional to the extra
  NLP work. RSS is flat because spaCy was already loaded."
```

---

## The parallel to the browser loop

| Browser loop | Python pipeline loop |
| ------------ | -------------------- |
| Agent sees DOM, console, network | Agent sees `.pipeline_status.json`, `.monitor.log`, `metrics.json` |
| Vite hot-reload shows code changes | Re-run with adjusted config shows effect |
| `make test-ui-e2e` is the gate | `make ci` / `make ci-fast` / acceptance is the gate |
| Chrome DevTools MCP attaches to browser | Agent reads pipeline artifacts as files |
| You direct: "fix this CSS" | You direct: "that stage is too slow, investigate" |

---

## What's not yet automated

- The agent cannot currently **start** a pipeline run and **watch** it autonomously
  in a single turn — it runs the command, waits for it to finish, then reads output.
  For long runs (acceptance tests, full CI), this means the agent blocks or you
  background the command and the agent polls the terminal file.
- Flamegraph capture (`f` keypress) requires your input during the run — the agent
  cannot press keys in your terminal.
- Memray `.bin` files require `memray flamegraph` to convert — the agent can run
  this but the output is an HTML file, not directly inspectable as text.

These are workflow friction points, not blockers. The structured artifacts
(`.pipeline_status.json`, `.monitor.log`, `metrics.json`) are the high-value path.

---

## Quick reference

```bash
# Run CI (unit + integration + E2E)
make ci-fast

# Run acceptance test (with monitor if config has monitor: true)
make test-acceptance CONFIGS="path/to/your_operator.yaml"

# Read pipeline status during a run
cat output/.pipeline_status.json

# Read monitor log after a run
cat output/.monitor.log

# Read per-stage metrics after a run
cat output/metrics.json

# Compare with frozen release profile
python scripts/eval/profile/diff_profiles.py data/profiles/v0.42.yaml data/profiles/v0.43.yaml

# Run comparison tool (Streamlit, visual)
make run-compare
```

---

## Related documentation

| Document | Relationship |
| -------- | ------------ |
| [Agent-Browser Closed Loop Guide](AGENT_BROWSER_LOOP_GUIDE.md) | Browser-side loops (Playwright E2E, Chrome DevTools MCP, live co-development) |
| [Live Pipeline Monitor](LIVE_PIPELINE_MONITOR.md) | `--monitor`, `.pipeline_status.json`, `.monitor.log` (RFC-065) |
| [Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md) | Frozen release profiles, `freeze_profile.py` (RFC-064) |
| [Pipeline and Workflow Guide](PIPELINE_AND_WORKFLOW.md) | Pipeline stages, module roles, `metrics.json` |
| [Testing Guide](TESTING_GUIDE.md) | `make ci-fast`, `make test`, test commands |
| [Run Comparison Tool](https://github.com/chipi/podcast_scraper/blob/main/tools/run_compare/README.md) | Streamlit UI for comparing runs (RFC-047/066) |

### RFCs

- [RFC-064: Performance Profiling and Release Freeze](../rfc/RFC-064-performance-profiling-release-freeze.md) — frozen profiles, `freeze_profile.py`
- [RFC-065: Live Pipeline Monitor](../rfc/RFC-065-live-pipeline-monitor.md) — `--monitor`, status file, dashboard, py-spy/memray
- [RFC-066: Run Comparison — Performance Tab](../rfc/RFC-066-run-compare-performance-tab.md) — Streamlit performance tab
