# RFC-065: Live Pipeline Monitor (macOS Developer Tooling)

## Status

**Draft — Stub** (split from RFC-064)

## RFC Number

065

## Authors

Podcast Scraper Team

## Date

2026-04-09

## Related RFCs

- `docs/rfc/RFC-064-performance-profiling-release-freeze.md` — Parent RFC; frozen profiles and release freeze framework
- `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md` — Quality benchmarking framework

## Related ADRs

- ADR-027: Deep Provider Fingerprinting

---

## Abstract

This RFC proposes a **live pipeline monitoring dashboard** for `podcast_scraper` — a `--monitor` CLI flag that spawns a real-time resource dashboard alongside a running pipeline. The developer sees CPU%, RSS, active pipeline stage, and elapsed time updating in real time without opening a second terminal manually.

This is macOS-specific developer tooling that complements the frozen profile system in RFC-064. Where RFC-064 captures a static snapshot at release time, this RFC provides a live view during development and debugging.

Split from RFC-064 to keep the release freeze framework focused on its core deliverable (frozen profiles + diff tool).

---

## Scope (To Be Designed)

The following capabilities were outlined in the original RFC-064 draft and are the starting point for this RFC's design:

- **`--monitor` CLI flag** on the main `podcast_scraper` command
- **Terminal split** — tmux pane split (if in tmux) or new Terminal.app window via `osascript` (macOS fallback)
- **`rich` Live dashboard** — CPU%, RSS, peak RSS, active stage, elapsed time, refreshing in-place
- **Stage awareness** — pipeline writes stage transitions to a status mechanism; monitor reads and displays
- **On-demand flamegraphs** — `py-spy` CPU flamegraph capture via keypress (`f`); SVG output to `debug/`
- **`memray` integration** — optional `--memray` flag wrapping the pipeline for memory allocation profiling
- **Final summary** — resource summary table displayed on clean pipeline exit

### Dependencies

```toml
[project.optional-dependencies]
monitor = [
    "psutil>=5.9.0",
    "rich>=13.0.0",
    "py-spy>=0.3.14",
    "memray>=1.5.0",
]
```

`py-spy` and `memray` are optional — the monitor runs without them; flamegraph capture is simply unavailable if not installed.

### Platform Considerations

- tmux detection and `osascript` Terminal.app fallback are macOS-specific
- `py-spy` requires SIP disabled or `sudo` on macOS for process attachment
- `memray` wraps the process, which may affect measurements

---

## Open Questions

1. Should the monitor be a separate process or an in-process thread?
2. How should stage transitions be communicated (status file, shared memory, pipe)?
3. Should the monitor work outside tmux/Terminal.app (e.g., plain terminal with cursor positioning)?
4. What is the minimum viable version — just RSS + stage name, or full dashboard from the start?

---

## References

- **Parent RFC**: `docs/rfc/RFC-064-performance-profiling-release-freeze.md`
- **Source Code**: `src/podcast_scraper/monitor/` (proposed)
