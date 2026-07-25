# Perf-trace reports

Timestamped performance-trace runs, one per release (or per defended tuning).
This is the cross-release history — the sibling of
[eval-reports](../../eval-reports/index.md), for performance.

Each report captures the standard scenarios (see [../index.md](../index.md)) at
a known git ref / release, records median-of-3 per scenario, and **compares to
the previous release's report** so regressions surface at release time.

## How a report is produced

1. Capture the current scenarios into `data/perf/traces/<surface>/` with a
   release label (see the per-area pages: [search](../search.md) ·
   [graph](../graph.md) · [surfaces](../surfaces.md)).
2. Author `<YYYY-MM-DD>-<release>.md` here from the raw `*.metrics.json`:
   a per-surface comparison table (this release vs previous), plus notes on any
   delta worth defending or investigating.
3. Add a row to the index below.

Producing a report is a **required, non-gating** release step — the release
process refuses to complete *silently* without one, but it never fails CI. You
iterate the numbers until you're happy, then it ships with the release.

## Report index

| Date | Release / ref | Surfaces | Report |
| --- | --- | --- | --- |
| 2026-07-24 | `2.7.0.dev1` | search-ui, graph-api, surfaces-ui | [2026-07-24-2.7.0.dev1.md](2026-07-24-2.7.0.dev1.md) |
| 2026-07-19 | `feat/graph-v3` tuning arc | graph | [graph-v3-tuning-2026-07-19.md](graph-v3-tuning-2026-07-19.md) |

> The 2026-07-24 report is the first unified per-release report under the
> resolidified framework. The graph-v3 arc below it is the pre-framework
> historical report, migrated in verbatim.
