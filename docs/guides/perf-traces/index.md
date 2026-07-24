# Performance Traces

> **The single entry point for the viewer/API performance-trace harness.**
> How we capture reproducible perf measurements per surface, where the raw
> artifacts live, and how each release's numbers are recorded so they can be
> compared to the previous release — the same discipline as
> [Evaluation Reports](../eval-reports/index.md), applied to performance.

This page is the framework. Per-surface recipes live in the area pages
([search](search.md) · [graph](graph.md) · [surfaces](surfaces.md)); each
release's captured numbers live under [reports/](reports/index.md).

---

## What this harness is (and is not)

**Is:** a set of Playwright + Chrome DevTools/CDP capture scripts that boot a
dedicated api + viewer on isolated ports, drive one scenario, emit a labelled
`*.metrics.json` (plus optional `*.trace.json.gz` / `*.screenshot.png`), and
tear everything down. Numbers are **scripted, deterministic settles** — good for
*relative* comparison across git refs / releases, indicative (not definitive)
for absolute user-facing latency.

**Is not:** a CI gate. Traces are captured on demand — with a hypothesis, before
a release, or to defend a tuning — never "A/B on a hunch." (ML model-inference
baselines are a *separate* family, gated on the nightly matrix; see
`scripts/dev/capture_*_baseline.py`.)

## Layout

```text
data/perf/traces/            <- raw artifacts (git-tracked, like data/eval/runs)
  search/                    <- *.metrics.json, *.api.metrics.json
  graph/                     <- *.metrics.json, *.trace.json.gz, *.screenshot.png
  surfaces/                  <- library / digest / entity-load
docs/guides/perf-traces/     <- documentation (this dir)
  index.md                   <- this framework page
  search.md  graph.md  surfaces.md   <- per-area recipes + scenario catalogs
  reports/
    index.md                 <- report hub (one row per release run)
    <YYYY-MM-DD>-<label>.md   <- per-release timestamped report (comparison tables)
```

**Split rationale (mirrors `data/eval`):** raw artifacts under `data/perf/`
(diffable, machine-readable, kept out of the prose); human-readable reports and
recipes under `docs/guides/` (tracked, gate the strict docs build, browsable
history).

## Capture families

| Family | Script | Measures | Area page |
| --- | --- | --- | --- |
| Search UI | `scripts/dev/capture-search-perf.{sh,mjs}` | TTI-to-query, filter-open, results-paint, workspace/palette/operator/compare | [search](search.md) |
| Search API | `scripts/dev/capture-search-api.{sh,py}` | per-query API latency (no browser) | [search](search.md) |
| Graph UI | `scripts/dev/capture-graph-lcp.{sh,mjs}` | shell LCP, graph time-to-canvas, main-thread block | [graph](graph.md) |
| Graph API | `scripts/dev/capture-graph-api.{sh,py}` *(Chunk 2)* | graph/relational endpoint latency (no browser) | [graph](graph.md) |
| Surfaces UI | `scripts/dev/capture-surface-perf.{sh,mjs}` *(Chunk 3)* | Library / Digest load, entity-load | [surfaces](surfaces.md) |

## Standard capture conditions

Keep these constant so runs are comparable:

- **Corpus:** `.test_outputs/manual/prod-v2/corpus` (the reference corpus) unless
  a change specifically targets fixture-corpus behavior.
- **Viewport:** 1440×900 @ DPR-2 (retina), headless Chromium.
- **Median-of-3** per scenario; record all three runs in the report.

## Adding a new surface

1. Add a `captureX` scenario to the relevant `capture-*-perf.mjs` (or a new
   script following the same arg contract: `--corpus --label --output-dir`).
2. Point `--output-dir` at `data/perf/traces/<surface>/`.
3. Document the scenario in the area page ([search](search.md) /
   [graph](graph.md) / [surfaces](surfaces.md)): what it measures, the exact
   selector/wait it keys on, and how to reproduce.
4. On the next release run, add its numbers to that release's
   [report](reports/index.md).

## Per release

Generating an official perf report is a **required, non-gating** release step —
see [reports/index.md](reports/index.md) and the release runbook. Capture →
iterate the numbers until happy → author the timestamped report → it ships with
the release. Not a CI gate; a discipline that can't be silently skipped.

## Related

- [Search perf trace runbook](../SEARCH_PERF_TRACE_RUNBOOK.md) — deep recipe.
- [Graph perf trace runbook](../GRAPH_PERF_TRACE_RUNBOOK.md) — deep recipe.
- [Performance Guide](../PERFORMANCE.md) — pipeline/audio/backend perf.
- [Evaluation Reports](../eval-reports/index.md) — the sibling discipline for
  summarization/quality.
