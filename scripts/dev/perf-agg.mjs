// perf-agg.mjs — pure aggregation math for the UI perf capturers.
//
// Extracted from capture-search-perf.mjs so the report-number math (median,
// cold-run-1 exclusion, per-scenario min/mean/median/max, and the optional
// net/render split) is unit-testable WITHOUT booting a browser or importing
// Playwright. capture-search-perf.mjs (and any sibling UI capturer) imports
// from here; the tests import from here directly.
//
// No side effects, no I/O, no third-party imports — keep it that way so the
// tests stay a pure function check.

/**
 * Median of a numeric list, ignoring null/undefined. Even-length → rounded mean
 * of the two middles. Returns null for an empty list.
 */
export function median(nums) {
  const s = nums.filter((n) => n != null).sort((a, b) => a - b)
  if (!s.length) return null
  const mid = Math.floor(s.length / 2)
  return s.length % 2 ? s[mid] : Math.round((s[mid - 1] + s[mid]) / 2)
}

/**
 * Aggregate per-run scenario captures into one summary per scenario name.
 *
 * `perRun` is an array (one entry per run) of arrays of scenario objects
 * `{ name, <metric>, [error], [net_ms], [render_ms] }`. The FIRST run is the
 * cold pass (index-open + model load) and is EXCLUDED from the warm
 * min/mean/median/max; its value is recorded separately as `cold_ms`. If only
 * one run exists (or every warm run errored), the warm stats fall back to all
 * runs so a single-run capture still yields numbers.
 *
 * The headline metric is the first non-`name`/`error` key on the scenario
 * object (e.g. `run_ms`, `paint_ms`, `tti_ms`). Scenarios that also carry a
 * `net_ms` sub-split (operator-compare) get a `split_median { net_ms, render_ms }`.
 */
export function aggregateScenarios(perRun) {
  if (!perRun.length) return []
  const names = perRun[0].map((s) => s.name)
  return names.map((name) => {
    const entries = perRun.map((run) => run.find((s) => s.name === name)).filter(Boolean)
    const metric = Object.keys(entries[0]).find((k) => k !== 'name' && k !== 'error') || 'ms'
    const allRuns = entries.map((e) => (e[metric] == null ? null : e[metric]))
    const cold = allRuns.length ? allRuns[0] : null
    let warm = allRuns.slice(1).filter((v) => v != null)
    if (!warm.length) warm = allRuns.filter((v) => v != null)
    const errors = entries.map((e) => e.error).filter(Boolean)
    const mean = warm.length ? Math.round(warm.reduce((a, b) => a + b, 0) / warm.length) : null
    // Scenarios that record a sub-split (operator-compare: net vs render) expose
    // warm medians so run_ms stays the single headline metric but the split is
    // auditable in the JSON.
    const warmEntries = entries.length > 1 ? entries.slice(1) : entries
    const splitMedian = (key) => median(warmEntries.map((e) => e[key]).filter((v) => v != null))
    const hasSplit = entries.some((e) => e.net_ms != null)
    return {
      name,
      metric,
      min_ms: warm.length ? Math.min(...warm) : null,
      mean_ms: mean,
      median_ms: median(warm),
      max_ms: warm.length ? Math.max(...warm) : null,
      cold_ms: cold,
      runs: allRuns,
      warm_samples: warm.length,
      ...(hasSplit ? { split_median: { net_ms: splitMedian('net_ms'), render_ms: splitMedian('render_ms') } } : {}),
      ...(errors.length ? { errors } : {}),
    }
  })
}
