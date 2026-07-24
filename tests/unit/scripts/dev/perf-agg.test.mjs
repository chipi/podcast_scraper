// Unit tests for the UI perf-capturer aggregation math (scripts/dev/perf-agg.mjs).
// Pure functions — no browser, no I/O. Run with: node --test <this file>
// (wired into `make test-perf-agg`).

import assert from 'node:assert/strict'
import { test } from 'node:test'
import { median, aggregateScenarios } from '../../../../scripts/dev/perf-agg.mjs'

test('median: odd length returns the middle', () => {
  assert.equal(median([30, 10, 20]), 20)
})

test('median: even length returns rounded mean of the two middles', () => {
  assert.equal(median([10, 20, 30, 40]), 25)
  assert.equal(median([10, 21]), 16) // (10+21)/2 = 15.5 → 16
})

test('median: empty (or all-null) is null', () => {
  assert.equal(median([]), null)
  assert.equal(median([null, undefined]), null)
})

test('median: ignores null/undefined entries', () => {
  assert.equal(median([10, null, 30, undefined]), 20)
})

test('aggregateScenarios: excludes cold run-1 from warm stats, records cold_ms', () => {
  const perRun = [
    [{ name: 'results-paint', paint_ms: 5000 }], // cold
    [{ name: 'results-paint', paint_ms: 180 }],
    [{ name: 'results-paint', paint_ms: 200 }],
    [{ name: 'results-paint', paint_ms: 190 }],
  ]
  const [s] = aggregateScenarios(perRun)
  assert.equal(s.metric, 'paint_ms')
  assert.equal(s.cold_ms, 5000)
  assert.equal(s.min_ms, 180)
  assert.equal(s.max_ms, 200)
  assert.equal(s.median_ms, 190)
  assert.equal(s.mean_ms, 190) // (180+200+190)/3 = 190
  assert.equal(s.warm_samples, 3)
  assert.deepEqual(s.runs, [5000, 180, 200, 190])
})

test('aggregateScenarios: single run falls back to using that run as warm', () => {
  const perRun = [[{ name: 'x', run_ms: 42 }]]
  const [s] = aggregateScenarios(perRun)
  assert.equal(s.cold_ms, 42)
  assert.equal(s.median_ms, 42)
  assert.equal(s.warm_samples, 1)
})

test('aggregateScenarios: all warm runs null → falls back to all runs (incl. cold)', () => {
  const perRun = [
    [{ name: 'x', run_ms: 100 }],
    [{ name: 'x', run_ms: null, error: 'boom' }],
  ]
  const [s] = aggregateScenarios(perRun)
  // warm slice is [null] → empty after filter → fall back to all non-null = [100]
  assert.equal(s.median_ms, 100)
  assert.deepEqual(s.errors, ['boom'])
})

test('aggregateScenarios: metric picker skips name and error keys', () => {
  const perRun = [[{ name: 'x', error: 'e', tti_ms: 7 }], [{ name: 'x', tti_ms: 9 }]]
  const [s] = aggregateScenarios(perRun)
  assert.equal(s.metric, 'tti_ms')
  assert.equal(s.cold_ms, 7)
  assert.equal(s.median_ms, 9)
})

test('aggregateScenarios: records net/render split_median when present (warm only)', () => {
  const perRun = [
    [{ name: 'operator-compare', run_ms: 900, net_ms: 890, render_ms: 10 }], // cold
    [{ name: 'operator-compare', run_ms: 356, net_ms: 352, render_ms: 4 }],
    [{ name: 'operator-compare', run_ms: 360, net_ms: 356, render_ms: 4 }],
  ]
  const [s] = aggregateScenarios(perRun)
  assert.equal(s.metric, 'run_ms')
  assert.equal(s.cold_ms, 900)
  assert.ok(s.split_median)
  assert.equal(s.split_median.net_ms, 354) // median(352,356)=(352+356)/2=354
  assert.equal(s.split_median.render_ms, 4)
})

test('aggregateScenarios: no split_median when net_ms absent', () => {
  const perRun = [[{ name: 'x', run_ms: 1 }], [{ name: 'x', run_ms: 2 }]]
  const [s] = aggregateScenarios(perRun)
  assert.equal(s.split_median, undefined)
})

test('aggregateScenarios: empty input returns empty array', () => {
  assert.deepEqual(aggregateScenarios([]), [])
})
