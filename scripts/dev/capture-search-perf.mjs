#!/usr/bin/env node
// capture-search-perf.mjs — Playwright + CDP UI perf capturer for Search v3.
//
// Search v3 §S1 stabilization pass (2026-07-20). Captures 3 UI scenarios
// that exist on the merged Search launcher (compact-launcher shape after S1):
//
//   * leftpanel-search-open — page.goto → #search-q visible (analog of
//     RFC-107 §P2 "ui-workspace-open" TTI, on the pre-S2 UI).
//   * filter-apply — click SearchTopKChip (deterministic + present today) →
//     popover-visible ms (analog of RFC-107 §P2 "ui-filter-apply").
//   * results-paint — submit a query, measure to first .search-result
//     card visible (analog of RFC-107 §P2 "ui-workspace-open" for the
//     hit-render side).
//
// The 3 remaining scenarios from RFC-107 §P2 emit NOT_APPLICABLE_YET rows
// with a clear "lands with slice S<N>" reason, so the report shape stays
// stable and the operator can diff across commits without gaps:
//
//   * workspace-open (S2 #1232)
//   * cmdk-open      (S3 #1233)
//   * operator-cluster (S4 #1234)
//
// Output: <output-dir>/<label>.ui.metrics.json — same shape family as
// scripts/dev/capture_search_api.py output, single file per capture.

import { chromium } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'

function parseArgs(argv) {
  const out = {}
  for (let i = 0; i < argv.length; i += 2) {
    const k = argv[i]
    const v = argv[i + 1]
    if (!k?.startsWith('--')) continue
    out[k.slice(2)] = v
  }
  return out
}

const args = parseArgs(process.argv.slice(2))
for (const need of ['viewer', 'corpus', 'label', 'output-dir']) {
  if (!args[need]) {
    console.error(`FATAL: --${need} required`)
    process.exit(2)
  }
}

const VIEWER = args.viewer.replace(/\/$/, '')
const CORPUS = args.corpus
const LABEL = args.label
const OUTPUT_DIR = args['output-dir']
const WAIT_MS = Number(args['wait-ms'] ?? '3000')
const VW = Number(args['viewport-w'] ?? '1440')
const VH = Number(args['viewport-h'] ?? '900')
const DPR = Number(args['viewport-dpr'] ?? '2')

fs.mkdirSync(OUTPUT_DIR, { recursive: true })
const OUT = path.join(OUTPUT_DIR, `${LABEL}.ui.metrics.json`)

// The compact launcher's query field id (unchanged since UXS-005).
const QUERY_FIELD_SEL = '#search-q'
const TOPK_CHIP_SEL = '[data-testid="search-chip-topk"]'
const TOPK_POPOVER_SEL = '[data-testid="search-popover-topk"]'
const RESULT_ROW_SEL = '[data-testid="search-filter-bar"] ~ * [data-testid="search-result-tier"]'

const scenarios = []

async function captureLeftpanelSearchOpen(page) {
  const t0 = Date.now()
  await page.goto(`${VIEWER}/?path=${encodeURIComponent(CORPUS)}`)
  await page.locator(QUERY_FIELD_SEL).waitFor({ state: 'visible', timeout: 30_000 })
  await page.locator(QUERY_FIELD_SEL).waitFor({ state: 'attached', timeout: 5_000 })
  const elapsed = Date.now() - t0
  return { name: 'leftpanel-search-open', tti_ms: elapsed }
}

async function captureFilterApply(page) {
  // Prereq: on the page from the previous scenario.
  const t0 = Date.now()
  await page.locator(TOPK_CHIP_SEL).click()
  await page.locator(TOPK_POPOVER_SEL).waitFor({ state: 'visible', timeout: 5_000 })
  const elapsed = Date.now() - t0
  // Close popover so it doesn't interfere with the next scenario.
  await page.keyboard.press('Escape')
  return { name: 'filter-apply', open_ms: elapsed }
}

async function captureResultsPaint(page) {
  // Prereq: on the launcher, api healthy. Fill + submit; wait for at least
  // one result card. Timed on the submit→first-card path.
  await page.locator(QUERY_FIELD_SEL).fill('trail building')
  const t0 = Date.now()
  await page
    .locator('form#semantic-search-form')
    .getByRole('button', { name: /^Search$/ })
    .click()
  try {
    await page.locator(RESULT_ROW_SEL).first().waitFor({ state: 'visible', timeout: 15_000 })
  } catch {
    return { name: 'results-paint', paint_ms: null, error: 'no result rendered in 15s' }
  }
  const elapsed = Date.now() - t0
  return { name: 'results-paint', paint_ms: elapsed }
}

async function main() {
  const browser = await chromium.launch({ headless: true })
  try {
    const ctx = await browser.newContext({
      viewport: { width: VW, height: VH },
      deviceScaleFactor: DPR,
    })
    const page = await ctx.newPage()

    scenarios.push(await captureLeftpanelSearchOpen(page))
    scenarios.push(await captureFilterApply(page))
    scenarios.push(await captureResultsPaint(page))

    // NOT_APPLICABLE_YET rows: keep report shape stable for cross-commit diffing.
    for (const [name, slice] of [
      ['workspace-open', 'S2 (#1232)'],
      ['cmdk-open', 'S3 (#1233)'],
      ['operator-cluster', 'S4 (#1234)'],
    ]) {
      scenarios.push({ name, status: 'NOT_APPLICABLE_YET', unblocks_with: slice })
    }

    // Warmup grace so any last CDP events settle.
    await page.waitForTimeout(WAIT_MS)
    await ctx.close()
  } finally {
    await browser.close()
  }

  const payload = {
    schema_version: '1',
    label: LABEL,
    captured_at: new Date().toISOString(),
    viewer: VIEWER,
    corpus: CORPUS,
    viewport: { width: VW, height: VH, device_scale_factor: DPR },
    scenarios,
  }
  fs.writeFileSync(OUT, JSON.stringify(payload, null, 2) + '\n')
  console.log(`\ncapture-search-perf: ${scenarios.length} scenarios → ${path.basename(OUT)}`)
  for (const s of scenarios) {
    if (s.status === 'NOT_APPLICABLE_YET') {
      console.log(`  ${s.name.padEnd(24)} NOT_APPLICABLE_YET (${s.unblocks_with})`)
    } else if (s.error) {
      console.log(`  ${s.name.padEnd(24)} ERROR: ${s.error}`)
    } else {
      const key = Object.keys(s).find((k) => k !== 'name' && s[k] !== null)
      console.log(`  ${s.name.padEnd(24)} ${key}=${s[key]} ms`)
    }
  }
}

main().catch((err) => {
  console.error('capture-search-perf: FATAL', err)
  process.exit(1)
})
