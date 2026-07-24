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

import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// @playwright/test is installed in the viewer's node_modules, not repo-root.
// This script lives in scripts/dev/, so ESM bare-import resolution never reaches
// it. Resolve explicitly from the viewer package via createRequire.
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const viewerRequire = createRequire(
  path.join(__dirname, '..', '..', 'web', 'gi-kg-viewer', 'package.json'),
)
const { chromium } = viewerRequire('@playwright/test')

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
// Query to drive results-paint / operator scenarios. Default suits the prod-v2
// reference corpus (finance); override with --query for other corpora.
const QUERY = args.query ?? 'the economy'

fs.mkdirSync(OUTPUT_DIR, { recursive: true })
const OUT = path.join(OUTPUT_DIR, `${LABEL}.ui.metrics.json`)

// Query field id (unchanged since UXS-005; lives on the Search main tab post-S2).
const QUERY_FIELD_SEL = '#search-q'
// Status-bar corpus path input — setting it enables search (?path= alone doesn't).
const STATUS_BAR_CORPUS_SEL = '[data-testid="status-bar-corpus-path"]'
const TOPK_CHIP_SEL = '[data-testid="search-chip-topk"]'
const TOPK_POPOVER_SEL = '[data-testid="search-popover-topk"]'
const RESULT_ROW_SEL = '[data-testid="search-workspace"] article'
// S2–S8 shipped surfaces (RFC-107). Testids mirror e2e/E2E_SURFACE_MAP.md.
const SEARCH_TAB_SEL = '[data-testid="main-tab-search"]'
const PALETTE_SEL = '[data-testid="command-palette"]'
const OPERATOR_CLUSTER_CHIP_SEL = '[data-testid="operator-chip-cluster"]'
const OPERATOR_CLUSTER_PANEL_SEL = '[data-testid="operator-cluster-panel"]'
const OPERATOR_COMPARE_CHIP_SEL = '[data-testid="operator-chip-compare"]'
const OPERATOR_COMPARE_RUN_SEL = '[data-testid="operator-compare-run"]'
const OPERATOR_COMPARE_COLUMNS_SEL = '[data-testid="operator-compare-columns"]'

const scenarios = []

async function captureWorkspaceOpen(page) {
  // S2 (#1232) — page load → set corpus path (enables search) → Search main tab
  // → #search-q visible. TTI to the Query Workspace query field (replaces the
  // pre-S2 compact-launcher metric).
  const t0 = Date.now()
  await page.goto(`${VIEWER}/?path=${encodeURIComponent(CORPUS)}`)
  await page.locator(SEARCH_TAB_SEL).waitFor({ state: 'visible', timeout: 30_000 })
  // The status-bar corpus path drives search enablement; ?path= alone does not.
  await page.locator(STATUS_BAR_CORPUS_SEL).fill(CORPUS)
  await page.locator(SEARCH_TAB_SEL).click()
  await page.locator(QUERY_FIELD_SEL).waitFor({ state: 'visible', timeout: 30_000 })
  const elapsed = Date.now() - t0
  return { name: 'workspace-open', tti_ms: elapsed }
}

async function captureFilterApply(page) {
  // Prereq: on the Search tab from captureWorkspaceOpen, corpus path set.
  const chip = page.locator(TOPK_CHIP_SEL)
  try {
    await chip.waitFor({ state: 'visible', timeout: 5_000 })
    const t0 = Date.now()
    await chip.click()
    await page.locator(TOPK_POPOVER_SEL).waitFor({ state: 'visible', timeout: 5_000 })
    const elapsed = Date.now() - t0
    await page.keyboard.press('Escape') // close popover for the next scenario
    return { name: 'filter-apply', open_ms: elapsed }
  } catch {
    return { name: 'filter-apply', open_ms: null, error: 'topk popover did not open in 5s' }
  }
}

async function captureResultsPaint(page) {
  // Prereq: on the Search tab, api healthy. Fill + submit; wait for the first
  // result card in the workspace. Timed on the submit→first-card path.
  const field = page.locator(QUERY_FIELD_SEL)
  try {
    // #search-q is disabled until the viewer sees a healthy API — wait for it.
    await field.waitFor({ state: 'visible', timeout: 10_000 })
    await page.waitForFunction(
      (sel) => {
        const el = document.querySelector(sel)
        return el && !el.hasAttribute('disabled')
      },
      QUERY_FIELD_SEL,
      { timeout: 15_000 },
    )
    await field.fill(QUERY)
  } catch {
    return { name: 'results-paint', paint_ms: null, error: 'query field never enabled (API unhealthy?)' }
  }
  const t0 = Date.now()
  await field.press('Enter')
  try {
    await page.locator(RESULT_ROW_SEL).first().waitFor({ state: 'visible', timeout: 15_000 })
  } catch {
    return { name: 'results-paint', paint_ms: null, error: 'no result rendered in 15s' }
  }
  const elapsed = Date.now() - t0
  return { name: 'results-paint', paint_ms: elapsed }
}

async function captureCmdkOpen(page) {
  // S3 (#1233) — '/' summons the command palette overlay. Blur to body first
  // so the key isn't typed into the query field.
  await page.locator('body').click({ position: { x: 5, y: 5 } })
  const t0 = Date.now()
  await page.keyboard.press('/')
  try {
    await page.locator(PALETTE_SEL).waitFor({ state: 'visible', timeout: 5_000 })
  } catch {
    return { name: 'cmdk-open', open_ms: null, error: 'palette did not open in 5s' }
  }
  const elapsed = Date.now() - t0
  await page.keyboard.press('Escape')
  return { name: 'cmdk-open', open_ms: elapsed }
}

async function captureOperatorCluster(page) {
  // S4 (#1234) — Cluster operator: chip click → server aggregation → panel
  // populated. Prereq: results present from captureResultsPaint.
  const chip = page.locator(OPERATOR_CLUSTER_CHIP_SEL)
  if ((await chip.count()) === 0) {
    return { name: 'operator-cluster', run_ms: null, error: 'cluster chip absent (no results?)' }
  }
  const t0 = Date.now()
  await chip.click()
  try {
    await page.locator(OPERATOR_CLUSTER_PANEL_SEL).waitFor({ state: 'visible', timeout: 15_000 })
  } catch {
    return { name: 'operator-cluster', run_ms: null, error: 'cluster panel not shown in 15s' }
  }
  const elapsed = Date.now() - t0
  await chip.click() // toggle off
  return { name: 'operator-cluster', run_ms: elapsed }
}

async function captureOperatorCompare(page) {
  // S8 — Compare operator: chip → Run compare → 2-column packs rendered. Hits
  // the real /api/search/compare endpoint. Prereq: ≥2 comparable subjects in
  // the current hit set (chip disabled otherwise).
  const chip = page.locator(OPERATOR_COMPARE_CHIP_SEL)
  if ((await chip.count()) === 0 || (await chip.isDisabled())) {
    return { name: 'operator-compare', run_ms: null, error: 'compare chip absent/disabled' }
  }
  await chip.click()
  const run = page.locator(OPERATOR_COMPARE_RUN_SEL)
  try {
    await run.waitFor({ state: 'visible', timeout: 5_000 })
  } catch {
    return { name: 'operator-compare', run_ms: null, error: 'compare picker did not open' }
  }
  const t0 = Date.now()
  await run.click()
  try {
    await page.locator(OPERATOR_COMPARE_COLUMNS_SEL).waitFor({ state: 'visible', timeout: 20_000 })
  } catch {
    return { name: 'operator-compare', run_ms: null, error: 'compare columns not shown in 20s' }
  }
  const elapsed = Date.now() - t0
  return { name: 'operator-compare', run_ms: elapsed }
}

async function main() {
  const browser = await chromium.launch({ headless: true })
  try {
    const ctx = await browser.newContext({
      viewport: { width: VW, height: VH },
      deviceScaleFactor: DPR,
    })
    const page = await ctx.newPage()

    // Order matters: workspace-open navigates to the Search tab; results-paint
    // must precede the operator scenarios (they read the current hit set).
    scenarios.push(await captureWorkspaceOpen(page))
    scenarios.push(await captureFilterApply(page))
    scenarios.push(await captureResultsPaint(page))
    scenarios.push(await captureCmdkOpen(page))
    scenarios.push(await captureOperatorCluster(page))
    scenarios.push(await captureOperatorCompare(page))

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
