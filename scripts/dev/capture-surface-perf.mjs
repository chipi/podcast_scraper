#!/usr/bin/env node
// capture-surface-perf.mjs — Playwright/CDP UI perf capturer for the "other"
// viewer surfaces (perf-traces Chunk 3): Library, Digest, and entity-load.
//
// The basic-but-important surfaces outside search and graph — the ones a user
// hits constantly and that must not regress silently. Same shape family as
// capture-search-perf.mjs (single <label>.ui.metrics.json, median-of-1 per
// scenario, graceful per-scenario error fields so a run always emits a report).
//
// Scenarios:
//   * library-load — goto → set corpus path → Library tab → library-root visible.
//   * digest-load  — Digest tab → digest-root visible.
//   * entity-load  — Library tab → click first episode row → episode-detail-rail.
//
// Output: <output-dir>/<label>.ui.metrics.json

import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// @playwright/test lives in the viewer node_modules, not repo-root; this script
// is in scripts/dev/ so ESM bare-import never resolves it. Resolve explicitly.
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

fs.mkdirSync(OUTPUT_DIR, { recursive: true })
const OUT = path.join(OUTPUT_DIR, `${LABEL}.ui.metrics.json`)

const STATUS_BAR_CORPUS_SEL = '[data-testid="status-bar-corpus-path"]'
const LIBRARY_TAB_SEL = '[data-testid="main-tab-library"]'
const LIBRARY_ROOT_SEL = '[data-testid="library-root"]'
const LIBRARY_ROW_SEL = '[data-library-episode-row]'
const DIGEST_TAB_SEL = '[data-testid="main-tab-digest"]'
const DIGEST_ROOT_SEL = '[data-testid="digest-root"]'
const EPISODE_RAIL_SEL = '[data-testid="episode-detail-rail"]'

const scenarios = []

async function landAndSetCorpus(page) {
  await page.goto(`${VIEWER}/?path=${encodeURIComponent(CORPUS)}`)
  await page.locator(LIBRARY_TAB_SEL).waitFor({ state: 'visible', timeout: 30_000 })
  await page.locator(STATUS_BAR_CORPUS_SEL).fill(CORPUS)
}

async function captureLibraryLoad(page) {
  // goto → set corpus path → Library tab → library-root visible.
  const t0 = Date.now()
  try {
    await page.locator(LIBRARY_TAB_SEL).click()
    await page.locator(LIBRARY_ROOT_SEL).waitFor({ state: 'visible', timeout: 20_000 })
  } catch {
    return { name: 'library-load', tti_ms: null, error: 'library-root not visible in 20s' }
  }
  return { name: 'library-load', tti_ms: Date.now() - t0 }
}

async function captureDigestLoad(page) {
  const t0 = Date.now()
  try {
    await page.locator(DIGEST_TAB_SEL).click()
    await page.locator(DIGEST_ROOT_SEL).waitFor({ state: 'visible', timeout: 20_000 })
  } catch {
    return { name: 'digest-load', tti_ms: null, error: 'digest-root not visible in 20s' }
  }
  return { name: 'digest-load', tti_ms: Date.now() - t0 }
}

async function captureEntityLoad(page) {
  // Library tab → click first episode row → episode-detail rail populated.
  try {
    await page.locator(LIBRARY_TAB_SEL).click()
    await page.locator(LIBRARY_ROOT_SEL).waitFor({ state: 'visible', timeout: 20_000 })
    const firstRow = page.locator(LIBRARY_ROW_SEL).first()
    await firstRow.waitFor({ state: 'visible', timeout: 10_000 })
    const t0 = Date.now()
    await firstRow.click()
    await page.locator(EPISODE_RAIL_SEL).waitFor({ state: 'visible', timeout: 15_000 })
    return { name: 'entity-load', open_ms: Date.now() - t0 }
  } catch {
    return { name: 'entity-load', open_ms: null, error: 'episode-detail rail not shown in 15s' }
  }
}

async function main() {
  const browser = await chromium.launch({ headless: true })
  try {
    const ctx = await browser.newContext({
      viewport: { width: VW, height: VH },
      deviceScaleFactor: DPR,
    })
    const page = await ctx.newPage()

    await landAndSetCorpus(page)
    scenarios.push(await captureLibraryLoad(page))
    scenarios.push(await captureDigestLoad(page))
    scenarios.push(await captureEntityLoad(page))

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
  console.log(`\ncapture-surface-perf: ${scenarios.length} scenarios → ${path.basename(OUT)}`)
  for (const s of scenarios) {
    if (s.error) {
      console.log(`  ${s.name.padEnd(16)} ERROR: ${s.error}`)
    } else {
      const key = Object.keys(s).find((k) => k !== 'name' && s[k] !== null)
      console.log(`  ${s.name.padEnd(16)} ${key}=${s[key]} ms`)
    }
  }
}

main().catch((err) => {
  console.error('capture-surface-perf: FATAL', err)
  process.exit(1)
})
