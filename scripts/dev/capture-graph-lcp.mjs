#!/usr/bin/env node
/**
 * capture-graph-lcp.mjs — headless Chromium capture of a Chrome DevTools
 * Performance recording (chrome://tracing format, `traceEvents` shape) of the
 * gi-kg-viewer graph route's initial paint.
 *
 * Output shape matches `docs/wip/graph-v3/traces/03-C-first-paint.json.json.gz`
 * so tier-C traces stay comparable to future runs. Also emits a summary JSON
 * with LCP / FCP / TTI / main-thread-blocking-time so a human can diff the
 * numbers without opening chrome://tracing.
 *
 * INVOKED BY: scripts/dev/capture-graph-lcp.sh (the orchestrator that boots
 * dedicated api + viewer servers on isolated ports and hands us the URL to
 * navigate to).
 *
 * ARGS via env vars (all required except VIEWPORT_DPR):
 *   LCP_TARGET_URL       — e.g. http://127.0.0.1:5600/?path=/abs/path/to/corpus
 *   LCP_OUTPUT_DIR       — dir to write ${label}.trace.json + ${label}.metrics.json
 *   LCP_LABEL            — file-name prefix (e.g. "main-baseline", "branch-tuning-A")
 *   VIEWPORT_WIDTH       — px (default 1440)
 *   VIEWPORT_HEIGHT      — px (default 900)
 *   VIEWPORT_DPR         — device scale factor (default 2, matches tier-C)
 *   LCP_WAIT_MS          — how long to wait after LCP before stopping trace (default 5000)
 *
 * Usage:
 *   node scripts/dev/capture-graph-lcp.mjs
 *
 * See docs/guides/GRAPH_PERF_TRACE_RUNBOOK.md for the full recipe.
 */

import { mkdirSync, writeFileSync, createWriteStream } from 'node:fs'
import { createGzip } from 'node:zlib'
import { pipeline } from 'node:stream/promises'
import { Readable } from 'node:stream'
import { join, dirname, resolve as pathResolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'

// Node's ES module resolver anchors at this script's location and doesn't
// walk into web/gi-kg-viewer/node_modules where @playwright/test lives.
// createRequire anchored at the viewer's package dir resolves it correctly
// (Playwright ships a CJS entry that works with require()).
const scriptDir = dirname(fileURLToPath(import.meta.url))
const repoRoot = pathResolve(scriptDir, '..', '..')
const viewerRequire = createRequire(`${repoRoot}/web/gi-kg-viewer/package.json`)
const { chromium } = viewerRequire('@playwright/test')

const {
  LCP_TARGET_URL,
  LCP_OUTPUT_DIR,
  LCP_LABEL,
  VIEWPORT_WIDTH = '1440',
  VIEWPORT_HEIGHT = '900',
  VIEWPORT_DPR = '2',
  LCP_WAIT_MS = '5000',
  /**
   * Optional: seed the viewer's `useGraphLoadModeStore` before the app
   * boots. `'topDown'` mounts the tier-8 super-theme rollup (3–8 nodes,
   * expand-on-tap); `'everything'` is today's default (full merged
   * artifact). See web/gi-kg-viewer/src/stores/graphLoadMode.ts.
   * When unset, the app reads whatever is in localStorage (blank → default).
   */
  LCP_LOAD_MODE = '',
  /**
   * Optional: after the initial canvas mount, tap the first SuperTheme
   * node and measure expand-on-tap resettle time. Only meaningful when
   * LCP_LOAD_MODE=topDown. `'1'` enables.
   */
  LCP_EXPAND_FIRST_SUPERTHEME = '',
} = process.env

for (const [k, v] of Object.entries({ LCP_TARGET_URL, LCP_OUTPUT_DIR, LCP_LABEL })) {
  if (!v) {
    console.error(`FATAL: env ${k} is required`)
    process.exit(2)
  }
}

mkdirSync(LCP_OUTPUT_DIR, { recursive: true })
const tracePathRaw = join(LCP_OUTPUT_DIR, `${LCP_LABEL}.trace.json`)
const tracePathGz = join(LCP_OUTPUT_DIR, `${LCP_LABEL}.trace.json.gz`)
const metricsPath = join(LCP_OUTPUT_DIR, `${LCP_LABEL}.metrics.json`)

// --- launch ---
const browser = await chromium.launch({
  headless: true,
  // devtools categories match Chrome DevTools Performance panel defaults;
  // extra "disabled-by-default-devtools.timeline.frame" gives us LCP frames.
})

const context = await browser.newContext({
  viewport: { width: parseInt(VIEWPORT_WIDTH), height: parseInt(VIEWPORT_HEIGHT) },
  deviceScaleFactor: parseFloat(VIEWPORT_DPR),
})

/**
 * Seed `localStorage['ps_graph_load_mode']` BEFORE any page script runs, so
 * the viewer's `useGraphLoadModeStore.readInitial()` picks it up on first
 * evaluation. `addInitScript` fires on every navigation in the context —
 * fine here because we only navigate once.
 *
 * We don't touch USERPREFS-1 (the server-side mirror) — the localStorage
 * value is authoritative on init, and the USERPREFS-1 watcher only
 * overwrites AFTER hydrate, which we skip through anyway. This is
 * deliberate: we want a measurement of the CODE PATH, not "what the
 * server happens to have persisted for the mock-signed-in test user".
 */
if (LCP_LOAD_MODE === 'topDown' || LCP_LOAD_MODE === 'everything') {
  await context.addInitScript((mode) => {
    try {
      localStorage.setItem('ps_graph_load_mode', mode)
    } catch {
      /* private mode / quota — measurement will fall through to the app default */
    }
  }, LCP_LOAD_MODE)
  console.log(`[lcp-capture] seeded localStorage ps_graph_load_mode=${LCP_LOAD_MODE}`)
}

const page = await context.newPage()
const client = await page.context().newCDPSession(page)

// --- Chrome DevTools trace: categories match the Performance panel default
// preset so the output is drop-in comparable to a hand-captured trace. ---
const tracingCategories = [
  '-*',
  'devtools.timeline',
  'v8.execute',
  'disabled-by-default-devtools.timeline',
  'disabled-by-default-devtools.timeline.frame',
  'disabled-by-default-v8.cpu_profiler',
  'disabled-by-default-devtools.timeline.stack',
  'toplevel',
  'blink.console',
  'blink.user_timing',
  'latencyInfo',
  'disabled-by-default-devtools.screenshot',
  'loading',
]

const traceChunks = []
client.on('Tracing.dataCollected', (event) => {
  if (event.value?.length) traceChunks.push(...event.value)
})

await client.send('Tracing.start', {
  categories: tracingCategories.join(','),
  options: 'sampling-frequency=10000',
  transferMode: 'ReportEvents',
})

console.log(`[lcp-capture] Navigating: ${LCP_TARGET_URL}`)
const navStart = Date.now()
await page.goto(LCP_TARGET_URL, { waitUntil: 'domcontentloaded' })

// The viewer's default landing tab is Digest, not Graph. To measure a
// graph-focused LCP (comparable to Tier-C's 1561ms which was captured on
// the graph tab) we click Graph after the shell renders, then wait for
// the `.graph-canvas` container to become visible. Two metrics reported:
//   - lcp_ms      — Web Vitals LCP of the whole page load (shell paint)
//   - graph_time_to_canvas_ms — click Graph → .graph-canvas visible
// The second is the one that's comparable to Tier-C. The first is a
// sanity number.
await page.waitForSelector('[data-testid="main-tab-graph"]', { timeout: 10_000 })

// The `?path=` query param DOESN'T auto-set the corpus in the app. The
// viewer reads the status-bar corpus-path input; e2e specs do the same
// via `statusBarCorpusPathInput(page).fill(...)`. Do the same here so the
// graph tab has a corpus to load.
const corpusFromUrl = new URL(LCP_TARGET_URL).searchParams.get('path')
if (corpusFromUrl) {
  console.log(`[lcp-capture] filling corpus path: ${corpusFromUrl}`)
  await page.getByTestId('status-bar-corpus-path').fill(corpusFromUrl)
  await page.getByTestId('status-bar-corpus-path').press('Enter')
}

const clickAt = await page.evaluate(() => performance.now())
await page.click('[data-testid="main-tab-graph"]')

// --- collect Web Vitals via the browser-side Performance API. LCP fires
//     late (paint after last-largest element), so we wait deterministically
//     for it to arrive or for LCP_WAIT_MS to elapse.
// Wait for the graph canvas container to appear + become non-empty.
// `.graph-canvas` mounts as soon as the GraphCanvas.vue container renders;
// we wait for it to be present + have children (Cytoscape mounted) as the
// "graph paint" signal.
let graphTimeToCanvasMs = null
try {
  await page.waitForFunction(
    () => {
      const el = document.querySelector('.graph-canvas')
      return !!el && (el.children.length > 0 || el.querySelector('canvas') !== null)
    },
    undefined,
    { timeout: parseInt(LCP_WAIT_MS) },
  )
  graphTimeToCanvasMs = await page.evaluate((clickAt) => performance.now() - clickAt, clickAt)
  console.log(`[lcp-capture] graph_time_to_canvas=${graphTimeToCanvasMs?.toFixed(0)}ms (click→cy mounted)`)
} catch (e) {
  console.log(`[lcp-capture] WARNING: graph canvas did not appear within ${LCP_WAIT_MS}ms — ${e.message.split('\n')[0]}`)
}

const metrics = await page.evaluate(async (waitMs) => {
  // Paint (FCP) and LCP both need PerformanceObserver — getEntriesBy*
  // returns nothing on some Chromium builds unless an observer buffered
  // them. Set both up before waiting.
  let fcpEntry = null
  const fcpObserver = new PerformanceObserver((list) => {
    for (const e of list.getEntries()) {
      if (e.name === 'first-contentful-paint') fcpEntry = e
    }
  })
  fcpObserver.observe({ type: 'paint', buffered: true })

  const lcpPromise = new Promise((resolve) => {
    let lastEntry = null
    const po = new PerformanceObserver((list) => {
      const entries = list.getEntries()
      if (entries.length) lastEntry = entries[entries.length - 1]
    })
    po.observe({ type: 'largest-contentful-paint', buffered: true })
    setTimeout(() => {
      po.disconnect()
      fcpObserver.disconnect()
      resolve(lastEntry)
    }, waitMs)
  })

  const navEntry = performance.getEntriesByType('navigation')[0]
  const lcp = await lcpPromise

  // Long-tasks summed = a proxy for main-thread block time. Cheap +
  // reproducible; not identical to the Performance panel's "Total Blocking
  // Time" (which subtracts 50ms floor per task) but close enough for a
  // trend signal.
  const longTasks = performance.getEntriesByType('longtask')
  const totalLongTaskMs = longTasks.reduce((s, t) => s + t.duration, 0)

  const mem = performance.memory
    ? {
        jsHeapUsedMB: performance.memory.usedJSHeapSize / (1024 * 1024),
        jsHeapLimitMB: performance.memory.jsHeapSizeLimit / (1024 * 1024),
      }
    : null

  return {
    lcp_ms: lcp?.startTime ?? null,
    lcp_element_tag: lcp?.element?.tagName ?? null,
    lcp_element_id: lcp?.element?.id ?? null,
    fcp_ms: fcpEntry?.startTime ?? null,
    ttfb_ms: navEntry?.responseStart ?? null,
    dom_content_loaded_ms: navEntry?.domContentLoadedEventEnd ?? null,
    load_event_end_ms: navEntry?.loadEventEnd ?? null,
    long_tasks_count: longTasks.length,
    long_tasks_total_ms: totalLongTaskMs,
    memory: mem,
  }
}, parseInt(LCP_WAIT_MS))

console.log(`[lcp-capture] LCP=${metrics.lcp_ms?.toFixed(0)}ms FCP=${metrics.fcp_ms?.toFixed(0)}ms longtasks=${metrics.long_tasks_count} (${metrics.long_tasks_total_ms.toFixed(0)}ms total)`)

metrics.graph_time_to_canvas_ms = graphTimeToCanvasMs
metrics.load_mode_seed = LCP_LOAD_MODE || null

/**
 * Optional second-phase measurement: expand-on-tap of the first
 * SuperTheme node in top-down mode. Reports:
 *   - expand_click_to_settle_ms: wall time from `mousedown` on the tap
 *     target until the next `finishLayoutPass` measure lands.
 *   - expand_node_count_before / after: nodes present in the cy core
 *     before + after the expand (proves the slice grew).
 * We poll the DOM for the first `[data-node-type=SuperTheme]` in cy's
 * rendered SVG-like canvas; cytoscape does NOT expose DOM per node, so
 * we drive the tap via `cy.$('node[type="SuperTheme"]').first().emit('tap')`
 * from a page.evaluate. That matches the app's tap-handler contract.
 */
let expandMetrics = null
if (LCP_EXPAND_FIRST_SUPERTHEME === '1' && graphTimeToCanvasMs != null) {
  try {
    // Give the initial fcose settle a moment to finish before we probe.
    await page.waitForTimeout(500)
    expandMetrics = await page.evaluate(async () => {
      /* eslint-disable no-undef */
      const win = window
      const cy = win.__GIKG_CY_DEV__ || null
      if (!cy) return { skipped: true, reason: 'window.__GIKG_CY_DEV__ not exposed (not a dev build?)' }
      const nodesBefore = cy.nodes().length
      const target = cy.nodes('[type="SuperTheme"]').first()
      if (target.empty()) return { skipped: true, reason: 'no SuperTheme node' }
      const targetId = target.id()
      // Mark before we tap so we can find the measure boundary later.
      performance.mark('expand:tap-fired')
      const settlePromise = new Promise((resolve) => {
        // Resolve on the next `finishLayoutPass` measure that arrives after the
        // mark. finishLayoutPass fires performance.measure('flp:total', ...).
        const po = new PerformanceObserver((list) => {
          for (const e of list.getEntries()) {
            if (e.name === 'flp:total' && e.startTime > performance.getEntriesByName('expand:tap-fired')[0].startTime) {
              po.disconnect()
              resolve({ resolvedBy: 'flp:total', duration_ms: e.duration })
              return
            }
          }
        })
        po.observe({ entryTypes: ['measure'] })
        // Fallback timeout so we don't hang if instrumentation is off.
        setTimeout(() => {
          po.disconnect()
          resolve({ resolvedBy: 'timeout', duration_ms: null })
        }, 4000)
      })
      const t0 = performance.now()
      target.emit('tap')
      const settled = await settlePromise
      const wall_ms = performance.now() - t0
      const nodesAfter = cy.nodes().length
      return {
        target_super_theme_id: targetId,
        expand_click_to_settle_ms: wall_ms,
        expand_settle_source: settled.resolvedBy,
        flp_total_last_duration_ms: settled.duration_ms,
        expand_node_count_before: nodesBefore,
        expand_node_count_after: nodesAfter,
      }
      /* eslint-enable no-undef */
    })
    console.log(`[lcp-capture] expand: ${expandMetrics.skipped ? `SKIP (${expandMetrics.reason})` : `${expandMetrics.expand_click_to_settle_ms?.toFixed(0)}ms (${expandMetrics.expand_node_count_before} → ${expandMetrics.expand_node_count_after} nodes)`}`)
  } catch (e) {
    expandMetrics = { skipped: true, reason: e.message.split('\n')[0] }
    console.log(`[lcp-capture] expand: ERROR ${expandMetrics.reason}`)
  }
}
metrics.expand_on_tap = expandMetrics

// Read redraw pipeline marks (rdw:*) — populated only when
// GraphCanvas.vue::redraw() is instrumented with performance.mark. Marks
// are absolute (performance.now() timestamps); we normalise to the earliest.
metrics.redraw_pipeline_marks = await page.evaluate(() => {
  const marks = performance.getEntriesByType('mark').filter((m) => m.name.startsWith('rdw:'))
  if (!marks.length) return []
  const base = marks[0].startTime
  return marks.map((m) => ({ name: m.name, t_ms: m.startTime - base }))
})
if (metrics.redraw_pipeline_marks.length) {
  console.log('[lcp-capture] redraw pipeline marks (ms from first mark):')
  let prev = 0
  for (const m of metrics.redraw_pipeline_marks) {
    const dt = m.t_ms - prev
    console.log(`  ${m.name.padEnd(35)}  t=${m.t_ms.toFixed(1).padStart(7)}ms  +${dt.toFixed(1)}ms`)
    prev = m.t_ms
  }
}

// Read finishLayoutPass phase measures if GraphCanvas.vue is instrumented
// (see GraphCanvas.vue::finishLayoutPass performance.mark calls).
// Reports one entry per phase — the LAST one, since finishLayoutPass can
// fire multiple times per settle.
metrics.finish_layout_pass_measures = await page.evaluate(() => {
  const all = performance.getEntriesByType('measure').filter((m) => m.name.startsWith('flp:'))
  const byName = new Map()
  for (const m of all) byName.set(m.name, { count: (byName.get(m.name)?.count ?? 0) + 1, last_duration_ms: m.duration, total_duration_ms: (byName.get(m.name)?.total_duration_ms ?? 0) + m.duration })
  return Object.fromEntries(byName)
})
if (Object.keys(metrics.finish_layout_pass_measures).length) {
  console.log('[lcp-capture] finishLayoutPass phase durations (last call):')
  for (const [name, m] of Object.entries(metrics.finish_layout_pass_measures)) {
    console.log(`  ${name.padEnd(45)} last=${m.last_duration_ms.toFixed(1).padStart(6)}ms  n=${m.count}  total=${m.total_duration_ms.toFixed(0)}ms`)
  }
}

// Sanity: did the graph actually render, or did the LCP measure the app
// shell / login screen? Capture a screenshot + probe DOM state so the
// reader can tell.
const sanity = await page.evaluate(() => ({
  title: document.title,
  url: location.href,
  hasCanvas: !!document.querySelector('canvas'),
  cytoscapeCanvasCount: document.querySelectorAll('canvas').length,
  graphMountEl: !!document.querySelector('[data-testid="graph-canvas"], .graph-canvas, #cy'),
  loginVisible: !!document.querySelector('[data-testid*="sign-in"], [data-testid*="login"], input[type="password"]'),
  bodyClasses: document.body.className,
  bodyText: document.body.innerText.slice(0, 200),
}))
console.log(`[lcp-capture] sanity: canvas=${sanity.hasCanvas} (${sanity.cytoscapeCanvasCount}) graph=${sanity.graphMountEl} login=${sanity.loginVisible}`)
if (sanity.loginVisible) {
  console.log(`[lcp-capture] WARNING: login screen visible — LCP is app-shell paint, NOT graph paint.`)
  console.log(`[lcp-capture]   Title: ${sanity.title}`)
  console.log(`[lcp-capture]   Body head: ${sanity.bodyText.replace(/\n/g, ' ')}`)
  console.log(`[lcp-capture]   Set LCP_SIGN_IN_ROLE to bypass (currently unimplemented — sign-in mock TODO).`)
}
Object.assign(metrics, { sanity })

// Screenshot next to metrics/trace so the reader can see what the browser
// saw at the end of the wait window.
const screenshotPath = join(LCP_OUTPUT_DIR, `${LCP_LABEL}.screenshot.png`)
await page.screenshot({ path: screenshotPath, fullPage: false })
console.log(`[lcp-capture]   ${screenshotPath}`)

// --- stop trace, wait for the final Tracing.tracingComplete event, then
//     dump chunks + gzip.
const tracingCompletePromise = new Promise((resolve) => {
  client.once('Tracing.tracingComplete', resolve)
})
await client.send('Tracing.end')
await tracingCompletePromise

const traceDoc = { traceEvents: traceChunks }
const traceJson = JSON.stringify(traceDoc)
writeFileSync(tracePathRaw, traceJson, 'utf-8')

// gzip alongside the raw so the trace file matches the shape checked in at
// docs/wip/graph-v3/traces/03-C-first-paint.json.json.gz (they compress ~10x).
await pipeline(Readable.from(traceJson), createGzip(), createWriteStream(tracePathGz))

const summary = {
  captured_at: new Date().toISOString(),
  label: LCP_LABEL,
  target_url: LCP_TARGET_URL,
  viewport: {
    width: parseInt(VIEWPORT_WIDTH),
    height: parseInt(VIEWPORT_HEIGHT),
    device_scale_factor: parseFloat(VIEWPORT_DPR),
  },
  wait_after_nav_ms: parseInt(LCP_WAIT_MS),
  nav_elapsed_ms: Date.now() - navStart,
  metrics,
  trace_events_count: traceChunks.length,
  trace_files: {
    raw: tracePathRaw,
    gz: tracePathGz,
  },
}
writeFileSync(metricsPath, JSON.stringify(summary, null, 2))
console.log(`[lcp-capture] wrote:\n  ${metricsPath}\n  ${tracePathRaw}\n  ${tracePathGz}`)

await context.close()
await browser.close()
