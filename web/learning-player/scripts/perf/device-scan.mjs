#!/usr/bin/env node
/**
 * Device performance scan for the player app — `make perf-android` (2026-10-08).
 *
 * Attaches to the app's WebView over the Chrome DevTools Protocol (Android: `adb forward` to its
 * `webview_devtools_remote_<pid>` socket, which `make perf-android` sets up) and walks every main
 * screen IN THE APP AS INSTALLED — whatever backend it points at and whoever is signed in. For each
 * screen it records what a listener feels and what the device pays:
 *
 *   settle      ms until no API request has been in flight for 1.5 s (the screen has its data)
 *   api         requests, KB, the slowest one
 *   long tasks  main-thread tasks > 50 ms (count, total, longest) under a CPU slowdown
 *   heap        JS heap used
 *   images      loaded <img>s, their decoded size, how many are hidden or larger than 1000 px
 *   layers      composited layers drawing content
 *   gpu         Chromium tile + GPU image memory (memory-infra), and the app's PSS from dumpsys
 *
 * Built from the probes that found the 2026-10-08 Android blank-render bug (73 layers, 200 MB of
 * tiles, 3000 px originals in 116 px cards) so the next regression is a report, not an afternoon.
 *
 * Usage: node scripts/perf/device-scan.mjs --cdp http://localhost:9333 --out <dir>
 *          [--cpu 4] [--adb <path>] [--package app.closelistening.player] [--routes a,b,c]
 * Writes <dir>/report.md and <dir>/report.json.
 */
import { createRequire } from 'module'
import { execFileSync } from 'child_process'
import { mkdirSync, writeFileSync } from 'fs'
import { join } from 'path'

const require = createRequire(import.meta.url)
const { chromium } = require('playwright')

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`)
  return i > 0 ? process.argv[i + 1] : dflt
}
const CDP = arg('cdp', 'http://localhost:9333')
const OUT = arg('out', '.test_outputs/perf')
const CPU = Number(arg('cpu', '4'))
const ADB = arg('adb', '')
const PKG = arg('package', 'app.closelistening.player')
const SETTLE_QUIET_MS = 1500
const SETTLE_CAP_MS = 25000

const BASE_ROUTES = [
  '/', '/browse', '/search?q=economy',
  '/library?tab=shows', '/library?tab=saved', '/library?tab=collections', '/library?tab=revisit',
  '/profile?tab=account', '/profile?tab=interests', '/profile?tab=stats', '/queue', '/settings',
]

const browser = await chromium.connectOverCDP(CDP)
const page = browser.contexts()[0].pages()[0]
const cdp = await page.context().newCDPSession(page)
const bcdp = await browser.newBrowserCDPSession()
await cdp.send('Network.enable')
await cdp.send('Performance.enable')
await cdp.send('LayerTree.enable')

// API calls do NOT go through the WebView's network stack: `CapacitorHttp` (capacitor.config.ts)
// routes fetch/XHR natively, so DevTools never sees them — a first version of this scan reported
// artwork and analytics as "the API". They are measured IN THE PAGE instead, by wrapping the
// (already native-patched) fetch: duration, decoded size, status, and an in-flight count for
// "settled". Images are real WebView loads, so those stay on the DevTools network log.
const WRAP = () => {
  if (window.__lpPerf) return
  const P = (window.__lpPerf = { calls: [], inflight: 0 })
  const orig = window.fetch.bind(window)
  window.fetch = async (input, init) => {
    const url = String(input && input.url ? input.url : input)
    const isApi = /\/api\//.test(url) && !/analytics\./.test(url)
    if (!isApi) return orig(input, init)
    const t = performance.now()
    P.inflight++
    try {
      const r = await orig(input, init)
      const rec = { url, method: (init && init.method) || 'GET', ms: Math.round(performance.now() - t), status: r.status }
      try {
        const body = await r.clone().text()
        rec.kb = Math.round(body.length / 1024)
        if (/\/(trending|storylines)\b/.test(url)) rec.body = body.slice(0, 50000)
      } catch {
        /* an opaque body still has its timing */
      }
      P.calls.push(rec)
      return r
    } catch (e) {
      P.calls.push({ url, ms: Math.round(performance.now() - t), status: 'failed' })
      throw e
    } finally {
      P.inflight--
    }
  }
  P.longTasks = []
  new PerformanceObserver((l) => P.longTasks.push(...l.getEntries().map((e) => e.duration))).observe({ type: 'longtask', buffered: true })
}

// A FRESH app: reload, so every screen is measured on its first visit after a start — the case a
// listener waits on. The wrapper goes in right after, before any screen is opened.
await page.reload()
await page.waitForTimeout(6000)
await page.evaluate(WRAP)
await cdp.send('LayerTree.disable').catch(() => {})
await cdp.send('LayerTree.enable')
await cdp.send('Emulation.setCPUThrottlingRate', { rate: CPU })

const push = (r) =>
  page.evaluate((to) => document.querySelector('#app').__vue_app__.config.globalProperties.$router.push(to), r)

// Entity routes from what the app itself loads — links it renders, and the ids in its own /trending
// and /storylines responses (topics and storylines open from buttons, not links) — so the scan works
// on any corpus, backend and account.
async function discoverRoutes() {
  const found = {}
  for (const from of ['/browse', '/']) {
    await push(from)
    await page.waitForTimeout(5000)
    const hrefs = await page.evaluate(() => [...document.querySelectorAll('a[href]')].map((a) => a.getAttribute('href')))
    for (const h of hrefs) {
      const m = h && h.match(/^\/(episode|topic|person|storyline|theme)\/[^?#]+/)
      if (m && !found[m[1]]) found[m[1]] = m[0]
    }
  }
  const bodies = await page.evaluate(() => window.__lpPerf.calls.filter((c) => c.body).map((c) => c.body))
  for (const body of bodies) {
    try {
      const data = JSON.parse(body)
      for (const it of data.items ?? data) {
        const eid = it.entity_id ?? it.id
        const kind = eid?.startsWith('topic:') ? 'topic' : eid?.startsWith('person:') ? 'person' : eid?.startsWith('thc:') ? 'storyline' : eid?.startsWith('tc:') ? 'theme' : null
        if (kind && !found[kind]) found[kind] = `/${kind}/${eid}`
      }
    } catch {
      /* not JSON — the links above still count */
    }
  }
  return Object.values(found)
}

let layers = []
cdp.on('LayerTree.layerTreeDidChange', (e) => { if (e.layers) layers = e.layers })
// Media (artwork, photos) — real WebView loads.
let media = new Map()
cdp.on('Network.requestWillBeSent', (e) => {
  if (e.type === 'Image' || /\/artwork|\/photo/.test(e.request.url)) media.set(e.requestId, { url: e.request.url, t0: e.timestamp })
})
cdp.on('Network.loadingFinished', (e) => {
  const r = media.get(e.requestId)
  if (r) { r.ms = Math.round((e.timestamp - r.t0) * 1000); r.kb = Math.round(e.encodedDataLength / 1024) }
})
cdp.on('Network.loadingFailed', (e) => {
  const r = media.get(e.requestId)
  if (r) { r.ms = Math.round((e.timestamp - r.t0) * 1000); r.failed = true }
})

async function gpuMemory() {
  const events = []
  const onData = (e) => events.push(...e.value)
  bcdp.on('Tracing.dataCollected', onData)
  const done = new Promise((r) => bcdp.once('Tracing.tracingComplete', r))
  await bcdp.send('Tracing.start', { traceConfig: { includedCategories: ['disabled-by-default-memory-infra'], memoryDumpConfig: { triggers: [] } }, transferMode: 'ReportEvents' })
  await bcdp.send('Tracing.requestMemoryDump', { levelOfDetail: 'detailed' })
  await bcdp.send('Tracing.end')
  await done
  bcdp.off('Tracing.dataCollected', onData)
  const tot = {}
  for (const ev of events) {
    const al = ev.args?.dumps?.allocators
    if (!al) continue
    for (const name of ['cc/tile_memory', 'gpu/shared_images', 'cc/image_memory']) {
      const v = al[name]?.attrs?.size?.value
      if (v) tot[name] = (tot[name] || 0) + parseInt(v, 16)
    }
  }
  const mb = (n) => Math.round((n || 0) / 1048576)
  return { tilesMB: mb(tot['cc/tile_memory']), gpuImagesMB: mb(tot['gpu/shared_images']), imageCacheMB: mb(tot['cc/image_memory']) }
}

function appPssMB() {
  if (!ADB) return null
  try {
    const pid = execFileSync(ADB, ['shell', 'pidof', PKG]).toString().trim()
    const m = execFileSync(ADB, ['shell', 'dumpsys', 'meminfo', pid]).toString()
    const t = m.match(/TOTAL PSS:\s+(\d+)/) || m.match(/^\s*TOTAL\s+(\d+)/m)
    return t ? Math.round(Number(t[1]) / 1024) : null
  } catch {
    return null
  }
}

const routes = [...BASE_ROUTES, ...(arg('routes', '') ? arg('routes').split(',') : await discoverRoutes())]
const rows = []
const allReqs = []
for (const route of routes) {
  media = new Map()
  await page.evaluate(() => {
    window.__lpPerf.calls.length = 0
    window.__lpPerf.longTasks.length = 0
    document.querySelectorAll('dialog[open]').forEach((d) => d.close())
  })
  const t0 = Date.now()
  await push(route)
  let quietSince = Date.now()
  while (Date.now() - t0 < SETTLE_CAP_MS) {
    const busy =
      (await page.evaluate(() => window.__lpPerf.inflight)) > 0 || [...media.values()].some((r) => r.ms == null)
    if (busy) quietSince = Date.now()
    else if (Date.now() - quietSince > SETTLE_QUIET_MS) break
    await page.waitForTimeout(100)
  }
  const settleMs = Math.max(0, quietSince - t0)
  const perf = await page.evaluate(() => ({
    calls: window.__lpPerf.calls.map(({ body, ...c }) => c),
    longTasks: window.__lpPerf.longTasks.slice(),
  }))
  const lt = perf.longTasks
  const metrics = Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map((m) => [m.name, m.value]))
  const images = await page.evaluate(() => {
    const L = [...document.images].filter((i) => i.complete && i.naturalWidth)
    const hidden = (i) => !i.offsetParent || i.getBoundingClientRect().width === 0
    const mb = (a) => Math.round(a.reduce((s, i) => s + i.naturalWidth * i.naturalHeight * 4, 0) / 1048576)
    return { count: L.length, decodedMB: mb(L), hiddenMB: mb(L.filter(hidden)), over1000: L.filter((i) => i.naturalWidth > 1000).length }
  })
  const list = perf.calls
  const slowest = [...list].sort((a, b) => b.ms - a.ms)[0]
  const m = [...media.values()].filter((r) => r.ms != null)
  const row = {
    route, settleMs,
    api: { count: list.length, kb: list.reduce((s, r) => s + (r.kb || 0), 0), slowestMs: slowest?.ms ?? null, slowest: slowest ? shortUrl(slowest.url) : null },
    media: { count: m.length, kb: m.reduce((s, r) => s + (r.kb || 0), 0), slowestMs: m.length ? Math.max(...m.map((r) => r.ms)) : null },
    longTasks: { count: lt.length, totalMs: Math.round(lt.reduce((a, b) => a + b, 0)), maxMs: Math.round(Math.max(0, ...lt)) },
    heapMB: Math.round(metrics.JSHeapUsedSize / 1048576),
    images,
    layers: layers.filter((l) => l.drawsContent).length,
    ...(await gpuMemory()),
    appPssMB: appPssMB(),
  }
  rows.push(row)
  for (const r of list) allReqs.push({ route, kind: 'api', ...r, url: shortUrl(r.url) })
  for (const r of m) allReqs.push({ route, kind: 'media', method: 'GET', status: r.failed ? 'failed' : 200, ...r, url: shortUrl(r.url) })
  console.log(`${route.slice(0, 40).padEnd(40)} settle ${row.settleMs}ms · api ${row.api.count}/${row.api.kb}KB slowest ${row.api.slowestMs ?? '-'}ms · media ${row.media.count}/${row.media.kb}KB · lt ${row.longTasks.count} max ${row.longTasks.maxMs}ms · img ${images.decodedMB}MB · layers ${row.layers} · tiles ${row.tilesMB}MB`)
}
await cdp.send('Emulation.setCPUThrottlingRate', { rate: 1 })

function shortUrl(u) {
  return u.replace(/^https?:\/\/[^/]+\/api\/app/, '').replace(/^https?:\/\/[^/]+/, '')
}

const stamp = new Date().toISOString()
const ua = await page.evaluate(() => navigator.userAgent)
mkdirSync(OUT, { recursive: true })
writeFileSync(join(OUT, 'report.json'), JSON.stringify({ stamp, cdp: CDP, cpuThrottle: CPU, userAgent: ua, rows, requests: allReqs }, null, 2))
const slow = allReqs.filter((r) => r.ms != null).sort((a, b) => b.ms - a.ms).slice(0, 25)
const big = allReqs.filter((r) => r.kb != null).sort((a, b) => b.kb - a.kb).slice(0, 15)
const md = [
  `# Device performance scan — ${stamp}`,
  '',
  `CPU slowdown ${CPU}× · ${ua}`,
  '',
  '| screen | settle ms | api req / KB | slowest api ms | media req / KB | long tasks (n / max ms) | heap MB | images (n / decoded MB / hidden MB / >1000px) | layers | tiles MB | GPU images MB | app PSS MB |',
  '|---|---|---|---|---|---|---|---|---|---|---|---|',
  ...rows.map((r) => `| \`${r.route}\` | ${r.settleMs} | ${r.api.count} / ${r.api.kb} | ${r.api.slowestMs ?? '-'} | ${r.media.count} / ${r.media.kb} | ${r.longTasks.count} / ${r.longTasks.maxMs} | ${r.heapMB} | ${r.images.count} / ${r.images.decodedMB} / ${r.images.hiddenMB} / ${r.images.over1000} | ${r.layers} | ${r.tilesMB} | ${r.gpuImagesMB} | ${r.appPssMB ?? '-'} |`),
  '',
  '## Slowest requests (api = app fetch timed in the page; media = WebView image loads)',
  '',
  '| ms | KB | kind | status | request | screen |',
  '|---|---|---|---|---|---|',
  ...slow.map((r) => `| ${r.ms} | ${r.kb ?? '-'} | ${r.kind} | ${r.status ?? '-'} | \`${r.method} ${r.url.slice(0, 100)}\` | \`${r.route}\` |`),
  '',
  '## Largest payloads',
  '',
  '| KB | ms | request |',
  '|---|---|---|',
  ...big.map((r) => `| ${r.kb} | ${r.ms} | \`${r.url.slice(0, 100)}\` |`),
  '',
]
writeFileSync(join(OUT, 'report.md'), md.join('\n'))
console.log(`\n✓ report: ${join(OUT, 'report.md')}`)
await browser.close().catch(() => {})
