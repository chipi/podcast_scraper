/**
 * "Copy debug info" (Settings, operator 2026-10-08): one text block a tester can paste back, taken
 * at the moment something looks wrong, so we never have to ask them for their phone model, Android
 * version, WebView version or GPU.
 *
 * The page's own view (user agent, GPU name via WebGL, approximate RAM, JS heap) plus, in the app,
 * the native `AppProcess.memoryInfo` (free / total RAM, low-memory flag, app footprint, thermal
 * state, WebView package). What nothing can read is said in the block rather than left out silently:
 * free GPU memory is exposed by neither Android nor iOS.
 */

import { nativeMemoryInfo } from '../services/lifecycle'
import { cacheStats } from '../services/contentCache'

export interface DebugContext {
  version: string
  sha: string
  builtAt: string
  platform: string
  target: string
  route: string
  userId: string | null
}

interface UAData {
  getHighEntropyValues?: (hints: string[]) => Promise<Record<string, unknown>>
}

async function uaDetails(): Promise<string[]> {
  const out: string[] = [`User agent: ${navigator.userAgent}`]
  const ua = (navigator as Navigator & { userAgentData?: UAData }).userAgentData
  if (ua?.getHighEntropyValues) {
    try {
      const v = await ua.getHighEntropyValues(['model', 'platform', 'platformVersion', 'fullVersionList'])
      const brands = Array.isArray(v.fullVersionList)
        ? (v.fullVersionList as { brand: string; version: string }[]).map((b) => `${b.brand} ${b.version}`).join(', ')
        : ''
      out.push(`Device model: ${String(v.model || '(not reported)')}`)
      out.push(`OS: ${String(v.platform || '')} ${String(v.platformVersion || '')}`.trim())
      if (brands) out.push(`Browser engine: ${brands}`)
    } catch {
      /* not offered — the user agent above still carries most of it */
    }
  }
  return out
}

function gpu(): string {
  try {
    const canvas = document.createElement('canvas')
    const gl = (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')) as WebGLRenderingContext | null
    if (!gl) return 'GPU: WebGL unavailable'
    const ext = gl.getExtension('WEBGL_debug_renderer_info')
    const renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)
    const vendor = ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR)
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE)
    gl.getExtension('WEBGL_lose_context')?.loseContext()
    return `GPU: ${String(renderer)} (${String(vendor)}), max texture ${String(maxTex)}`
  } catch {
    return 'GPU: could not be read'
  }
}

function memory(): string[] {
  const out: string[] = []
  const dm = (navigator as Navigator & { deviceMemory?: number }).deviceMemory
  out.push(`Device memory: ${dm != null ? `~${dm} GB` : '(not reported)'}`)
  out.push(`CPU cores: ${navigator.hardwareConcurrency ?? '(not reported)'}`)
  const pm = (performance as Performance & {
    memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number }
  }).memory
  if (pm) {
    const mb = (n: number): string => `${Math.round(n / 1048576)} MB`
    out.push(`App JS memory: ${mb(pm.usedJSHeapSize)} used of ${mb(pm.jsHeapSizeLimit)} limit`)
  }
  return out
}

/** Native facts, one per line, in a stable order; absent on the web or an older app build. */
async function nativeLines(): Promise<string[]> {
  const n = await nativeMemoryInfo()
  if (!n) return ['Native: (not available — web, or an app build before 2026-10-08)']
  const order = [
    'manufacturer', 'model', 'osVersion', 'webView', 'availMb', 'totalMb', 'thresholdMb', 'lowMemory',
    'lowRamDevice', 'memoryClassMb', 'appPssMb', 'appFootprintMb', 'thermalStatus', 'thermalState',
    'lowPowerMode',
  ]
  const keys = [...order.filter((k) => k in n), ...Object.keys(n).filter((k) => !order.includes(k))]
  return keys.map((k) => `Native ${k}: ${String(n[k])}`)
}

function screenInfo(): string {
  const dark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  return (
    `Screen: ${screen.width}x${screen.height} @${window.devicePixelRatio}x, ` +
    `viewport ${window.innerWidth}x${window.innerHeight}, ${dark} mode`
  )
}

function network(): string {
  const c = (navigator as Navigator & { connection?: { effectiveType?: string; type?: string; saveData?: boolean } })
    .connection
  const kind = c ? [c.type, c.effectiveType, c.saveData ? 'data saver' : ''].filter(Boolean).join(', ') : ''
  return `Network: ${navigator.onLine ? 'online' : 'offline'}${kind ? ` (${kind})` : ''}`
}

/**
 * The images the page holds right now and what they cost decoded — the measure behind the Android
 * blank-render bug (2026-10-08: compositor memory, not JS heap). Natural size, 4 bytes a pixel.
 */
function images(): string[] {
  try {
    const imgs = Array.from(document.querySelectorAll('img')).filter((i) => i.complete && i.naturalWidth > 0)
    const px = imgs.reduce((n, i) => n + i.naturalWidth * i.naturalHeight, 0)
    const largest = imgs.reduce((m, i) => Math.max(m, i.naturalWidth, i.naturalHeight), 0)
    return [
      `Images on the page: ${imgs.length} decoded, ~${Math.round((px * 4) / 1048576)} MB as bitmaps, ` +
        `largest side ${largest}px`,
    ]
  } catch {
    return ['Images on the page: could not be read']
  }
}

/** The on-device content cache for this account, per key (operator 2026-10-08: dump the cache). */
async function cacheLines(): Promise<string[]> {
  const { namespace, entries } = await cacheStats()
  const kb = (n: number): string => `${Math.round(n / 1024)} KB`
  const total = entries.reduce((n, e) => n + e.bytes, 0)
  return [
    `Cache (${namespace}): ${entries.length} keys, ${kb(total)}`,
    ...entries
      .sort((a, b) => b.bytes - a.bytes)
      .map((e) => `  ${e.key}: ${kb(e.bytes)}`),
  ]
}

/** The whole block, in the order a reader triages: when, what app, what device, what screen. */
export async function collectDebugInfo(ctx: DebugContext): Promise<string> {
  const now = new Date()
  const lines = [
    'Close Listening debug info',
    `Time: ${now.toLocaleString()} (UTC ${now.toISOString()})`,
    `App: ${ctx.version} · ${ctx.sha} · built ${ctx.builtAt} · ${ctx.platform} · backend ${ctx.target}`,
    `Screen in app: ${ctx.route}`,
    `Account: ${ctx.userId ?? '(signed out)'}`,
    ...(await uaDetails()),
    gpu(),
    ...memory(),
    ...(await nativeLines()),
    'Not readable by any app: free GPU memory, GPU load',
    ...images(),
    ...(await cacheLines()),
    screenInfo(),
    network(),
  ]
  return lines.join('\n')
}
