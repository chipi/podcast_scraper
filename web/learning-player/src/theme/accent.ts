/**
 * Per-show adaptive accent: derive `--lp-accent` from the current episode/show artwork (UXS-011,
 * #1598). Wired in `App.vue` off `player.currentArtwork`.
 *
 * Flow: load the artwork → sample a downscaled copy → pick the dominant vibrant colour → clamp it
 * to ≥4.5:1 against the surface (`contrast.ts`) → `setShowAccent`. Any failure (image error,
 * cross-origin canvas taint, or an artwork with no vibrant colour) resolves to `null` and falls
 * back to the brand default — a missing accent is never an error, just the Ember baseline.
 *
 * Extraction is a best-effort visual nicety; the app must never block or throw on it.
 */

import { clampToContrast, rgbToHslChannels, ACCENT_TEXT_BG, MIN_CONTRAST } from './contrast'
import { setShowAccent } from './theme'

const SAMPLE_SIZE = 24 // downscale target; enough hue signal, ~576 pixels to scan
const HUE_BUCKETS = 12
const MIN_SAT = 0.15 // below this a pixel is effectively greyscale — no accent signal
const MIN_L = 0.15
const MAX_L = 0.85

function toHex(r: number, g: number, b: number): string {
  const h = (n: number): string =>
    Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, '0')
  return `#${h(r)}${h(g)}${h(b)}`
}

/**
 * The dominant vibrant colour of an RGBA pixel buffer, or `null` if the image has no colour worth
 * accenting (all grey / transparent / too dark or light). Buckets qualifying pixels by hue, weights
 * each bucket by saturation, and returns the saturation-weighted mean of the heaviest bucket — so a
 * large muted background does not outvote a smaller vivid subject. Pure; unit-tested directly.
 */
export function vibrantColorFromPixels(data: Uint8ClampedArray | number[]): string | null {
  const weight = new Array<number>(HUE_BUCKETS).fill(0)
  const sumR = new Array<number>(HUE_BUCKETS).fill(0)
  const sumG = new Array<number>(HUE_BUCKETS).fill(0)
  const sumB = new Array<number>(HUE_BUCKETS).fill(0)
  let any = false

  for (let i = 0; i + 3 < data.length; i += 4) {
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]
    const a = data[i + 3]
    if (a < 125) continue
    const { h, s, l } = rgbToHslChannels(r, g, b)
    if (s < MIN_SAT || l < MIN_L || l > MAX_L) continue
    const bucket = Math.min(HUE_BUCKETS - 1, Math.floor(h * HUE_BUCKETS))
    weight[bucket] += s
    sumR[bucket] += r * s
    sumG[bucket] += g * s
    sumB[bucket] += b * s
    any = true
  }
  if (!any) return null

  let top = 0
  for (let k = 1; k < HUE_BUCKETS; k++) if (weight[k] > weight[top]) top = k
  const w = weight[top]
  if (w <= 0) return null
  return toHex(sumR[top] / w, sumG[top] / w, sumB[top] / w)
}

/**
 * Load *url*, sample it, and return its dominant vibrant colour as hex, or `null` on any failure.
 * Uses an anonymous cross-origin request so a same-origin (our stored) artwork can be read back;
 * a tainted cross-origin image makes `getImageData` throw and we resolve `null`.
 */
export function extractAccentFromImage(url: string): Promise<string | null> {
  return new Promise((resolve) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas')
        canvas.width = SAMPLE_SIZE
        canvas.height = SAMPLE_SIZE
        const ctx = canvas.getContext('2d')
        if (!ctx) return resolve(null)
        ctx.drawImage(img, 0, 0, SAMPLE_SIZE, SAMPLE_SIZE)
        const { data } = ctx.getImageData(0, 0, SAMPLE_SIZE, SAMPLE_SIZE)
        resolve(vibrantColorFromPixels(data))
      } catch {
        resolve(null) // cross-origin taint or a canvas without pixel access
      }
    }
    img.onerror = () => resolve(null)
    img.src = url
  })
}

/**
 * Derive the accent from *url* and apply it to *el*'s `--lp-accent`. `null`/failed extraction clears
 * back to the brand default. Injectable *extract* keeps the DOM-free path testable.
 */
export async function deriveShowAccent(
  url: string | null,
  el: HTMLElement = document.documentElement,
  extract: (u: string) => Promise<string | null> = extractAccentFromImage,
): Promise<void> {
  if (!url) {
    setShowAccent(null, el)
    return
  }
  const raw = await extract(url)
  if (!raw) {
    setShowAccent(null, el)
    return
  }
  setShowAccent(clampToContrast(raw, ACCENT_TEXT_BG, MIN_CONTRAST), el)
}
