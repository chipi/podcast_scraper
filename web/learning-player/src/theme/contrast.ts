/**
 * WCAG contrast math for the per-show adaptive accent (UXS-011, #1598).
 *
 * The accent is derived from show artwork (`accent.ts`) and then clamped HERE so it stays legible:
 * it is used both as `text-accent` on the dark surface and as `bg-accent` behind dark
 * `accent-foreground` text, so a low-luminance artwork colour would fail either way. The clamp
 * lifts (or, over a light surface, lowers) the colour's HSL lightness — preserving its hue and
 * saturation, i.e. the show's character — until it meets the minimum ratio. Pure functions, no DOM.
 */

/** The app surface the accent must stay legible against — `--lp-surface` in `tokens.css`. */
export const SURFACE_BG = '#161419'

/** WCAG AA body-text minimum. */
export const MIN_CONTRAST = 4.5

interface Rgb {
  r: number
  g: number
  b: number
}

function parseHex(hex: string): Rgb | null {
  const m = hex.trim().replace(/^#/, '')
  const s =
    m.length === 3
      ? m
          .split('')
          .map((c) => c + c)
          .join('')
      : m
  if (s.length !== 6 || /[^0-9a-fA-F]/.test(s)) return null
  return {
    r: parseInt(s.slice(0, 2), 16),
    g: parseInt(s.slice(2, 4), 16),
    b: parseInt(s.slice(4, 6), 16),
  }
}

function toHex({ r, g, b }: Rgb): string {
  const h = (n: number): string =>
    Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, '0')
  return `#${h(r)}${h(g)}${h(b)}`
}

/** Relative luminance per WCAG 2.x (sRGB → linear, weighted). Input is a hex colour. */
export function relativeLuminance(hex: string): number {
  const rgb = parseHex(hex)
  if (!rgb) return 0
  const lin = (c: number): number => {
    const s = c / 255
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4)
  }
  return 0.2126 * lin(rgb.r) + 0.7152 * lin(rgb.g) + 0.0722 * lin(rgb.b)
}

/** WCAG contrast ratio between two hex colours (1..21, order-independent). */
export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a)
  const lb = relativeLuminance(b)
  const [hi, lo] = la >= lb ? [la, lb] : [lb, la]
  return (hi + 0.05) / (lo + 0.05)
}

function rgbToHsl({ r, g, b }: Rgb): { h: number; s: number; l: number } {
  const rn = r / 255
  const gn = g / 255
  const bn = b / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  const l = (max + min) / 2
  const d = max - min
  let h = 0
  let s = 0
  if (d !== 0) {
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    if (max === rn) h = ((gn - bn) / d + (gn < bn ? 6 : 0)) / 6
    else if (max === gn) h = ((bn - rn) / d + 2) / 6
    else h = ((rn - gn) / d + 4) / 6
  }
  return { h, s, l }
}

/** HSL of raw sRGB channels (0..255). Exposed for the artwork extractor in `accent.ts`. */
export function rgbToHslChannels(r: number, g: number, b: number): { h: number; s: number; l: number } {
  return rgbToHsl({ r, g, b })
}

function hslToRgb(h: number, s: number, l: number): Rgb {
  if (s === 0) {
    const v = l * 255
    return { r: v, g: v, b: v }
  }
  const hue = (t: number): number => {
    let tt = t
    if (tt < 0) tt += 1
    if (tt > 1) tt -= 1
    if (tt < 1 / 6) return p + (q - p) * 6 * tt
    if (tt < 1 / 2) return q
    if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6
    return p
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q
  return { r: hue(h + 1 / 3) * 255, g: hue(h) * 255, b: hue(h - 1 / 3) * 255 }
}

/**
 * Return *hex* unchanged if it already meets *minRatio* against *bg*; otherwise walk its HSL
 * lightness in the direction that raises contrast (away from the background's luminance) until it
 * does, preserving hue + saturation. Returns the best achievable colour if the ratio can't be met
 * (it always can over a dark surface, where even a pure hue at high lightness clears 4.5:1).
 */
export function clampToContrast(hex: string, bg: string = SURFACE_BG, minRatio: number = MIN_CONTRAST): string {
  const rgb = parseHex(hex)
  if (!rgb) return hex
  if (contrastRatio(hex, bg) >= minRatio) return hex
  const { h, s } = rgbToHsl(rgb)
  const lighten = relativeLuminance(bg) < 0.5
  let best = hex
  for (let step = 1; step <= 50; step++) {
    const l = lighten ? Math.min(1, step * 0.02) : Math.max(0, 1 - step * 0.02)
    const candidate = toHex(hslToRgb(h, s, l))
    best = candidate
    if (contrastRatio(candidate, bg) >= minRatio) return candidate
  }
  return best
}
