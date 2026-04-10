import { getTokenVar } from '../theme/theme'

/** Parse `#RRGGBB` or return null. */
function parseHex(hex: string): { r: number; g: number; b: number } | null {
  const h = hex.replace('#', '').trim()
  if (h.length !== 6) {
    return null
  }
  const n = Number.parseInt(h, 16)
  if (Number.isNaN(n)) {
    return null
  }
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 }
}

export function rgbaFromToken(varName: `--ps-${string}`, alpha: number): string {
  const raw = getTokenVar(varName)
  const rgb = parseHex(raw)
  if (!rgb) {
    return `rgba(76, 144, 240, ${alpha})`
  }
  return `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`
}

export function chartGridColor(): string {
  return rgbaFromToken('--ps-border', 0.35)
}

/** Ordered palette for multi-series charts (semantic tokens). */
export function chartSeriesColors(count: number): string[] {
  const keys: `--ps-${string}`[] = [
    '--ps-primary',
    '--ps-gi',
    '--ps-kg',
    '--ps-warning',
    '--ps-success',
    '--ps-danger',
    '--ps-link',
    '--ps-muted',
  ]
  const out: string[] = []
  for (let i = 0; i < count; i += 1) {
    out.push(rgbaFromToken(keys[i % keys.length]!, 0.75))
  }
  return out
}
