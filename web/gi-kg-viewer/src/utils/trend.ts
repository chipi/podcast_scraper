/** Shared trend-direction helpers (temporal_velocity). Green rising / red cooling / amber
 *  steady, with a neutral band around the 1.0 flat line so tiny wobbles don't flip. Defined
 *  once and reused across the dashboard trending views and the digest topic bands. */

export type TrendDirection = 'up' | 'down' | 'steady'

export function trendDirection(v: number): TrendDirection {
  if (v >= 1.15) return 'up'
  if (v <= 0.85) return 'down'
  return 'steady'
}

/** Hex colour so callers can drive both SVG ``fill`` and CSS ``color`` without depending on the
 *  configured Tailwind palette. */
export function trendColor(v: number): string {
  const d = trendDirection(v)
  return d === 'up' ? '#22c55e' : d === 'down' ? '#ef4444' : '#f59e0b'
}

/** ↑ rising / ↓ cooling / → steady — pairs with {@link trendColor}. */
export function trendArrow(v: number): string {
  const d = trendDirection(v)
  return d === 'up' ? '↑' : d === 'down' ? '↓' : '→'
}
