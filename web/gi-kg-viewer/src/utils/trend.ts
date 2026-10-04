/** Shared trend-direction helpers (temporal_velocity). Green rising / red cooling / grey
 *  steady, with a neutral band around the 1.0 flat line so tiny wobbles don't flip. Defined
 *  once and reused across the dashboard trending views and the digest topic bands. */

export type TrendDirection = 'up' | 'down' | 'steady'

export function trendDirection(v: number): TrendDirection {
  if (v >= 1.15) return 'up'
  if (v <= 0.85) return 'down'
  return 'steady'
}

/** A theme token, so the colour follows light / dark. Usable as a CSS ``color`` or an inline-style
 *  ``fill`` — NOT as an SVG ``fill=`` attribute, which does not resolve ``var()`` everywhere.
 *  Steady is ``muted``, not amber: it is the "not moving" state, and amber read as a warning. The old
 *  Tailwind hexes failed AA as text in light (2.15–3.76:1) and "down" failed even in dark (4.31:1). */
export function trendColor(v: number): string {
  const d = trendDirection(v)
  return d === 'up' ? 'var(--ps-success)' : d === 'down' ? 'var(--ps-danger)' : 'var(--ps-muted)'
}

/** ↑ rising / ↓ cooling / → steady — pairs with {@link trendColor}. */
export function trendArrow(v: number): string {
  const d = trendDirection(v)
  return d === 'up' ? '↑' : d === 'down' ? '↓' : '→'
}
