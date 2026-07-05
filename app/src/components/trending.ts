/** A corpus topic that's "heating up" (temporal_velocity), shared by the Home
 *  trending views. `series` is its monthly counts aligned to a common month axis. */
export interface RisingTopic {
  id: string
  label: string
  /** velocity_last_over_6mo, rounded to 1dp (e.g. 2.1 → "2.1×"). */
  v: number
  total: number
  series: number[]
}

export type TrendDirection = 'up' | 'down' | 'steady'

/** Velocity → trend direction. A neutral band around 1.0 (flat) stops tiny wobbles from
 *  flipping green↔red: ≥1.15 rising, ≤0.85 cooling, else steady. */
export function trendDirection(v: number): TrendDirection {
  if (v >= 1.15) return 'up'
  if (v <= 0.85) return 'down'
  return 'steady'
}

/** Green (rising) / red (cooling) / amber (steady). Returned as a hex so callers can drive
 *  both SVG ``fill`` and CSS ``color`` without depending on the configured Tailwind palette. */
export function trendColor(v: number): string {
  const d = trendDirection(v)
  return d === 'up' ? '#22c55e' : d === 'down' ? '#ef4444' : '#f59e0b'
}

/** ↑ rising / ↓ cooling / → steady — pairs with {@link trendColor}. */
export function trendArrow(v: number): string {
  const d = trendDirection(v)
  return d === 'up' ? '↑' : d === 'down' ? '↓' : '→'
}
