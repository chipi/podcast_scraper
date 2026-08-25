/** A corpus topic that's "heating up" (temporal_velocity), shared by the Home
 *  trending views. `series` is its monthly counts aligned to a common month axis. */
export interface RisingTopic {
  id: string
  label: string
  /** velocity_last_over_6mo, rounded to 1dp (e.g. 2.1 → "2.1×"). */
  v: number
  total: number
  series: number[]
  /** Optional speaker role (host/guest/mentioned) — set for people, drives a role badge. */
  role?: string | null
}

/** Per-topic theme ("storyline") colouring for the sparkline view: topics in the same
 *  co-occurrence cluster share a hue; unclustered topics use {@link THEME_NEUTRAL}. */
export interface TopicTheme {
  color: string
  label: string | null
  /** Stable group index for ordering (clusters by declaration order; unclustered sorts last). */
  group: number
}

/** Distinct, dark-theme-legible categorical hues (sky / violet / emerald / amber / pink / cyan /
 *  lime / orange). Cycled across theme clusters. */
export const THEME_PALETTE = [
  '#38bdf8',
  '#a78bfa',
  '#34d399',
  '#fbbf24',
  '#f472b6',
  '#22d3ee',
  '#a3e635',
  '#fb923c',
]

/** Fallback hue for topics not in any theme cluster. */
export const THEME_NEUTRAL = '#94a3b8'

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
  // Cooling uses red-400 (#f87171), not red-500 — the darker red failed WCAG AA (4.49:1) as small
  // bold text on the dark surface / over artwork scrims; the lighter red clears it (~6:1).
  return d === 'up' ? '#22c55e' : d === 'down' ? '#f87171' : '#f59e0b'
}

/** ↑ rising / ↓ cooling / → steady — pairs with {@link trendColor}. */
export function trendArrow(v: number): string {
  const d = trendDirection(v)
  return d === 'up' ? '↑' : d === 'down' ? '↓' : '→'
}
