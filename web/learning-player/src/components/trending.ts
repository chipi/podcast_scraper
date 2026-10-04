/** A corpus topic that's "heating up" (temporal_velocity), shared by the Home
 *  trending views. `series` is its monthly counts aligned to a common month axis. */
export interface RisingTopic {
  id: string
  label: string
  /** velocity_last_over_6mo, rounded to 1dp (e.g. 2.1 → "2.1×"). An acceleration RATIO —
   *  informative, but NOT what the rail is ordered by. See {@link RisingTopic.score}. */
  v: number
  /** #1931 ``trend_score`` — volume-with-recency, the server's ranking signal. Any client-side
   *  re-grouping must order by THIS, or the rail contradicts the order the server chose.
   *  Optional: artifacts written before #1931 have no trend_score; callers fall back to `v`. */
  score?: number
  total: number
  series: number[]
  /** Optional speaker role (host/guest/mentioned) — set for people, drives a role badge. */
  role?: string | null
  /** Optional hosted-photo route — set for people with a hosted photo, drives a chip avatar. */
  image_url?: string | null
}

/** Per-topic theme ("storyline") colouring for the sparkline view: topics in the same
 *  co-occurrence cluster share a hue; unclustered topics use {@link THEME_NEUTRAL}. */
export interface TopicTheme {
  color: string
  label: string | null
  /** Stable group index for ordering (clusters by declaration order; unclustered sorts last). */
  group: number
}

/** Categorical hues (sky / violet / emerald / amber / pink / cyan / lime / orange), cycled across
 *  storyline groups. Tokens `--lp-cat-1..8` (tokens.css, UXS-011), so a direction can repaint them;
 *  inline-style only (`color` / `background-color`), where `var()` always resolves. */
export const THEME_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8].map((i) => `var(--lp-cat-${i})`)

/** Hue for topics not in any storyline group. */
export const THEME_NEUTRAL = "var(--lp-cat-neutral)"

export type TrendDirection = "up" | "down" | "steady"

/** Velocity → trend direction. A neutral band around 1.0 (flat) stops tiny wobbles from
 *  flipping green↔red: ≥1.15 rising, ≤0.85 cooling, else steady. */
export function trendDirection(v: number): TrendDirection {
  if (v >= 1.15) return "up"
  if (v <= 0.85) return "down"
  return "steady"
}

/** Green (rising) / red (cooling) / amber (steady) as the `--lp-trend-*` tokens, so a visual
 *  direction can repaint them (`paper` does). Inline-style only (`color`, `drop-shadow()`), where
 *  `var()` resolves; not for an SVG ``fill=`` attribute. */
export function trendColor(v: number): string {
  const d = trendDirection(v)
  return d === "up"
    ? "var(--lp-trend-rising)"
    : d === "down"
      ? "var(--lp-trend-cooling)"
      : "var(--lp-trend-steady)"
}

/** Trend colour for text and strokes that sit on a dark scrim OVER ARTWORK (the trending-shows
 *  rail), not on a direction's ground — so it must not follow a light direction's darker set.
 *  These are the dark-ground values `--lp-trend-*` ship with. Cooling is red-400 (#f87171), not
 *  red-500: the darker red failed AA (4.49:1) as small bold text over scrims; this clears ~6:1. */
export function trendColorOnArtwork(v: number): string {
  const d = trendDirection(v)
  return d === "up" ? "#22c55e" : d === "down" ? "#f87171" : "#f59e0b"
}

/** ↑ rising / ↓ cooling / → steady — pairs with {@link trendColor}. */
export function trendArrow(v: number): string {
  const d = trendDirection(v)
  return d === "up" ? "↑" : d === "down" ? "↓" : "→"
}
