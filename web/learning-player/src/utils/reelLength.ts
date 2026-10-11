/**
 * A reel's length in whole minutes, as its pill, door and Discover row say it (operator 2026-10-11:
 * "real length everywhere" — nothing promises fifteen minutes a small corpus does not have). Rounded,
 * never below one, never above the budget asked for.
 */
export function reelMinutes(totalSeconds: number, budget = 15): number {
  return Math.min(budget, Math.max(1, Math.round(totalSeconds / 60)))
}
