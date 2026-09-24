import type { RecapRecurring } from '../services/types'

/**
 * How a recurring topic or person moved against the previous window of the same length.
 *
 * Shared by the Profile recap panel and the Home prompt so the two cannot drift into saying the
 * same movement two different ways.
 *
 * A delta of zero renders NOTHING: an arrow meaning "unchanged" is noise on every entry that did
 * not move, which is most of them. `is_new` wins over delta — absent from the previous window
 * entirely reads as "new", not as "+3".
 *
 * `newLabel` is passed in rather than translated here so this stays a pure function.
 */
export function trendLabel(item: RecapRecurring, newLabel: string): string {
  if (item.is_new) return newLabel
  if (item.delta > 0) return `↑${item.delta}`
  if (item.delta < 0) return `↓${Math.abs(item.delta)}`
  return ''
}
