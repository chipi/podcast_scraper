/**
 * Newest first, for anything stamped with `created_at` in whole Unix SECONDS (notes, highlights).
 *
 * Seconds are too coarse to order on alone: two notes written within one second tie, and a plain
 * sort left them oldest-first — a just-written note could land behind "Show more". Ties break on
 * position instead: the capture store appends, and the server returns in creation order, so of two
 * same-second items the LATER one is the newer.
 */
export function newestFirst<T extends { created_at?: number | null }>(items: readonly T[]): T[] {
  return items
    .map((item, i) => ({ item, i }))
    .sort((a, b) => (b.item.created_at ?? 0) - (a.item.created_at ?? 0) || b.i - a.i)
    .map(({ item }) => item)
}
