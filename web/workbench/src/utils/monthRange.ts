/**
 * Inclusive filter of `{ month: 'YYYY-MM' }` rows to a `[from, to]` range.
 * Empty `from`/`to` means open-ended on that side. Lexicographic compare is
 * correct for zero-padded `YYYY-MM`. Used by the Index timeseries date filter.
 */
export function filterByMonthRange<T extends { month: string }>(
  items: T[],
  from: string,
  to: string,
): T[] {
  const lo = from.trim()
  const hi = to.trim()
  return items.filter((it) => {
    if (lo && it.month < lo) return false
    if (hi && it.month > hi) return false
    return true
  })
}
