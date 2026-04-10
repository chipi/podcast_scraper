/**
 * Canonical feed id for viewer UI and API filters.
 * Matches ``normalize_feed_id`` in ``corpus_scope.py`` for non-empty strings (strip only).
 */
export function normalizeFeedIdForViewer(feedId: string | null | undefined): string {
  return (feedId ?? '').trim()
}
