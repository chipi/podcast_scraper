/**
 * Native ``title`` / hover text for feed name — same string everywhere:
 * Digest (topic + recent rows), Library (episode list + episode detail meta line).
 */

export type EpisodeRowFeedHoverRow = {
  feed_id?: string | null
  feed_display_title?: string | null
  feed_rss_url?: string | null
  feed_description?: string | null
}

export type CatalogFeedHover = {
  rss_url?: string | null
  description?: string | null
}

export type CatalogFeedForHoverLookup = CatalogFeedHover & { feed_id: string }

/**
 * Multi-line hover: RSS URL, feed id, description (episode row and/or one catalog feed row).
 */
export function buildEpisodeRowFeedHoverTitle(
  row: EpisodeRowFeedHoverRow,
  catalogFeed?: CatalogFeedHover | null,
): string {
  const lines: string[] = []
  const rss = row.feed_rss_url?.trim() || catalogFeed?.rss_url?.trim()
  if (rss) {
    lines.push(`RSS: ${rss}`)
  }
  const fid = row.feed_id?.trim()
  if (fid) {
    lines.push(`Feed id: ${fid}`)
  }
  const desc = row.feed_description?.trim() || catalogFeed?.description?.trim()
  if (desc) {
    lines.push(desc)
  }
  return lines.join('\n')
}

/**
 * Same hover as ``buildEpisodeRowFeedHoverTitle``, resolving the catalog row via
 * ``normalizeFeedId(feed_id)`` (Digest + Library both use this).
 */
export function feedNameHoverWithCatalogLookup(
  row: EpisodeRowFeedHoverRow,
  catalogFeeds: readonly CatalogFeedForHoverLookup[],
  normalizeFeedId: (raw: string) => string,
): string {
  const fid = row.feed_id?.trim() ?? ''
  const want = fid ? normalizeFeedId(fid) : ''
  const hit = want
    ? catalogFeeds.find((f) => normalizeFeedId(f.feed_id) === want)
    : undefined
  return buildEpisodeRowFeedHoverTitle(row, hit ?? null)
}
