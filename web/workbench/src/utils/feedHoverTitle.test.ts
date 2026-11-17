import { describe, expect, it } from 'vitest'
import {
  buildEpisodeRowFeedHoverTitle,
  feedNameHoverWithCatalogLookup,
} from './feedHoverTitle'

describe('buildEpisodeRowFeedHoverTitle', () => {
  it('joins rss, id, and description from row', () => {
    const t = buildEpisodeRowFeedHoverTitle({
      feed_id: 'f1',
      feed_rss_url: 'https://x/rss',
      feed_description: 'About',
    })
    expect(t).toContain('RSS: https://x/rss')
    expect(t).toContain('Feed id: f1')
    expect(t).toContain('About')
  })

  it('falls back to catalog rss and description', () => {
    const t = buildEpisodeRowFeedHoverTitle(
      { feed_id: 'z' },
      { rss_url: 'https://c/rss', description: 'Cat desc' },
    )
    expect(t).toContain('RSS: https://c/rss')
    expect(t).toContain('Cat desc')
  })

  it('returns empty when nothing to show', () => {
    expect(buildEpisodeRowFeedHoverTitle({})).toBe('')
  })
})

describe('feedNameHoverWithCatalogLookup', () => {
  it('merges row and catalog feed by feed_id', () => {
    const t = feedNameHoverWithCatalogLookup(
      { feed_id: 'f1' },
      [{ feed_id: 'f1', rss_url: 'https://x/rss', description: 'D' }],
      (s) => s,
    )
    expect(t).toContain('RSS: https://x/rss')
    expect(t).toContain('Feed id: f1')
    expect(t).toContain('D')
  })
})
