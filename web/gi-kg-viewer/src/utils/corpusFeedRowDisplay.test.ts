import { describe, expect, it } from 'vitest'
import type { CorpusFeedItem } from '../api/corpusLibraryApi'
import {
  CORPUS_FEED_FILTER_SEARCH_THRESHOLD,
  feedRowAccessibleName,
  feedRowTitleAttr,
  feedRowVisibleLabel,
  filterFeedsByQuery,
} from './corpusFeedRowDisplay'

function makeFeed(overrides: Partial<CorpusFeedItem> = {}): CorpusFeedItem {
  return {
    feed_id: 'feed-1',
    display_title: null,
    episode_count: 0,
    ...overrides,
  }
}

describe('CORPUS_FEED_FILTER_SEARCH_THRESHOLD', () => {
  it('is the documented constant', () => {
    expect(CORPUS_FEED_FILTER_SEARCH_THRESHOLD).toBe(15)
  })
})

describe('feedRowVisibleLabel', () => {
  it('returns placeholder when feed_id is empty', () => {
    expect(feedRowVisibleLabel(makeFeed({ feed_id: '' }))).toBe('(No feed id)')
  })
  it('prefers a trimmed display title', () => {
    expect(feedRowVisibleLabel(makeFeed({ display_title: '  My Show  ' }))).toBe('My Show')
  })
  it('falls back to trimmed feed_id when title is null', () => {
    expect(feedRowVisibleLabel(makeFeed({ feed_id: '  feed-x  ', display_title: null }))).toBe(
      'feed-x',
    )
  })
  it('falls back to feed_id when title is whitespace-only', () => {
    expect(feedRowVisibleLabel(makeFeed({ feed_id: 'feed-x', display_title: '   ' }))).toBe(
      'feed-x',
    )
  })
  it('falls back to feed_id when title is an empty string', () => {
    expect(feedRowVisibleLabel(makeFeed({ feed_id: 'feed-x', display_title: '' }))).toBe('feed-x')
  })
})

describe('feedRowTitleAttr', () => {
  it('returns placeholder when feed_id is empty and nothing else present', () => {
    expect(feedRowTitleAttr(makeFeed({ feed_id: '' }))).toBe('(No feed id)')
  })
  it('joins title and id with a middle dot', () => {
    expect(feedRowTitleAttr(makeFeed({ feed_id: 'feed-x', display_title: 'My Show' }))).toBe(
      'My Show · feed-x',
    )
  })
  it('uses id alone when title is null', () => {
    expect(feedRowTitleAttr(makeFeed({ feed_id: 'feed-x', display_title: null }))).toBe('feed-x')
  })
  it('uses id alone when title is whitespace-only', () => {
    expect(feedRowTitleAttr(makeFeed({ feed_id: 'feed-x', display_title: '  ' }))).toBe('feed-x')
  })
  it('falls back to title when feed_id is non-empty but trims to empty', () => {
    // feed_id is whitespace (passes the `=== ""` guard) so id is empty -> use title
    expect(feedRowTitleAttr(makeFeed({ feed_id: '   ', display_title: 'My Show' }))).toBe(
      'My Show',
    )
  })
  it('yields empty title/id segment when feed_id trims empty and title absent', () => {
    // Both id and title empty -> the final `|| ""` fallback, only RSS survives
    expect(feedRowTitleAttr(makeFeed({ feed_id: '   ', display_title: null, rss_url: 'r' }))).toBe(
      'RSS: r',
    )
  })
  it('trims title and id in the combined form', () => {
    expect(
      feedRowTitleAttr(makeFeed({ feed_id: '  feed-x  ', display_title: '  My Show  ' })),
    ).toBe('My Show · feed-x')
  })
  it('appends RSS line when present', () => {
    expect(
      feedRowTitleAttr(
        makeFeed({ feed_id: 'feed-x', display_title: 'My Show', rss_url: '  http://x/rss  ' }),
      ),
    ).toBe('My Show · feed-x\nRSS: http://x/rss')
  })
  it('appends description line when present', () => {
    expect(
      feedRowTitleAttr(
        makeFeed({ feed_id: 'feed-x', display_title: 'My Show', description: '  Desc  ' }),
      ),
    ).toBe('My Show · feed-x\nDesc')
  })
  it('appends all parts in order: title/id, RSS, description', () => {
    expect(
      feedRowTitleAttr(
        makeFeed({
          feed_id: 'feed-x',
          display_title: 'My Show',
          rss_url: 'http://x/rss',
          description: 'Desc',
        }),
      ),
    ).toBe('My Show · feed-x\nRSS: http://x/rss\nDesc')
  })
  it('ignores whitespace-only rss_url and description', () => {
    expect(
      feedRowTitleAttr(
        makeFeed({ feed_id: 'feed-x', display_title: 'My Show', rss_url: '   ', description: '  ' }),
      ),
    ).toBe('My Show · feed-x')
  })
  it('ignores null/undefined rss_url and description', () => {
    expect(
      feedRowTitleAttr(
        makeFeed({ feed_id: 'feed-x', display_title: 'My Show', rss_url: null, description: null }),
      ),
    ).toBe('My Show · feed-x')
  })
  it('combines empty feed_id placeholder with RSS and description', () => {
    expect(
      feedRowTitleAttr(makeFeed({ feed_id: '', rss_url: 'http://x/rss', description: 'Desc' })),
    ).toBe('(No feed id)\nRSS: http://x/rss\nDesc')
  })
})

describe('feedRowAccessibleName', () => {
  it('returns placeholder with episode count when feed_id is empty', () => {
    expect(feedRowAccessibleName(makeFeed({ feed_id: '', episode_count: 3 }))).toBe(
      '(No feed id), 3 episodes',
    )
  })
  it('includes title, feed id and count when title present', () => {
    expect(
      feedRowAccessibleName(
        makeFeed({ feed_id: 'feed-x', display_title: 'My Show', episode_count: 5 }),
      ),
    ).toBe('My Show, feed id feed-x, 5 episodes')
  })
  it('falls back to id and count when title is null', () => {
    expect(
      feedRowAccessibleName(makeFeed({ feed_id: 'feed-x', display_title: null, episode_count: 2 })),
    ).toBe('feed-x, 2 episodes')
  })
  it('falls back to id and count when title is whitespace-only', () => {
    expect(
      feedRowAccessibleName(makeFeed({ feed_id: 'feed-x', display_title: '  ', episode_count: 1 })),
    ).toBe('feed-x, 1 episodes')
  })
  it('trims title and id', () => {
    expect(
      feedRowAccessibleName(
        makeFeed({ feed_id: '  feed-x  ', display_title: '  My Show  ', episode_count: 0 }),
      ),
    ).toBe('My Show, feed id feed-x, 0 episodes')
  })
})

describe('filterFeedsByQuery', () => {
  const feeds: CorpusFeedItem[] = [
    makeFeed({ feed_id: 'tech-talk', display_title: 'Tech Talk' }),
    makeFeed({ feed_id: 'history-pod', display_title: 'History Podcast' }),
    makeFeed({ feed_id: 'no-title', display_title: null }),
  ]

  it('returns a copy of all feeds when query is empty', () => {
    const result = filterFeedsByQuery(feeds, '')
    expect(result).toEqual(feeds)
    expect(result).not.toBe(feeds)
  })
  it('returns a copy of all feeds when query is whitespace-only', () => {
    const result = filterFeedsByQuery(feeds, '   ')
    expect(result).toHaveLength(feeds.length)
    expect(result).not.toBe(feeds)
  })
  it('matches on display title case-insensitively', () => {
    expect(filterFeedsByQuery(feeds, 'TECH').map((f) => f.feed_id)).toEqual(['tech-talk'])
  })
  it('matches on feed_id case-insensitively', () => {
    expect(filterFeedsByQuery(feeds, 'HISTORY').map((f) => f.feed_id)).toEqual(['history-pod'])
  })
  it('matches a feed whose only label is its id', () => {
    expect(filterFeedsByQuery(feeds, 'no-title').map((f) => f.feed_id)).toEqual(['no-title'])
  })
  it('trims the query before matching', () => {
    expect(filterFeedsByQuery(feeds, '  tech  ').map((f) => f.feed_id)).toEqual(['tech-talk'])
  })
  it('returns an empty array when nothing matches', () => {
    expect(filterFeedsByQuery(feeds, 'zzz')).toEqual([])
  })
  it('returns multiple matches by partial substring', () => {
    // "pod" matches "History Podcast" (label) and "history-pod" id is the same row
    const result = filterFeedsByQuery(feeds, 'pod')
    expect(result.map((f) => f.feed_id)).toEqual(['history-pod'])
  })
  it('matches the (No feed id) visible label for empty feed_id rows', () => {
    const withEmpty = [...feeds, makeFeed({ feed_id: '', display_title: null })]
    expect(filterFeedsByQuery(withEmpty, 'no feed id').map((f) => f.feed_id)).toEqual([''])
  })
  it('handles an empty feed list', () => {
    expect(filterFeedsByQuery([], 'anything')).toEqual([])
  })
})
