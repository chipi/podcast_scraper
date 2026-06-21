import { describe, expect, it } from 'vitest'
import { buildFeedEntry, feedEntryHasOverrides, splitFeedEntry } from './feedOverrides'

describe('splitFeedEntry', () => {
  it('treats a bare string as a URL with no overrides', () => {
    expect(splitFeedEntry('https://a.example/rss')).toEqual({
      urlKey: 'url',
      url: 'https://a.example/rss',
      must: {},
      advanced: {},
      extras: {},
    })
  })

  it('extracts must-fields, known advanced fields, and preserves unknown keys', () => {
    const s = splitFeedEntry({
      url: 'https://a.example/rss',
      max_episodes: 5,
      episode_order: 'oldest',
      episode_offset: 2,
      episode_since: '2024-01-01',
      circuit_breaker_enabled: true,
      delay_ms: 250,
      some_future_key: 'keep me',
    })
    expect(s.url).toBe('https://a.example/rss')
    expect(s.must).toEqual({
      max_episodes: 5,
      episode_order: 'oldest',
      episode_offset: 2,
      episode_since: '2024-01-01',
    })
    expect(s.advanced).toEqual({ circuit_breaker_enabled: true, delay_ms: 250 })
    expect(s.extras).toEqual({ some_future_key: 'keep me' })
  })

  it('remembers the rss key variant', () => {
    const s = splitFeedEntry({ rss: 'https://b.example/feed' })
    expect(s.urlKey).toBe('rss')
    expect(s.url).toBe('https://b.example/feed')
  })

  it('ignores invalid episode_order values', () => {
    const s = splitFeedEntry({ url: 'https://a.example', episode_order: 'sideways' })
    expect(s.must.episode_order).toBeUndefined()
  })
})

describe('buildFeedEntry', () => {
  it('collapses to a plain string when no overrides or extras remain', () => {
    expect(buildFeedEntry('url', 'https://a.example/rss', {}, {})).toBe('https://a.example/rss')
  })

  it('omits empty must-fields (never persists null/blank)', () => {
    const out = buildFeedEntry(
      'url',
      'https://a.example/rss',
      { max_episodes: 3, episode_order: undefined, episode_since: '' as unknown as string },
      {},
    )
    expect(out).toEqual({ url: 'https://a.example/rss', max_episodes: 3 })
  })

  it('keeps the rss key and merges extras', () => {
    const out = buildFeedEntry(
      'rss',
      'https://b.example/feed',
      { episode_order: 'newest' },
      { delay_ms: 250 },
    )
    expect(out).toEqual({ rss: 'https://b.example/feed', delay_ms: 250, episode_order: 'newest' })
  })

  it('round-trips split → build (advanced + unknown merged)', () => {
    const entry = { url: 'https://a.example', max_episodes: 7, user_agent: 'x', future_key: 1 }
    const s = splitFeedEntry(entry)
    expect(buildFeedEntry(s.urlKey, s.url, s.must, { ...s.extras, ...s.advanced })).toEqual(entry)
  })
})

describe('feedEntryHasOverrides', () => {
  it('is false for a bare string and url-only object', () => {
    expect(feedEntryHasOverrides('https://a.example')).toBe(false)
    expect(feedEntryHasOverrides({ url: 'https://a.example' })).toBe(false)
  })

  it('is true when any non-URL key is present', () => {
    expect(feedEntryHasOverrides({ url: 'https://a.example', max_episodes: 2 })).toBe(true)
  })
})
