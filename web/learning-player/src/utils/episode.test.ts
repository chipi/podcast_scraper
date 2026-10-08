import { describe, expect, it } from 'vitest'
import type { EpisodeDetail } from '../services/types'
import { episodeArtwork, episodePlayerArtwork, summaryFromDetail } from './episode'

const detail: EpisodeDetail = {
  slug: 'ep-1',
  title: 'Title',
  feed_id: 'feed-x',
  podcast_title: 'Show',
  publish_date: '2026-05-01',
  duration_seconds: 1800,
  episode_image_url: 'e.jpg',
  feed_image_url: 'f.jpg',
  artwork_url: 'a.jpg',
  summary_title: 'Headline',
  summary_bullets: ['one', 'two'],
  summary_text: 'A prose summary.',
  has_transcript: true,
  has_summary: true,
  has_gi: true,
  has_kg: false,
  has_bridge: true,
}

describe('summaryFromDetail', () => {
  it('adapts a detail into the card summary shape', () => {
    const s = summaryFromDetail(detail)
    expect(s.slug).toBe('ep-1')
    expect(s.status).toBe('ready')
    expect(s.summary_text).toBe('A prose summary.')
    // `summary_title`, matching the server (#2004 item 4). It used to be the first sentence of the
    // prose — a different shape in the same slot, which is the inconsistency that fix removed.
    expect(s.summary_preview).toBe('Headline') // lede falls back to the prose summary
    expect(s.topics).toEqual([])
    expect(s.has_gi).toBe(true)
    expect(s.has_kg).toBe(false)
  })
})

describe('artwork size by surface (Pixel 8 blank-render fix, 2026-10-08)', () => {
  const sized = { ...detail, artwork_url: '/api/app/artwork?ref=x&size=medium', artwork_thumb_url: '/api/app/artwork?ref=x&size=thumb' }

  it('a card built from a detail gets the thumb, never the player-size image', () => {
    expect(episodeArtwork(sized)).toContain('size=thumb')
    expect(summaryFromDetail(sized).artwork_url).toContain('size=thumb')
  })

  it('the player, lock screen and offline copy get the player size', () => {
    expect(episodePlayerArtwork(sized)).toContain('size=medium')
  })

  it('a detail without a thumb (offline record, older server) keeps its own artwork', () => {
    expect(episodeArtwork(detail)).toContain('a.jpg')
    expect(summaryFromDetail(detail).artwork_url).toBe('a.jpg')
  })
})
