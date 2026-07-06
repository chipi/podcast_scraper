import { describe, expect, it } from 'vitest'
import type { EpisodeDetail } from '../services/types'
import { summaryFromDetail } from './episode'

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
    expect(s.summary_preview).toBe('A prose summary.') // lede falls back to the prose summary
    expect(s.topics).toEqual([])
    expect(s.has_gi).toBe(true)
    expect(s.has_kg).toBe(false)
  })
})
