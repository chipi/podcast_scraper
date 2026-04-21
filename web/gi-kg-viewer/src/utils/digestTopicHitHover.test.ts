import { describe, expect, it } from 'vitest'
import {
  buildTopicHitHoverPanel,
  topicHitFirstRowHasContent,
  topicHitHoverPanelIsEmpty,
} from './digestTopicHitHover'

describe('buildTopicHitHoverPanel', () => {
  it('splits published date, timing extras, score, feed, summary, about, rss', () => {
    const p = buildTopicHitHoverPanel(
      {
        metadata_relative_path: 'm.json',
        episode_title: 'T',
        feed_id: 'f1',
        publish_date: '2024-01-02',
        episode_number: 12,
        duration_seconds: 3661,
        score: 0.8123,
        feed_description: 'A'.repeat(230),
        feed_rss_url: 'https://example.com/feed.xml',
        summary_preview: 'Short recap text.',
      },
      {
        feedDisplayLabel: 'My Feed',
        showFeedLine: true,
        summaryPreview: 'Short recap text.',
      },
    )
    expect(p.publishDateValue).toBe('2024-01-02')
    expect(p.timingExtras).toBe('E12 · 1h 1m')
    expect(p.similarityScore).toBe('0.812')
    expect(p.feedLine).toBe('My Feed')
    expect(p.summaryPreview).toBe('Short recap text.')
    expect(p.aboutFeed?.endsWith('…')).toBe(true)
    expect(p.rssLine).toBe('https://example.com/feed.xml')
    expect(topicHitFirstRowHasContent(p)).toBe(true)
    expect(topicHitHoverPanelIsEmpty(p)).toBe(false)
  })

  it('topicHitHoverPanelIsEmpty when nothing applies', () => {
    const p = buildTopicHitHoverPanel(
      {
        metadata_relative_path: null,
        episode_title: 'T',
        feed_id: '',
        score: null,
      },
      { feedDisplayLabel: 'Unknown feed', showFeedLine: false, summaryPreview: '' },
    )
    expect(topicHitHoverPanelIsEmpty(p)).toBe(true)
  })

  it('first row can be score-only', () => {
    const p = buildTopicHitHoverPanel(
      {
        metadata_relative_path: 'm.json',
        episode_title: 'T',
        feed_id: 'x',
        score: 0.5,
      },
      { feedDisplayLabel: 'F', showFeedLine: false, summaryPreview: '' },
    )
    expect(p.similarityScore).toBe('0.500')
    expect(p.publishDateValue).toBeNull()
    expect(p.timingExtras).toBeNull()
    expect(topicHitFirstRowHasContent(p)).toBe(true)
  })
})
