import { describe, expect, it } from 'vitest'
import {
  digestRowFeedLabel,
  digestRowFeedLabelWithCatalog,
  digestRowSummaryPreview,
  digestTopicHitSimilarityDisplay,
  libraryEpisodeSummaryLine,
} from './digestRowDisplay'

describe('digestRowDisplay', () => {
  it('digestTopicHitSimilarityDisplay tiers and raw title', () => {
    expect(digestTopicHitSimilarityDisplay(0.9)).toMatchObject({
      label: 'Strong match',
      labelClass: 'text-gi',
      rawTitle: 'Similarity: 0.900',
    })
    expect(digestTopicHitSimilarityDisplay(0.75).label).toBe('Good match')
    expect(digestTopicHitSimilarityDisplay(0.5).label).toBe('Weak match')
  })

  it('digestRowSummaryPreview prefers summary_preview', () => {
    expect(
      digestRowSummaryPreview({
        summary_preview: '  server line  ',
        summary_title: 'T',
        summary_bullets_preview: ['a'],
      }),
    ).toBe('server line')
  })

  it('digestRowSummaryPreview composes from title and bullets', () => {
    expect(
      digestRowSummaryPreview({
        summary_title: 'Head',
        summary_bullets_preview: ['one', 'two'],
      }),
    ).toBe('Head — one · two')
  })

  it('digestRowSummaryPreview joins all bullets when no summary_preview', () => {
    expect(
      digestRowSummaryPreview({
        summary_title: 'Head',
        summary_bullets_preview: ['a', 'b', 'c'],
      }),
    ).toBe('Head — a · b · c')
    expect(
      digestRowSummaryPreview({
        summary_bullets_preview: ['x', 'y', 'z'],
      }),
    ).toBe('x · y · z')
  })

  it('digestRowFeedLabel prefers display title', () => {
    expect(
      digestRowFeedLabel({ feed_display_title: ' Show ', feed_id: 'fid' }),
    ).toBe('Show')
  })

  it('digestRowFeedLabel falls back to feed_id', () => {
    expect(digestRowFeedLabel({ feed_id: 'x' })).toBe('x')
  })

  it('digestRowFeedLabelWithCatalog prefers feeds API title over raw feed_id', () => {
    expect(
      digestRowFeedLabelWithCatalog(
        { feed_id: 'uuid-here', feed_display_title: null },
        { 'uuid-here': 'Human Podcast Name' },
      ),
    ).toBe('Human Podcast Name')
  })

  it('digestRowFeedLabelWithCatalog falls back when catalog misses', () => {
    expect(
      digestRowFeedLabelWithCatalog({ feed_id: 'only-id', feed_display_title: 'From row' }, {}),
    ).toBe('From row')
  })

  it('libraryEpisodeSummaryLine uses topics when summary_bullets_preview empty', () => {
    expect(
      libraryEpisodeSummaryLine({
        summary_preview: null,
        summary_title: null,
        summary_bullets_preview: [],
        topics: ['Alpha point', 'Beta point'],
      }),
    ).toBe('Alpha point · Beta point')
  })

  it('libraryEpisodeSummaryLine prefers summary_bullets_preview over topics', () => {
    expect(
      libraryEpisodeSummaryLine({
        summary_title: 'Head',
        summary_bullets_preview: ['one'],
        topics: ['ignored'],
      }),
    ).toBe('Head — one')
  })
})
