import { describe, expect, it } from 'vitest'

import { graphNodeFill, graphNodeLegendLabel, graphNodeTypeStyles } from './colors'
import { visualGroupForNode } from './visualGroup'

describe('visualGroupForNode: news nodes (UXS-019)', () => {
  it('draws a news item as an Article and its source as a NewsSource', () => {
    expect(
      visualGroupForNode({
        id: 'item:news_article:abc',
        type: 'Episode',
        properties: { content_type: 'news_article' },
      }),
    ).toBe('Article')
    expect(
      visualGroupForNode({
        id: 'source:abc',
        type: 'Podcast',
        properties: { source_kind: 'news' },
      }),
    ).toBe('NewsSource')
  })

  it('leaves podcast episodes and shows as they were', () => {
    expect(visualGroupForNode({ id: 'episode:e1', type: 'Episode', properties: {} })).toBe('Episode')
    expect(
      visualGroupForNode({ id: 'podcast:p1', type: 'Podcast', properties: { source_kind: 'podcast' } }),
    ).toBe('Podcast')
  })

  it('gives both a style and a legend label', () => {
    expect(graphNodeFill('Article')).toBe(graphNodeTypeStyles.Article.background)
    expect(graphNodeFill('NewsSource')).toBe(graphNodeTypeStyles.NewsSource.background)
    expect(graphNodeLegendLabel('NewsSource')).toBe('News source')
  })
})
