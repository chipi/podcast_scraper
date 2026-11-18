// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import FeedOverrideEditor from './FeedOverrideEditor.vue'
import { buildFeedEntry, splitFeedEntry } from '../../utils/feedOverrides'

const PODCAST = { name: 'podcast_episode', default_transport: 'rss_enclosure', source_kind: 'podcast' }
const NEWS = { name: 'news_article', default_transport: 'rss_atom', source_kind: 'news' }

describe('feed entries name a content type and transport (UXS-019)', () => {
  it('round-trips a news source and keeps a podcast a bare URL', () => {
    const news = { url: 'https://www.dailyexample.test/feed.xml', content_type: 'news_article', transport: 'rss_atom' }
    const s = splitFeedEntry(news)
    expect(s.must).toMatchObject({ content_type: 'news_article', transport: 'rss_atom' })
    expect(s.extras).toEqual({})
    expect(buildFeedEntry(s.urlKey, s.url, s.must, s.extras)).toEqual(news)
    const pod = splitFeedEntry('https://example.com/podcast.xml')
    expect(buildFeedEntry(pod.urlKey, pod.url, pod.must, pod.extras)).toBe('https://example.com/podcast.xml')
  })
})

describe('FeedOverrideEditor content type (UXS-019)', () => {
  it('hides the fields when only podcasts are registered', () => {
    const w = mount(FeedOverrideEditor, {
      props: { entry: 'https://example.com/podcast.xml', contentTypeOptions: [PODCAST], transports: ['rss_enclosure'] },
    })
    expect(w.find('[data-testid="feed-override-content-type"]').exists()).toBe(false)
  })

  it('offers registered types and saves a news source with its transport', async () => {
    const w = mount(FeedOverrideEditor, {
      props: {
        entry: 'https://www.dailyexample.test/feed.xml',
        contentTypeOptions: [PODCAST, NEWS],
        transports: ['rss_atom', 'rss_enclosure'],
      },
    })
    const options = w.findAll('[data-testid="feed-override-content-type"] option').map((o) => o.text())
    expect(options).toEqual(['Podcast (default)', 'News'])
    await w.get('[data-testid="feed-override-content-type"]').setValue('news_article')
    expect(w.get('[data-testid="feed-override-transport"] option').text()).toBe('Default (rss_atom)')
    await w.get('[data-testid="feed-override-transport"]').setValue('rss_atom')
    const save = w.findAll('button').find((b) => /save/i.test(b.text()))
    expect(save).toBeTruthy()
    await save!.trigger('click')
    expect(w.emitted('save')?.[0]?.[0]).toEqual({
      url: 'https://www.dailyexample.test/feed.xml',
      content_type: 'news_article',
      transport: 'rss_atom',
    })
  })
})
