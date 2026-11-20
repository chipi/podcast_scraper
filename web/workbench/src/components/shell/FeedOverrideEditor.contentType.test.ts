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

describe('a news source edits its news: block as fields (UXS-019)', () => {
  const NATURE = {
    url: 'https://www.nature.example/nature.rss',
    content_type: 'news_article',
    news: {
      max_items: 15,
      include_url_pattern: '/articles/d41586-',
      text_status: 'teaser',
      serveable: 'summary',
      mineable: true,
      text_from: 'standfirst',
      personal: false,
      translate: false,
    },
  }
  const mountNews = (entry: unknown) =>
    mount(FeedOverrideEditor, {
      props: { entry: entry as never, contentTypeOptions: [PODCAST, NEWS], transports: ['rss_atom'] },
    })
  const save = async (w: ReturnType<typeof mountNews>) => {
    await w.get('[data-testid="feed-override-save"]').trigger('click')
    return w.emitted('save')?.[0]?.[0]
  }

  it('shows each news setting in its own field, not as raw JSON', () => {
    const w = mountNews(NATURE)
    expect(w.find('[data-testid="feed-override-news"]').exists()).toBe(true)
    expect((w.get('[data-testid="feed-override-news-serveable"]').element as HTMLSelectElement).value).toBe('summary')
    expect((w.get('[data-testid="feed-override-news-max_items"]').element as HTMLInputElement).value).toBe('15')
    expect((w.get('[data-testid="feed-override-news-text_from"]').element as HTMLSelectElement).value).toBe('standfirst')
    expect((w.get('[data-testid="feed-override-news-mineable"]').element as HTMLSelectElement).value).toBe('true')
    expect((w.get('[data-testid="feed-override-news-include_url_pattern"]').element as HTMLInputElement).value).toBe(
      '/articles/d41586-',
    )
    expect((w.get('[data-testid="feed-override-extras"]').element as HTMLTextAreaElement).value).toBe('')
  })

  it('hides the podcast-only fields for a news source', () => {
    const w = mountNews(NATURE)
    for (const id of ['max-episodes', 'order', 'offset', 'since', 'until', 'adv-use_publisher_transcripts']) {
      expect(w.find(`[data-testid="feed-override-${id}"]`).exists()).toBe(false)
    }
    expect(w.find('[data-testid="feed-override-profile"]').exists()).toBe(true)
    expect(w.find('[data-testid="feed-override-adv-user_agent"]').exists()).toBe(true)
  })

  it('still shows a podcast-only field a news source sets, so no saved value is hidden', () => {
    const w = mountNews({ ...NATURE, max_episodes: 5 })
    expect((w.get('[data-testid="feed-override-max-episodes"]').element as HTMLInputElement).value).toBe('5')
    expect(w.find('[data-testid="feed-override-order"]').exists()).toBe(false)
  })

  it('saves the entry unchanged when nothing is edited', async () => {
    expect(await save(mountNews(NATURE))).toEqual(NATURE)
  })

  it('saves an edited setting, and leaves a cleared one to the server default', async () => {
    const w = mountNews(NATURE)
    await w.get('[data-testid="feed-override-news-serveable"]').setValue('metadata')
    await w.get('[data-testid="feed-override-news-max_items"]').setValue('30')
    await w.get('[data-testid="feed-override-news-text_status"]').setValue('')
    const out = (await save(w)) as { news: Record<string, unknown> }
    expect(out.news).toMatchObject({ serveable: 'metadata', max_items: 30, text_from: 'standfirst' })
    expect(out.news).not.toHaveProperty('text_status')
  })

  it('keeps a news key the form does not know', async () => {
    const entry = { ...NATURE, news: { ...NATURE.news, some_future_key: 'keep me' } }
    expect(await save(mountNews(entry))).toEqual(entry)
  })

  it('refuses a cap outside 1 to 500 and a bad key variable, and saves nothing', async () => {
    const w = mountNews(NATURE)
    await w.get('[data-testid="feed-override-news-max_items"]').setValue('0')
    expect(await save(w)).toBeUndefined()
    expect(w.get('[data-testid="feed-override-error"]').text()).toContain('1 to 500')
    await w.get('[data-testid="feed-override-news-max_items"]').setValue('15')
    await w.get('[data-testid="feed-override-news-api_key_env"]').setValue('guardian key')
    expect(await save(w)).toBeUndefined()
    expect(w.get('[data-testid="feed-override-error"]').text()).toContain('environment variable')
  })

  it('shows no news fields for a podcast', () => {
    const w = mountNews('https://example.com/podcast.xml')
    expect(w.find('[data-testid="feed-override-news"]').exists()).toBe(false)
    expect(w.find('[data-testid="feed-override-max-episodes"]').exists()).toBe(true)
  })
})
