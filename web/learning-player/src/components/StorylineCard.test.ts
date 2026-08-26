import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { TopicCard } from '../services/types'
import StorylineCard from './StorylineCard.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function card(): TopicCard {
  return {
    id: 'topic:sanctions',
    label: 'Sanctions',
    cluster_id: null,
    cluster_label: null,
    cluster_size: 0,
    theme_cluster_id: 'thc:shadow-fleet',
    theme_cluster_label: 'Shadow-fleet economics',
    theme_cluster_size: 3,
    theme_sibling_topics: [
      { id: 'topic:tankers', label: 'Tankers', cluster_id: null, cluster_label: null, cluster_size: 0 },
      { id: 'topic:insurance', label: 'Insurance', cluster_id: null, cluster_label: null, cluster_size: 0 },
    ],
    episode_count: 0,
    episodes: [],
    related_people: [],
  }
}

const mountIt = () =>
  mount(StorylineCard, {
    props: { id: 'thc:shadow-fleet', label: 'Shadow-fleet economics', anchorTopicId: 'topic:sanctions' },
    global: { plugins: [i18n], stubs: { teleport: true } },
  })

afterEach(() => vi.restoreAllMocks())

describe('StorylineCard (#9)', () => {
  it('titles itself with the storyline and lists the anchor + its theme siblings', async () => {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(card())
    const w = mountIt()
    await flushPromises()
    // Header is the storyline, NOT the first member topic — the whole point of #9.
    expect(w.get('h2').text()).toBe('Shadow-fleet economics')
    const rows = w.findAll('[data-testid="storyline-topic-row"]')
    expect(rows).toHaveLength(3) // anchor + 2 siblings
    expect(rows[0].text()).toContain('Sanctions')
    expect(rows[1].text()).toContain('Tankers')
    expect(rows[2].text()).toContain('Insurance')
  })

  it('emits open-topic with the member id when a row is tapped', async () => {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(card())
    const w = mountIt()
    await flushPromises()
    await w.findAll('[data-testid="storyline-topic-row"]')[1].trigger('click')
    expect(w.emitted('open-topic')![0]).toEqual(['topic:tankers'])
  })

  it('de-dupes if the API lists the anchor among its own siblings', async () => {
    const c = card()
    c.theme_sibling_topics = [
      { id: 'topic:sanctions', label: 'Sanctions', cluster_id: null, cluster_label: null, cluster_size: 0 },
      { id: 'topic:tankers', label: 'Tankers', cluster_id: null, cluster_label: null, cluster_size: 0 },
    ]
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(c)
    const w = mountIt()
    await flushPromises()
    expect(w.findAll('[data-testid="storyline-topic-row"]')).toHaveLength(2)
  })

  it('shows an empty message when the anchor card fails to load', async () => {
    vi.spyOn(api, 'getTopicCard').mockRejectedValue(new Error('offline'))
    const w = mountIt()
    await flushPromises()
    expect(w.text()).toContain("Couldn't load the topics")
    expect(w.findAll('[data-testid="storyline-topic-row"]')).toHaveLength(0)
  })
})
