import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { TrendingEntity } from '../services/types'
import InterestSections from './InterestSections.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function trend(entity_id: string, label: string, anchor_topic_id: string | null = null): TrendingEntity {
  return {
    entity_id,
    kind: '',
    label,
    velocity: 1,
    volume: 1,
    heating_up: false,
    total: 1,
    series: [],
    anchor_topic_id,
  }
}

const mountSections = (selected: string[], openable = false) =>
  mount(InterestSections, { props: { selected, openable }, global: { plugins: [i18n] } })

const section = (w: ReturnType<typeof mountSections>, kind: string) =>
  w.get(`[data-testid="interests-section-${kind}"]`)

async function openAdd(w: ReturnType<typeof mountSections>, kind: string) {
  await w.get(`[data-testid="interest-add-${kind}"]`).trigger('click')
  await flushPromises()
}

beforeEach(() => {
  vi.spyOn(api, 'getTrending').mockImplementation(async (kind: string) => {
    if (kind === 'topic') return [trend('topic:sleep', 'Sleep'), trend('topic:ai', 'AI')]
    if (kind === 'person') return [trend('person:jane', 'Jane Doe')]
    if (kind === 'storyline') return [trend('thc:fleet', 'Shadow fleet', 'topic:sanctions')]
    return []
  })
  vi.spyOn(api, 'getTopClusters').mockResolvedValue([{ id: 'tc:health', label: 'Health', size: 3 }])
  vi.spyOn(api, 'getStorylines').mockResolvedValue([])
})
afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('InterestSections', () => {
  it('has a section per kind, in the order Topics, People, Themes, Storylines', async () => {
    const w = mountSections([])
    await flushPromises()
    expect(w.findAll('h2').map((h) => h.text())).toEqual(['Topics', 'People', 'Themes', 'Storylines'])
  })

  it('suggests what is trending, minus what is already followed, and follows on tap', async () => {
    const w = mountSections(['topic:sleep'])
    await flushPromises()
    await openAdd(w, 'topic')
    const topics = section(w, 'topic')
    const offered = topics.findAll('[data-testid="interest-suggestion"]')
    expect(offered.map((b) => b.text())).toEqual(['+ AI'])
    expect(offered[0].attributes('aria-label')).toBe('Follow AI')
    await offered[0].trigger('click')
    expect(w.emitted('toggle')).toEqual([['topic:ai']])
  })

  it('falls back to the top themes when nothing is trending', async () => {
    const w = mountSections([])
    await flushPromises()
    await openAdd(w, 'theme')
    const themes = section(w, 'theme').findAll('[data-testid="interest-suggestion"]')
    expect(themes.map((b) => b.text())).toEqual(['+ Health'])
  })

  it('× on a followed chip asks to stop following it', async () => {
    const w = mountSections(['person:jane'])
    await flushPromises()
    const remove = section(w, 'person').get('[data-testid="interest-remove"]')
    expect(remove.attributes('aria-label')).toBe('Stop following Jane Doe')
    await remove.trigger('click')
    expect(w.emitted('toggle')).toEqual([['person:jane']])
  })

  it('each kind wears its own hue, followed or suggested alike — and never the accent', async () => {
    // The colours teach the kind: each kind its own hue token, never the accent (2026-10-04).
    vi.mocked(api.getTrending).mockImplementation(async (kind: string) =>
      ({
        topic: [trend('topic:ai', 'AI')],
        person: [trend('person:jane', 'Jane Doe')],
        theme: [trend('tc:health', 'Health')],
        storyline: [trend('thc:fleet', 'Shadow fleet', 'topic:sanctions')],
      })[kind] ?? []
    )
    const w = mountSections(['topic:sleep', 'person:ada', 'tc:mind', 'thc:tides'])
    await flushPromises()
    const pill = { topic: 'text-topic', person: 'text-person', theme: 'text-theme', storyline: 'text-storyline' }
    for (const [kind, cls] of Object.entries(pill)) {
      await openAdd(w, kind)
      expect(section(w, kind).get(`[data-testid="interest-following-${kind}"]`).classes(), kind).toContain(cls)
      expect(section(w, kind).get('[data-testid="interest-suggestion"]').classes(), kind).toContain(cls)
      expect(section(w, kind).get(`[data-testid="interest-following-${kind}"]`).classes().join(' '), kind).not.toMatch(/accent/)
    }
  })

  it('says so when a section follows nothing', async () => {
    const w = mountSections([])
    await flushPromises()
    expect(section(w, 'person').get('[data-testid="interest-none"]').text()).toBe(
      "You're not following anyone yet."
    )
  })

  it('a topic and its same-named theme each stay in their own section', async () => {
    // De-duplicating by label across the whole set would drop one of these: they read the same and
    // are different kinds.
    const w = mountSections(['topic:health', 'tc:health'])
    await flushPromises()
    expect(section(w, 'topic').findAll('[data-testid="interest-following-topic"]')).toHaveLength(1)
    expect(section(w, 'theme').findAll('[data-testid="interest-following-theme"]')).toHaveLength(1)
  })

  describe('+ Add — one search on screen at a time', () => {
    it('a closed section is its heading and pills: no search box, no suggestions', async () => {
      const w = mountSections(['topic:sleep'])
      await flushPromises()
      for (const kind of ['topic', 'person', 'theme', 'storyline']) {
        expect(section(w, kind).find('input').exists(), kind).toBe(false)
        expect(section(w, kind).find('[data-testid="interest-suggestion"]').exists(), kind).toBe(false)
        expect(section(w, kind).find(`[data-testid="interest-add-${kind}"]`).exists(), kind).toBe(true)
      }
    })

    it('Add sits in the followed row, and in the empty one too', async () => {
      const w = mountSections(['topic:sleep'])
      await flushPromises()
      const topicRow = section(w, 'topic').get('[data-testid="interest-following-topic"]').element.parentElement!
      expect(topicRow.querySelector('[data-testid="interest-add-topic"]')).not.toBeNull()
      const personRow = section(w, 'person').get('[data-testid="interest-none"]').element.parentElement!
      expect(personRow.querySelector('[data-testid="interest-add-person"]')).not.toBeNull()
    })

    it('opening another section closes the first and drops its query', async () => {
      vi.useFakeTimers()
      vi.spyOn(api, 'searchInterests').mockResolvedValue([])
      const w = mountSections([])
      await flushPromises()
      await openAdd(w, 'topic')
      await section(w, 'topic').get('[data-testid="interest-search-topic"]').setValue('sle')
      await openAdd(w, 'person')
      expect(w.findAll('input')).toHaveLength(1)
      expect(section(w, 'person').find('input').exists()).toBe(true)
      await openAdd(w, 'topic')
      expect((section(w, 'topic').get('input').element as HTMLInputElement).value).toBe('')
    })

    it('Done closes it', async () => {
      const w = mountSections([])
      await flushPromises()
      await openAdd(w, 'theme')
      await w.get('[data-testid="interest-add-done-theme"]').trigger('click')
      expect(w.find('input').exists()).toBe(false)
      expect(w.find('[data-testid="interest-add-theme"]').exists()).toBe(true)
    })
  })

  describe('search', () => {
    beforeEach(() => vi.useFakeTimers())

    async function type(w: ReturnType<typeof mountSections>, kind: string, q: string) {
      if (!section(w, kind).find(`[data-testid="interest-search-${kind}"]`).exists()) {
        await w.get(`[data-testid="interest-add-${kind}"]`).trigger('click')
        await flushPromises()
      }
      await section(w, kind).get(`[data-testid="interest-search-${kind}"]`).setValue(q)
      await vi.advanceTimersByTimeAsync(300)
      await flushPromises()
    }

    it('searches every item of the kind, not just the suggestions, and follows a hit', async () => {
      const search = vi
        .spyOn(api, 'searchInterests')
        .mockResolvedValue([{ id: 'person:ada', kind: 'person', label: 'Ada Lovelace' }])
      const w = mountSections([])
      await flushPromises()
      await type(w, 'person', 'lov')
      expect(search).toHaveBeenCalledWith('person', 'lov')
      const hit = section(w, 'person').get('[data-testid="interest-result"]')
      expect(hit.text()).toBe('+ Ada Lovelace')
      expect(hit.attributes('aria-pressed')).toBe('false')
      // Results replace the suggestions while there is a query.
      expect(section(w, 'person').find('[data-testid="interest-suggestion"]').exists()).toBe(false)
      await hit.trigger('click')
      expect(w.emitted('toggle')).toEqual([['person:ada']])
    })

    it('marks a hit that is already followed, and keeps its label once the box is cleared', async () => {
      vi.spyOn(api, 'searchInterests').mockResolvedValue([
        { id: 'person:ada', kind: 'person', label: 'Ada Lovelace' },
      ])
      const w = mountSections(['person:ada'])
      await flushPromises()
      await type(w, 'person', 'ada')
      const hit = section(w, 'person').get('[data-testid="interest-result"]')
      expect(hit.attributes('aria-pressed')).toBe('true')
      expect(hit.attributes('aria-label')).toBe('Stop following Ada Lovelace')
      await type(w, 'person', '')
      // Not the de-slugged "ada": the label the search taught it is kept.
      expect(section(w, 'person').get('[data-testid="interest-following-person"]').text()).toBe(
        'Ada Lovelace'
      )
    })

    it('does not search on a single character', async () => {
      const search = vi.spyOn(api, 'searchInterests').mockResolvedValue([])
      const w = mountSections([])
      await flushPromises()
      await type(w, 'topic', 'a')
      expect(search).not.toHaveBeenCalled()
    })

    it('an answer for an older query never overwrites a newer one', async () => {
      let releaseOld: (v: unknown) => void = () => {}
      vi.spyOn(api, 'searchInterests').mockImplementation((_kind, q) =>
        q === 'sl'
          ? new Promise((r) => (releaseOld = () => r([{ id: 'topic:slow', kind: 'topic', label: 'Slow' }])))
          : Promise.resolve([{ id: 'topic:sleep', kind: 'topic', label: 'Sleep' }])
      )
      const w = mountSections([])
      await flushPromises()
      await type(w, 'topic', 'sl')
      await type(w, 'topic', 'sle')
      releaseOld(undefined)
      await flushPromises()
      const hits = section(w, 'topic').findAll('[data-testid="interest-result"]')
      expect(hits.map((h) => h.text())).toEqual(['+ Sleep'])
    })

    it('says when nothing matches, and when search itself failed', async () => {
      const search = vi.spyOn(api, 'searchInterests').mockResolvedValue([])
      const w = mountSections([])
      await flushPromises()
      await type(w, 'theme', 'zzz')
      expect(section(w, 'theme').get('[data-testid="interest-no-match"]').text()).toBe(
        'Nothing matches “zzz”.'
      )
      search.mockRejectedValue(new Error('offline'))
      await type(w, 'theme', 'zzzz')
      expect(section(w, 'theme').find('[data-testid="interest-search-failed"]').exists()).toBe(true)
    })
  })

  describe('opening a followed item', () => {
    it('a storyline opens on its anchor topic', async () => {
      const w = mountSections(['thc:fleet'], true)
      await flushPromises()
      await section(w, 'storyline').get('[data-testid="interest-open"]').trigger('click')
      expect(w.emitted('open')).toEqual([[{ kind: 'storyline', id: 'topic:sanctions' }]])
    })

    it('a theme has nowhere to open yet, so it is text, not a button', async () => {
      const w = mountSections(['tc:health'], true)
      await flushPromises()
      expect(section(w, 'theme').find('[data-testid="interest-open"]').exists()).toBe(false)
    })

    it('nothing opens when the parent did not ask for it (the onboarding sheet)', async () => {
      const w = mountSections(['topic:sleep'])
      await flushPromises()
      expect(w.find('[data-testid="interest-open"]').exists()).toBe(false)
    })
  })
})
