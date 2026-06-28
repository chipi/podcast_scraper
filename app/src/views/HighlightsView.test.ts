import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, Highlight, Note } from '../services/types'
import HighlightsView from './HighlightsView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
})

function hl(over: Partial<Highlight> = {}): Highlight {
  return {
    id: 'h1', episode_slug: 'show-ep01', kind: 'span', start_ms: 65_000, end_ms: 68_000,
    char_start: 0, char_end: 5, segment_ids: ['s1'], quote_text: 'a captured line',
    speaker: 'person:guest', source_insight_id: null, color: null, created_at: 1, anchor_status: null,
    ...over,
  }
}

function detail(slug: string, title: string): EpisodeDetail {
  return {
    slug, title, feed_id: 'f', podcast_title: 'Show', publish_date: null, duration_seconds: null,
    episode_image_url: null, feed_image_url: null, artwork_url: null, summary_title: null,
    summary_bullets: [], summary_text: null, has_transcript: true, has_summary: false,
    has_gi: false, has_kg: false, has_bridge: false,
  }
}

const mountView = () => mount(HighlightsView, { global: { plugins: [i18n, router] } })

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(api, 'getNotes').mockResolvedValue([])
})
afterEach(() => vi.restoreAllMocks())

describe('HighlightsView', () => {
  it('shows the empty state when nothing is captured', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([])
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('No highlights yet')
  })

  it('groups by episode (title hydrated), renders the jump link, and the export link', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([hl()])
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail('show-ep01', 'How Sleep Works'))
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('How Sleep Works') // group heading from hydrated title
    expect(w.text()).toContain('a captured line')
    expect(w.text()).toContain('1:05') // 65_000ms
    const exportLink = w.findAll('a').find((a) => (a.attributes('href') ?? '').includes('export.md'))
    expect(exportLink?.attributes('download')).toBe('my-highlights.md')
  })

  it('flags a drifted anchor', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([hl({ anchor_status: 'drifted' })])
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail('show-ep01', 'Ep'))
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('anchor drifted')
  })

  it('removes a highlight', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([hl()])
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail('show-ep01', 'Ep'))
    const del = vi.spyOn(api, 'deleteHighlight').mockResolvedValue([])
    const w = mountView()
    await flushPromises()
    await w.find('[aria-label="Remove highlight"]').trigger('click')
    expect(del).toHaveBeenCalledWith('h1')
  })

  it('adds a note to a highlight through the inline editor', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([hl()])
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail('show-ep01', 'Ep'))
    const created: Note = {
      id: 'n1', target: 'highlight', target_id: 'h1', text: 'my thought', created_at: 1, updated_at: 1,
    }
    const create = vi.spyOn(api, 'createNote').mockResolvedValue(created)
    const w = mountView()
    await flushPromises()
    // open the add-note editor
    const addBtn = w.findAll('button').find((b) => b.text().includes('Add note'))!
    await addBtn.trigger('click')
    await w.find('textarea').setValue('my thought')
    await w.findAll('button').find((b) => b.text() === 'Save')!.trigger('click')
    await flushPromises()
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ target: 'highlight', target_id: 'h1', text: 'my thought' }),
    )
    expect(w.text()).toContain('my thought')
  })
})
