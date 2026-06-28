import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Highlight, ResurfacingItem } from '../services/types'
import ResurfacingInbox from './ResurfacingInbox.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
})

function hl(over: Partial<Highlight> = {}): Highlight {
  return {
    id: 'h1', episode_slug: 'show-ep01', kind: 'moment', start_ms: 65_000, end_ms: null,
    char_start: null, char_end: null, segment_ids: [], quote_text: null, speaker: null,
    source_insight_id: null, color: null, created_at: 1, anchor_status: null, ...over,
  }
}

const item = (over: Partial<ResurfacingItem> = {}): ResurfacingItem => ({
  highlight: hl(),
  reflection_prompt: 'What still resonates about this?',
  ...over,
})

const mountInbox = () => mount(ResurfacingInbox, { global: { plugins: [i18n, router] } })

beforeEach(() => {
  vi.spyOn(api, 'markSurfaced').mockResolvedValue()
  vi.spyOn(api, 'putResurfacingSettings').mockResolvedValue({ paused: true })
})
afterEach(() => vi.restoreAllMocks())

describe('ResurfacingInbox', () => {
  it('shows the empty state when nothing is due', async () => {
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [], paused: false })
    const w = mountInbox()
    await flushPromises()
    expect(w.text()).toContain('Nothing to revisit right now')
  })

  it('renders due items with the reflection prompt + jump link, and dismisses one', async () => {
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [item()], paused: false })
    const w = mountInbox()
    await flushPromises()
    expect(w.text()).toContain('What still resonates about this?')
    // jump link carries ?t=65 (65_000ms) to the player
    const link = w.findAll('a').find((a) => (a.attributes('href') ?? '').includes('/episode/show-ep01'))
    expect(link?.attributes('href')).toContain('t=65')
    // dismiss removes it locally + advances the ladder server-side
    await w.findAll('button').find((b) => b.text() === 'Got it')!.trigger('click')
    expect(api.markSurfaced).toHaveBeenCalledWith('h1')
    await flushPromises()
    expect(w.text()).not.toContain('What still resonates about this?')
  })

  it('pause toggles the pacing setting and reloads', async () => {
    const get = vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [item()], paused: false })
    const w = mountInbox()
    await flushPromises()
    get.mockResolvedValue({ items: [], paused: true }) // server now reports paused
    await w.findAll('button').find((b) => b.text() === 'Pause')!.trigger('click')
    await flushPromises()
    expect(api.putResurfacingSettings).toHaveBeenCalledWith(true)
    expect(w.text()).toContain('Resurfacing is paused.')
  })
})
