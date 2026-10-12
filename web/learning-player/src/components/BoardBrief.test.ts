/**
 * A board's Brief, in a sheet (RFC-119 amendment 2026-10-12, #2340; operator 2026-10-12: behind a
 * button, laid out as the episode's Brief): what the board holds, its top voices, its key points
 * opening where they are said, "What this board's about" in rows per kind, and the play button
 * into its own moments.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import type { BoardBrief as Brief } from '../services/types'

const api = vi.hoisted(() => ({ getBoardBrief: vi.fn(), getTopicReel: vi.fn() }))
const tracked = vi.hoisted(() => [] as Array<{ name: string; props: unknown }>)
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))
vi.mock('../services/analytics', async (orig) => ({
  ...(await orig<typeof import('../services/analytics')>()),
  track: (name: string, props: unknown) => tracked.push({ name, props }),
}))

import BoardBrief from './BoardBrief.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', component: stub },
    { path: '/episode/:slug', name: 'player', component: stub },
    { path: '/topic/:id', name: 'topic', component: stub },
    { path: '/person/:id', name: 'person', component: stub },
    { path: '/theme/:id', name: 'theme', component: stub },
    { path: '/storyline/:id', name: 'storyline', component: stub },
    { path: '/listen/topic/:id', name: 'topic-reel', component: stub },
  ],
})

// jsdom has <dialog> without showModal()/close() — the stub ConfirmDialog.test.ts uses.
beforeAll(() => {
  if (!('showModal' in HTMLDialogElement.prototype)) {
    Object.assign(HTMLDialogElement.prototype, {
      showModal(this: HTMLDialogElement) {
        this.open = true
      },
      close(this: HTMLDialogElement) {
        this.open = false
        this.dispatchEvent(new Event('close'))
      },
    })
  }
})

const episode = (slug: string, title: string) => ({ slug, title }) as Brief['points'][number]['episode']
function brief(over: Partial<Brief> = {}): Brief {
  return {
    board_id: 'col_a',
    name: 'Deep dive',
    holds: { episodes: 3, moments: 1, shows: 1, subjects: 0, searches: 0, links: 2 },
    voices: [
      { id: 'person:maya', name: 'Maya', episodes: 3, image_url: null, saved: false },
      { id: 'person:noah', name: 'Noah Brier', episodes: 1, image_url: '/photo/noah', saved: false },
    ],
    about: [
      { id: 'topic:risk', kind: 'topic', label: 'risk management', episodes: 3, saved: true },
      { id: 'topic:soil', kind: 'topic', label: 'soil erosion', episodes: 1, saved: false },
    ],
    points: [
      {
        episode: episode('ep-a', 'The Mechanics of a Quiet, Fast Bike'),
        insight_id: 'i1',
        text: 'Service intervals are a risk decision.',
        quote_text: 'Service intervals are a risk decision.',
        speaker: 'Noah Brier',
        start_ms: 96_400,
      },
      {
        episode: episode('ep-b', 'Enduro Skills'),
        insight_id: 'i2',
        text: 'Pick your crashes.',
        quote_text: null,
        speaker: null,
        start_ms: null,
      },
    ],
    threads: [
      { id: 'topic:anchor', kind: 'storyline', label: 'Managing risk', episodes: 3 },
      { id: 'tc:rel', kind: 'theme', label: 'reliability', episodes: 2 },
    ],
    ...over,
  }
}

async function mountBrief(open = true, version = 3) {
  const w = mount(BoardBrief, {
    props: { open, boardId: 'col_a', boardName: 'Deep dive', version },
    global: { plugins: [i18n, router] },
    attachTo: document.body,
  })
  await flushPromises()
  return w
}

describe('BoardBrief', () => {
  beforeEach(() => {
    api.getBoardBrief.mockReset()
    api.getTopicReel.mockResolvedValue({ clips: [{}, {}, {}], total_seconds: 95 })
    tracked.length = 0
  })

  it('loads nothing until it is opened, then opens as a modal sheet', async () => {
    api.getBoardBrief.mockResolvedValue(brief())
    const w = await mountBrief(false)
    expect(api.getBoardBrief).not.toHaveBeenCalled()
    await w.setProps({ open: true })
    await flushPromises()
    expect(api.getBoardBrief).toHaveBeenCalledWith('col_a')
    expect((w.get('[data-testid="board-brief"]').element as HTMLDialogElement).open).toBe(true)
    w.unmount()
  })

  it('reads as the episode Brief: the board, its voices, its key points, what it is about', async () => {
    api.getBoardBrief.mockResolvedValue(brief())
    const w = await mountBrief()
    expect(w.get('[data-testid="board-brief-dossier"]').text()).toContain('Deep dive')
    expect(w.get('[data-testid="board-brief-holds"]').text()).toBe(
      '3 episodes · 1 saved moment · 1 show · 2 links',
    )
    // Top voices, the topic page's grid, each opening the person.
    expect(w.findAll('[data-testid="ec-top-voice"]').map((v) => v.attributes('href'))).toEqual([
      '/person/person:maya',
      '/person/person:noah',
    ])
    // Key points open their episode where the quote is said.
    expect(w.findAll('[data-testid="board-brief-point"]').map((p) => p.attributes('href'))).toEqual([
      '/episode/ep-a?t=96',
      '/episode/ep-b',
    ])
    // What this board's about: storylines, themes, then topics, every one opening its page.
    expect(w.findAll('[data-testid="board-brief-thread"]').map((x) => x.attributes('href'))).toEqual([
      '/storyline/topic:anchor',
      '/theme/tc:rel',
    ])
    const topics = w.findAll('[data-testid="board-brief-subject"]')
    expect(topics.map((x) => x.text())).toEqual(['risk management', 'soil erosion'])
    expect(topics[0].attributes('title')).toBe('saved on this board')
    w.unmount()
  })

  it('plays the board’s own moments through the short-listen player', async () => {
    api.getBoardBrief.mockResolvedValue(brief())
    const w = await mountBrief()
    expect(api.getTopicReel).toHaveBeenCalledWith('board:col_a', 15)
    const play = w.get('[data-testid="board-brief-play"]')
    expect(play.text()).toContain('Play 3 moments')
    await play.trigger('click')
    await flushPromises()
    expect(router.currentRoute.value.fullPath).toBe('/listen/topic/board:col_a?minutes=15')
    expect(tracked.at(-1)).toEqual({ name: 'reel_open', props: { source: 'board', kind: 'board' } })
    expect(w.emitted('close')).toBeTruthy()
    w.unmount()
  })

  it('says a board with no episodes has nothing to brief yet', async () => {
    api.getBoardBrief.mockResolvedValue(
      brief({
        holds: { episodes: 0, moments: 0, shows: 2, subjects: 0, searches: 1, links: 1 },
        voices: [],
        about: [],
        points: [],
        threads: [],
      }),
    )
    const w = await mountBrief()
    expect(w.find('[data-testid="board-brief-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="board-brief-play"]').exists()).toBe(false)
    w.unmount()
  })

  it('follows the board when an item is added or removed while it is open', async () => {
    api.getBoardBrief.mockResolvedValue(brief())
    const w = await mountBrief(true, 3)
    await w.setProps({ version: 4 })
    await flushPromises()
    expect(api.getBoardBrief).toHaveBeenCalledTimes(2)
    w.unmount()
  })

  it('says so when the Brief cannot be loaded', async () => {
    api.getBoardBrief.mockRejectedValue(new Error('503'))
    const w = await mountBrief()
    expect(w.get('[data-testid="board-brief-failed"]').text()).toBe('The Brief could not be loaded.')
    w.unmount()
  })
})
