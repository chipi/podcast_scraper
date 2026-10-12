/**
 * A board's Brief (RFC-119 amendment 2026-10-12, #2340): what the board holds, what its episodes
 * share, each key point opened where it is said, and nothing at all for a board with no episodes.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import type { BoardBrief as Brief } from '../services/types'

const api = vi.hoisted(() => ({ getBoardBrief: vi.fn(), getTopicReel: vi.fn() }))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))

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

const episode = (slug: string, title: string) => ({ slug, title }) as Brief['points'][number]['episode']
function brief(over: Partial<Brief> = {}): Brief {
  return {
    board_id: 'col_a',
    name: 'Deep dive',
    holds: { episodes: 3, moments: 1, shows: 1, subjects: 0, searches: 0, links: 2 },
    about: [
      { id: 'topic:risk', kind: 'topic', label: 'risk management', episodes: 3, saved: true },
      { id: 'person:noah', kind: 'person', label: 'Noah Brier', episodes: 2, saved: false },
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
    threads: [{ id: 'topic:anchor', kind: 'storyline', label: 'Managing risk', episodes: 2 }],
    ...over,
  }
}

function mountBrief(version = 3) {
  return mount(BoardBrief, {
    props: { boardId: 'col_a', boardName: 'Deep dive', version },
    global: { plugins: [i18n, router] },
  })
}

describe('BoardBrief', () => {
  beforeEach(() => {
    api.getBoardBrief.mockReset()
    api.getTopicReel.mockResolvedValue({ clips: [{}, {}, {}], total_seconds: 300 })
  })

  it('says what the board holds and what its episodes share, each opening where it leads', async () => {
    api.getBoardBrief.mockResolvedValue(brief())
    const w = mountBrief()
    await flushPromises()
    expect(api.getBoardBrief).toHaveBeenCalledWith('col_a')
    expect(w.get('[data-testid="board-brief-holds"]').text()).toBe(
      '3 episodes · 1 saved moment · 1 show · 2 links',
    )
    const subjects = w.findAll('[data-testid="board-brief-subject"]')
    expect(subjects.map((s) => s.attributes('href'))).toEqual(['/topic/topic:risk', '/person/person:noah'])
    expect(subjects[0].text()).toContain('in 3 episodes')
    expect(subjects[0].attributes('title')).toBe('saved on this board')
    const points = w.findAll('[data-testid="board-brief-point"]')
    // A point with a time opens its episode there; one without opens the episode.
    expect(points.map((p) => p.attributes('href'))).toEqual(['/episode/ep-a?t=96', '/episode/ep-b'])
    expect(points[0].text()).toContain('Noah Brier · The Mechanics of a Quiet, Fast Bike')
    expect(w.get('[data-testid="board-brief-thread"]').attributes('href')).toBe('/storyline/topic:anchor')
    // The board's own short listen, through the same door as a subject's.
    expect(w.get('[data-testid="reel-door"]').attributes('href')).toBe('/listen/topic/board:col_a?minutes=15')
  })

  it('shows nothing for a board with no episodes: shows, searches and links only', async () => {
    api.getBoardBrief.mockResolvedValue(
      brief({
        holds: { episodes: 0, moments: 0, shows: 2, subjects: 0, searches: 1, links: 1 },
        about: [],
        points: [],
        threads: [],
      }),
    )
    const w = mountBrief()
    await flushPromises()
    expect(w.find('[data-testid="board-brief"]').exists()).toBe(false)
  })

  it('follows the board when an item is added or removed', async () => {
    api.getBoardBrief.mockResolvedValue(brief())
    const w = mountBrief(3)
    await flushPromises()
    await w.setProps({ version: 4 })
    await flushPromises()
    expect(api.getBoardBrief).toHaveBeenCalledTimes(2)
  })

  it('says so when the Brief cannot be loaded', async () => {
    api.getBoardBrief.mockRejectedValue(new Error('503'))
    const w = mountBrief()
    await flushPromises()
    expect(w.find('[data-testid="board-brief"]').exists()).toBe(false)
    expect(w.get('[data-testid="board-brief-failed"]').text()).toBe('The Brief could not be loaded.')
  })
})
