/**
 * "▶ 15 min on this" on a topic, theme or storyline (operator 2026-10-11): a door into its reel that
 * appears only when the reel has at least two clips, so it never opens an empty reel.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'

const api = vi.hoisted(() => ({ getTopicReel: vi.fn() }))
const tracked = vi.hoisted(() => [] as Array<{ name: string; props: unknown }>)
vi.mock('../services/analytics', async (orig) => ({
  ...(await orig<typeof import('../services/analytics')>()),
  track: (name: string, props: unknown) => tracked.push({ name, props }),
}))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))

import ReelDoor from './ReelDoor.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', component: { template: '<div/>' } },
    { path: '/listen/topic/:id', name: 'topic-reel', component: { template: '<div/>' } },
  ],
})
const reel = (clips: number, seconds = 144) => ({ clips: Array.from({ length: clips }, () => ({})), total_seconds: seconds })

describe('ReelDoor', () => {
  it('opens the subject’s reel, named by what it actually runs', async () => {
    api.getTopicReel.mockResolvedValue(reel(3))
    const w = mount(ReelDoor, { props: { subjectId: 'thc:risk', label: 'Managing risk' }, global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(api.getTopicReel).toHaveBeenCalledWith('thc:risk', 15)
    const door = w.get('[data-testid="reel-door"]')
    expect(door.text()).toBe('▶ 2 min on this') // 2:24
    expect(door.attributes('href')).toBe('/listen/topic/thc:risk?minutes=15')
    expect(door.attributes('aria-label')).toBe('Play 2 minutes of clips on Managing risk')
    await door.trigger('click')
    expect(tracked.at(-1)).toEqual({ name: 'reel_open', props: { source: 'subject_page', kind: 'storyline' } })
  })

  it('opens a board’s own short listen from the board, and says so (#2340)', async () => {
    api.getTopicReel.mockResolvedValue(reel(4, 600))
    const w = mount(ReelDoor, {
      props: { subjectId: 'board:col_abc', label: 'Deep dive', source: 'board' },
      global: { plugins: [i18n, router] },
    })
    await flushPromises()
    expect(api.getTopicReel).toHaveBeenCalledWith('board:col_abc', 15)
    const door = w.get('[data-testid="reel-door"]')
    expect(door.attributes('href')).toBe('/listen/topic/board:col_abc?minutes=15')
    await door.trigger('click')
    expect(tracked.at(-1)).toEqual({ name: 'reel_open', props: { source: 'board', kind: 'board' } })
  })

  it('has no door for a reel of fewer than two clips, or one that fails to load', async () => {
    api.getTopicReel.mockResolvedValueOnce(reel(1))
    const one = mount(ReelDoor, { props: { subjectId: 'topic:a', label: 'a' }, global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(one.find('[data-testid="reel-door"]').exists()).toBe(false)
    api.getTopicReel.mockRejectedValueOnce(new Error('503'))
    const failed = mount(ReelDoor, { props: { subjectId: 'topic:b', label: 'b' }, global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(failed.find('[data-testid="reel-door"]').exists()).toBe(false)
  })
})
