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
const reel = (clips: number) => ({ clips: Array.from({ length: clips }, () => ({})) })

describe('ReelDoor', () => {
  it('opens the subject’s fifteen-minute reel, named for screen readers', async () => {
    api.getTopicReel.mockResolvedValue(reel(3))
    const w = mount(ReelDoor, { props: { subjectId: 'thc:risk', label: 'Managing risk' }, global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(api.getTopicReel).toHaveBeenCalledWith('thc:risk', 15)
    const door = w.get('[data-testid="reel-door"]')
    expect(door.text()).toBe('▶ 15 min on this')
    expect(door.attributes('href')).toBe('/listen/topic/thc:risk?minutes=15')
    expect(door.attributes('aria-label')).toBe('Play 15 minutes of clips on Managing risk')
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
