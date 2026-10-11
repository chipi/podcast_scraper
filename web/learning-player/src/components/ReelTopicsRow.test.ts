/**
 * Discover's "Minutes on" row (ADR-163 §8 amendment): chips into topic reels, in the page scope,
 * hidden with none; a scope switch mid-load never shows the old scope's topics.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'

const api = vi.hoisted(() => ({ getReelTopics: vi.fn() }))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))

import ReelTopicsRow from './ReelTopicsRow.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', component: { template: '<div/>' } },
    { path: '/listen/topic/:id', name: 'topic-reel', component: { template: '<div/>' } },
  ],
})
const topic = (id: string, label: string) => ({ topic_id: id, topic_label: label, clips: 3, total_seconds: 40 })

describe('ReelTopicsRow', () => {
  it('offers each topic as a chip into its fifteen-minute reel', async () => {
    api.getReelTopics.mockResolvedValue([topic('topic:a', 'alpha')])
    const w = mount(ReelTopicsRow, { props: { scope: 'mine' }, global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(api.getReelTopics).toHaveBeenCalledWith('mine')
    const chip = w.get('[data-testid="reel-topic-chip"]')
    expect(chip.text()).toBe('▶ 15 min on alpha')
    expect(chip.attributes('href')).toBe('/listen/topic/topic:a?minutes=15')
  })

  it('is hidden with no topic, and the newest scope wins a race', async () => {
    let releaseOld: (v: unknown) => void = () => {}
    api.getReelTopics.mockImplementationOnce(() => new Promise((r) => (releaseOld = r)))
    api.getReelTopics.mockResolvedValueOnce([])
    const w = mount(ReelTopicsRow, { props: { scope: 'mine' }, global: { plugins: [i18n, router] } })
    await w.setProps({ scope: 'corpus' })
    await flushPromises()
    releaseOld([topic('topic:old', 'old')])
    await flushPromises()
    expect(w.find('[data-testid="reel-topics"]').exists()).toBe(false)
  })
})
