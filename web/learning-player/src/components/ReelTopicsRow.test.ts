/**
 * Discover's "15 minutes on" row (ADR-163 §8 amendment): chips into topic, theme and storyline reels,
 * in the page scope,
 * hidden with none; a scope switch mid-load never shows the old scope's topics.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'

const api = vi.hoisted(() => ({ getReelTopics: vi.fn() }))
const tracked = vi.hoisted(() => [] as Array<{ name: string; props: unknown }>)
vi.mock('../services/analytics', async (orig) => ({
  ...(await orig<typeof import('../services/analytics')>()),
  track: (name: string, props: unknown) => tracked.push({ name, props }),
}))
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
const topic = (id: string, label: string, kind: 'topic' | 'theme' | 'storyline' = 'topic') => ({
  subject_id: id,
  subject_kind: kind,
  subject_label: label,
  clips: 3,
  total_seconds: 40,
})

describe('ReelTopicsRow', () => {
  it('offers each topic as a chip into its fifteen-minute reel', async () => {
    api.getReelTopics.mockResolvedValue([topic('topic:a', 'alpha')])
    const w = mount(ReelTopicsRow, { props: { scope: 'mine' }, global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(api.getReelTopics).toHaveBeenCalledWith('mine')
    const chip = w.get('[data-testid="reel-topic-chip"]')
    expect(chip.text()).toBe('▶ alpha')
    expect(chip.attributes('aria-label')).toBe('Play 15 minutes on alpha')
    expect(w.get('h2').text()).toBe('15 minutes on')
    expect(chip.attributes('href')).toBe('/listen/topic/topic:a?minutes=15')
  })

  it("colours each chip by its kind, as topics, themes and storylines are on episode notes", async () => {
    api.getReelTopics.mockResolvedValue([
      topic('topic:a', 'alpha'),
      topic('tc:b', 'beta', 'theme'),
      topic('thc:c', 'gamma', 'storyline'),
    ])
    const w = mount(ReelTopicsRow, { props: { scope: 'corpus' }, global: { plugins: [i18n, router] } })
    await flushPromises()
    const chips = w.findAll('[data-testid="reel-topic-chip"]')
    expect(chips.map((c) => c.attributes('data-kind'))).toEqual(['topic', 'theme', 'storyline'])
    expect(chips[0].classes()).toContain('text-topic')
    expect(chips[1].classes()).toContain('text-theme')
    expect(chips[2].classes()).toContain('text-storyline')
    expect(chips[2].attributes('href')).toBe('/listen/topic/thc:c?minutes=15')
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

  it('a chip tap reports reel_open from the Discover row, with the kind — never the subject', async () => {
    tracked.length = 0
    api.getReelTopics.mockResolvedValue([topic('thc:c', 'gamma', 'storyline')])
    const w = mount(ReelTopicsRow, { props: { scope: 'corpus' }, global: { plugins: [i18n, router] } })
    await flushPromises()
    await w.get('[data-testid="reel-topic-chip"]').trigger('click')
    expect(tracked).toEqual([{ name: 'reel_open', props: { source: 'discover_row', kind: 'storyline' } }])
  })
})
