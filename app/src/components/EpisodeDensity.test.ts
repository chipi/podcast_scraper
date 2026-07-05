import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeEnrichmentSignals } from '../services/types'
import EpisodeDensity from './EpisodeDensity.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountIt = (slug = 'ep-1') =>
  mount(EpisodeDensity, { props: { slug }, global: { plugins: [i18n] } })

const withDensity = (over: Partial<NonNullable<EpisodeEnrichmentSignals['insight_density']>> = {}) =>
  vi.spyOn(api, 'getEpisodeEnrichment').mockResolvedValue({
    insight_density: { counts: { early: 1, mid: 5, late: 2 }, duration_seconds: 1800, ...over },
  })

afterEach(() => vi.restoreAllMocks())

describe('EpisodeDensity', () => {
  it('renders the three thirds with counts and marks the densest', async () => {
    withDensity()
    const w = mountIt()
    await flushPromises()
    expect(w.get('[data-testid="density-early"]').text()).toContain('1')
    expect(w.get('[data-testid="density-mid"]').text()).toContain('5')
    expect(w.get('[data-testid="density-late"]').text()).toContain('2')
    // Peak caption names the densest third (mid).
    expect(w.get('[data-testid="density-peak"]').text()).toContain('Mid')
  })

  it('seeks to the start of a third (2/3 of duration for late)', async () => {
    withDensity()
    const w = mountIt()
    await flushPromises()
    await w.get('[data-testid="density-late"]').trigger('click')
    // 2/3 of 1800s = 1200s.
    expect(w.emitted('seek')![0]).toEqual([1200])
  })

  it('disables seek when duration is unknown', async () => {
    withDensity({ duration_seconds: 0 })
    const w = mountIt()
    await flushPromises()
    expect(w.get('[data-testid="density-mid"]').attributes('disabled')).toBeDefined()
    await w.get('[data-testid="density-mid"]').trigger('click')
    expect(w.emitted('seek')).toBeUndefined()
  })

  it('renders nothing when the episode has no insight-density signal', async () => {
    vi.spyOn(api, 'getEpisodeEnrichment').mockResolvedValue({})
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="episode-density"]').exists()).toBe(false)
  })

  it('renders nothing when all thirds are empty', async () => {
    vi.spyOn(api, 'getEpisodeEnrichment').mockResolvedValue({
      insight_density: { counts: { early: 0, mid: 0, late: 0 }, duration_seconds: 900 },
    })
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="episode-density"]').exists()).toBe(false)
  })
})
