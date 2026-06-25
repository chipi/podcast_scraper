import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import InterestsPicker from './InterestsPicker.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountPicker = () => mount(InterestsPicker, { global: { plugins: [i18n] } })

afterEach(() => vi.restoreAllMocks())

describe('InterestsPicker', () => {
  it('renders top clusters, pre-selects saved interests, and saves the toggled set', async () => {
    vi.spyOn(api, 'getTopClusters').mockResolvedValue([
      { id: 'tc:ai', label: 'AI', size: 5 },
      { id: 'tc:health', label: 'Health', size: 3 },
    ])
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['tc:ai']) // ai pre-selected
    const put = vi.spyOn(api, 'putUserInterests').mockResolvedValue(['tc:ai', 'tc:health'])

    const w = mountPicker()
    await flushPromises()
    const aiChip = w.findAll('button').find((b) => b.text() === 'AI')!
    const healthChip = w.findAll('button').find((b) => b.text() === 'Health')!
    expect(aiChip.attributes('aria-pressed')).toBe('true')
    expect(healthChip.attributes('aria-pressed')).toBe('false')

    await healthChip.trigger('click') // add Health
    await w.findAll('button').find((b) => b.text() === 'Save')!.trigger('click')
    await flushPromises()
    expect(put).toHaveBeenCalledWith(['tc:ai', 'tc:health'])
    expect(w.emitted('saved')?.[0]).toEqual([['tc:ai', 'tc:health']])
    expect(w.emitted('close')).toBeTruthy()
  })

  it('shows the empty state when no clusters exist', async () => {
    vi.spyOn(api, 'getTopClusters').mockResolvedValue([])
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = mountPicker()
    await flushPromises()
    expect(w.text()).toContain('No topics to choose yet.')
  })

  it('closes on the dimmed backdrop', async () => {
    vi.spyOn(api, 'getTopClusters').mockResolvedValue([])
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = mountPicker()
    await flushPromises()
    await w.find('[role="dialog"]').trigger('click')
    expect(w.emitted('close')).toBeTruthy()
  })
})
