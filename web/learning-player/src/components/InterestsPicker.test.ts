import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import InterestsPicker from './InterestsPicker.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
// Stub <Teleport> so the modal renders inline in the wrapper (it teleports to <body> in the app).
const mountPicker = () => mount(InterestsPicker, { global: { plugins: [i18n], stubs: { teleport: true } } })

// Default the storylines fetch to empty; the storyline-specific tests override it.
beforeEach(() => vi.spyOn(api, 'getStorylines').mockResolvedValue([]))
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

  it('renders a Storylines section and saves a toggled storyline (thc:) token', async () => {
    vi.spyOn(api, 'getTopClusters').mockResolvedValue([{ id: 'tc:ai', label: 'AI', size: 5 }])
    vi.spyOn(api, 'getStorylines').mockResolvedValue([
      { id: 'thc:shadow-fleet', label: 'Shadow fleet', size: 4, anchor_topic_id: 'topic:sanctions' },
    ])
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const put = vi.spyOn(api, 'putUserInterests').mockResolvedValue(['thc:shadow-fleet'])

    const w = mountPicker()
    await flushPromises()
    expect(w.find('[data-testid="interests-storylines"]').exists()).toBe(true)

    await w.findAll('button').find((b) => b.text() === 'Shadow fleet')!.trigger('click')
    await w.findAll('button').find((b) => b.text() === 'Save')!.trigger('click')
    await flushPromises()
    expect(put).toHaveBeenCalledWith(['thc:shadow-fleet'])
  })

  it('preserves followed tokens the picker does not offer (topic:/person:) across save', async () => {
    vi.spyOn(api, 'getTopClusters').mockResolvedValue([{ id: 'tc:ai', label: 'AI', size: 5 }])
    // The user follows a person (from an entity card) plus already has the AI cluster selected.
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['person:jane', 'tc:ai'])
    const put = vi.spyOn(api, 'putUserInterests').mockResolvedValue(['person:jane'])

    const w = mountPicker()
    await flushPromises()
    // Deselect the AI cluster; person:jane isn't a picker chip and must still survive the PUT.
    await w.findAll('button').find((b) => b.text() === 'AI')!.trigger('click')
    await w.findAll('button').find((b) => b.text() === 'Save')!.trigger('click')
    await flushPromises()
    expect(put).toHaveBeenCalledWith(['person:jane'])
  })
})
