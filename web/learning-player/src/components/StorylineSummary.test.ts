/**
 * #2344: a storyline's summary — shown when written, absent until then, and says when it is stale.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'

const api = vi.hoisted(() => ({ getStorylineSummary: vi.fn() }))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))

import StorylineSummary from './StorylineSummary.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

describe('StorylineSummary', () => {
  it('shows the paragraph, when it was written, and that the storyline grew since', async () => {
    api.getStorylineSummary.mockResolvedValue({ text: 'A thread on risk.', created_at: 1_760_000_000, stale: true })
    const w = mount(StorylineSummary, { props: { storylineId: 'thc:risk' }, global: { plugins: [i18n] } })
    await flushPromises()
    expect(api.getStorylineSummary).toHaveBeenCalledWith('thc:risk')
    expect(w.get('[data-testid="storyline-summary"]').text()).toContain('A thread on risk.')
    expect(w.get('[data-testid="storyline-summary-when"]').text()).toContain('the storyline grew since')
  })

  it('is absent until one is written, and when it cannot be read', async () => {
    api.getStorylineSummary.mockResolvedValueOnce(null)
    const none = mount(StorylineSummary, { props: { storylineId: 'thc:x' }, global: { plugins: [i18n] } })
    await flushPromises()
    expect(none.find('[data-testid="storyline-summary"]').exists()).toBe(false)
    api.getStorylineSummary.mockRejectedValueOnce(new Error('offline'))
    const down = mount(StorylineSummary, { props: { storylineId: 'thc:y' }, global: { plugins: [i18n] } })
    await flushPromises()
    expect(down.find('[data-testid="storyline-summary"]').exists()).toBe(false)
  })
})
