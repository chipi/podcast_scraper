import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeSummary, Me } from '../services/types'
import { useAuthStore } from '../stores/auth'
import FavoriteButton from './FavoriteButton.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const item = { kind: 'episode' as const, ref: 'ep1', label: 'Ep' }
const mountBtn = () => mount(FavoriteButton, { props: { item }, global: { plugins: [i18n] } })

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

describe('FavoriteButton', () => {
  it('renders signed out as a sign-in teaser, not hidden (#1590)', () => {
    const w = mountBtn()
    const btn = w.find('button')
    expect(btn.exists()).toBe(true)
    expect(btn.attributes('aria-label')).toContain('Sign in')
    expect(btn.attributes('aria-pressed')).toBeUndefined()
  })

  it('renders and toggles via the favorites store when signed in', async () => {
    useAuthStore().user = { user_id: 'u' } as unknown as Me
    const add = vi
      .spyOn(api, 'addFavorite')
      .mockResolvedValue({ episodes: [{ slug: 'ep1' } as EpisodeSummary], insights: [] })
    const w = mountBtn()
    expect(w.text()).toBe('♡') // not yet saved
    await w.find('button').trigger('click')
    await flushPromises()
    expect(add).toHaveBeenCalledWith(item)
    expect(w.text()).toBe('♥') // store now reports it saved
  })
})
