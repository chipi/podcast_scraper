import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import PlayerSkeleton from './PlayerSkeleton.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

describe('PlayerSkeleton', () => {
  it('reserves the player shape and announces the load to a screen reader', () => {
    const w = mount(PlayerSkeleton, { global: { plugins: [i18n] } })
    const root = w.find('[data-testid="player-skeleton"]')
    expect(root.exists()).toBe(true)
    expect(root.attributes('aria-busy')).toBe('true')
    // Reserves the square hero footprint (the biggest jump source) + at least a few placeholder
    // blocks for title/controls/transcript.
    expect(w.find('.aspect-square').exists()).toBe(true)
    expect(w.findAll('.animate-pulse').length).toBeGreaterThan(5)
    // Decorative blocks are hidden from AT; one sr-only line carries the status.
    expect(w.find('.sr-only').text()).toBeTruthy()
  })
})
