import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import FollowButton from './FollowButton.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountBtn = (props: Record<string, unknown> = {}) =>
  mount(FollowButton, { props: { following: false, ...props }, global: { plugins: [i18n] } })

describe('FollowButton (F2.4)', () => {
  it('shows Follow / Following from one control, and emits toggle', async () => {
    const w = mountBtn()
    expect(w.get('[data-testid="follow-show"]').text()).toContain('Follow')
    await w.get('[data-testid="follow-show"]').trigger('click')
    expect(w.emitted('toggle')).toHaveLength(1)

    const on = mountBtn({ following: true })
    expect(on.get('[data-testid="follow-show"]').text()).toContain('Following')
    expect(on.get('[data-testid="follow-show"]').attributes('aria-pressed')).toBe('true')
  })

  it('the overlay variant is plated to read over artwork, and positions NOTHING', () => {
    // `overlay` sets the look; the HOST places it. It used to hard-code `absolute right-1.5 top-1.5`,
    // which meant the button chose its own corner and nothing could sit beside or under it — so
    // ShowTile could not stack Follow and the heart into one column (operator 2026-09-17).
    const overlay = mountBtn({ variant: 'overlay' }).get('button').classes()
    expect(overlay, 'the overlay variant lost its plate').toContain('backdrop-blur')
    expect(overlay, 'the button still positions itself').not.toContain('absolute')
    expect(mountBtn({ variant: 'inline' }).get('button').classes()).not.toContain('backdrop-blur')
  })

  it('signed out, the label routes to sign-in and no pressed state is asserted', () => {
    const btn = mountBtn({ gated: true }).get('[data-testid="follow-show"]')
    expect(btn.attributes('aria-label')).toBe(en.auth.signInToFollow)
    expect(btn.attributes('aria-pressed')).toBeUndefined()
  })

  it('is disabled while a toggle is in flight', () => {
    expect(mountBtn({ busy: true }).get('button').attributes('disabled')).toBeDefined()
  })
})
