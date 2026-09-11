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

  it('the overlay variant floats over artwork; the inline variant does not', () => {
    expect(mountBtn({ variant: 'overlay' }).get('button').classes()).toContain('absolute')
    expect(mountBtn({ variant: 'inline' }).get('button').classes()).not.toContain('absolute')
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
