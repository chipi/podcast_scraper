import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import SectionStatus from './SectionStatus.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function mountIt(props: { phase: 'loading' | 'ready' | 'error'; rows?: number }) {
  return mount(SectionStatus, { props, global: { plugins: [i18n] } })
}

/**
 * The one loading/error surface every section shares (#1591). Tested directly because it is the
 * reason the same class of failure stopped being styled three different ways.
 */
describe('SectionStatus', () => {
  it('renders a skeleton while loading, marked busy for assistive tech', () => {
    const w = mountIt({ phase: 'loading', rows: 3 })
    const skeleton = w.get('[data-testid="section-loading"]')
    expect(skeleton.attributes('aria-busy')).toBe('true')
    // Skeleton rows are decorative; the announcement is the sr-only text, not N pulsing divs.
    expect(skeleton.findAll('[aria-hidden="true"]')).toHaveLength(3)
    expect(w.text()).toContain('Loading')
  })

  it('defaults to two skeleton rows', () => {
    expect(mountIt({ phase: 'loading' }).findAll('[aria-hidden="true"]')).toHaveLength(2)
  })

  it('renders an error with a retry, and announces it', () => {
    const w = mountIt({ phase: 'error' })
    const err = w.get('[data-testid="section-error"]')
    expect(err.attributes('role')).toBe('status')
    expect(w.get('[data-testid="section-retry"]').exists()).toBe(true)
    expect(w.text()).toContain('Try again')
  })

  it('emits retry when the button is used', async () => {
    const w = mountIt({ phase: 'error' })
    await w.get('[data-testid="section-retry"]').trigger('click')
    expect(w.emitted('retry')).toHaveLength(1)
  })

  it('renders nothing when ready — the caller owns success and empty', () => {
    // Only the caller knows whether ITS emptiness is actionable (UXS-012's state contract), so this
    // component deliberately has no opinion about the empty case.
    const w = mountIt({ phase: 'ready' })
    expect(w.find('[data-testid="section-loading"]').exists()).toBe(false)
    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
  })
})
