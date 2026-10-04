import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import TrendMomentum from './TrendMomentum.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountIt = (props: {
  velocity: number
  series?: number[]
  variant?: 'rail' | 'badge'
  hideSpark?: boolean
}) => mount(TrendMomentum, { props, global: { plugins: [i18n] } })

describe('TrendMomentum', () => {
  it('badge variant renders the rising pill + sparkline with ≥2 series points', () => {
    const w = mountIt({ variant: 'badge', velocity: 2.14, series: [1, 3, 6] })
    const el = w.get('[data-testid="trend-momentum"]')
    expect(el.text()).toContain('Rising')
    expect(el.text()).toContain('2.1×') // rounded to one decimal
    expect(el.find('svg').exists()).toBe(true)
  })

  it('badge variant says "Cooling" with a down arrow below the flat band, never "Rising"', () => {
    // Repro: the storyline page showed "↑ Rising · 0.4× vs its 6-month average" — the pill
    // hard-coded the rising arrow and word for any velocity.
    const w = mountIt({ variant: 'badge', velocity: 0.4, series: [6, 3, 1] })
    const el = w.get('[data-testid="trend-momentum"]')
    expect(el.text()).not.toContain('Rising')
    expect(el.text()).not.toContain('↑')
    expect(el.text()).toContain('Cooling')
    expect(el.text()).toContain('↓')
    expect(el.text()).toContain('0.4×')
  })

  it('badge variant says "Steady" with a level arrow inside the flat band', () => {
    const el = mountIt({ variant: 'badge', velocity: 1.0 }).get('[data-testid="trend-momentum"]')
    expect(el.text()).toContain('Steady')
    expect(el.text()).toContain('→')
    expect(el.text()).not.toContain('Rising')
  })

  it('rail variant renders the arrow + velocity + sparkline', () => {
    const w = mountIt({ variant: 'rail', velocity: 1.8, series: [2, 4, 5] })
    const el = w.get('[data-testid="trend-momentum"]')
    expect(el.text()).toContain('1.8×')
    expect(el.find('svg').exists()).toBe(true)
  })

  it('omits the sparkline when there are fewer than 2 points', () => {
    const w = mountIt({ variant: 'rail', velocity: 1.5, series: [3] })
    expect(w.get('[data-testid="trend-momentum"]').find('svg').exists()).toBe(false)
  })

  it('renders the velocity even with no series at all', () => {
    const w = mountIt({ variant: 'rail', velocity: 3.0 })
    expect(w.get('[data-testid="trend-momentum"]').text()).toContain('3×')
  })
})
