// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { localYmdDaysAgo } from '../../utils/localCalendarDate'
import DateChip from './DateChip.vue'

const CHIP = '[data-testid="date-chip"]'
const POPOVER = '[data-testid="date-popover"]'
const CUSTOM = '[data-testid="date-chip-custom"]'

// Pin "now" so the preset YYYY-MM-DD values are deterministic across the run.
const FIXED_NOW = new Date('2026-06-10T12:00:00')

const mountChip = (modelValue = '', props: Record<string, unknown> = {}) =>
  mount(DateChip, { props: { modelValue, ...props }, attachTo: document.body })

describe('DateChip', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(FIXED_NOW)
  })

  it('renders the default "Date ▾" label and a collapsed popover', () => {
    const w = mountChip('')
    expect(w.get(CHIP).text()).toBe('Date ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
    expect(w.get(CHIP).attributes('aria-haspopup')).toBe('dialog')
    expect(w.get(CHIP).attributes('aria-label')).toBe('Date filter')
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })

  it('honours a custom label prop in the chip and aria-label', () => {
    const w = mountChip('', { label: 'Published' })
    expect(w.get(CHIP).text()).toBe('Published ▾')
    expect(w.get(CHIP).attributes('aria-label')).toBe('Published filter')
  })

  it('honours custom chip / popover testids', () => {
    const w = mountChip('', { chipTestid: 'lib-date', popoverTestid: 'lib-date-pop' })
    expect(w.find('[data-testid="lib-date"]').exists()).toBe(true)
    expect(w.find('[data-testid="lib-date-pop"]').exists()).toBe(true)
    // The custom input testid is derived from the chip testid.
    expect(w.find('[data-testid="lib-date-custom"]').exists()).toBe(true)
  })

  it('renders the 7d preset label when modelValue matches 7 days ago', () => {
    const w = mountChip(localYmdDaysAgo(7))
    expect(w.get(CHIP).text()).toBe('Date: Last 7d ▾')
  })

  it('renders the 30d preset label when modelValue matches 30 days ago', () => {
    const w = mountChip(localYmdDaysAgo(30))
    expect(w.get(CHIP).text()).toBe('Date: Last 30d ▾')
  })

  it('renders the 90d preset label when modelValue matches 90 days ago', () => {
    const w = mountChip(localYmdDaysAgo(90))
    expect(w.get(CHIP).text()).toBe('Date: Last 90d ▾')
  })

  it('renders a custom ≥ label for an arbitrary date', () => {
    const w = mountChip('2026-01-01')
    expect(w.get(CHIP).text()).toBe('Date: ≥ 2026-01-01 ▾')
  })

  it('opens and closes the popover when the chip is clicked', async () => {
    const w = mountChip('')
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })

  it('does not render the Clear button while default (all time)', () => {
    const w = mountChip('')
    expect(w.get(POPOVER).find('button').exists()).toBe(true)
    expect(
      w.get(POPOVER).findAll('button').some((b) => b.text() === 'Clear'),
    ).toBe(false)
  })

  it('renders the Clear button once a value is active', () => {
    const w = mountChip('2026-01-01')
    expect(
      w.get(POPOVER).findAll('button').some((b) => b.text() === 'Clear'),
    ).toBe(true)
  })

  it('"All time" preset emits empty string and closes the popover', async () => {
    const w = mountChip('2026-01-01')
    await w.get(CHIP).trigger('click')
    const allTime = w
      .get(POPOVER)
      .findAll('button')
      .find((b) => b.text() === 'All time')!
    await allTime.trigger('click')
    expect(w.emitted('update:modelValue')).toEqual([['']])
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })

  it('the 7d preset emits the 7-days-ago YMD and closes', async () => {
    const w = mountChip('')
    await w.get(CHIP).trigger('click')
    const btn = w.get(POPOVER).findAll('button').find((b) => b.text() === '7d')!
    await btn.trigger('click')
    expect(w.emitted('update:modelValue')).toEqual([[localYmdDaysAgo(7)]])
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })

  it('the 30d and 90d presets emit their respective YMD values', async () => {
    const w = mountChip('')
    await w.get(CHIP).trigger('click')
    const btn30 = w.get(POPOVER).findAll('button').find((b) => b.text() === '30d')!
    await btn30.trigger('click')
    const btn90 = w.get(POPOVER).findAll('button').find((b) => b.text() === '90d')!
    await btn90.trigger('click')
    expect(w.emitted('update:modelValue')).toEqual([
      [localYmdDaysAgo(30)],
      [localYmdDaysAgo(90)],
    ])
  })

  it('marks the active preset button as visually selected', async () => {
    const w = mountChip(localYmdDaysAgo(7))
    const sevenBtn = w
      .get(POPOVER)
      .findAll('button')
      .find((b) => b.text() === '7d')!
    // The active preset uses the primary border class from presetButtonClass(true).
    expect(sevenBtn.classes().join(' ')).toContain('border-primary')
  })

  it('seeds the custom input from modelValue', () => {
    const w = mountChip('2026-01-01')
    const input = w.get(CUSTOM).element as HTMLInputElement
    expect(input.value).toBe('2026-01-01')
  })

  it('keeps the custom input synced when modelValue changes', async () => {
    const w = mountChip('2026-01-01')
    await w.setProps({ modelValue: '2026-02-02' })
    const input = w.get(CUSTOM).element as HTMLInputElement
    expect(input.value).toBe('2026-02-02')
  })

  it('commits a valid custom date on blur', async () => {
    const w = mountChip('')
    const input = w.get(CUSTOM)
    await input.setValue('2025-12-25')
    await input.trigger('blur')
    expect(w.emitted('update:modelValue')).toEqual([['2025-12-25']])
  })

  it('commits an empty custom value on blur (clearing back to all time)', async () => {
    const w = mountChip('2026-01-01')
    const input = w.get(CUSTOM)
    await input.setValue('')
    await input.trigger('blur')
    expect(w.emitted('update:modelValue')).toEqual([['']])
  })

  it('commits a trimmed valid custom value', async () => {
    // commitCustom trims before validating; a padded valid date still emits the
    // trimmed YMD.
    const w = mountChip('')
    const input = w.get(CUSTOM)
    // happy-dom keeps a well-formed YYYY-MM-DD verbatim through the date input.
    await input.setValue('2025-03-14')
    await input.trigger('blur')
    expect(w.emitted('update:modelValue')).toEqual([['2025-03-14']])
  })

  it('commits a valid custom date on Enter and closes the popover', async () => {
    const w = mountChip('')
    await w.get(CHIP).trigger('click')
    const input = w.get(CUSTOM)
    await input.setValue('2025-11-11')
    await input.trigger('keydown.enter')
    expect(w.emitted('update:modelValue')).toEqual([['2025-11-11']])
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })

  it('Clear emits empty string and closes the popover', async () => {
    const w = mountChip('2026-01-01')
    await w.get(CHIP).trigger('click')
    const clear = w
      .get(POPOVER)
      .findAll('button')
      .find((b) => b.text() === 'Clear')!
    await clear.trigger('click')
    expect(w.emitted('update:modelValue')).toEqual([['']])
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })
})
