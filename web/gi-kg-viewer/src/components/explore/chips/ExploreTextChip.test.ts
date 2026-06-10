// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import ExploreTextChip from './ExploreTextChip.vue'

const CHIP = '[data-testid="explore-chip-topic"]'
const POPOVER = '[data-testid="explore-popover-topic"]'
const INPUT = '[data-testid="explore-chip-topic-input"]'
const CLEAR = '[data-testid="explore-chip-topic-clear"]'

type Props = {
  modelValue?: string
  label?: string
  chipTestid?: string
  popoverTestid?: string
  placeholder?: string
  enabled?: boolean
  disabledTitle?: string
}

const mountChip = (props: Props = {}) =>
  mount(ExploreTextChip, {
    props: {
      modelValue: '',
      label: 'Topic',
      chipTestid: 'explore-chip-topic',
      popoverTestid: 'explore-popover-topic',
      ...props,
    },
    attachTo: document.body,
  })

describe('ExploreTextChip', () => {
  // Uses the popover composable (document listeners) but no store; pinia
  // is set for parity with sibling chip tests.
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the inactive label by default', () => {
    const w = mountChip()
    expect(w.get(CHIP).text()).toBe('Topic ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
    expect(w.get(CHIP).attributes('aria-haspopup')).toBe('dialog')
    expect(w.get(CHIP).attributes('aria-label')).toBe('Topic contains')
  })

  it('renders the active label with the (trimmed) model value', () => {
    const w = mountChip({ modelValue: 'rust' })
    expect(w.get(CHIP).text()).toBe('Topic: rust ▾')
    expect(w.get(CHIP).classes()).toContain('font-medium')
  })

  it('treats a whitespace-only model value as inactive', () => {
    const w = mountChip({ modelValue: '   ' })
    expect(w.get(CHIP).text()).toBe('Topic ▾')
  })

  it('truncates a long active value to 17 chars + ellipsis', () => {
    const long = 'abcdefghijklmnopqrstuvwxyz' // 26 chars
    const w = mountChip({ modelValue: long })
    expect(w.get(CHIP).text()).toBe(`Topic: ${long.slice(0, 17)}… ▾`)
  })

  it('keeps the popover hidden until the chip is clicked, then toggles aria-expanded', async () => {
    const w = mountChip()
    expect(w.get(POPOVER).isVisible()).toBe(false)
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(false)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
  })

  it('renders the placeholder and the label heading inside the popover', async () => {
    const w = mountChip({ placeholder: 'type a topic…' })
    await w.get(CHIP).trigger('click')
    expect(w.get(INPUT).attributes('placeholder')).toBe('type a topic…')
    expect(w.get(POPOVER).text()).toContain('Topic contains')
    expect(w.get(POPOVER).attributes('role')).toBe('dialog')
    expect(w.get(POPOVER).attributes('aria-label')).toBe('Topic contains')
  })

  it('seeds the input draft from modelValue', async () => {
    const w = mountChip({ modelValue: 'seeded' })
    await w.get(CHIP).trigger('click')
    expect((w.get(INPUT).element as HTMLInputElement).value).toBe('seeded')
  })

  it('syncs the draft when the modelValue prop changes', async () => {
    const w = mountChip({ modelValue: 'first' })
    await w.get(CHIP).trigger('click')
    await w.setProps({ modelValue: 'second' })
    expect((w.get(INPUT).element as HTMLInputElement).value).toBe('second')
  })

  it('emits update:modelValue on blur (commit)', async () => {
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    await w.get(INPUT).setValue('hello')
    await w.get(INPUT).trigger('blur')
    const ev = w.emitted('update:modelValue')
    expect(ev).toBeTruthy()
    expect(ev!.at(-1)).toEqual(['hello'])
  })

  it('commits, closes, and submits on Enter', async () => {
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    await w.get(INPUT).setValue('go')
    await w.get(INPUT).trigger('keydown.enter')
    expect(w.emitted('update:modelValue')!.at(-1)).toEqual(['go'])
    expect(w.emitted('submit')).toHaveLength(1)
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })

  it('hides the Clear button when inactive and shows it when active', async () => {
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    expect(w.find(CLEAR).exists()).toBe(false)

    await w.setProps({ modelValue: 'x' })
    expect(w.find(CLEAR).exists()).toBe(true)
  })

  it('Clear emits an empty value, resets the draft, and closes the popover', async () => {
    const w = mountChip({ modelValue: 'topic-value' })
    await w.get(CHIP).trigger('click')
    await w.get(CLEAR).trigger('click')
    expect(w.emitted('update:modelValue')!.at(-1)).toEqual([''])
    expect(w.get(POPOVER).isVisible()).toBe(false)
    // Draft reset to empty.
    expect((w.get(INPUT).element as HTMLInputElement).value).toBe('')
  })

  it('is disabled and tooltipped when enabled=false', () => {
    const w = mountChip({ enabled: false, disabledTitle: 'no corpus loaded' })
    const btn = w.get(CHIP)
    expect(btn.attributes('disabled')).toBeDefined()
    expect(btn.attributes('title')).toBe('no corpus loaded')
  })

  it('has no title attribute when enabled', () => {
    const w = mountChip({ enabled: true })
    expect(w.get(CHIP).attributes('title')).toBeUndefined()
  })

  it('honors a custom label across chip + popover', async () => {
    const w = mountChip({ label: 'Speaker' })
    expect(w.get(CHIP).text()).toBe('Speaker ▾')
    expect(w.get(CHIP).attributes('aria-label')).toBe('Speaker contains')
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).text()).toContain('Speaker contains')
  })
})
