// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import LibraryClusteredChip from './LibraryClusteredChip.vue'

const CHIP = '[data-testid="library-chip-clustered"]'

describe('LibraryClusteredChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  const mountChip = (modelValue = false) =>
    mount(LibraryClusteredChip, { props: { modelValue }, attachTo: document.body })

  it('renders the inactive label and aria-pressed=false by default', () => {
    const w = mountChip(false)
    const chip = w.get(CHIP)
    expect(chip.text()).toBe('Clustered')
    expect(chip.attributes('aria-pressed')).toBe('false')
    expect(chip.attributes('aria-label')).toBe('Toggle clustered episodes only')
    expect(chip.attributes('type')).toBe('button')
  })

  it('renders the active label and aria-pressed=true when modelValue is true', () => {
    const w = mountChip(true)
    const chip = w.get(CHIP)
    expect(chip.text()).toBe('Clustered ✓')
    expect(chip.attributes('aria-pressed')).toBe('true')
  })

  it('applies the filled/active class set when active', () => {
    const active = mountChip(true).get(CHIP)
    expect(active.classes()).toContain('bg-primary/15')
    expect(active.classes()).toContain('font-medium')

    const inactive = mountChip(false).get(CHIP)
    expect(inactive.classes()).not.toContain('bg-primary/15')
    expect(inactive.classes()).toContain('text-muted')
  })

  it('emits update:modelValue=true when toggled on from inactive', async () => {
    const w = mountChip(false)
    await w.get(CHIP).trigger('click')
    expect(w.emitted('update:modelValue')).toEqual([[true]])
  })

  it('emits update:modelValue=false when toggled off from active', async () => {
    const w = mountChip(true)
    await w.get(CHIP).trigger('click')
    expect(w.emitted('update:modelValue')).toEqual([[false]])
  })

  it('reflects a reactive prop change in the label without a re-mount', async () => {
    const w = mountChip(false)
    expect(w.get(CHIP).text()).toBe('Clustered')
    await w.setProps({ modelValue: true })
    expect(w.get(CHIP).text()).toBe('Clustered ✓')
    expect(w.get(CHIP).attributes('aria-pressed')).toBe('true')
  })
})
