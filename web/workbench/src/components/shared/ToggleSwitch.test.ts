// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import ToggleSwitch from './ToggleSwitch.vue'

describe('ToggleSwitch', () => {
  it('reflects modelValue via aria-checked', () => {
    const on = mount(ToggleSwitch, { props: { modelValue: true, label: 'Enable x' } })
    expect(on.get('[role="switch"]').attributes('aria-checked')).toBe('true')
    const off = mount(ToggleSwitch, { props: { modelValue: false } })
    expect(off.get('[role="switch"]').attributes('aria-checked')).toBe('false')
  })

  it('emits the toggled value on click', async () => {
    const w = mount(ToggleSwitch, { props: { modelValue: false } })
    await w.get('[role="switch"]').trigger('click')
    expect(w.emitted('update:modelValue')?.[0]).toEqual([true])
  })

  it('does not emit when disabled', async () => {
    const w = mount(ToggleSwitch, { props: { modelValue: false, disabled: true } })
    await w.get('[role="switch"]').trigger('click')
    expect(w.emitted('update:modelValue')).toBeUndefined()
  })

  it('exposes a custom testid', () => {
    const w = mount(ToggleSwitch, { props: { modelValue: true, testid: 'my-toggle' } })
    expect(w.find('[data-testid="my-toggle"]').exists()).toBe(true)
  })
})
