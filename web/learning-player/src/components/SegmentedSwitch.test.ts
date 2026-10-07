import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import SegmentedSwitch from './SegmentedSwitch.vue'

describe('SegmentedSwitch (operator 2026-10-07)', () => {
  const options = [
    { value: 'mine', label: 'You', testid: 'opt-mine' },
    { value: 'corpus', label: 'Everyone', testid: 'opt-corpus' },
  ]

  it('shows every option in words, in a named group, with the chosen one pressed', () => {
    const w = mount(SegmentedSwitch, { props: { options, label: 'Whose trends', modelValue: 'mine' } })
    expect(w.get('[role="group"]').attributes('aria-label')).toBe('Whose trends')
    expect(w.findAll('button').map((b) => b.text())).toEqual(['You', 'Everyone'])
    expect(w.get('[data-testid="opt-mine"]').attributes('aria-pressed')).toBe('true')
    expect(w.get('[data-testid="opt-corpus"]').attributes('aria-pressed')).toBe('false')
  })

  it('emits the option pressed', async () => {
    const w = mount(SegmentedSwitch, { props: { options, label: 'Whose trends', modelValue: 'mine' } })
    await w.get('[data-testid="opt-corpus"]').trigger('click')
    expect(w.emitted('update:modelValue')?.[0]).toEqual(['corpus'])
  })
})
