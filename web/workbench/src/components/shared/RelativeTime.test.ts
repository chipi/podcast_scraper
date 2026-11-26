// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import RelativeTime from './RelativeTime.vue'

describe('RelativeTime', () => {
  it('renders — for null and has no title', () => {
    const w = mount(RelativeTime, { props: { iso: null } })
    expect(w.text()).toBe('—')
    expect(w.get('[data-testid="relative-time"]').attributes('title')).toBeUndefined()
  })

  it('renders a relative label and an absolute local-time title for a real time', () => {
    const future = new Date(Date.now() + 3 * 3600_000).toISOString()
    const w = mount(RelativeTime, { props: { iso: future } })
    expect(w.text()).toMatch(/^in /)
    expect(w.get('[data-testid="relative-time"]').attributes('title')).toMatch(/\d{4}-\d{2}-\d{2} \d{2}:\d{2} \S+$/)
  })
})
