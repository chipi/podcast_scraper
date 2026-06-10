// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { CilDigestTopicPill } from '../../api/digestApi'
import CilTopicPillsRow from './CilTopicPillsRow.vue'

const pill = (over: Partial<CilDigestTopicPill> = {}): CilDigestTopicPill => ({
  topic_id: 't1',
  label: 'Climate policy',
  ...over,
})

const mountRow = (props: Record<string, unknown> = {}) =>
  mount(CilTopicPillsRow, {
    props: { pills: [pill()], ...props },
    attachTo: document.body,
  })

describe('CilTopicPillsRow', () => {
  it('renders one button per pill', () => {
    const w = mountRow({
      pills: [pill({ topic_id: 'a', label: 'Alpha' }), pill({ topic_id: 'b', label: 'Beta' })],
    })
    const buttons = w.findAll('button')
    expect(buttons.length).toBe(2)
    expect(buttons[0].text()).toBe('Alpha')
    expect(buttons[1].text()).toBe('Beta')
  })

  it('renders nothing when the pills array is empty', () => {
    const w = mountRow({ pills: [] })
    expect(w.find('button').exists()).toBe(false)
    // The v-if wrapper group is also absent.
    expect(w.find('[role="group"]').exists()).toBe(false)
  })

  it('exposes the group role + aria-label wrapper for a11y', () => {
    const w = mountRow()
    const group = w.get('[role="group"]')
    expect(group.attributes('aria-label')).toBe('Topic chips')
  })

  it('applies the dataTestid hook to the group wrapper', () => {
    const w = mountRow({ dataTestid: 'cil-pills' })
    expect(w.find('[data-testid="cil-pills"]').exists()).toBe(true)
  })

  it('sets aria-label and title per pill from the full label', () => {
    const w = mountRow({ pills: [pill({ label: 'Renewable energy' })] })
    const btn = w.get('button')
    expect(btn.attributes('aria-label')).toBe('Open graph for topic: Renewable energy')
    expect(btn.attributes('title')).toBe('Renewable energy')
  })

  it('omits the title attribute for a blank label', () => {
    const w = mountRow({ pills: [pill({ label: '   ' })] })
    expect(w.get('button').attributes('title')).toBeUndefined()
  })

  it('emits pill-click with the index when a pill is clicked', async () => {
    const w = mountRow({
      pills: [pill({ topic_id: 'a', label: 'Alpha' }), pill({ topic_id: 'b', label: 'Beta' })],
    })
    await w.findAll('button')[1].trigger('click')
    expect(w.emitted('pill-click')).toEqual([[1]])
  })

  it('truncates long labels with an ellipsis under the default strategy', () => {
    const long = 'A'.repeat(40)
    const w = mountRow({ pills: [pill({ label: long })] })
    const text = w.get('button').text()
    expect(text.endsWith('…')).toBe(true)
    // Default maxPillChars is 24 → 23 chars + ellipsis.
    expect(text.length).toBe(24)
  })

  it('respects a custom maxPillChars cap', () => {
    const w = mountRow({ pills: [pill({ label: 'A'.repeat(40) })], maxPillChars: 10 })
    expect(w.get('button').text().length).toBe(10)
  })

  it('renders the full label with truncation="none"', () => {
    const long = 'B'.repeat(40)
    const w = mountRow({ pills: [pill({ label: long })], truncation: 'none' })
    expect(w.get('button').text()).toBe(long)
  })

  it('renders the full label and wrap classes with truncation="wrap"', () => {
    const long = 'C'.repeat(40)
    const w = mountRow({ pills: [pill({ label: long })], truncation: 'wrap' })
    const btn = w.get('button')
    expect(btn.text()).toBe(long)
    expect(btn.classes().join(' ')).toContain('whitespace-normal')
  })

  it('uses the truncate class for the ellipsis strategy', () => {
    const w = mountRow()
    expect(w.get('button').classes()).toContain('truncate')
  })

  it('applies the default max-width class to pills', () => {
    const w = mountRow()
    expect(w.get('button').classes()).toContain('max-w-[11rem]')
  })

  it('drops the max-width class when maxWidthClass="auto"', () => {
    const w = mountRow({ maxWidthClass: 'auto' })
    expect(w.get('button').classes()).not.toContain('max-w-[11rem]')
  })

  it('applies legacy quote chrome (inline style) for clustered pills by default', () => {
    const w = mountRow({ pills: [pill({ in_topic_cluster: true })] })
    const btn = w.get('button')
    // Legacy quote appearance sets the inline gradient chrome style.
    expect(btn.attributes('style')).toBeTruthy()
    expect(btn.classes().join(' ')).toContain('border-transparent')
  })

  it('applies kg chrome classes (no inline style) for clustered pills when appearance=kg', () => {
    const w = mountRow({
      pills: [pill({ in_topic_cluster: true })],
      clusterMemberAppearance: 'kg',
    })
    const btn = w.get('button')
    expect(btn.classes().join(' ')).toContain('border-kg')
    // kg appearance does not apply the inline quote chrome style.
    expect(btn.attributes('style') ?? '').not.toContain('background')
  })

  it('uses the non-clustered chrome for plain pills', () => {
    const w = mountRow({ pills: [pill({ in_topic_cluster: false })] })
    expect(w.get('button').classes().join(' ')).toContain('bg-canvas')
  })
})
