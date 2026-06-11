// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useGraphHandoffStore } from '../../stores/graphHandoff'
import HandoffErrorStrip from './HandoffErrorStrip.vue'

const STRIP = '[data-testid="handoff-error-strip"]'
const DISMISS = '[data-testid="handoff-error-strip-dismiss"]'

describe('HandoffErrorStrip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  const mountStrip = () => mount(HandoffErrorStrip, { attachTo: document.body })

  it('renders nothing when there is no last result', () => {
    const w = mountStrip()
    expect(w.find(STRIP).exists()).toBe(false)
  })

  it('renders nothing when the last result was applied (not failed)', async () => {
    const w = mountStrip()
    const handoff = useGraphHandoffStore()
    handoff.lastResult = { status: 'applied', appliedCyId: 'node-1' }
    await w.vm.$nextTick()
    expect(w.find(STRIP).exists()).toBe(false)
  })

  it('renders nothing when the last result was superseded', async () => {
    const w = mountStrip()
    const handoff = useGraphHandoffStore()
    handoff.lastResult = { status: 'superseded' }
    await w.vm.$nextTick()
    expect(w.find(STRIP).exists()).toBe(false)
  })

  it('shows the strip with the failure reason when the handoff failed', async () => {
    const w = mountStrip()
    const handoff = useGraphHandoffStore()
    handoff.lastResult = { status: 'failed', reason: 'node not found' }
    await w.vm.$nextTick()

    const strip = w.get(STRIP)
    expect(strip.isVisible()).toBe(true)
    expect(strip.text()).toContain('Could not open episode in graph: node not found')
    // Accessibility wiring.
    expect(strip.attributes('role')).toBe('alert')
    expect(strip.attributes('aria-live')).toBe('polite')
  })

  it('falls back to "unknown error" when the failed result has no reason', async () => {
    const w = mountStrip()
    const handoff = useGraphHandoffStore()
    handoff.lastResult = { status: 'failed' }
    await w.vm.$nextTick()

    expect(w.get(STRIP).text()).toContain('Could not open episode in graph: unknown error')
  })

  it('dismiss button clears the store result and hides the strip', async () => {
    const w = mountStrip()
    const handoff = useGraphHandoffStore()
    handoff.lastResult = { status: 'failed', reason: 'boom' }
    await w.vm.$nextTick()
    expect(w.find(STRIP).exists()).toBe(true)

    await w.get(DISMISS).trigger('click')

    expect(handoff.lastResult).toBeNull()
    await w.vm.$nextTick()
    expect(w.find(STRIP).exists()).toBe(false)
  })

  it('reactively re-shows when a new failure lands after a dismiss', async () => {
    const w = mountStrip()
    const handoff = useGraphHandoffStore()

    handoff.lastResult = { status: 'failed', reason: 'first' }
    await w.vm.$nextTick()
    await w.get(DISMISS).trigger('click')
    await w.vm.$nextTick()
    expect(w.find(STRIP).exists()).toBe(false)

    handoff.lastResult = { status: 'failed', reason: 'second' }
    await w.vm.$nextTick()
    expect(w.get(STRIP).text()).toContain('second')
  })
})
