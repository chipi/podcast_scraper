// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { graphPanels } from '../../extensions'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphExpansionStore } from '../../stores/graphExpansion'
import GraphTabPanel from './GraphTabPanel.vue'

vi.mock('../../lib/analytics', () => ({ track: vi.fn() }))

const clearInteractionState = vi.hoisted(() => vi.fn())

vi.mock('./GraphCanvas.vue', async () => {
  const { defineComponent, h } = await import('vue')
  return {
    default: defineComponent({
      name: 'GraphCanvasStub',
      emits: ['request-corpus-graph-sync', 'request-graph-full-reset'],
      setup(_p, { expose }) {
        expose({ clearInteractionState })
        return () => h('div', { 'data-testid': 'canvas-stub' })
      },
    }),
  }
})

const globalStubs = { HandoffErrorStrip: true }

function withArtifact(): void {
  const artifacts = useArtifactsStore()
  vi.spyOn(artifacts, 'displayArtifact', 'get').mockReturnValue({ name: 'x' } as never)
}

describe('GraphTabPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    clearInteractionState.mockClear()
  })
  afterEach(() => graphPanels.reset())

  it('shows the load-a-graph placeholder and no canvas without an artifact', () => {
    const w = mount(GraphTabPanel, { global: { stubs: globalStubs } })
    expect(w.find('[data-testid="canvas-stub"]').exists()).toBe(false)
    expect(w.text()).toContain('Corpus path')
    expect(w.text()).toContain('graph time slice')
    expect(w.text()).toMatch(/\d+ episodes max/)
  })

  it('renders the canvas instead of the placeholder when an artifact is displayed', () => {
    withArtifact()
    const w = mount(GraphTabPanel, { global: { stubs: globalStubs } })
    expect(w.find('[data-testid="canvas-stub"]').exists()).toBe(true)
    expect(w.text()).not.toContain('graph time slice')
  })

  it('shows the expansion truncation line and clears it on dismiss', async () => {
    const exp = useGraphExpansionStore()
    const w = mount(GraphTabPanel, { global: { stubs: globalStubs } })
    expect(w.find('[data-testid="graph-expansion-truncation-line"]').exists()).toBe(false)
    exp.setTruncationLine('Showing 40 of 90 episodes')
    await w.vm.$nextTick()
    expect(w.get('[data-testid="graph-expansion-truncation-line"]').text()).toContain(
      'Showing 40 of 90 episodes',
    )
    await w.get('[data-testid="graph-expansion-truncation-dismiss"]').trigger('click')
    expect(exp.truncationLine).toBeNull()
    expect(w.find('[data-testid="graph-expansion-truncation-line"]').exists()).toBe(false)
  })

  it('re-emits the canvas sync and full-reset requests', async () => {
    withArtifact()
    const w = mount(GraphTabPanel, { global: { stubs: globalStubs } })
    const canvas = w.findComponent({ name: 'GraphCanvasStub' })
    canvas.vm.$emit('request-corpus-graph-sync')
    canvas.vm.$emit('request-graph-full-reset')
    expect(w.emitted('request-corpus-graph-sync')).toHaveLength(1)
    expect(w.emitted('request-graph-full-reset')).toHaveLength(1)
  })

  it('forwards clearInteractionState to the canvas, passing options through', () => {
    withArtifact()
    const w = mount(GraphTabPanel, { global: { stubs: globalStubs } })
    ;(w.vm as unknown as { clearInteractionState: (o?: object) => void }).clearInteractionState({
      skipRedraw: true,
    })
    expect(clearInteractionState).toHaveBeenCalledWith({ skipRedraw: true })
  })

  it('tolerates clearInteractionState when no canvas is mounted', () => {
    const w = mount(GraphTabPanel, { global: { stubs: globalStubs } })
    expect(() =>
      (w.vm as unknown as { clearInteractionState: () => void }).clearInteractionState(),
    ).not.toThrow()
    expect(clearInteractionState).not.toHaveBeenCalled()
  })

  it('renders notice panels always, and above-canvas panels only with an artifact', () => {
    graphPanels.register({ id: 'n', region: 'notice', component: { template: '<p data-testid="notice" />' } })
    graphPanels.register({ id: 'a', region: 'above-canvas', component: { template: '<p data-testid="above" />' } })
    const without = mount(GraphTabPanel, { global: { stubs: globalStubs } })
    expect(without.find('[data-testid="notice"]').exists()).toBe(true)
    expect(without.find('[data-testid="above"]').exists()).toBe(false)
    without.unmount()

    withArtifact()
    const withGraph = mount(GraphTabPanel, { global: { stubs: globalStubs } })
    expect(withGraph.find('[data-testid="notice"]').exists()).toBe(true)
    expect(withGraph.find('[data-testid="above"]').exists()).toBe(true)
  })
})
