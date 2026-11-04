// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { graphPanels } from '../../extensions'
import { useArtifactsStore } from '../../stores/artifacts'
import GraphTabPanel from './GraphTabPanel.vue'

vi.mock('../../lib/analytics', () => ({ track: vi.fn() }))

const stubs = { GraphCanvas: { template: '<div data-testid="canvas" />' }, HandoffErrorStrip: true }

describe('GraphTabPanel registered panels', () => {
  beforeEach(() => setActivePinia(createPinia()))
  afterEach(() => graphPanels.reset())

  it('shows notices always and above-canvas panels only with a graph, before the canvas', async () => {
    graphPanels.register({ id: 'n', region: 'notice', component: { template: '<p data-testid="notice" />' } })
    graphPanels.register({ id: 'a', region: 'above-canvas', component: { template: '<p data-testid="above" />' } })
    const w = mount(GraphTabPanel, { global: { stubs } })
    expect(w.find('[data-testid="notice"]').exists()).toBe(true)
    expect(w.find('[data-testid="above"]').exists()).toBe(false)
    const artifacts = useArtifactsStore()
    vi.spyOn(artifacts, 'displayArtifact', 'get').mockReturnValue({ name: 'x' } as never)
    w.unmount()
    const w2 = mount(GraphTabPanel, { global: { stubs } })
    const above = w2.get('[data-testid="above"]').element
    const canvas = w2.get('[data-testid="canvas"]').element
    expect(above.compareDocumentPosition(canvas) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    w2.unmount()
  })
})
