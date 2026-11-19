// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import ReplayControls from './ReplayControls.vue'
import { useGraphReplayStore } from '../../stores/graphReplay'

const SESSION = [
  { action: 'graph_rail_nav', to_id: 'topic:a' },
  { action: 'graph_rail_nav', to_id: 'person:b' },
]

describe('ReplayControls', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('shows the REPLAY banner + step, and next/prev/exit drive the store', async () => {
    const r = useGraphReplayStore()
    r.load('s1', SESSION)
    const w = mount(ReplayControls)
    expect(w.find('[data-testid="graph-replay-bar"]').exists()).toBe(true)
    expect(w.find('[data-testid="replay-step"]').text()).toContain('step 0 / 2')
    await w.get('[data-testid="replay-next"]').trigger('click')
    expect(r.step).toBe(1)
    expect(w.find('[data-testid="replay-step"]').text()).toContain('step 1 / 2')
    await w.get('[data-testid="replay-prev"]').trigger('click')
    expect(r.step).toBe(0)
    await w.get('[data-testid="replay-exit"]').trigger('click')
    expect(r.active).toBe(false)
  })

  it('the scrub slider sets the step', async () => {
    const r = useGraphReplayStore()
    r.load('s1', SESSION)
    const w = mount(ReplayControls)
    await w.get('[data-testid="replay-scrub"]').setValue('2')
    expect(r.step).toBe(2)
  })
})
