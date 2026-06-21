// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

const fetchOpsSummary = vi.fn()
vi.mock('../../api/opsApi', () => ({
  fetchOpsSummary: (...a: unknown[]) => fetchOpsSummary(...a),
}))

import OpsView from './OpsView.vue'

afterEach(() => {
  fetchOpsSummary.mockReset()
})

describe('OpsView', () => {
  it('renders source cards with the right bucket status + summary line', async () => {
    fetchOpsSummary.mockResolvedValue({
      target: 'default',
      live: ['health'],
      unconfigured: ['cost'],
      failed: ['alerts'],
      sources: {
        health: { ok: true, source: 'prod_api.health', data: { status: 'ok' } },
        cost: { ok: false, source: 'loki.cost', configured: false, error: 'not set' },
        alerts: { ok: false, source: 'grafana.alerts', configured: true, error: '401' },
      },
    })
    const w = mount(OpsView)
    await flushPromises()
    expect(w.get('[data-testid="ops-source-health"]').exists()).toBe(true)
    expect(w.get('[data-testid="ops-status-health"]').text()).toBe('live')
    expect(w.get('[data-testid="ops-status-cost"]').text()).toBe('unconfigured')
    expect(w.get('[data-testid="ops-status-alerts"]').text()).toBe('failed')
    expect(w.text()).toContain('status: ok')
    expect(w.text()).toContain('not configured') // cost: configured=false summary
  })

  it('shows an error when the fetch fails', async () => {
    fetchOpsSummary.mockRejectedValue(new Error('Network error'))
    const w = mount(OpsView)
    await flushPromises()
    expect(w.get('[data-testid="ops-error"]').text()).toContain('Network error')
  })
})
