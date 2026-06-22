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

  it('shows a loading state before data arrives', async () => {
    fetchOpsSummary.mockImplementation(() => new Promise(() => {})) // never resolves
    const w = mount(OpsView)
    await flushPromises() // let onMounted's refresh() flip loading=true and re-render
    expect(w.find('[data-testid="ops-loading"]').exists()).toBe(true)
  })

  it('formats per-source summary lines', async () => {
    fetchOpsSummary.mockResolvedValue({
      target: 'prod',
      live: ['deploys', 'cost', 'errors', 'alerts'],
      unconfigured: [],
      failed: [],
      sources: {
        deploys: { ok: true, source: 'github.deploys', data: { count: 2, failure_rate: 0.5 } },
        cost: { ok: true, source: 'loki.cost', data: { estimated_cost_usd: 0.5142 } },
        errors: { ok: true, source: 'sentry.errors', data: { total_issues: 3 } },
        alerts: { ok: true, source: 'grafana.alerts', data: { firing: 1, count: 4 } },
      },
    })
    const w = mount(OpsView)
    await flushPromises()
    const text = w.text()
    expect(text).toContain('2 deploys · 50% fail')
    expect(text).toContain('$0.5142 (24h)')
    expect(text).toContain('3 unresolved')
    expect(text).toContain('1 firing / 4')
  })

  it("renders 'n/a' when cost is null", async () => {
    fetchOpsSummary.mockResolvedValue({
      target: 'prod',
      live: ['cost'],
      unconfigured: [],
      failed: [],
      sources: { cost: { ok: true, source: 'loki.cost', data: { estimated_cost_usd: null } } },
    })
    const w = mount(OpsView)
    await flushPromises()
    expect(w.get('[data-testid="ops-source-cost"]').text()).toContain('n/a')
  })
})
