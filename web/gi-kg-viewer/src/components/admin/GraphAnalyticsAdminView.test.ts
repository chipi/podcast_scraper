// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import GraphAnalyticsAdminView from './GraphAnalyticsAdminView.vue'
import * as authApi from '../../api/authApi'

vi.mock('../../api/authApi', () => ({ fetchGraphAnalyticsSummary: vi.fn() }))
const fetchSummary = vi.mocked(authApi.fetchGraphAnalyticsSummary)

const SUMMARY = {
  total_events: 42,
  users: 3,
  by_action: { graph_node_tap: 20, graph_redraw: 15, graph_broke: 2 },
  node_taps_by_kind: { topic: 12, person: 8 },
  size: {
    samples: 15,
    nodes: { min: 5, avg: 22.3, max: 60, p50: 20, p95: 55 },
    edges: { min: 4, avg: 30, max: 80, p50: 28, p95: 70 },
    trail: { min: 0, avg: 3, max: 12, p50: 2, p95: 9 },
  },
  breakage: { count: 2, by_reason: { 'stuck-timeout': 2 } },
}
const clone = () => JSON.parse(JSON.stringify(SUMMARY)) as typeof SUMMARY

describe('GraphAnalyticsAdminView', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    fetchSummary.mockResolvedValue(clone())
  })

  it('renders usage, size and breakage from the summary', async () => {
    const w = mount(GraphAnalyticsAdminView)
    await flushPromises()
    expect(w.find('[data-testid="ga-total"]').text()).toBe('42')
    expect(w.find('[data-testid="ga-users"]').text()).toBe('3')
    expect(w.find('[data-testid="ga-actions"]').text()).toContain('graph_node_tap')
    expect(w.find('[data-testid="ga-taps"]').text()).toContain('topic')
    expect(w.find('[data-testid="ga-size-nodes"]').text()).toContain('60') // max node count
    expect(w.find('[data-testid="ga-breakage"]').text()).toContain('stuck-timeout')
  })

  it('surfaces a load error', async () => {
    fetchSummary.mockRejectedValue(new Error('nope'))
    const w = mount(GraphAnalyticsAdminView)
    await flushPromises()
    expect(w.find('[data-testid="graph-analytics-error"]').text()).toContain('nope')
  })
})
