// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import GraphAnalyticsAdminView from './GraphAnalyticsAdminView.vue'
import * as authApi from '../../api/authApi'

vi.mock('../../api/authApi', () => ({
  fetchGraphAnalyticsSummary: vi.fn(),
  fetchGraphSessions: vi.fn(),
  fetchGraphSession: vi.fn(),
}))
const fetchSummary = vi.mocked(authApi.fetchGraphAnalyticsSummary)
const fetchSessions = vi.mocked(authApi.fetchGraphSessions)
const fetchSession = vi.mocked(authApi.fetchGraphSession)

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
    fetchSessions.mockResolvedValue([])
    fetchSession.mockResolvedValue([])
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

  it('lists sessions and shows a step-by-step timeline on click', async () => {
    fetchSessions.mockResolvedValue([
      { session_id: 's1', user_id: 'u1', started: 1, ended: 2, count: 3, size_min: 5, size_max: 40 },
    ])
    fetchSession.mockResolvedValue([
      { action: 'graph_node_tap', kind: 'topic' },
      { action: 'graph_rail_nav', to_kind: 'person', trail_size: 1 },
      { action: 'graph_redraw', nodes: 20, edges: 30 },
    ])
    const w = mount(GraphAnalyticsAdminView)
    await flushPromises()
    expect(w.find('[data-testid="ga-sessions"]').text()).toContain('u1')
    await w.get('[data-testid="ga-session-s1"]').trigger('click')
    await flushPromises()
    const tl = w.find('[data-testid="ga-timeline"]').text()
    expect(tl).toContain('tapped topic')
    expect(tl).toContain('navigated → person')
    expect(tl).toContain('20 nodes')
  })

  it('emits replay with the session id', async () => {
    fetchSessions.mockResolvedValue([
      { session_id: 's1', user_id: 'u1', started: 1, ended: 2, count: 3, size_min: 5, size_max: 40 },
    ])
    const w = mount(GraphAnalyticsAdminView)
    await flushPromises()
    await w.get('[data-testid="ga-replay"]').trigger('click')
    expect(w.emitted('replay')?.[0]).toEqual(['s1'])
  })
})
