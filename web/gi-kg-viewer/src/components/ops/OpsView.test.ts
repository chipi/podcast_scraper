// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

const fetchOpsSummary = vi.fn()
const fetchResilience = vi.fn()
const resetResilience = vi.fn()
const fetchUsage = vi.fn()
vi.mock('../../api/opsApi', () => ({
  fetchOpsSummary: (...a: unknown[]) => fetchOpsSummary(...a),
  fetchResilience: (...a: unknown[]) => fetchResilience(...a),
  resetResilience: (...a: unknown[]) => resetResilience(...a),
  fetchUsage: (...a: unknown[]) => fetchUsage(...a),
}))

import OpsView from './OpsView.vue'

const CLEAR_RESILIENCE = {
  llm_breakers: { openai: { open: false, recent_failures: 0, cooldown_remaining_seconds: 0, trips_total: 0 } },
  llm_breakers_open: [],
  rss: {},
  fuses: { llm_max_calls_per_episode: 500, llm_max_calls_per_run: 8000, note: 'x' },
  any_open: false,
}

const USAGE = {
  group_by: ['provider', 'model'],
  total: {
    calls: 3,
    input_tokens: 3000,
    output_tokens: 300,
    cached_input_tokens: 800,
    cache_write_tokens: 0,
    estimated_cost_usd: 0.0123,
    guardrail_calls: 0,
  },
  groups: [
    {
      provider: 'openai',
      model: 'gpt-5.4-mini',
      calls: 2,
      input_tokens: 2000,
      output_tokens: 200,
      cached_input_tokens: 800,
      cache_write_tokens: 0,
      estimated_cost_usd: 0.01,
      guardrail_calls: 0,
    },
  ],
  dimensions: ['provider', 'model', 'operation', 'episode_id'],
  source_files: ['x/run.log'],
  run_id: null,
  uninstrumented: false,
}

afterEach(() => {
  fetchOpsSummary.mockReset()
  fetchResilience.mockReset()
  resetResilience.mockReset()
  fetchUsage.mockReset()
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

describe('OpsView resilience panel (ADR-113)', () => {
  it("shows 'all clear' and no reset button when nothing is open", async () => {
    fetchOpsSummary.mockResolvedValue({ target: 't', live: [], unconfigured: [], failed: [], sources: {} })
    fetchResilience.mockResolvedValue(CLEAR_RESILIENCE)
    const w = mount(OpsView)
    await flushPromises()
    expect(w.get('[data-testid="resilience-status"]').text()).toBe('all clear')
    expect(w.find('[data-testid="resilience-reset"]').exists()).toBe(false)
    expect(w.text()).toContain('500/episode')
  })

  it('shows an open breaker with cooldown and a reset button when backing off', async () => {
    fetchOpsSummary.mockResolvedValue({ target: 't', live: [], unconfigured: [], failed: [], sources: {} })
    fetchResilience.mockResolvedValue({
      ...CLEAR_RESILIENCE,
      llm_breakers: {
        gemini: { open: true, recent_failures: 3, cooldown_remaining_seconds: 42.4, trips_total: 5 },
      },
      llm_breakers_open: ['gemini'],
      any_open: true,
    })
    const w = mount(OpsView)
    await flushPromises()
    expect(w.get('[data-testid="resilience-status"]').text()).toBe('backing off')
    expect(w.get('[data-testid="resilience-breaker-gemini"]').text()).toContain('43s')
    expect(w.find('[data-testid="resilience-reset"]').exists()).toBe(true)
  })

  it('resets breakers and refetches on click', async () => {
    fetchOpsSummary.mockResolvedValue({ target: 't', live: [], unconfigured: [], failed: [], sources: {} })
    fetchResilience
      .mockResolvedValueOnce({ ...CLEAR_RESILIENCE, llm_breakers_open: ['openai'], any_open: true })
      .mockResolvedValueOnce(CLEAR_RESILIENCE)
    resetResilience.mockResolvedValue(undefined)
    const w = mount(OpsView)
    await flushPromises()
    await w.get('[data-testid="resilience-reset"]').trigger('click')
    await flushPromises()
    expect(resetResilience).toHaveBeenCalledWith('all')
    expect(w.get('[data-testid="resilience-status"]').text()).toBe('all clear')
  })
})

describe('OpsView usage panel (token/cost)', () => {
  const baseMocks = () => {
    fetchOpsSummary.mockResolvedValue({ target: 't', live: [], unconfigured: [], failed: [], sources: {} })
    fetchResilience.mockResolvedValue(CLEAR_RESILIENCE)
  }

  it('renders the rollup total and a per-group row', async () => {
    baseMocks()
    fetchUsage.mockResolvedValue(USAGE)
    const w = mount(OpsView)
    await flushPromises()
    expect(w.get('[data-testid="usage-total"]').text()).toContain('$0.0123')
    expect(w.get('[data-testid="usage-total"]').text()).toContain('800 cached')
    const rows = w.findAll('[data-testid="usage-row"]')
    expect(rows).toHaveLength(1)
    expect(rows[0].text()).toContain('gpt-5.4-mini')
  })

  it('re-fetches with the chosen group_by dimension on click', async () => {
    baseMocks()
    fetchUsage.mockResolvedValue({ ...USAGE, group_by: ['operation'] })
    const w = mount(OpsView)
    await flushPromises()
    await w.get('[data-testid="usage-groupby-operation"]').trigger('click')
    await flushPromises()
    expect(fetchUsage).toHaveBeenLastCalledWith('operation')
  })

  it('shows the uninstrumented warning instead of zeroing cost', async () => {
    baseMocks()
    fetchUsage.mockResolvedValue({ ...USAGE, uninstrumented: true })
    const w = mount(OpsView)
    await flushPromises()
    expect(w.find('[data-testid="usage-uninstrumented"]').exists()).toBe(true)
  })
})
