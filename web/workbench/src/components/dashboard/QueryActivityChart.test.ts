// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchQueryActivity } from '../../api/queryActivityApi'
import { useShellStore } from '../../stores/shell'
import QueryActivityChart from './QueryActivityChart.vue'

vi.mock('../../api/queryActivityApi', () => ({ fetchQueryActivity: vi.fn() }))
vi.mock('../../lib/analytics', () => ({ track: vi.fn() }))

const mockFetch = vi.mocked(fetchQueryActivity)

const VerticalBarChartStub = {
  name: 'VerticalBarChart',
  props: ['title', 'labels', 'values', 'yAxisLabel', 'insightText', 'helpText'],
  template: '<div data-testid="vbc" />',
}

function setup(corpusPath: string | null = '/corpus', health: string | null = 'ok') {
  const shell = useShellStore()
  shell.corpusPath = corpusPath ?? ''
  shell.healthStatus = health
  return shell
}

async function mountChart() {
  const w = mount(QueryActivityChart, { global: { stubs: { VerticalBarChart: VerticalBarChartStub } } })
  await flushPromises()
  return w
}

describe('QueryActivityChart', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mockFetch.mockReset()
  })

  it('does not fetch or render without a corpus path', async () => {
    setup('   ')
    const w = await mountChart()
    expect(mockFetch).not.toHaveBeenCalled()
    expect(w.find('[data-testid="query-activity-chart"]').exists()).toBe(false)
  })

  it('does not fetch until the API health is known', async () => {
    setup('/corpus', null)
    const w = await mountChart()
    expect(mockFetch).not.toHaveBeenCalled()
    expect(w.find('[data-testid="query-activity-chart"]').exists()).toBe(false)
  })

  it('renders daily buckets with month-day labels and a total insight', async () => {
    setup()
    mockFetch.mockResolvedValue({
      total: 7,
      buckets: [
        { date: '2026-06-01', count: 3 },
        { date: '2026-06-02', count: 4 },
      ],
    })
    const w = await mountChart()
    expect(mockFetch).toHaveBeenCalledWith('/corpus', 30)
    const chart = w.findComponent({ name: 'VerticalBarChart' })
    expect(chart.props('labels')).toEqual(['06-01', '06-02'])
    expect(chart.props('values')).toEqual([3, 4])
    expect(chart.props('title')).toBe('Search activity (last 30 days)')
    expect(chart.props('insightText')).toBe('7 searches in the last 30 days.')
    expect(chart.props('yAxisLabel')).toBe('Searches')
  })

  it('renders nothing when no searches were logged', async () => {
    setup()
    mockFetch.mockResolvedValue({ total: 0, buckets: [{ date: '2026-06-01', count: 0 }] })
    const w = await mountChart()
    expect(w.find('[data-testid="query-activity-chart"]').exists()).toBe(false)
  })

  it('renders nothing when the request fails', async () => {
    setup()
    mockFetch.mockRejectedValue(new Error('boom'))
    const w = await mountChart()
    expect(w.find('[data-testid="query-activity-chart"]').exists()).toBe(false)
  })

  it('reloads when the corpus path changes and clears when it is emptied', async () => {
    const shell = setup()
    mockFetch.mockResolvedValue({ total: 2, buckets: [{ date: '2026-06-01', count: 2 }] })
    const w = await mountChart()
    expect(w.find('[data-testid="query-activity-chart"]').exists()).toBe(true)

    mockFetch.mockResolvedValue({ total: 5, buckets: [{ date: '2026-06-03', count: 5 }] })
    shell.corpusPath = '/other'
    await flushPromises()
    expect(mockFetch).toHaveBeenLastCalledWith('/other', 30)
    expect(w.findComponent({ name: 'VerticalBarChart' }).props('values')).toEqual([5])

    shell.corpusPath = ''
    await flushPromises()
    expect(w.find('[data-testid="query-activity-chart"]').exists()).toBe(false)
  })

  it('drops a stale response that resolves after a newer load started', async () => {
    const shell = setup()
    let resolveFirst!: (v: { total: number; buckets: { date: string; count: number }[] }) => void
    mockFetch.mockImplementationOnce(
      () => new Promise((r) => { resolveFirst = r }),
    )
    const w = mount(QueryActivityChart, { global: { stubs: { VerticalBarChart: VerticalBarChartStub } } })
    mockFetch.mockResolvedValueOnce({ total: 9, buckets: [{ date: '2026-06-09', count: 9 }] })
    shell.corpusPath = '/newer'
    await flushPromises()
    resolveFirst({ total: 1, buckets: [{ date: '2026-01-01', count: 1 }] })
    await flushPromises()
    expect(w.findComponent({ name: 'VerticalBarChart' }).props('values')).toEqual([9])
  })

  it('ignores a stale failure that arrives after a newer load', async () => {
    const shell = setup()
    let rejectFirst!: (e: Error) => void
    mockFetch.mockImplementationOnce(() => new Promise((_r, rej) => { rejectFirst = rej }))
    const w = mount(QueryActivityChart, { global: { stubs: { VerticalBarChart: VerticalBarChartStub } } })
    mockFetch.mockResolvedValueOnce({ total: 4, buckets: [{ date: '2026-06-09', count: 4 }] })
    shell.corpusPath = '/newer'
    await flushPromises()
    rejectFirst(new Error('late'))
    await flushPromises()
    expect(w.find('[data-testid="query-activity-chart"]').exists()).toBe(true)
  })
})
