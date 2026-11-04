// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useThemeStore } from '../../stores/theme'
import VerticalBarChart from './VerticalBarChart.vue'

const h = vi.hoisted(() => ({
  instances: [] as Array<{ config: any; destroy: ReturnType<typeof vi.fn> }>,
}))

vi.mock('chart.js', () => ({
  Chart: vi.fn().mockImplementation(function (this: any, _ctx: unknown, config: unknown) {
    this.config = config
    this.destroy = vi.fn()
    h.instances.push(this)
  }),
}))
vi.mock('../../utils/chartRegister', () => ({ ensureChartJsRegistered: vi.fn() }))

const ctx2d = {}
let getContext: ReturnType<typeof vi.spyOn>

describe('VerticalBarChart', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    h.instances.length = 0
    getContext = vi
      .spyOn(HTMLCanvasElement.prototype, 'getContext')
      .mockReturnValue(ctx2d as never)
  })
  afterEach(() => {
    getContext.mockRestore()
  })

  it('shows the title and help/insight text', () => {
    const w = mount(VerticalBarChart, {
      props: {
        title: 'Episodes per month',
        labels: ['2026-01'],
        values: [3],
        insightText: 'Busy January',
        helpText: 'Counts by publish month',
      },
    })
    expect(w.text()).toContain('Episodes per month')
    expect(w.text()).toContain('Busy January')
    expect(w.text()).toContain('Counts by publish month')
    expect(w.text()).not.toContain('No data to chart.')
  })

  it('renders the empty state and builds no chart without data', () => {
    const w = mount(VerticalBarChart, { props: { title: 'T', labels: [], values: [] } })
    expect(w.text()).toContain('No data to chart.')
    expect(w.find('canvas').exists()).toBe(false)
    expect(h.instances).toHaveLength(0)
  })

  it('treats labels without values as no data', () => {
    const w = mount(VerticalBarChart, { props: { title: 'T', labels: ['a'], values: [] } })
    expect(w.text()).toContain('No data to chart.')
  })

  it('builds a bar chart from labels and values', () => {
    mount(VerticalBarChart, {
      props: { title: 'T', labels: ['a', 'b'], values: [1, 2] },
      attachTo: document.body,
    })
    expect(h.instances).toHaveLength(1)
    const cfg = h.instances[0].config
    expect(cfg.type).toBe('bar')
    expect(cfg.data.labels).toEqual(['a', 'b'])
    expect(cfg.data.datasets[0].data).toEqual([1, 2])
    expect(cfg.options.scales.y.beginAtZero).toBe(true)
  })

  it('does not build when the 2d context is unavailable', () => {
    getContext.mockReturnValue(null as never)
    mount(VerticalBarChart, { props: { title: 'T', labels: ['a'], values: [1] } })
    expect(h.instances).toHaveLength(0)
  })

  it('formats tooltip title and label, honouring the y-axis label', () => {
    mount(VerticalBarChart, {
      props: { title: 'T', labels: ['2026-01', '2026-02'], values: [1, 2], yAxisLabel: 'Searches' },
    })
    const cb = h.instances[0].config.options.plugins.tooltip.callbacks
    expect(cb.title([{ dataIndex: 1 }])).toBe('Publish month: 2026-02')
    expect(cb.title([])).toBe('')
    expect(cb.title([{ dataIndex: 9 }])).toBe('Publish month: ')
    expect(cb.label({ parsed: { y: 1234 } })).toBe(`Searches: ${(1234).toLocaleString()}`)
    expect(cb.label({ parsed: { y: null } })).toBe('Searches: null')
  })

  it('defaults the tooltip series name to Episodes', () => {
    mount(VerticalBarChart, { props: { title: 'T', labels: ['a'], values: [5] } })
    const cb = h.instances[0].config.options.plugins.tooltip.callbacks
    expect(cb.label({ parsed: { y: 5 } })).toBe('Episodes: 5')
  })

  it('rebuilds (destroying the old chart) when props change', async () => {
    const w = mount(VerticalBarChart, { props: { title: 'T', labels: ['a'], values: [1] } })
    await w.setProps({ labels: ['a', 'b'], values: [1, 2] })
    expect(h.instances).toHaveLength(2)
    expect(h.instances[0].destroy).toHaveBeenCalledTimes(1)
    expect(h.instances[1].config.data.labels).toEqual(['a', 'b'])
  })

  it('drops the chart when data becomes empty', async () => {
    const w = mount(VerticalBarChart, { props: { title: 'T', labels: ['a'], values: [1] } })
    await w.setProps({ labels: [], values: [] })
    expect(h.instances).toHaveLength(1)
    expect(h.instances[0].destroy).toHaveBeenCalledTimes(1)
    expect(w.text()).toContain('No data to chart.')
  })

  it('rebuilds on theme change', async () => {
    mount(VerticalBarChart, { props: { title: 'T', labels: ['a'], values: [1] } })
    const theme = useThemeStore()
    theme.choice = theme.choice === 'dark' ? 'light' : 'dark'
    await Promise.resolve()
    await Promise.resolve()
    expect(h.instances.length).toBeGreaterThanOrEqual(2)
  })

  it('destroys the chart on unmount', () => {
    const w = mount(VerticalBarChart, { props: { title: 'T', labels: ['a'], values: [1] } })
    w.unmount()
    expect(h.instances[0].destroy).toHaveBeenCalledTimes(1)
  })
})
