// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CoverageByMonthItem } from '../../api/corpusCoverageApi'
import CoverageByMonthChart from './CoverageByMonthChart.vue'

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

let getContext: ReturnType<typeof vi.spyOn>

/** A month with `with_kg` covered (the chart's default layer); `with_gi` holds a decoy. */
function m(month: string, total: number, with_kg: number): CoverageByMonthItem {
  return { month, total, with_gi: total, with_kg, with_both: 0 }
}

describe('CoverageByMonthChart', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    h.instances.length = 0
    getContext = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as never)
  })
  afterEach(() => {
    getContext.mockRestore()
  })

  it('shows the empty message when there are no rows with episodes', () => {
    const w = mount(CoverageByMonthChart, { props: { rows: [m('2026-01', 0, 0)] } })
    expect(w.text()).toContain('No episode metadata found.')
    expect(w.get('h3').text()).toBe('KG coverage by month')
    expect(w.find('canvas').exists()).toBe(false)
    expect(h.instances).toHaveLength(0)
  })

  it('flags months below the average and names the lowest one', () => {
    const w = mount(CoverageByMonthChart, {
      props: { rows: [m('2026-01', 10, 10), m('2026-02', 10, 2), m('2026-03', 10, 9)] },
    })
    // 100, 20, 90 -> avg 70 -> only 2026-02 below.
    expect(w.get('h3').text()).toBe('1 months below average KG coverage')
    expect(w.text()).toContain('Lowest: 2026-02 — 20%')
    expect(w.text()).toContain('1 month below average — 2026-02 need attention')
  })

  it('pluralises the below-average insight and lists every month', () => {
    const w = mount(CoverageByMonthChart, {
      props: { rows: [m('2026-01', 10, 10), m('2026-02', 10, 10), m('2026-03', 10, 0), m('2026-04', 10, 1)] },
    })
    expect(w.text()).toContain('2 months below average — 2026-03, 2026-04 need attention')
  })

  it('reports a trend when every month sits at the average', () => {
    const w = mount(CoverageByMonthChart, {
      props: { rows: [m('2026-01', 10, 5), m('2026-02', 10, 5)] },
    })
    expect(w.get('h3').text()).toBe('KG coverage by month — at or above average')
    expect(w.text()).toContain('Coverage stable: 0pp from 2026-01 to 2026-02')
  })

  it('does not write a trend insight for a single month', () => {
    const w = mount(CoverageByMonthChart, { props: { rows: [m('2026-01', 10, 5)] } })
    expect(w.text()).not.toContain('Coverage ')
    expect(w.text()).toContain('Lowest: 2026-01 — 50%')
  })

  it('builds a bar chart of KG percentages, colouring below-average months differently', () => {
    document.documentElement.style.setProperty('--ps-kg', '#112233')
    document.documentElement.style.setProperty('--ps-warning', '#aa5500')
    mount(CoverageByMonthChart, {
      props: { rows: [m('2026-01', 10, 10), m('2026-02', 10, 2), m('2026-00', 0, 0)] },
    })
    const cfg = h.instances[0].config
    expect(cfg.data.labels).toEqual(['2026-01', '2026-02'])
    expect(cfg.data.datasets[0].data).toEqual([100, 20])
    const [ok, low] = cfg.data.datasets[0].backgroundColor
    expect(ok).toBe('rgba(17,34,51,0.75)')
    expect(low).toBe('rgba(170,85,0,0.75)')
    expect(cfg.options.scales.y.max).toBe(100)
  })

  it('does not build when the 2d context is unavailable', () => {
    getContext.mockReturnValue(null as never)
    mount(CoverageByMonthChart, { props: { rows: [m('2026-01', 10, 5)] } })
    expect(h.instances).toHaveLength(0)
  })

  it('formats the tooltip as a percentage', () => {
    mount(CoverageByMonthChart, { props: { rows: [m('2026-01', 10, 5)] } })
    const label = h.instances[0].config.options.plugins.tooltip.callbacks.label
    expect(label({ parsed: { y: 66.6 } })).toBe('67% KG')
    expect(label({ parsed: { y: null } })).toBe('null')
  })

  it('emits select-month for a clicked bar and ignores clicks on empty space', () => {
    const w = mount(CoverageByMonthChart, {
      props: { rows: [m('2026-01', 10, 5), m('2026-02', 10, 6)] },
    })
    const onClick = h.instances[0].config.options.onClick
    onClick({}, [])
    onClick({}, [{ index: 1 }])
    onClick({}, [{ index: 7 }])
    expect(w.emitted('select-month')).toEqual([['2026-02']])
  })

  it('draws the average line plugin', () => {
    mount(CoverageByMonthChart, { props: { rows: [m('2026-01', 10, 10), m('2026-02', 10, 0)] } })
    const plugin = h.instances[0].config.plugins[0]
    expect(plugin.id).toBe('psAvgLine')
    const ctx = {
      save: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      fillText: vi.fn(),
    } as Record<string, any>
    const getPixelForValue = vi.fn(() => 40)
    plugin.afterDatasetsDraw({
      ctx,
      scales: { y: { getPixelForValue }, x: { left: 5, right: 95 } },
    })
    expect(getPixelForValue).toHaveBeenCalledWith(50)
    expect(ctx.moveTo).toHaveBeenCalledWith(5, 40)
    expect(ctx.lineTo).toHaveBeenCalledWith(95, 40)
    expect(ctx.fillText).toHaveBeenCalledWith('avg 50%', 95, 36)

    ctx.moveTo.mockClear()
    plugin.afterDatasetsDraw({ ctx, scales: {} })
    expect(ctx.moveTo).not.toHaveBeenCalled()
  })

  it('rebuilds when rows change and destroys on unmount', async () => {
    const w = mount(CoverageByMonthChart, { props: { rows: [m('2026-01', 10, 5)] } })
    await w.setProps({ rows: [m('2026-01', 10, 5), m('2026-02', 10, 6)] })
    expect(h.instances).toHaveLength(2)
    expect(h.instances[0].destroy).toHaveBeenCalledTimes(1)
    w.unmount()
    expect(h.instances[1].destroy).toHaveBeenCalledTimes(1)
  })

  it('charts another layer from the count, label, token and testid suffix it is given', () => {
    document.documentElement.style.setProperty('--ps-probe', '#010203')
    const w = mount(CoverageByMonthChart, {
      props: {
        rows: [m('2026-01', 10, 0), m('2026-02', 10, 0)],
        count: (r: CoverageByMonthItem) => r.with_gi / 2,
        label: 'Probe',
        token: '--ps-probe',
        testidSuffix: 'probe',
      },
    })
    expect(w.find('[data-testid="coverage-by-month-chart-probe"]').exists()).toBe(true)
    expect(w.get('h3').text()).toBe('Probe coverage by month — at or above average')
    const cfg = h.instances[0].config
    expect(cfg.data.datasets[0].data).toEqual([50, 50])
    expect(cfg.data.datasets[0].backgroundColor[0]).toBe('rgba(1,2,3,0.75)')
    expect(cfg.options.plugins.tooltip.callbacks.label({ parsed: { y: 50 } })).toBe('50% Probe')
  })
})
