// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import PipelineStageChart from './PipelineStageChart.vue'

const h = vi.hoisted(() => ({
  instances: [] as Array<{ config: any; destroy: ReturnType<typeof vi.fn>; getDatasetMeta?: any }>,
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

describe('PipelineStageChart', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    h.instances.length = 0
    getContext = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as never)
  })
  afterEach(() => {
    getContext.mockRestore()
  })

  it('uses a generic title and builds no chart for no stages', () => {
    const w = mount(PipelineStageChart, { props: { stages: [] } })
    expect(w.get('h3').text()).toBe('Stage timings')
    expect(h.instances).toHaveLength(0)
  })

  it('names the dominant stage and its share of total run time', () => {
    const w = mount(PipelineStageChart, {
      props: {
        stages: [
          { label: 'Scraping', seconds: 30 },
          { label: 'Parsing', seconds: 10 },
        ],
      },
    })
    expect(w.get('h3').text()).toBe('Scraping: 75% of total run time')
  })

  it('reports 0% when every stage is zero seconds, and builds no chart', () => {
    const w = mount(PipelineStageChart, {
      props: { stages: [{ label: 'Idle', seconds: 0 }] },
    })
    expect(w.get('h3').text()).toBe('Idle: 0% of total run time')
    expect(h.instances).toHaveLength(0)
  })

  it('charts stages sorted longest first, dropping zero-second stages', () => {
    mount(PipelineStageChart, {
      props: {
        stages: [
          { label: 'Parsing', seconds: 10 },
          { label: 'Zero', seconds: 0 },
          { label: 'Scraping', seconds: 30 },
        ],
      },
    })
    const cfg = h.instances[0].config
    expect(cfg.type).toBe('bar')
    expect(cfg.options.indexAxis).toBe('y')
    expect(cfg.data.labels).toEqual(['Scraping', 'Parsing'])
    expect(cfg.data.datasets[0].data).toEqual([30, 10])
    const [first, second] = cfg.data.datasets[0].backgroundColor
    expect(first).not.toBe(second)
  })

  it('does not build when the 2d context is unavailable', () => {
    getContext.mockReturnValue(null as never)
    mount(PipelineStageChart, { props: { stages: [{ label: 'A', seconds: 5 }] } })
    expect(h.instances).toHaveLength(0)
  })

  it('formats the tooltip as a human duration', () => {
    mount(PipelineStageChart, { props: { stages: [{ label: 'A', seconds: 5 }] } })
    const label = h.instances[0].config.options.plugins.tooltip.callbacks.label
    expect(label({ parsed: { x: 125 } })).toMatch(/2/)
  })

  it('draws a duration label at the end of each bar', () => {
    mount(PipelineStageChart, {
      props: {
        stages: [
          { label: 'A', seconds: 60 },
          { label: 'B', seconds: 30 },
        ],
      },
    })
    const plugin = h.instances[0].config.plugins[0]
    expect(plugin.id).toBe('barEndDur')
    const ctx = { save: vi.fn(), restore: vi.fn(), fillText: vi.fn() } as Record<string, any>
    const chart = {
      ctx,
      getDatasetMeta: () => ({
        data: [{ x: 100, y: 10 }, { x: 50, y: 30 }, { y: 1 }, null],
      }),
    }
    plugin.afterDatasetsDraw(chart)
    expect(ctx.save).toHaveBeenCalled()
    expect(ctx.restore).toHaveBeenCalled()
    // Only bars with a numeric x get text; text is offset 6px right of the bar end.
    expect(ctx.fillText).toHaveBeenCalledTimes(2)
    expect(ctx.fillText.mock.calls[0][1]).toBe(106)
    expect(ctx.fillText.mock.calls[0][2]).toBe(10)
    expect(ctx.fillText.mock.calls[1][1]).toBe(56)
  })

  it('rebuilds when stages change and destroys on unmount', async () => {
    const w = mount(PipelineStageChart, { props: { stages: [{ label: 'A', seconds: 5 }] } })
    await w.setProps({ stages: [{ label: 'A', seconds: 5 }, { label: 'B', seconds: 9 }] })
    expect(h.instances).toHaveLength(2)
    expect(h.instances[0].destroy).toHaveBeenCalledTimes(1)
    w.unmount()
    expect(h.instances[1].destroy).toHaveBeenCalledTimes(1)
  })
})
