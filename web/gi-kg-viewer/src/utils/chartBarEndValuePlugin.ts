/**
 * Draws formatted values at the end of horizontal bar segments (Tufte: label the data).
 */
import type { Chart, Plugin } from 'chart.js'

export type BarEndValueFormat = (raw: number, dataIndex: number) => string

/**
 * Chart.js resolves any function under `options.plugins.*` as scriptable; passing bar formatters
 * there breaks at runtime. Attach real formatters here and read them from the plugin.
 */
const barEndFormatters = new WeakMap<Chart, BarEndValueFormat>()

export function setBarEndValueFormatter(chart: Chart, fn: BarEndValueFormat | null): void {
  if (fn == null) {
    barEndFormatters.delete(chart)
  } else {
    barEndFormatters.set(chart, fn)
  }
}

export const barEndValuePlugin: Plugin<'bar'> = {
  id: 'barEndValue',

  afterDatasetsDraw(chart: Chart<'bar'>) {
    if (chart.options.indexAxis !== 'y') {
      return
    }
    const rawFmt = barEndFormatters.get(chart)
    const fmt: BarEndValueFormat =
      typeof rawFmt === 'function'
        ? rawFmt
        : (v: number) => (Number.isFinite(v) ? v.toLocaleString() : '—')
    const { ctx, chartArea } = chart
    if (!chartArea) {
      return
    }
    ctx.save()
    ctx.font = '10px system-ui, -apple-system, sans-serif'
    ctx.fillStyle = getComputedStyle(document.documentElement)
      .getPropertyValue('--ps-muted')
      .trim() || '#888'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'

    chart.data.datasets.forEach((dataset, di) => {
      const meta = chart.getDatasetMeta(di)
      if (meta.hidden) {
        return
      }
      meta.data.forEach((el, j) => {
        if (!el || typeof el.getProps !== 'function') {
          return
        }
        const props = el.getProps(['x', 'y'], true)
        const raw = dataset.data[j]
        const v = typeof raw === 'number' ? raw : Number(raw)
        const text = fmt(v, j)
        const pad = 4
        ctx.fillText(text, Math.min(props.x + pad, chartArea.right - 2), props.y)
      })
    })
    ctx.restore()
  },
}
