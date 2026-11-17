/**
 * Draws each line series label at its last point (Tufte: direct labeling vs legend).
 */
import type { Chart, Plugin } from 'chart.js'

export const lineEndLabelsPlugin: Plugin = {
  id: 'lineEndLabels',

  afterDatasetsDraw(chart: Chart) {
    const { ctx, chartArea } = chart
    if (!chartArea) {
      return
    }
    ctx.save()
    ctx.font = '10px system-ui, -apple-system, sans-serif'
    ctx.textBaseline = 'middle'

    const lineCount = chart.data.datasets.filter(
      (_, i) => chart.getDatasetMeta(i).type === 'line',
    ).length

    chart.data.datasets.forEach((dataset, i) => {
      const meta = chart.getDatasetMeta(i)
      if (meta.type !== 'line' || meta.hidden) {
        return
      }
      const pts = meta.data
      if (!pts?.length) {
        return
      }
      const last = pts[pts.length - 1]
      if (!last || typeof last.x !== 'number' || typeof last.y !== 'number') {
        return
      }
      const stroke = (dataset as { borderColor?: string | string[] }).borderColor
      const color = typeof stroke === 'string' ? stroke : '#888'
      ctx.fillStyle = color
      ctx.textAlign = 'left'
      let label = dataset.label ?? ''
      if (label.length > 30) {
        label = `${label.slice(0, 29)}…`
      }
      const yStack = (i - (lineCount - 1) / 2) * 11
      const x = Math.min(last.x + 5, chartArea.right - 4)
      ctx.fillText(label, x, last.y + yStack)
    })
    ctx.restore()
  },
}
