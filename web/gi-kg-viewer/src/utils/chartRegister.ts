/**
 * Central Chart.js registration (tree-shaken controllers/elements).
 * Call `ensureChartJsRegistered()` before constructing any Chart instance.
 */
import {
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  Filler,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip,
  type Chart as ChartInstance,
} from 'chart.js'

let registered = false

const endLabelPlugin = {
  id: 'psEndLabel',
  afterDraw(chart: ChartInstance) {
    const kind = (chart.config as { type?: string }).type
    if (kind !== 'line') {
      return
    }
    const ctx = chart.ctx
    chart.data.datasets.forEach((dataset, i) => {
      const meta = chart.getDatasetMeta(i)
      if (meta.hidden || !meta.data.length) {
        return
      }
      const lastPoint = meta.data[meta.data.length - 1] as { x?: number; y?: number; skip?: boolean }
      if (!lastPoint || lastPoint.skip || typeof lastPoint.x !== 'number' || typeof lastPoint.y !== 'number') {
        return
      }
      const x = lastPoint.x
      const y = lastPoint.y
      const color =
        typeof dataset.borderColor === 'string'
          ? dataset.borderColor
          : Array.isArray(dataset.borderColor)
            ? String(dataset.borderColor[0] ?? '')
            : 'var(--ps-muted)'
      ctx.save()
      ctx.fillStyle = color || 'var(--ps-muted)'
      ctx.font = '10px Inter, system-ui, sans-serif'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'middle'
      ctx.fillText(String(dataset.label ?? ''), x + 6, y + 3)
      ctx.restore()
    })
  },
}

function applyTufteDefaults(): void {
  Chart.defaults.plugins.legend.display = false
  Chart.defaults.plugins.tooltip.enabled = true

  Chart.defaults.scales.linear = {
    ...Chart.defaults.scales.linear,
    grid: { display: false },
    border: { display: true } as (typeof Chart.defaults.scales.linear)['border'],
    ticks: {
      ...Chart.defaults.scales.linear.ticks,
      maxTicksLimit: 5,
      color: 'var(--ps-muted)',
    },
  }
  Chart.defaults.scales.category = {
    ...Chart.defaults.scales.category,
    grid: { display: false },
    border: { display: true } as (typeof Chart.defaults.scales.category)['border'],
    ticks: {
      ...Chart.defaults.scales.category.ticks,
      color: 'var(--ps-muted)',
    },
  }
}

export function ensureChartJsRegistered(): void {
  if (registered) {
    return
  }
  // Register built-ins first so `Chart.defaults.plugins.*` exists (Chart.js v4 tree-shaking).
  Chart.register(
    BarController,
    BarElement,
    CategoryScale,
    LinearScale,
    LineController,
    LineElement,
    PointElement,
    Filler,
    Tooltip,
    Legend,
    endLabelPlugin,
  )
  applyTufteDefaults()
  registered = true
}
