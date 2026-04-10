/**
 * Central Chart.js registration (tree-shaken controllers/elements).
 * Call `ensureChartJsRegistered()` before constructing any Chart instance.
 */
import {
  ArcElement,
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  DoughnutController,
  Filler,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip,
} from 'chart.js'

let registered = false

export function ensureChartJsRegistered(): void {
  if (registered) {
    return
  }
  Chart.register(
    ArcElement,
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
    DoughnutController,
  )
  registered = true
}
