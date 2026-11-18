import type { ChartType } from 'chart.js'

/**
 * Bar end labels use `setBarEndValueFormatter` in `chartBarEndValuePlugin.ts` — not `options.plugins`.
 */
declare module 'chart.js' {
  // Merging into Chart.js's interface needs its type parameter under the same name.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface PluginOptionsByType<TType extends ChartType = ChartType> {
    barEndValue?: Record<string, never>
  }
}

export {}
