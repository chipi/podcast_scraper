import type { ChartType } from 'chart.js'

/**
 * Bar end labels use `setBarEndValueFormatter` in `chartBarEndValuePlugin.ts` — not `options.plugins`.
 */
declare module 'chart.js' {
  interface PluginOptionsByType<TType extends ChartType = ChartType> {
    barEndValue?: Record<string, never>
  }
}

export {}
