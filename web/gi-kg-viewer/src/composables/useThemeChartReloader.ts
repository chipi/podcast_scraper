import { watch } from 'vue'
import { useThemeStore } from '../stores/theme'

/**
 * Rebuild Chart.js instances when the viewer theme changes so tick/grid colors
 * match the active ``data-theme`` token values (canvas does not inherit CSS).
 */
export function useThemeChartReloader(rebuild: () => void): void {
  const theme = useThemeStore()
  watch(
    () => theme.choice,
    () => {
      queueMicrotask(() => rebuild())
    },
  )
}
