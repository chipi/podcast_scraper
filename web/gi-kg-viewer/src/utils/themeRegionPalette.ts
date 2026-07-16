/**
 * Shared palette + stable hash for graph-v3 theme-cluster region tints.
 *
 * Kept in a plain util (not the Vue component or the Cytoscape stylesheet)
 * so GraphCanvas, GraphThemeLegend, and unit tests all resolve the same
 * hex for a given `thc:...` cluster id.
 */

export const THEME_REGION_PALETTE_SIZE = 8

/** 8 pastel hues, roughly evenly spaced around HSL, saturation ~45%,
 *  lightness ~65%. At the stylesheet's default 0.14 underlay opacity
 *  they read as soft coloured mist against the darker canvas. Kept in
 *  sync with the theme-region-N selectors in `cyGraphStylesheet.ts`. */
export const THEME_REGION_PALETTE: readonly string[] = [
  '#7ba3d9', // 210° cool blue
  '#d9d97b', // 60°  warm yellow
  '#d97ba3', // 330° warm pink
  '#7bd9a3', // 150° cool green
  '#d9a37b', // 30°  warm orange
  '#a37bd9', // 270° cool purple
  '#a3d97b', // 90°  cool lime
  '#7bd9d9', // 180° cool cyan
] as const

/** Stable, cheap djb2-style hash → palette index. Same `thc:...` id
 *  always maps to the same colour across sessions and browsers (no
 *  `Math.random`, no `Date.now`). */
export function themeRegionIndex(clusterId: string): number {
  let h = 0
  for (let i = 0; i < clusterId.length; i++) {
    h = (h * 31 + clusterId.charCodeAt(i)) | 0
  }
  return Math.abs(h) % THEME_REGION_PALETTE_SIZE
}

/** Convenience: hex swatch for a `thc:...` id. */
export function themeRegionColor(clusterId: string): string {
  return THEME_REGION_PALETTE[themeRegionIndex(clusterId)] ?? THEME_REGION_PALETTE[0]
}
