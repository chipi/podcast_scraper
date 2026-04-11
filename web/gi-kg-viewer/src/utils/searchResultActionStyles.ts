/**
 * Action chip styles for semantic search result cards — reuse anywhere we mirror **G** / **E** affordances.
 * @see components/search/ResultCard.vue
 */
export const SEARCH_RESULT_GRAPH_BUTTON_CLASS =
  'flex size-6 shrink-0 items-center justify-center rounded-sm bg-gi text-[11px] font-semibold leading-none text-gi-foreground hover:opacity-90 disabled:opacity-40'

/**
 * **E** (graph / corpus id chip) — same hex as Cytoscape legend **Episode** (``graphNodeTypeStyles.Episode``).
 * Border-2 + shadow so the glyph stays readable on all themes.
 */
export const SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS =
  'flex size-6 shrink-0 cursor-help items-center justify-center rounded-sm border-2 border-[#364fc7] bg-[#4c6ef5] text-[11px] font-bold leading-none text-white shadow-sm hover:brightness-110'

/**
 * Diagnostics **?** next to **E** — black/neutral glyph on light chip; inverted in dark mode for contrast.
 */
export const SEARCH_RESULT_DIAGNOSTICS_HELP_CHIP_CLASS =
  'flex size-6 shrink-0 cursor-help items-center justify-center rounded-sm border-2 border-[#212529] bg-white text-[11px] font-bold leading-none text-[#212529] shadow-sm hover:bg-zinc-100 dark:border-zinc-300 dark:bg-zinc-950 dark:text-zinc-100 dark:hover:bg-zinc-900'
