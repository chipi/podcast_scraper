/**
 * The fixed highlight colour palette (PRD-040 FR1.4) — a small, lightweight categorisation set.
 * Tokens are persisted in `highlight.color`; the swatch/border classes are literal Tailwind
 * utilities (kept as full strings so the JIT scanner includes them — never build them dynamically).
 */

export interface HighlightColor {
  /** Persisted token (highlight.color). */
  token: string
  /** i18n key for the swatch's accessible name. */
  labelKey: string
  /** Filled swatch class. */
  swatch: string
  /** Left-border accent class for the highlight card. */
  border: string
}

export const HIGHLIGHT_COLORS: readonly HighlightColor[] = [
  { token: 'amber', labelKey: 'highlights.color.amber', swatch: 'bg-amber-400', border: 'border-l-amber-400' },
  { token: 'rose', labelKey: 'highlights.color.rose', swatch: 'bg-rose-400', border: 'border-l-rose-400' },
  { token: 'sky', labelKey: 'highlights.color.sky', swatch: 'bg-sky-400', border: 'border-l-sky-400' },
  { token: 'emerald', labelKey: 'highlights.color.emerald', swatch: 'bg-emerald-400', border: 'border-l-emerald-400' },
  { token: 'violet', labelKey: 'highlights.color.violet', swatch: 'bg-violet-400', border: 'border-l-violet-400' },
] as const

const BY_TOKEN = new Map(HIGHLIGHT_COLORS.map((c) => [c.token, c]))

/** The left-border accent class for a highlight's colour (transparent when unset/unknown). */
export function borderClass(token: string | null | undefined): string {
  return (token && BY_TOKEN.get(token)?.border) || 'border-l-transparent'
}
