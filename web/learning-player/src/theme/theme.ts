/**
 * Theme application for the consumer app (UXS-011).
 *
 * MVP is dark-only; the data-theme hook is in place for a future light theme. The per-show
 * adaptive accent is set at runtime via `setShowAccent` — components read `var(--lp-accent)`
 * (the `accent` Tailwind key), never a hard-coded hue. `accent.ts` derives the colour from the
 * current episode's artwork and contrast-clamps it (`contrast.ts`); `App.vue` calls it off
 * `player.currentArtwork` (#1598). This module owns only the CSS-variable write.
 */

export type ThemeMode = 'dark'

/** Apply the baseline theme (dark-only MVP). */
export function applyTheme(mode: ThemeMode = 'dark'): void {
  document.documentElement.setAttribute('data-theme', mode)
}

/**
 * Set the per-show accent on a root element (default: the app root). Pass `null` to clear back to
 * the brand default. The caller (`accent.ts`) has already contrast-clamped the colour; this only
 * writes the variable.
 */
export function setShowAccent(color: string | null, el: HTMLElement = document.documentElement): void {
  if (color && color.trim()) {
    el.style.setProperty('--lp-accent', color.trim())
  } else {
    el.style.removeProperty('--lp-accent')
  }
}
