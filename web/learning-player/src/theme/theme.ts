/**
 * Theme application for the consumer app (UXS-011).
 *
 * MVP is dark-only; the data-theme hook is in place for a future light theme. The per-show
 * adaptive accent is set at runtime via `setShowAccent` — components read `var(--lp-accent)`
 * (the `accent` Tailwind key), never a hard-coded hue. A real contrast-clamp + artwork
 * extraction lands with the Player surface (#1083); this is the wiring seam.
 */

export type ThemeMode = 'dark'

/** Apply the baseline theme (dark-only MVP). */
export function applyTheme(mode: ThemeMode = 'dark'): void {
  document.documentElement.setAttribute('data-theme', mode)
}

/**
 * Set the per-show accent on a root element (default: the app root). Pass `null` to clear
 * back to the brand default. Contrast-clamping against `surface` is a Player-surface concern
 * (#1083); here we only wire the variable.
 */
export function setShowAccent(color: string | null, el: HTMLElement = document.documentElement): void {
  if (color && color.trim()) {
    el.style.setProperty('--lp-accent', color.trim())
  } else {
    el.style.removeProperty('--lp-accent')
  }
}
