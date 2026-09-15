/**
 * The ONE positioning rule for every popover panel (⋯ overflow, share menu, add-to-collection) —
 * so a drawer can never open off the edge of the screen again (operator 2026-09-13).
 *
 * Before this, each menu positioned itself: the ⋯ hung from the trigger's right edge with no clamp,
 * the share menu was hard-anchored `right-0` with no flip, and only add-to-collection flipped. A
 * trigger near the LEFT edge (a card control under the artwork, a storyline share on Home) pushed the
 * right-anchored panels straight off the left of the viewport. This function is the single source of
 * truth they now all call: given the trigger rect, the panel's measured size and the viewport, it
 * returns a `{top,left}` that keeps the panel fully on screen.
 *
 * Coordinates are viewport-relative (for a `position: fixed` panel teleported to <body>), which is
 * what lets it escape any clipped/transformed ancestor.
 */
export interface Rect {
  top: number
  bottom: number
  left: number
  right: number
}
export interface Size {
  width: number
  height: number
}
export interface AnchorOptions {
  /**
   * Which trigger edge the panel prefers to line up with before clamping.
   * `end` (default) lines the panel's RIGHT edge to the trigger's right edge (the established idiom
   * for a control at a row's right); `start` lines its LEFT edge to the trigger's left.
   */
  align?: "start" | "end"
  /** Vertical gap between trigger and panel. */
  gap?: number
  /** Minimum distance the panel keeps from every viewport edge. */
  margin?: number
}

export interface AnchorResult {
  top: number
  left: number
}

export function anchorPanel(
  trigger: Rect,
  panel: Size,
  viewport: Size,
  opts: AnchorOptions = {}
): AnchorResult {
  const align = opts.align ?? "end"
  const gap = opts.gap ?? 4
  const margin = opts.margin ?? 8

  // Horizontal: start from the preferred edge, then clamp so neither side overflows. When the panel
  // is wider than the viewport allows, pin to the left margin and let the panel's own max-width keep
  // it on screen rather than producing a negative (off-left) coordinate.
  const preferredLeft = align === "end" ? trigger.right - panel.width : trigger.left
  const maxLeft = viewport.width - panel.width - margin
  const left = maxLeft < margin ? margin : Math.min(Math.max(preferredLeft, margin), maxLeft)

  // Vertical: below the trigger by default; flip above only when there is not enough room below AND
  // there is room above. Otherwise clamp to the top margin (a panel taller than the viewport is a
  // content problem, not a placement one — it scrolls).
  let top = trigger.bottom + gap
  const overflowsBottom = top + panel.height > viewport.height - margin
  const roomAbove = trigger.top - gap - panel.height >= margin
  if (overflowsBottom && roomAbove) top = trigger.top - gap - panel.height
  else if (overflowsBottom) top = margin

  return { top: Math.round(top), left: Math.round(left) }
}
