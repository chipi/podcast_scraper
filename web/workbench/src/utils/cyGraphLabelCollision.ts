/**
 * Label collision culling: a label is drawn only when it does not overlap a label already kept.
 *
 * Cytoscape draws every label wherever its node sits, so close nodes drew their names over each
 * other (public-only walk, 2026-11-17: "Diego Ferrer" over an episode title, two topics on one
 * spot). Counting labels in the layout's node size did not separate them. Here labels are kept
 * greedily by priority (the selected node, its neighbours, then the most connected), and the rest
 * get `graph-label-collides` (the stylesheet hides their text, the node stays).
 *
 * Label sizes are in graph units, so overlap does not change with pan or zoom; it changes when the
 * label tier switches (short/full), when nodes move, and when the selection moves the priorities.
 */
import type { Core, NodeSingular } from 'cytoscape'

export const LABEL_COLLIDES_CLASS = 'graph-label-collides'

export type LabelBox = {
  id: string
  x1: number
  y1: number
  x2: number
  y2: number
  /** Higher wins. */
  priority: number
}

/** Pure: the ids whose label overlaps a higher-priority kept label (ties: input order). */
export function labelsToHide(boxes: readonly LabelBox[], gap = 0): Set<string> {
  const order = boxes
    .map((b, i) => ({ b, i }))
    .sort((p, q) => q.b.priority - p.b.priority || p.i - q.i)
  // A uniform grid keeps the check near-linear; a cell is the median label's size.
  const widths = boxes.map((b) => b.x2 - b.x1).sort((a, z) => a - z)
  const heights = boxes.map((b) => b.y2 - b.y1).sort((a, z) => a - z)
  const cw = Math.max(widths[Math.floor(widths.length / 2)] ?? 1, 1)
  const ch = Math.max(heights[Math.floor(heights.length / 2)] ?? 1, 1)
  const grid = new Map<string, LabelBox[]>()
  const hidden = new Set<string>()
  const cells = (b: LabelBox): string[] => {
    const out: string[] = []
    for (let cx = Math.floor(b.x1 / cw); cx <= Math.floor(b.x2 / cw); cx++) {
      for (let cy = Math.floor(b.y1 / ch); cy <= Math.floor(b.y2 / ch); cy++) out.push(`${cx},${cy}`)
    }
    return out
  }
  const overlaps = (a: LabelBox, b: LabelBox): boolean =>
    a.x1 < b.x2 + gap && b.x1 < a.x2 + gap && a.y1 < b.y2 + gap && b.y1 < a.y2 + gap
  for (const { b } of order) {
    const keys = cells(b)
    const clash = keys.some((k) => (grid.get(k) ?? []).some((kept) => overlaps(b, kept)))
    if (clash) {
      hidden.add(b.id)
      continue
    }
    for (const k of keys) {
      const list = grid.get(k)
      if (list) list.push(b)
      else grid.set(k, [b])
    }
  }
  return hidden
}

function labelText(n: NodeSingular): string {
  return String(n.style('label') ?? '').trim()
}

/**
 * Hide the labels that collide on `core`. `focusId` (the selected node) and its neighbours win
 * first; then degree. Nodes whose label is empty or not drawn (hidden by zoom tier or type) are
 * left out, so they neither win nor lose.
 */
export function syncGraphLabelCollisions(core: Core, focusId?: string | null): void {
  const nodes = core.nodes(':visible')
  const focus = focusId ? core.getElementById(focusId) : null
  const near = new Set<string>()
  if (focus && focus.nonempty()) {
    near.add(focus.id())
    focus.neighborhood('node').forEach((n) => {
      near.add(n.id())
    })
  }
  const boxes: LabelBox[] = []
  core.batch(() => {
    nodes.removeClass(LABEL_COLLIDES_CLASS)
  })
  nodes.forEach((n) => {
    if (n.hasClass('graph-label-tier-none') || n.hasClass('graph-node-zoom-hidden')) return
    if (!labelText(n)) return
    const bb = n.boundingBox({ includeNodes: false, includeEdges: false, includeLabels: true })
    if (!Number.isFinite(bb.x1) || bb.w <= 0 || bb.h <= 0) return
    const id = n.id()
    const priority = (id === focusId ? 3e6 : 0) + (near.has(id) ? 2e6 : 0) + n.degree(false)
    boxes.push({ id, x1: bb.x1, y1: bb.y1, x2: bb.x2, y2: bb.y2, priority })
  })
  const hide = labelsToHide(boxes, 2)
  core.batch(() => {
    for (const id of hide) core.getElementById(id).addClass(LABEL_COLLIDES_CLASS)
  })
}
