/**
 * Transcript sub-segment capture (PRD-040 FR1.2) — turn a live text selection inside one transcript
 * line into exact character offsets + the verbatim quote, so "save" can capture a phrase, not just
 * the whole line. Uses the pre-caret Range-length technique so it's robust to the grounded-quote
 * span splitting within a segment (a segment's text may be several child spans).
 */

export interface SubRange {
  char_start: number
  char_end: number
  quote_text: string
}

/** Length of the text from the start of `el` up to (container, offset), via a collapsed Range. */
function offsetWithin(el: HTMLElement, container: Node, offset: number): number {
  const pre = el.ownerDocument.createRange()
  pre.selectNodeContents(el)
  pre.setEnd(container, offset)
  return pre.toString().length
}

/**
 * The current selection as character offsets within `textEl`, or `null` when there is no usable
 * selection inside it (collapsed, empty, or anchored elsewhere). Defensive: never throws.
 */
export function selectionSubRange(textEl: HTMLElement): SubRange | null {
  const sel = typeof window !== 'undefined' ? window.getSelection() : null
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return null
  let range: Range
  try {
    range = sel.getRangeAt(0)
  } catch {
    return null
  }
  if (!textEl.contains(range.commonAncestorContainer)) return null
  const quote = sel.toString().trim()
  if (!quote) return null
  try {
    const a = offsetWithin(textEl, range.startContainer, range.startOffset)
    const b = offsetWithin(textEl, range.endContainer, range.endOffset)
    const char_start = Math.min(a, b)
    const char_end = Math.max(a, b)
    if (char_end <= char_start) return null
    return { char_start, char_end, quote_text: quote }
  } catch {
    return null
  }
}
