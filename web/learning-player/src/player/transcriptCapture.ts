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

/** A captured span over one or more transcript segments (a paragraph or a selected phrase). */
export interface ParagraphSpan {
  start_ms: number
  end_ms: number
  segment_ids: string[]
  char_start: number
  char_end: number
  quote_text: string
  speaker: string | null
}

interface SegLike {
  id: string
  start: number
  end: number
  text: string
  speaker: string | null
}

const ms = (s: number): number => Math.max(0, Math.round(s * 1000))

/**
 * Build a span highlight from a paragraph's segments (PRD-040 FR1.2). With no selection it captures
 * the WHOLE paragraph; with a selection (offsets into the paragraph's joined text, single-space
 * separators) it captures just that phrase, anchored to the segments the selection actually touches —
 * so `start_ms`/`end_ms` + `segment_ids` reflect the real spoken range, and `quote_text` is verbatim.
 */
export function spanFromParagraph(segments: SegLike[], sub: SubRange | null): ParagraphSpan {
  const joined = segments.map((s) => s.text).join(' ')
  if (!sub) {
    const last = segments[segments.length - 1]
    return {
      start_ms: ms(segments[0].start),
      end_ms: ms(last.end),
      segment_ids: segments.map((s) => s.id),
      char_start: 0,
      char_end: joined.length,
      quote_text: joined,
      speaker: segments[0].speaker ?? null,
    }
  }
  // Map the selection's char range onto the segments it overlaps in the joined text.
  let offset = 0
  const ranges = segments.map((s) => {
    const r = { seg: s, start: offset, end: offset + s.text.length }
    offset += s.text.length + 1 // + the space separator
    return r
  })
  const touched = ranges.filter((r) => r.start < sub.char_end && r.end > sub.char_start)
  const span = touched.length ? touched : [ranges[0]]
  const anchor = span[0]
  const last = span[span.length - 1]
  return {
    start_ms: ms(anchor.seg.start),
    end_ms: ms(last.seg.end),
    segment_ids: span.map((r) => r.seg.id),
    char_start: Math.max(0, sub.char_start - anchor.start),
    char_end: Math.max(0, sub.char_end - anchor.start),
    quote_text: sub.quote_text,
    speaker: anchor.seg.speaker ?? null,
  }
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
