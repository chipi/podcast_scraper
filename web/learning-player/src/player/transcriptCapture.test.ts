import { afterEach, describe, expect, it, vi } from 'vitest'
import { selectionSubRange, spanFromParagraph } from './transcriptCapture'

afterEach(() => vi.restoreAllMocks())

const seg = (id: string, start: number, end: number, text: string, speaker: string | null = null) =>
  ({ id, start, end, text, speaker })

describe('spanFromParagraph', () => {
  const para = [seg('a', 1, 2, 'Deep sleep', 'person:g'), seg('b', 2, 4, 'consolidates memory')]

  it('captures the WHOLE paragraph when there is no selection', () => {
    const span = spanFromParagraph(para, null)
    expect(span.quote_text).toBe('Deep sleep consolidates memory')
    expect(span.segment_ids).toEqual(['a', 'b'])
    expect(span.start_ms).toBe(1000) // first seg start
    expect(span.end_ms).toBe(4000) // last seg end
    expect(span.speaker).toBe('person:g')
  })

  it('captures a phrase anchored to the segments it actually touches', () => {
    // "sleep consolidates" spans the boundary of seg a (0–10) and seg b (11–31) in the joined text.
    const span = spanFromParagraph(para, { char_start: 5, char_end: 21, quote_text: 'sleep consolidates' })
    expect(span.quote_text).toBe('sleep consolidates')
    expect(span.segment_ids).toEqual(['a', 'b']) // both touched
    expect(span.start_ms).toBe(1000) // anchor = first touched seg (a)
    expect(span.end_ms).toBe(4000) // last touched seg (b)
    expect(span.char_start).toBe(5) // relative to the anchor segment
  })

  it('a phrase inside one segment touches only that segment', () => {
    const span = spanFromParagraph(para, { char_start: 0, char_end: 4, quote_text: 'Deep' })
    expect(span.segment_ids).toEqual(['a'])
    expect(span.end_ms).toBe(2000)
  })
})

function fakeSelection(range: Range | null, text: string): Selection {
  return {
    isCollapsed: range == null || range.collapsed,
    rangeCount: range ? 1 : 0,
    getRangeAt: () => range as Range,
    toString: () => text,
  } as unknown as Selection
}

/** Real element with text + a Range over [start,end) of its text node; spy window.getSelection. */
function withSelection(text: string, start: number, end: number): HTMLElement {
  const el = document.createElement('span')
  el.textContent = text
  document.body.appendChild(el)
  const node = el.firstChild as Text
  const range = document.createRange()
  range.setStart(node, start)
  range.setEnd(node, end)
  vi.spyOn(window, 'getSelection').mockReturnValue(fakeSelection(range, text.slice(start, end)))
  return el
}

describe('selectionSubRange', () => {
  it('returns null when there is no selection', () => {
    vi.spyOn(window, 'getSelection').mockReturnValue(null)
    expect(selectionSubRange(document.createElement('span'))).toBeNull()
  })

  it('returns null for a collapsed selection', () => {
    const el = withSelection('Hello world.', 3, 3)
    expect(selectionSubRange(el)).toBeNull()
  })

  it('computes char offsets + verbatim quote for a sub-range within the line', () => {
    const el = withSelection('Deep sleep consolidates memory.', 5, 10) // "sleep"
    expect(selectionSubRange(el)).toEqual({ char_start: 5, char_end: 10, quote_text: 'sleep' })
  })

  it('returns null when the selection is anchored outside the element', () => {
    const inside = document.createElement('span')
    inside.textContent = 'inside'
    const outside = document.createElement('span')
    outside.textContent = 'outside text'
    document.body.append(inside, outside)
    const node = outside.firstChild as Text
    const range = document.createRange()
    range.setStart(node, 0)
    range.setEnd(node, 4)
    vi.spyOn(window, 'getSelection').mockReturnValue(fakeSelection(range, 'outs'))
    expect(selectionSubRange(inside)).toBeNull()
  })
})
