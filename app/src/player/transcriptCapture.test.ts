import { afterEach, describe, expect, it, vi } from 'vitest'
import { selectionSubRange } from './transcriptCapture'

afterEach(() => vi.restoreAllMocks())

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
