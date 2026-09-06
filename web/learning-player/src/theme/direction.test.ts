import { beforeEach, describe, expect, it } from 'vitest'
import { applyDirection, DIRECTION_KEY, resolveDirection } from './direction'

/** A minimal in-memory Storage, so these tests say nothing about jsdom's sessionStorage. */
function store(initial: Record<string, string> = {}) {
  const map = new Map(Object.entries(initial))
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    get size() {
      return map.size
    },
  }
}

describe('resolveDirection', () => {
  it('applies and persists a direction from the URL', () => {
    const s = store()
    expect(resolveDirection('?direction=paper', s)).toBe('paper')
    expect(s.getItem(DIRECTION_KEY)).toBe('paper')
  })

  it('restores the stored direction when the URL says nothing', () => {
    expect(resolveDirection('', store({ [DIRECTION_KEY]: 'terminal' }))).toBe('terminal')
    expect(resolveDirection('?other=1', store({ [DIRECTION_KEY]: 'terminal' }))).toBe('terminal')
  })

  it('CLEARS the stored direction for an explicitly empty value', () => {
    // The bug: `?direction=` parses to '' and `'' ?? stored` is '', so the old code neither stored
    // nor applied it — and the stored direction survived to be restored on the next navigation.
    // The switch could be turned on and moved, but never turned off.
    const s = store({ [DIRECTION_KEY]: 'paper' })
    expect(resolveDirection('?direction=', s)).toBeNull()
    expect(s.getItem(DIRECTION_KEY)).toBeNull()
  })

  it('stays cleared on the next navigation', () => {
    // The half that actually mattered: clearing for one page load while leaving storage intact
    // would look fixed and revert the moment you clicked anything.
    const s = store({ [DIRECTION_KEY]: 'paper' })
    resolveDirection('?direction=', s)
    expect(resolveDirection('', s)).toBeNull()
  })

  it('switches directly from one direction to another', () => {
    const s = store({ [DIRECTION_KEY]: 'paper' })
    expect(resolveDirection('?direction=signal', s)).toBe('signal')
    expect(s.getItem(DIRECTION_KEY)).toBe('signal')
  })

  it('treats an empty stored value as the default rather than an empty attribute', () => {
    expect(resolveDirection('', store({ [DIRECTION_KEY]: '' }))).toBeNull()
  })
})

describe('applyDirection', () => {
  let root: HTMLElement
  beforeEach(() => {
    root = document.createElement('html')
  })

  it('sets the attribute the stylesheet keys on', () => {
    applyDirection(root, 'paper')
    expect(root.getAttribute('data-direction')).toBe('paper')
  })

  it('REMOVES the attribute for the default, rather than setting it empty', () => {
    // `data-direction=""` is not the same as no attribute: it is an attribute selectors can match,
    // and leaving it behind is how "cleared" turns into "cleared, mostly".
    applyDirection(root, 'paper')
    applyDirection(root, null)
    expect(root.hasAttribute('data-direction')).toBe(false)
  })
})
