import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { useSearchStore } from './search'

describe('useSearchStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('applyLibrarySearchHandoff normalizes feed and sets hint', () => {
    const s = useSearchStore()
    s.applyLibrarySearchHandoff('  f1  ', 'alpha bravo')
    expect(s.filters.feed).toBe('f1')
    expect(s.query).toBe('alpha bravo')
    expect(s.libraryHandoffHint).toContain('Library')
  })

  it('clearResults clears library handoff hint', () => {
    const s = useSearchStore()
    s.applyLibrarySearchHandoff('f1', 'q')
    s.clearResults()
    expect(s.libraryHandoffHint).toBeNull()
  })
})
