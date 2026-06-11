// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// collapseSeed dynamically imports ./artifacts and calls removeRelativeArtifacts.
// Mock the module so the expansion store can be exercised in isolation.
const removeRelativeArtifacts = vi.fn(async (_paths: string[]) => {})

vi.mock('./artifacts', () => ({
  useArtifactsStore: () => ({ removeRelativeArtifacts }),
}))

import { useGraphExpansionStore } from './graphExpansion'

describe('useGraphExpansionStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    removeRelativeArtifacts.mockClear()
  })

  it('initial state is empty', () => {
    const s = useGraphExpansionStore()
    expect(s.expandedBySeed).toEqual({})
    expect(s.truncationLine).toBeNull()
    expect(s.expansionBusyCyId).toBeNull()
    expect(s.peekCorpusBeyondProbeGen()).toBe(0)
  })

  describe('isExpanded', () => {
    it('returns false for an unknown seed', () => {
      const s = useGraphExpansionStore()
      expect(s.isExpanded('seed-1')).toBe(false)
    })

    it('returns true once a seed is recorded', () => {
      const s = useGraphExpansionStore()
      s.recordExpand('seed-1', ['a/b.json'])
      expect(s.isExpanded('seed-1')).toBe(true)
    })

    it('trims the queried seed id before lookup', () => {
      const s = useGraphExpansionStore()
      s.recordExpand('seed-1', ['a/b.json'])
      expect(s.isExpanded('  seed-1  ')).toBe(true)
    })
  })

  describe('recordExpand', () => {
    it('stores the added rel paths keyed by trimmed seed id', () => {
      const s = useGraphExpansionStore()
      s.recordExpand('  seed-1  ', ['a.json', 'b.json'])
      expect(s.expandedBySeed['seed-1']).toEqual({ addedRelPaths: ['a.json', 'b.json'] })
      expect(s.isExpanded('seed-1')).toBe(true)
    })

    it('ignores an empty seed id', () => {
      const s = useGraphExpansionStore()
      s.recordExpand('', ['a.json'])
      expect(s.expandedBySeed).toEqual({})
    })

    it('ignores a whitespace-only seed id', () => {
      const s = useGraphExpansionStore()
      s.recordExpand('   ', ['a.json'])
      expect(s.expandedBySeed).toEqual({})
    })

    it('overwrites a prior record for the same seed', () => {
      const s = useGraphExpansionStore()
      s.recordExpand('seed-1', ['a.json'])
      s.recordExpand('seed-1', ['c.json'])
      expect(s.expandedBySeed['seed-1']).toEqual({ addedRelPaths: ['c.json'] })
    })

    it('keeps separate seeds independent', () => {
      const s = useGraphExpansionStore()
      s.recordExpand('seed-1', ['a.json'])
      s.recordExpand('seed-2', ['b.json'])
      expect(Object.keys(s.expandedBySeed).sort()).toEqual(['seed-1', 'seed-2'])
    })
  })

  describe('truncation line', () => {
    it('setTruncationLine sets the message', () => {
      const s = useGraphExpansionStore()
      s.setTruncationLine('truncated at 200')
      expect(s.truncationLine).toBe('truncated at 200')
    })

    it('setTruncationLine can set null', () => {
      const s = useGraphExpansionStore()
      s.setTruncationLine('msg')
      s.setTruncationLine(null)
      expect(s.truncationLine).toBeNull()
    })

    it('clearTruncationLine resets to null', () => {
      const s = useGraphExpansionStore()
      s.setTruncationLine('msg')
      s.clearTruncationLine()
      expect(s.truncationLine).toBeNull()
    })
  })

  describe('setBusy', () => {
    it('sets a trimmed cy id', () => {
      const s = useGraphExpansionStore()
      s.setBusy('  node-1  ')
      expect(s.expansionBusyCyId).toBe('node-1')
    })

    it('null clears busy', () => {
      const s = useGraphExpansionStore()
      s.setBusy('node-1')
      s.setBusy(null)
      expect(s.expansionBusyCyId).toBeNull()
    })

    it('empty string clears busy', () => {
      const s = useGraphExpansionStore()
      s.setBusy('node-1')
      s.setBusy('')
      expect(s.expansionBusyCyId).toBeNull()
    })

    it('whitespace-only string clears busy', () => {
      const s = useGraphExpansionStore()
      s.setBusy('node-1')
      s.setBusy('   ')
      expect(s.expansionBusyCyId).toBeNull()
    })
  })

  describe('collapseSeed', () => {
    it('removes the seed and its artifacts, clearing truncation line', async () => {
      const s = useGraphExpansionStore()
      s.recordExpand('seed-1', ['a.json', 'b.json'])
      s.setTruncationLine('msg')
      await s.collapseSeed('seed-1')
      expect(removeRelativeArtifacts).toHaveBeenCalledWith(['a.json', 'b.json'])
      expect(s.isExpanded('seed-1')).toBe(false)
      expect(s.expandedBySeed).toEqual({})
      expect(s.truncationLine).toBeNull()
    })

    it('trims the seed id before collapsing', async () => {
      const s = useGraphExpansionStore()
      s.recordExpand('seed-1', ['a.json'])
      await s.collapseSeed('  seed-1  ')
      expect(removeRelativeArtifacts).toHaveBeenCalledWith(['a.json'])
      expect(s.isExpanded('seed-1')).toBe(false)
    })

    it('is a no-op for an unknown seed', async () => {
      const s = useGraphExpansionStore()
      s.setTruncationLine('keep-me')
      await s.collapseSeed('missing')
      expect(removeRelativeArtifacts).not.toHaveBeenCalled()
      expect(s.truncationLine).toBe('keep-me')
    })

    it('leaves other seeds intact', async () => {
      const s = useGraphExpansionStore()
      s.recordExpand('seed-1', ['a.json'])
      s.recordExpand('seed-2', ['b.json'])
      await s.collapseSeed('seed-1')
      expect(s.isExpanded('seed-1')).toBe(false)
      expect(s.isExpanded('seed-2')).toBe(true)
    })
  })

  describe('resetExpansionState', () => {
    it('clears all state and bumps the probe generation', () => {
      const s = useGraphExpansionStore()
      s.recordExpand('seed-1', ['a.json'])
      s.setTruncationLine('msg')
      s.setBusy('node-1')
      s.commitCorpusBeyondProbe(s.peekCorpusBeyondProbeGen(), 'node-1', true)
      const genBefore = s.peekCorpusBeyondProbeGen()

      s.resetExpansionState()

      expect(s.expandedBySeed).toEqual({})
      expect(s.truncationLine).toBeNull()
      expect(s.expansionBusyCyId).toBeNull()
      expect(s.corpusBeyondAppendKnown('node-1')).toBeUndefined()
      expect(s.peekCorpusBeyondProbeGen()).toBe(genBefore + 1)
    })
  })

  describe('invalidateCorpusBeyondHints', () => {
    it('clears probe results and increments the generation', () => {
      const s = useGraphExpansionStore()
      s.commitCorpusBeyondProbe(s.peekCorpusBeyondProbeGen(), 'node-1', true)
      expect(s.corpusBeyondAppendKnown('node-1')).toBe(true)

      s.invalidateCorpusBeyondHints()

      expect(s.corpusBeyondAppendKnown('node-1')).toBeUndefined()
      expect(s.peekCorpusBeyondProbeGen()).toBe(1)
    })
  })

  describe('corpus-beyond probe commit', () => {
    it('commits when the wave generation matches the current generation', () => {
      const s = useGraphExpansionStore()
      const gen = s.peekCorpusBeyondProbeGen()
      s.commitCorpusBeyondProbe(gen, 'node-1', true)
      expect(s.corpusBeyondAppendKnown('node-1')).toBe(true)
    })

    it('can commit a false (no-append) result', () => {
      const s = useGraphExpansionStore()
      const gen = s.peekCorpusBeyondProbeGen()
      s.commitCorpusBeyondProbe(gen, 'node-1', false)
      expect(s.corpusBeyondAppendKnown('node-1')).toBe(false)
    })

    it('trims the cy id before committing', () => {
      const s = useGraphExpansionStore()
      const gen = s.peekCorpusBeyondProbeGen()
      s.commitCorpusBeyondProbe(gen, '  node-1  ', true)
      expect(s.corpusBeyondAppendKnown('node-1')).toBe(true)
    })

    it('drops a commit whose wave generation is stale', () => {
      const s = useGraphExpansionStore()
      const staleGen = s.peekCorpusBeyondProbeGen()
      s.invalidateCorpusBeyondHints()
      s.commitCorpusBeyondProbe(staleGen, 'node-1', true)
      expect(s.corpusBeyondAppendKnown('node-1')).toBeUndefined()
    })

    it('ignores a commit with an empty cy id', () => {
      const s = useGraphExpansionStore()
      const gen = s.peekCorpusBeyondProbeGen()
      s.commitCorpusBeyondProbe(gen, '', true)
      expect(s.corpusBeyondAppendKnown('')).toBeUndefined()
    })

    it('ignores a commit with a whitespace-only cy id', () => {
      const s = useGraphExpansionStore()
      const gen = s.peekCorpusBeyondProbeGen()
      s.commitCorpusBeyondProbe(gen, '   ', true)
      expect(s.peekCorpusBeyondProbeGen()).toBe(gen)
    })

    it('overwrites a prior probe result for the same id', () => {
      const s = useGraphExpansionStore()
      const gen = s.peekCorpusBeyondProbeGen()
      s.commitCorpusBeyondProbe(gen, 'node-1', true)
      s.commitCorpusBeyondProbe(gen, 'node-1', false)
      expect(s.corpusBeyondAppendKnown('node-1')).toBe(false)
    })
  })

  describe('corpusBeyondAppendKnown', () => {
    it('returns undefined for an unknown id', () => {
      const s = useGraphExpansionStore()
      expect(s.corpusBeyondAppendKnown('never-probed')).toBeUndefined()
    })

    it('returns undefined for an empty id', () => {
      const s = useGraphExpansionStore()
      expect(s.corpusBeyondAppendKnown('')).toBeUndefined()
    })

    it('returns undefined for a whitespace-only id', () => {
      const s = useGraphExpansionStore()
      expect(s.corpusBeyondAppendKnown('   ')).toBeUndefined()
    })

    it('trims the queried id before lookup', () => {
      const s = useGraphExpansionStore()
      const gen = s.peekCorpusBeyondProbeGen()
      s.commitCorpusBeyondProbe(gen, 'node-1', true)
      expect(s.corpusBeyondAppendKnown('  node-1  ')).toBe(true)
    })

    it('distinguishes a committed false from an absent key', () => {
      const s = useGraphExpansionStore()
      const gen = s.peekCorpusBeyondProbeGen()
      s.commitCorpusBeyondProbe(gen, 'node-1', false)
      expect(s.corpusBeyondAppendKnown('node-1')).toBe(false)
      expect(s.corpusBeyondAppendKnown('node-2')).toBeUndefined()
    })
  })
})
