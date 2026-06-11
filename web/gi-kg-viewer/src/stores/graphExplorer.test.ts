// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { localYmdDaysAgo } from '../utils/localCalendarDate'
import { useCorpusLensStore } from './corpusLens'
import {
  GRAPH_LAYOUT_CYCLE_ORDER,
  useGraphExplorerStore,
} from './graphExplorer'

describe('useGraphExplorerStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    // Default: no build-time graph-lens override (seedDefaultSinceYmd → 7d).
    vi.unstubAllEnvs()
  })

  describe('initial state', () => {
    it('exposes the documented defaults', () => {
      const s = useGraphExplorerStore()
      expect(s.preferredLayout).toBe('cose')
      expect(s.minimapOpen).toBe(false)
      expect(s.activeDegreeBucket).toBe(null)
      expect(s.sinceYmd).toBe('')
      expect(s.seeded).toBe(false)
      expect(s.graphTabOpenedThisSession).toBe(false)
      expect(s.lastAutoLoadWasCapped).toBe(false)
    })
  })

  describe('degree bucket', () => {
    it('toggleDegreeBucket sets the bucket when none is active', () => {
      const s = useGraphExplorerStore()
      s.toggleDegreeBucket('hi')
      expect(s.activeDegreeBucket).toBe('hi')
    })

    it('toggleDegreeBucket on the same key clears it (toggle off)', () => {
      const s = useGraphExplorerStore()
      s.toggleDegreeBucket('hi')
      s.toggleDegreeBucket('hi')
      expect(s.activeDegreeBucket).toBe(null)
    })

    it('toggleDegreeBucket on a different key switches buckets', () => {
      const s = useGraphExplorerStore()
      s.toggleDegreeBucket('hi')
      s.toggleDegreeBucket('lo')
      expect(s.activeDegreeBucket).toBe('lo')
    })

    it('clearDegreeBucket resets to null', () => {
      const s = useGraphExplorerStore()
      s.toggleDegreeBucket('hi')
      s.clearDegreeBucket()
      expect(s.activeDegreeBucket).toBe(null)
    })

    it('clearDegreeBucket is a no-op when already null', () => {
      const s = useGraphExplorerStore()
      s.clearDegreeBucket()
      expect(s.activeDegreeBucket).toBe(null)
    })

    it('resetForNewArtifact clears the active degree bucket', () => {
      const s = useGraphExplorerStore()
      s.toggleDegreeBucket('hi')
      s.resetForNewArtifact()
      expect(s.activeDegreeBucket).toBe(null)
    })
  })

  describe('markGraphTabOpenedOnce', () => {
    it('flips the flag on first call', () => {
      const s = useGraphExplorerStore()
      s.markGraphTabOpenedOnce()
      expect(s.graphTabOpenedThisSession).toBe(true)
    })

    it('is idempotent (early-return branch when already opened)', () => {
      const s = useGraphExplorerStore()
      s.markGraphTabOpenedOnce()
      s.markGraphTabOpenedOnce()
      expect(s.graphTabOpenedThisSession).toBe(true)
    })
  })

  describe('seedFromCorpusLensIfNeeded', () => {
    it('seeds from the corpus lens when corpus has a since value', () => {
      const corpus = useCorpusLensStore()
      corpus.sinceYmd = '2024-01-15'
      const s = useGraphExplorerStore()
      s.seedFromCorpusLensIfNeeded()
      expect(s.sinceYmd).toBe('2024-01-15')
      expect(s.seeded).toBe(true)
    })

    it('trims whitespace from the corpus since value', () => {
      const corpus = useCorpusLensStore()
      corpus.sinceYmd = '  2024-02-20  '
      const s = useGraphExplorerStore()
      s.seedFromCorpusLensIfNeeded()
      expect(s.sinceYmd).toBe('2024-02-20')
      expect(s.seeded).toBe(true)
    })

    it('falls back to the 7-day default when corpus since is empty', () => {
      const corpus = useCorpusLensStore()
      corpus.sinceYmd = ''
      const s = useGraphExplorerStore()
      s.seedFromCorpusLensIfNeeded()
      expect(s.sinceYmd).toBe(localYmdDaysAgo(7))
      expect(s.seeded).toBe(true)
    })

    it('falls back to the default when corpus since is whitespace-only', () => {
      const corpus = useCorpusLensStore()
      corpus.sinceYmd = '   '
      const s = useGraphExplorerStore()
      s.seedFromCorpusLensIfNeeded()
      expect(s.sinceYmd).toBe(localYmdDaysAgo(7))
      expect(s.seeded).toBe(true)
    })

    it('is a no-op once already seeded (early-return branch)', () => {
      const corpus = useCorpusLensStore()
      corpus.sinceYmd = '2024-03-01'
      const s = useGraphExplorerStore()
      s.seedFromCorpusLensIfNeeded()
      // Change corpus afterwards; a second call must not re-seed.
      corpus.sinceYmd = '2099-12-31'
      s.seedFromCorpusLensIfNeeded()
      expect(s.sinceYmd).toBe('2024-03-01')
    })
  })

  describe('resetSinceYmdToInitialCorpusSeed', () => {
    it('uses the corpus since value when present', () => {
      const corpus = useCorpusLensStore()
      corpus.sinceYmd = '2024-04-04'
      const s = useGraphExplorerStore()
      s.setAllTime() // mutate first so we can observe the reset
      s.resetSinceYmdToInitialCorpusSeed()
      expect(s.sinceYmd).toBe('2024-04-04')
      expect(s.seeded).toBe(true)
    })

    it('falls back to the 7-day default when corpus is empty', () => {
      const corpus = useCorpusLensStore()
      corpus.sinceYmd = ''
      const s = useGraphExplorerStore()
      s.resetSinceYmdToInitialCorpusSeed()
      expect(s.sinceYmd).toBe(localYmdDaysAgo(7))
      expect(s.seeded).toBe(true)
    })

    it('re-seeds even when already seeded (no early-return guard)', () => {
      const corpus = useCorpusLensStore()
      corpus.sinceYmd = '2024-05-05'
      const s = useGraphExplorerStore()
      s.setSinceYmd('2024-06-06')
      expect(s.seeded).toBe(true)
      s.resetSinceYmdToInitialCorpusSeed()
      expect(s.sinceYmd).toBe('2024-05-05')
    })
  })

  describe('resetGraphLensForNewCorpus', () => {
    it('clears seeded, sinceYmd, and the capped flag', () => {
      const s = useGraphExplorerStore()
      s.setSinceYmd('2024-07-07')
      s.setLastAutoLoadCapped(true)
      s.resetGraphLensForNewCorpus()
      expect(s.seeded).toBe(false)
      expect(s.sinceYmd).toBe('')
      expect(s.lastAutoLoadWasCapped).toBe(false)
    })
  })

  describe('setSinceYmd', () => {
    it('trims and stores the value, marking seeded', () => {
      const s = useGraphExplorerStore()
      s.setSinceYmd('  2024-08-08 ')
      expect(s.sinceYmd).toBe('2024-08-08')
      expect(s.seeded).toBe(true)
    })

    it('stores an empty string when given only whitespace', () => {
      const s = useGraphExplorerStore()
      s.setSinceYmd('   ')
      expect(s.sinceYmd).toBe('')
      expect(s.seeded).toBe(true)
    })
  })

  describe('setPresetDays', () => {
    it.each([7, 30, 90] as const)('sets sinceYmd to %d days ago and seeds', (days) => {
      const s = useGraphExplorerStore()
      s.setPresetDays(days)
      expect(s.sinceYmd).toBe(localYmdDaysAgo(days))
      expect(s.seeded).toBe(true)
    })
  })

  describe('setAllTime', () => {
    it('clears sinceYmd but marks seeded', () => {
      const s = useGraphExplorerStore()
      s.setSinceYmd('2024-09-09')
      s.setAllTime()
      expect(s.sinceYmd).toBe('')
      expect(s.seeded).toBe(true)
    })
  })

  describe('setLastAutoLoadCapped', () => {
    it('sets the flag true', () => {
      const s = useGraphExplorerStore()
      s.setLastAutoLoadCapped(true)
      expect(s.lastAutoLoadWasCapped).toBe(true)
    })

    it('sets the flag false', () => {
      const s = useGraphExplorerStore()
      s.setLastAutoLoadCapped(true)
      s.setLastAutoLoadCapped(false)
      expect(s.lastAutoLoadWasCapped).toBe(false)
    })
  })

  describe('cyclePreferredLayout', () => {
    it('advances through the full cycle order and wraps back to the start', () => {
      const s = useGraphExplorerStore()
      const seen: string[] = []
      for (let i = 0; i < GRAPH_LAYOUT_CYCLE_ORDER.length; i++) {
        seen.push(s.cyclePreferredLayout())
      }
      // Starting at 'cose' (index 0), cycling N times visits each subsequent
      // entry and wraps back to 'cose'.
      const expected = [
        ...GRAPH_LAYOUT_CYCLE_ORDER.slice(1),
        GRAPH_LAYOUT_CYCLE_ORDER[0],
      ]
      expect(seen).toEqual(expected)
      expect(s.preferredLayout).toBe('cose')
    })

    it('returns the next layout and mutates preferredLayout', () => {
      const s = useGraphExplorerStore()
      const next = s.cyclePreferredLayout()
      expect(next).toBe('breadthfirst')
      expect(s.preferredLayout).toBe(next)
    })

    it('wraps from the last entry back to the first', () => {
      const s = useGraphExplorerStore()
      s.preferredLayout = GRAPH_LAYOUT_CYCLE_ORDER[GRAPH_LAYOUT_CYCLE_ORDER.length - 1]
      const next = s.cyclePreferredLayout()
      expect(next).toBe(GRAPH_LAYOUT_CYCLE_ORDER[0])
    })

    it('handles an unknown current layout (indexOf -1 → Math.max(0,…) branch)', () => {
      const s = useGraphExplorerStore()
      // Force an out-of-cycle value; indexOf returns -1, clamped to 0,
      // so the next layout is order[1] = 'breadthfirst'.
      s.preferredLayout = 'bogus' as never
      const next = s.cyclePreferredLayout()
      expect(next).toBe('breadthfirst')
    })
  })

  describe('GRAPH_LAYOUT_CYCLE_ORDER', () => {
    it('contains the RFC-080 layout set in order', () => {
      expect(GRAPH_LAYOUT_CYCLE_ORDER).toEqual([
        'cose',
        'breadthfirst',
        'circle',
        'grid',
        'timeline',
      ])
    })
  })
})
