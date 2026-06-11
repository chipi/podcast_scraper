// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { localYmdDaysAgo } from '../utils/localCalendarDate'

describe('useCorpusLensStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('defaults to all-time (empty sinceYmd, activePreset "all")', async () => {
    const { useCorpusLensStore } = await import('./corpusLens')
    const s = useCorpusLensStore()
    expect(s.sinceYmd).toBe('')
    expect(s.activePreset).toBe('all')
  })

  describe('setPreset', () => {
    it('"all" clears sinceYmd and yields preset "all"', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.setPreset(7)
      s.setPreset('all')
      expect(s.sinceYmd).toBe('')
      expect(s.activePreset).toBe('all')
    })

    it('7 sets sinceYmd to today-7 and preset "7"', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.setPreset(7)
      expect(s.sinceYmd).toBe(localYmdDaysAgo(7))
      expect(s.activePreset).toBe('7')
    })

    it('30 sets sinceYmd to today-30 and preset "30"', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.setPreset(30)
      expect(s.sinceYmd).toBe(localYmdDaysAgo(30))
      expect(s.activePreset).toBe('30')
    })

    it('90 sets sinceYmd to today-90 and preset "90"', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.setPreset(90)
      expect(s.sinceYmd).toBe(localYmdDaysAgo(90))
      expect(s.activePreset).toBe('90')
    })

    it('switching between numeric presets overwrites sinceYmd', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.setPreset(7)
      expect(s.activePreset).toBe('7')
      s.setPreset(90)
      expect(s.sinceYmd).toBe(localYmdDaysAgo(90))
      expect(s.activePreset).toBe('90')
    })
  })

  describe('activePreset (custom + edge cases)', () => {
    it('arbitrary date that matches no preset is "custom"', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.sinceYmd = '2000-01-01'
      expect(s.activePreset).toBe('custom')
    })

    it('whitespace-only sinceYmd is treated as all-time', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.sinceYmd = '   '
      expect(s.activePreset).toBe('all')
    })

    it('preset boundaries are recognized after manual assignment', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.sinceYmd = localYmdDaysAgo(30)
      expect(s.activePreset).toBe('30')
    })

    it('activePreset reacts to direct sinceYmd mutation', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      expect(s.activePreset).toBe('all')
      s.sinceYmd = localYmdDaysAgo(7)
      expect(s.activePreset).toBe('7')
      s.sinceYmd = 'not-a-preset'
      expect(s.activePreset).toBe('custom')
    })
  })

  describe('reset', () => {
    it('clears a numeric preset back to all-time', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.setPreset(30)
      s.reset()
      expect(s.sinceYmd).toBe('')
      expect(s.activePreset).toBe('all')
    })

    it('clears a custom date back to all-time', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.sinceYmd = '1999-12-31'
      expect(s.activePreset).toBe('custom')
      s.reset()
      expect(s.sinceYmd).toBe('')
      expect(s.activePreset).toBe('all')
    })

    it('is idempotent when already all-time', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      s.reset()
      s.reset()
      expect(s.sinceYmd).toBe('')
      expect(s.activePreset).toBe('all')
    })
  })
})
