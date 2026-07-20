// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
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

  describe('USERPREFS-1 (#1215) adoption', () => {
    it('setPreset(30) write-throughs to userPrefs.set with the persisted preset value', async () => {
      const { useCorpusLensStore } = await import('./corpusLens')
      const { useUserPreferencesStore } = await import('./userPreferences')
      const setSpy = vi.spyOn(useUserPreferencesStore(), 'set')
      const s = useCorpusLensStore()
      s.setPreset(30)
      await new Promise((r) => setTimeout(r, 0))
      // activePreset is the string '30' per CorpusLensPreset type; the
      // watcher writes it through to USERPREFS-1 verbatim.
      expect(setSpy).toHaveBeenCalledWith('corpusLensPreset', '30')
    })

    it('hydrated preset from server applies via setPreset (echo-suppressed)', async () => {
      const { useUserPreferencesStore } = await import('./userPreferences')
      const userPrefs = useUserPreferencesStore()
      // Seed a server-side preset BEFORE loading corpusLens so the immediate
      // watcher picks it up. Pinia setup-stores unwrap refs on the store
      // proxy — assign directly, not via `.value`.
      ;(userPrefs as unknown as { local: Record<string, unknown> }).local = {
        corpusLensPreset: '7',
      }
      const setSpy = vi.spyOn(userPrefs, 'set')
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      await new Promise((r) => setTimeout(r, 0))
      // sinceYmd was moved to today-minus-7 by setPreset.
      expect(s.sinceYmd).toBe(localYmdDaysAgo(7))
      expect(s.activePreset).toBe('7')
      // And the applyingRemote guard suppressed the echo write-back.
      expect(setSpy).not.toHaveBeenCalled()
    })

    it('numeric legacy write (7 instead of "7") is normalised on read', async () => {
      const { useUserPreferencesStore } = await import('./userPreferences')
      const userPrefs = useUserPreferencesStore()
      ;(userPrefs as unknown as { local: Record<string, unknown> }).local = {
        corpusLensPreset: 30,
      }
      const { useCorpusLensStore } = await import('./corpusLens')
      const s = useCorpusLensStore()
      await new Promise((r) => setTimeout(r, 0))
      expect(s.activePreset).toBe('30')
    })
  })
})
