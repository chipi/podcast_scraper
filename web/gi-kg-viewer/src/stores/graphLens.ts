import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useCorpusLensStore } from './corpusLens'
import { localYmdDaysAgo } from '../utils/localCalendarDate'

/**
 * Graph-tab-only time lens (independent from Digest/Library ``corpusLens`` after first seed).
 */
export const useGraphLensStore = defineStore('graphLens', () => {
  /** Lower bound YYYY-MM-DD (local calendar); empty string = all time. */
  const sinceYmd = ref('')
  /** True after ``seedFromCorpusLensIfNeeded`` for this corpus session. */
  const seeded = ref(false)
  /** True after the user has opened the Graph tab at least once this browser session. */
  const graphTabOpenedThisSession = ref(false)
  /** Last corpus API auto-sync applied the episode cap (pool larger than cap). */
  const lastAutoLoadWasCapped = ref(false)

  function markGraphTabOpenedOnce(): void {
    if (graphTabOpenedThisSession.value) return
    graphTabOpenedThisSession.value = true
  }

  function seedFromCorpusLensIfNeeded(): void {
    if (seeded.value) return
    const corpus = useCorpusLensStore()
    const s = corpus.sinceYmd?.trim() ?? ''
    sinceYmd.value = s ? s : localYmdDaysAgo(7)
    seeded.value = true
  }

  function resetForNewCorpus(): void {
    seeded.value = false
    sinceYmd.value = ''
    lastAutoLoadWasCapped.value = false
  }

  function setSinceYmd(next: string): void {
    sinceYmd.value = next.trim()
    seeded.value = true
  }

  function setPresetDays(days: 7 | 30 | 90): void {
    sinceYmd.value = localYmdDaysAgo(days)
    seeded.value = true
  }

  function setAllTime(): void {
    sinceYmd.value = ''
    seeded.value = true
  }

  function setLastAutoLoadCapped(v: boolean): void {
    lastAutoLoadWasCapped.value = v
  }

  return {
    sinceYmd,
    seeded,
    graphTabOpenedThisSession,
    lastAutoLoadWasCapped,
    markGraphTabOpenedOnce,
    seedFromCorpusLensIfNeeded,
    resetForNewCorpus,
    setSinceYmd,
    setPresetDays,
    setAllTime,
    setLastAutoLoadCapped,
  }
})
