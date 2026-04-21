import { defineStore } from 'pinia'
import { ref } from 'vue'
import { localYmdDaysAgo } from '../utils/localCalendarDate'
import { useCorpusLensStore } from './corpusLens'

export type GraphLayoutName = 'cose' | 'breadthfirst' | 'circle' | 'grid'

/** Order for the graph layout cycle control (toolbar / bottom bar). */
export const GRAPH_LAYOUT_CYCLE_ORDER: readonly GraphLayoutName[] = [
  'cose',
  'breadthfirst',
  'circle',
  'grid',
] as const

export const useGraphExplorerStore = defineStore('graphExplorer', () => {
  const preferredLayout = ref<GraphLayoutName>('cose')
  const minimapOpen = ref(false)
  /** Degree histogram bucket id or null = no filter. */
  const activeDegreeBucket = ref<string | null>(null)

  /** Graph tab time lens (independent from Digest/Library ``corpusLens`` after first seed). */
  const sinceYmd = ref('')
  const seeded = ref(false)
  /** True after the user has opened the Graph tab at least once this browser session. */
  const graphTabOpenedThisSession = ref(false)
  /** Last corpus API auto-sync applied the episode cap (pool larger than cap). */
  const lastAutoLoadWasCapped = ref(false)

  function clearDegreeBucket(): void {
    activeDegreeBucket.value = null
  }

  function toggleDegreeBucket(key: string): void {
    activeDegreeBucket.value = activeDegreeBucket.value === key ? null : key
  }

  function resetForNewArtifact(): void {
    activeDegreeBucket.value = null
  }

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

  /**
   * Graph tab “Reset”: same date lower bound as the first auto-sync for this corpus session
   * (Digest/Library corpus lens, else last 7 days local).
   */
  function resetSinceYmdToInitialCorpusSeed(): void {
    const corpus = useCorpusLensStore()
    const s = corpus.sinceYmd?.trim() ?? ''
    sinceYmd.value = s ? s : localYmdDaysAgo(7)
    seeded.value = true
  }

  function resetGraphLensForNewCorpus(): void {
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

  /** Advance ``preferredLayout`` to the next algorithm in ``GRAPH_LAYOUT_CYCLE_ORDER``. */
  function cyclePreferredLayout(): GraphLayoutName {
    const order = GRAPH_LAYOUT_CYCLE_ORDER as readonly GraphLayoutName[]
    const cur = preferredLayout.value
    const i = Math.max(0, order.indexOf(cur))
    const next = order[(i + 1) % order.length]!
    preferredLayout.value = next
    return next
  }

  return {
    preferredLayout,
    minimapOpen,
    activeDegreeBucket,
    clearDegreeBucket,
    toggleDegreeBucket,
    resetForNewArtifact,
    sinceYmd,
    seeded,
    graphTabOpenedThisSession,
    lastAutoLoadWasCapped,
    markGraphTabOpenedOnce,
    seedFromCorpusLensIfNeeded,
    resetSinceYmdToInitialCorpusSeed,
    resetGraphLensForNewCorpus,
    setSinceYmd,
    setPresetDays,
    setAllTime,
    setLastAutoLoadCapped,
    cyclePreferredLayout,
  }
})
