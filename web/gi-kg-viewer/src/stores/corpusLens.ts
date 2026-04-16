import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import {
  inferCorpusLensPreset,
  type CorpusLensPreset,
  localYmdDaysAgo,
} from '../utils/localCalendarDate'

/**
 * Shared publish-date lower bound for **Digest** and **Library** (YYYY-MM-DD local
 * calendar). Empty string means **all time**, matching Library ``since`` omitted.
 */
export const useCorpusLensStore = defineStore('corpusLens', () => {
  const sinceYmd = ref('')

  const activePreset = computed((): CorpusLensPreset =>
    inferCorpusLensPreset(sinceYmd.value),
  )

  function setPreset(kind: 'all' | 7 | 30 | 90): void {
    if (kind === 'all') {
      sinceYmd.value = ''
    } else {
      sinceYmd.value = localYmdDaysAgo(kind)
    }
  }

  function reset(): void {
    sinceYmd.value = ''
  }

  return {
    sinceYmd,
    activePreset,
    setPreset,
    reset,
  }
})
