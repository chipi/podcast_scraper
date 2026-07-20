import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import {
  inferCorpusLensPreset,
  type CorpusLensPreset,
  localYmdDaysAgo,
} from '../utils/localCalendarDate'
import { useUserPreferencesStore } from './userPreferences'

/**
 * Shared publish-date lower bound for **Digest** and **Library** (YYYY-MM-DD local
 * calendar). Empty string means **all time**, matching Library ``since`` omitted.
 *
 * USERPREFS-1 (#1215): we persist the **preset** ('all' | 7 | 30 | 90), not the
 * calculated YYYY-MM-DD, because the calculated value shifts as time passes.
 * On hydrate the preset is re-applied via ``setPreset`` so ``sinceYmd`` is
 * always today-relative.
 */

const PREF_KEY = 'corpusLensPreset'

type PresetInput = 'all' | 7 | 30 | 90
type PersistedPreset = 'all' | '7' | '30' | '90'

function parsePreset(v: unknown): PersistedPreset | null {
  if (v === 'all' || v === '7' || v === '30' || v === '90') return v
  // Tolerate a numeric write from an older version.
  if (v === 7) return '7'
  if (v === 30) return '30'
  if (v === 90) return '90'
  return null
}

function presetInputFromPersisted(p: PersistedPreset): PresetInput {
  if (p === 'all') return 'all'
  return Number(p) as 7 | 30 | 90
}

export const useCorpusLensStore = defineStore('corpusLens', () => {
  const sinceYmd = ref('')
  const userPrefs = useUserPreferencesStore()
  let applyingRemote = false

  const activePreset = computed((): CorpusLensPreset =>
    inferCorpusLensPreset(sinceYmd.value),
  )

  function setPreset(kind: PresetInput): void {
    if (kind === 'all') {
      sinceYmd.value = ''
    } else {
      sinceYmd.value = localYmdDaysAgo(kind)
    }
  }

  function reset(): void {
    sinceYmd.value = ''
  }

  /* Write-through to USERPREFS-1 whenever the local preset changes,
     but only if the change was user-initiated (not hydration echo). */
  watch(activePreset, (preset) => {
    if (applyingRemote) return
    void userPrefs.set(PREF_KEY, preset)
  })

  /* Apply server-hydrated value once the preferences store lands one.
     applyingRemote suppresses the write-through echo. */
  watch(
    () => userPrefs.get(PREF_KEY),
    (v) => {
      const parsed = parsePreset(v)
      if (parsed === null) return
      if (parsed === activePreset.value) return
      applyingRemote = true
      setPreset(presetInputFromPersisted(parsed))
      void Promise.resolve().then(() => {
        applyingRemote = false
      })
    },
    { immediate: true },
  )

  return {
    sinceYmd,
    activePreset,
    setPreset,
    reset,
  }
})
