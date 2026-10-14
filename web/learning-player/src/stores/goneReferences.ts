/**
 * The listener's stored references that no longer exist in the corpus (#2345).
 *
 * A theme, storyline, topic or person a listener followed, saved, put on a board or started a path
 * on can leave the corpus. A RENAME is re-keyed on the server and never shows here; what is truly
 * gone is listed here, so each list can show it muted, "No longer available", with its own remove
 * action — never silently dropped. Loaded once a session; best effort (an error leaves it empty).
 */
import { defineStore } from 'pinia'
import { getGoneReferences } from '../services/api'

export const useGoneReferencesStore = defineStore('goneReferences', {
  state: () => ({ refs: new Set<string>(), loaded: false }),
  getters: {
    isGone:
      (s) =>
      (ref: string | null | undefined): boolean =>
        !!ref && s.refs.has(ref),
  },
  actions: {
    async ensureLoaded(): Promise<void> {
      if (this.loaded) return
      this.loaded = true
      try {
        this.refs = new Set((await getGoneReferences()).map((g) => g.ref))
      } catch {
        // Signed out or offline: nothing is marked, every list renders as before.
      }
    },
  },
})
