import { computed, ref } from 'vue'
import { getPodcasts } from '../services/api'

/**
 * Whether the corpus holds more than one language (PRD-047 FR7.5) — the one condition under which a
 * language badge carries information. In a single-language corpus every badge reads the same code,
 * which is exactly the constant #2115 deleted for costing a wrap and saying nothing.
 *
 * Module-level: every badge on a page asks the same question, and one catalogue fetch answers all
 * of them. A failed fetch leaves the answer unknown (no badges) and is retried by the next caller,
 * so an offline start does not pin the app to "monolingual" for the session.
 */
const languages = ref<ReadonlySet<string> | null>(null)
let pending: Promise<void> | null = null

/** `en-US` / `EN` / ` en ` → `en`; empty or missing → null. */
export function primaryLanguage(tag: string | null | undefined): string | null {
  const primary = (tag ?? '').trim().split(/[-_]/)[0].toLowerCase()
  return primary || null
}

function load(): void {
  if (pending || languages.value) return
  pending = getPodcasts()
    .then((shows) => {
      const found = new Set<string>()
      for (const show of shows) {
        const lang = primaryLanguage(show.language)
        if (lang) found.add(lang)
      }
      languages.value = found
    })
    .catch(() => undefined)
    .finally(() => {
      pending = null
    })
}

export function useCorpusLanguages() {
  load()
  return { multilingual: computed(() => (languages.value?.size ?? 0) > 1) }
}

/** Test seam: forget the cached answer so each test starts from an unloaded catalogue. */
export function resetCorpusLanguagesForTests(): void {
  languages.value = null
  pending = null
}
