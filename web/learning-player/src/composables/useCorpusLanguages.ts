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
/** Each show's language, from the same catalogue fetch — Search's fallback for a result whose hits
 *  carry none (only source-layer chunks of a translated episode do). */
const byFeed = ref<ReadonlyMap<string, string>>(new Map())
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
      const feeds = new Map<string, string>()
      for (const show of shows) {
        const lang = primaryLanguage(show.language)
        if (!lang) continue
        found.add(lang)
        feeds.set(show.feed_id, lang)
      }
      byFeed.value = feeds
      languages.value = found
    })
    .catch(() => undefined)
    .finally(() => {
      pending = null
    })
}

export function useCorpusLanguages() {
  load()
  const multilingual = computed(() => (languages.value?.size ?? 0) > 1)
  /** Whether a badge for `tag` renders — what a CONTAINER whose only content may be the badge must
   *  gate on. Gating on `tag` alone left an empty row wherever the badge itself stays hidden. */
  const badgeShown = (tag: string | null | undefined): boolean =>
    multilingual.value && primaryLanguage(tag) !== null
  const languageOfFeed = (feedId: unknown): string | null =>
    typeof feedId === 'string' ? (byFeed.value.get(feedId) ?? null) : null
  return { multilingual, badgeShown, languageOfFeed }
}

/** Test seam: forget the cached answer so each test starts from an unloaded catalogue. */
export function resetCorpusLanguagesForTests(): void {
  languages.value = null
  byFeed.value = new Map()
  pending = null
}
