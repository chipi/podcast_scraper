import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import type { SearchHit } from '../api/searchApi'

/**
 * RFC-094 §3 OQ-2 — the **active search/filter context**, shared read-only across
 * surfaces. One writer (the search store, on a successful run); many readers.
 *
 * Library reads it to rank episodes by **hybrid relevance** (PRD-033 FR2.2) and to
 * render the "why this episode" snippet on rows when a context is active (FR2.1).
 * Graph will later weight nodes by the same context. The store is a thin projection
 * of ``/api/search`` hits onto episodes — it does not call the API itself.
 */

/** Best-scoring search hit projected onto one episode. */
export interface EpisodeRelevance {
  episodeId: string
  /** Best hybrid score among this episode's hits (higher = more relevant). */
  score: number
  /** "Why this episode" — the top-scoring segment/insight text for the context. */
  snippet: string
  /** doc_type of the snippet source (insight, transcript, quote, …). */
  docType: string
}

function episodeIdOf(hit: SearchHit): string {
  const raw = hit.metadata?.['episode_id']
  return typeof raw === 'string' ? raw.trim() : ''
}

function docTypeOf(hit: SearchHit): string {
  const raw = hit.metadata?.['doc_type']
  return typeof raw === 'string' ? raw : ''
}

function snippetOf(hit: SearchHit, maxLen = 220): string {
  const text = (hit.text ?? '').trim().replace(/\s+/g, ' ')
  return text.length > maxLen ? `${text.slice(0, maxLen - 1).trimEnd()}…` : text
}

export const useActiveSearchContextStore = defineStore('activeSearchContext', () => {
  /** The query that produced the current context ('' when no context is active). */
  const query = ref('')
  /** episode_id → its best relevance projection. */
  const byEpisode = ref<Map<string, EpisodeRelevance>>(new Map())

  /** True when a non-empty query produced at least one episode-attributable hit. */
  const active = computed(() => query.value.length > 0 && byEpisode.value.size > 0)

  /**
   * Project ``/api/search`` hits onto episodes (best score wins per episode) and
   * record the query. Called by the search store after a successful run.
   */
  function setContext(queryText: string, hits: readonly SearchHit[]): void {
    const map = new Map<string, EpisodeRelevance>()
    for (const hit of hits) {
      const episodeId = episodeIdOf(hit)
      if (!episodeId) continue
      const score = Number.isFinite(hit.score) ? Number(hit.score) : 0
      const existing = map.get(episodeId)
      if (!existing || score > existing.score) {
        map.set(episodeId, { episodeId, score, snippet: snippetOf(hit), docType: docTypeOf(hit) })
      }
    }
    query.value = queryText.trim()
    byEpisode.value = map
  }

  /** Drop the context (search cleared). */
  function clear(): void {
    query.value = ''
    byEpisode.value = new Map()
  }

  /** Relevance for an episode under the active context, or null if none/inactive. */
  function relevanceFor(episodeId: string | null | undefined): EpisodeRelevance | null {
    if (!active.value || !episodeId) return null
    return byEpisode.value.get(episodeId) ?? null
  }

  return { query, byEpisode, active, setContext, clear, relevanceFor }
})
