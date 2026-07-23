/**
 * Aggregate ``metadata.query_enrichments.related_topics`` from a search-result
 * page into a deduped, ranked list for the "Also about:" chip row above episode
 * groups (#1261-2).
 *
 * The server (RFC-088 QueryEnricher chain) attaches a per-hit list of
 * ``{ topic_id, topic_label?, similarity }`` when ``enrich_results=true``. We
 * union across hits: each topic's ``score`` is the *max* similarity seen
 * (higher = more relevant to *some* hit on the page) and ``count`` is how many
 * hits referenced it (higher = more consistent signal across the page).
 *
 * Sort order: score desc, then count desc, then topicId asc — deterministic
 * so component tests can assert positions without race conditions.
 */

import type { SearchHit } from '../services/types'

export interface RelatedTopicChip {
  topicId: string
  label: string
  score: number
  count: number
}

interface ServerRelatedTopic {
  topic_id?: unknown
  topic_label?: unknown
  similarity?: unknown
}

function readRelated(hit: SearchHit): ServerRelatedTopic[] {
  const md = hit.metadata as Record<string, unknown>
  const enrich = md.query_enrichments
  if (!enrich || typeof enrich !== 'object') return []
  const raw = (enrich as Record<string, unknown>).related_topics
  return Array.isArray(raw) ? (raw as ServerRelatedTopic[]) : []
}

export function aggregateRelatedTopics(
  hits: readonly SearchHit[],
  limit = 8,
): RelatedTopicChip[] {
  const byId = new Map<string, RelatedTopicChip>()
  for (const hit of hits) {
    for (const related of readRelated(hit)) {
      const id = typeof related.topic_id === 'string' ? related.topic_id.trim() : ''
      if (!id) continue
      const sim = typeof related.similarity === 'number' && Number.isFinite(related.similarity)
        ? related.similarity
        : 0
      const label = typeof related.topic_label === 'string' && related.topic_label.trim()
        ? related.topic_label.trim()
        : id
      const existing = byId.get(id)
      if (existing) {
        if (sim > existing.score) existing.score = sim
        existing.count += 1
        // A late-arriving label wins only if the current one is the raw id (no
        // real label was ever attached).
        if (existing.label === existing.topicId && label !== id) existing.label = label
      } else {
        byId.set(id, { topicId: id, label, score: sim, count: 1 })
      }
    }
  }
  const chips = Array.from(byId.values())
  chips.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score
    if (b.count !== a.count) return b.count - a.count
    return a.topicId.localeCompare(b.topicId)
  })
  return chips.slice(0, Math.max(0, limit))
}
