/**
 * Summarize which corpus fields matched the query for one episode's hit list —
 * shown in the episode-group header as "Matched: Title · Summary · Transcript"
 * (#1261-5). Ported from the operator viewer's TranscriptClusterCard label
 * logic (see gi-kg-viewer/src/components/search/TranscriptClusterCard.vue),
 * with the same ``matched_field`` / ``doc_type`` fallback chain.
 *
 * Priority:
 *   1. ``metadata.matched_field`` — set explicitly by the indexer (commit
 *      049b7736) for episode-level metadata surfaces.
 *   2. ``metadata.doc_type`` — the row's own kind carries the same signal
 *      when ``matched_field`` didn't round-trip through the aux table.
 *
 * Rows that resolve to nothing (kg_topic, kg_entity, quote, insight without a
 * matched_field) are excluded — they aren't episode-level metadata surfaces.
 */

import type { SearchHit } from '../services/types'

export type MatchedFieldLabel =
  | 'Title'
  | 'Description'
  | 'Summary'
  | 'Summary bullet'
  | 'Transcript'
  | 'Insight'

export interface MatchedFieldBreakdown {
  label: MatchedFieldLabel
  count: number
}

/** Display order — fixed so the summary reads left-to-right predictably. */
export const MATCHED_FIELD_ORDER: readonly MatchedFieldLabel[] = [
  'Title',
  'Description',
  'Summary',
  'Summary bullet',
  'Transcript',
  'Insight',
] as const

export function matchedFieldLabel(hit: SearchHit): MatchedFieldLabel | null {
  const md = (hit.metadata ?? {}) as Record<string, unknown>
  const raw = typeof md.matched_field === 'string' ? md.matched_field : ''
  switch (raw) {
    case 'title':
      return 'Title'
    case 'description':
      return 'Description'
    case 'summary':
      return 'Summary'
    case 'summary_bullet':
      return 'Summary bullet'
    case 'transcript':
      return 'Transcript'
    default:
      break
  }
  const docType = typeof md.doc_type === 'string' ? md.doc_type : ''
  switch (docType) {
    case 'transcript':
      return 'Transcript'
    case 'episode_title':
      return 'Title'
    case 'episode_description':
      return 'Description'
    case 'summary_short':
      return 'Summary'
    case 'summary':
      return 'Summary bullet'
    case 'insight':
      return 'Insight'
    default:
      return null
  }
}

export function summarizeMatchedFields(hits: readonly SearchHit[]): MatchedFieldBreakdown[] {
  const counts = new Map<MatchedFieldLabel, number>()
  for (const hit of hits) {
    const label = matchedFieldLabel(hit)
    if (label === null) continue
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  const out: MatchedFieldBreakdown[] = []
  for (const label of MATCHED_FIELD_ORDER) {
    const c = counts.get(label)
    if (c) out.push({ label, count: c })
  }
  return out
}
