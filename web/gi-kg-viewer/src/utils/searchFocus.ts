import type { SearchHit } from '../api/searchApi'

/** Map a search hit to a Cytoscape node id when the indexed doc maps to a graph node. */
export function graphNodeIdFromSearchHit(row: SearchHit): string | null {
  const dt = String(row.metadata?.doc_type ?? '').toLowerCase()
  const sid = row.metadata?.source_id
  if (typeof sid !== 'string' || !sid.trim()) {
    return null
  }
  const focusable = ['insight', 'quote', 'kg_topic', 'kg_entity']
  if (!focusable.includes(dt)) {
    return null
  }
  return sid.trim()
}
