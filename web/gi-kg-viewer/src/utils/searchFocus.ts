import type { Core } from 'cytoscape'
import type { SearchHit } from '../api/searchApi'

const FOCUSABLE_DOC_TYPES = new Set(['insight', 'quote', 'kg_topic', 'kg_entity'])

/**
 * Return the raw source_id for a search hit when the doc_type maps to a graph
 * node.  This is the *canonical* id stored in the FAISS index — it does NOT
 * include the `g:` / `k:` prefix that `mergeGiKg.ts` adds when combining
 * artifacts.  Use `resolveCyNodeId` to find the actual Cytoscape node.
 */
export function graphNodeIdFromSearchHit(row: SearchHit): string | null {
  const dt = String(row.metadata?.doc_type ?? '').toLowerCase()
  const sid = row.metadata?.source_id
  if (typeof sid !== 'string' || !sid.trim()) {
    return null
  }
  if (!FOCUSABLE_DOC_TYPES.has(dt)) {
    return null
  }
  return sid.trim()
}

/**
 * Resolve a raw source_id (from the search index) to the actual Cytoscape node
 * id.  The merged GI+KG graph prefixes every node id with `g:` (GI) or `k:`
 * (KG), and KG files sometimes use `kg:` internally, so the Cytoscape id may
 * be `k:kg:topic:foo` instead of `topic:foo`.
 *
 * Tries the bare id first, then common prefixed variants.  Returns the first
 * match or null.
 */
export function resolveCyNodeId(core: Core, rawId: string): string | null {
  if (!rawId) return null
  const candidates = [
    rawId,
    `g:${rawId}`,
    `k:${rawId}`,
    `k:kg:${rawId}`,
    `g:gi:${rawId}`,
  ]
  for (const c of candidates) {
    if (!core.$id(c).empty()) return c
  }
  return null
}
