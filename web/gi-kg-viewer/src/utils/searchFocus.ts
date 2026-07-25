import type { Core } from 'cytoscape'
import type { CompareSubjectRef, SearchHit } from '../api/searchApi'

const FOCUSABLE_DOC_TYPES = new Set(['insight', 'quote', 'kg_topic', 'kg_entity'])

/**
 * Search v3 §S8 — map a search hit to the single subject it most naturally
 * represents, for the "Pin to Compare" action. A hit can mention several
 * comparable subjects (the CompareOperatorPanel discovery walk surfaces them
 * all for the picker); pinning needs ONE — the hit's own primary anchor:
 *
 *   * ``kg_topic``  → that Topic (``source_id``).
 *   * ``kg_entity`` → that Person / entity (``source_id``).
 *   * otherwise (insight / quote / transcript) → the speaker (Person) when
 *     one is present, else the hit's Episode.
 *
 * Returns null when nothing usable is on the hit (the caller no-ops).
 */
export function primaryCompareSubjectFromHit(row: SearchHit): CompareSubjectRef | null {
  const md = (row.metadata ?? {}) as Record<string, unknown>
  const docType = typeof md.doc_type === 'string' ? md.doc_type : ''
  const sourceId = typeof md.source_id === 'string' ? md.source_id.trim() : ''
  if (docType === 'kg_topic' && sourceId) {
    const label = typeof md.topic_label === 'string' ? md.topic_label : null
    return { kind: 'topic', id: sourceId, label: label || sourceId }
  }
  if (docType === 'kg_entity' && sourceId) {
    const label = typeof md.entity_label === 'string' ? md.entity_label : null
    return { kind: 'person', id: sourceId, label: label || sourceId }
  }
  const speaker =
    typeof md.speaker_name === 'string'
      ? md.speaker_name
      : typeof md.speaker === 'string'
        ? md.speaker
        : ''
  if (speaker.trim()) {
    return { kind: 'person', id: speaker.trim(), label: speaker.trim() }
  }
  const epId = typeof md.episode_id === 'string' ? md.episode_id.trim() : ''
  if (epId) {
    const epTitle = typeof md.episode_title === 'string' ? md.episode_title : null
    return { kind: 'episode', id: epId, label: epTitle || epId }
  }
  return null
}

/**
 * Return the raw source_id for a search hit when the doc_type maps to a graph
 * node.  This is the *canonical* id stored in the search index — it does NOT
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
 * The episode a search hit belongs to, returned as a focus *fallback*.
 *
 * Some focusable doc types have no node of their own in the merged graph — most
 * notably `quote`: quotes are evidence under Insights and are never rendered as
 * graph nodes (the merged graph carries Episode / Insight / Topic / Person /
 * Organization / Podcast). "Show on graph" on a quote therefore targets a node
 * that does not exist; without a resolvable fallback the focus never lands and
 * the handoff FSM hangs until its 15s stuck-timeout. Falling back to the hit's
 * Episode node (resolved via `resolveCyNodeId` → `__unified_ep__:<id>`) makes
 * the click center the quote's episode instead — useful, and never stuck.
 *
 * Harmless for hits whose own node IS present (insight / kg_topic / kg_entity):
 * the fallback is only consulted when the primary id fails to resolve.
 */
export function episodeFallbackForSearchHit(row: SearchHit): string | null {
  const ep = row.metadata?.episode_id
  return typeof ep === 'string' && ep.trim() ? ep.trim() : null
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
    `__unified_ep__:${rawId}`,
    `episode:${rawId}`,
    `g:episode:${rawId}`,
    `g:ep:${rawId}`,
    `k:episode:${rawId}`,
    `k:kg:episode:${rawId}`,
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
