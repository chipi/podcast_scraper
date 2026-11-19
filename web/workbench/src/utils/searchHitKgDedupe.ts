import type { SearchHit } from '../api/searchApi'

/**
 * True when this hit is a merged kg_entity / kg_topic row (same embedded text, multiple episodes).
 * For these rows we only offer **G**; **L** / **E** would misleadingly target a single representative episode.
 */
export function isKgSurfaceMultiEpisodeDedupe(hit: SearchHit): boolean {
  const dt = String(hit.metadata?.doc_type ?? '').toLowerCase()
  if (dt !== 'kg_entity' && dt !== 'kg_topic') {
    return false
  }
  const n = Number(hit.metadata?.kg_surface_match_count)
  return Number.isFinite(n) && n > 1
}
