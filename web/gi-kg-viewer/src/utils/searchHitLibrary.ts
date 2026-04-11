import type { SearchHit } from '../api/searchApi'

/**
 * Corpus-relative path to episode metadata (``*.metadata.json``), stamped by the
 * vector indexer as ``source_metadata_relative_path`` on each hit.
 */
export function sourceMetadataRelativePathFromSearchHit(hit: SearchHit): string | null {
  const p = hit.metadata?.source_metadata_relative_path
  if (typeof p !== 'string') {
    return null
  }
  const t = p.trim()
  return t.length ? t : null
}
