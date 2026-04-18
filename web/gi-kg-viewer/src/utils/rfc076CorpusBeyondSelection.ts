import type { CorpusNodeEpisodeItem } from '../api/corpusLibraryApi'

/** Normalize relative artifact paths the same way RFC-076 expand compares selections. */
export function normalizeArtifactRelPath(p: string): string {
  return p.trim().replace(/\\/g, '/').replace(/^\.\/+/, '')
}

/**
 * True when ``POST /api/corpus/node-episodes`` would yield at least one GI/KG file not already in
 * ``selectedRelPaths`` (so merging would add something beyond the current graph selection).
 */
export function wouldRfc076AppendNewArtifacts(
  episodes: CorpusNodeEpisodeItem[],
  selectedRelPaths: string[],
): boolean {
  const sel = new Set(
    selectedRelPaths.map((p) => normalizeArtifactRelPath(p)).filter(Boolean),
  )
  for (const ep of episodes) {
    const gi = normalizeArtifactRelPath(ep.gi_relative_path ?? '')
    const kg = normalizeArtifactRelPath(ep.kg_relative_path ?? '')
    if (gi && !sel.has(gi)) {
      return true
    }
    if (kg && !sel.has(kg)) {
      return true
    }
  }
  return false
}
