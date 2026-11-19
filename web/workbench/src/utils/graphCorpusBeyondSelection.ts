import type { CorpusNodeEpisodeItem } from '../api/corpusLibraryApi'
import { episodeArtifactRelPaths } from './episodeArtifactPaths'

/** Normalize relative artifact paths the same way cross-episode expand compares selections. */
export function normalizeArtifactRelPath(p: string): string {
  return p.trim().replace(/\\/g, '/').replace(/^\.\/+/, '')
}

/**
 * True when ``POST /api/corpus/node-episodes`` would yield at least one artifact the expand loads
 * (``episodeArtifactRelPaths``: KG, plus what a downstream build adds) not already in
 * ``selectedRelPaths``, so merging would add something beyond the current graph selection.
 */
export function wouldCrossEpisodeExpandAppendNewArtifacts(
  episodes: CorpusNodeEpisodeItem[],
  selectedRelPaths: string[],
): boolean {
  const sel = new Set(
    selectedRelPaths.map((p) => normalizeArtifactRelPath(p)).filter(Boolean),
  )
  return episodes.some((ep) =>
    episodeArtifactRelPaths(ep).some((p) => {
      const n = normalizeArtifactRelPath(p)
      return n !== '' && !sel.has(n)
    }),
  )
}
