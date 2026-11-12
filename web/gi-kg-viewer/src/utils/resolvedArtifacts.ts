/**
 * Catalog-resolved episode artifacts: newest-first ordering, and the
 * paths a resolved row contributes (its KG artifact, and what a build adds).
 */
import { episodeArtifactRelPaths } from './episodeArtifactPaths'
import type { CorpusResolvedEpisodeArtifact } from '../api/corpusLibraryApi'

function publishDateKey(iso: string | null | undefined): number {
  if (!iso?.trim()) {
    return 0
  }
  const t = Date.parse(iso.slice(0, 10))
  return Number.isFinite(t) ? t : 0
}

/** Newest first; tie-break episode_id ascending. */
export function sortResolvedArtifactsNewestFirst(
  rows: CorpusResolvedEpisodeArtifact[],
): CorpusResolvedEpisodeArtifact[] {
  return [...rows].sort((a, b) => {
    const da = publishDateKey(a.publish_date)
    const db = publishDateKey(b.publish_date)
    if (db !== da) {
      return db - da
    }
    return a.episode_id.localeCompare(b.episode_id)
  })
}

/** The row's KG artifact, then the files a build adds for it (extension point `episodeArtifactPaths`). */
export function artifactRelPathsForResolvedRow(r: CorpusResolvedEpisodeArtifact): string[] {
  return episodeArtifactRelPaths(r as unknown as Parameters<typeof episodeArtifactRelPaths>[0])
}
