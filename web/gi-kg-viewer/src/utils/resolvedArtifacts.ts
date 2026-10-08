/**
 * Catalog-resolved episode artifacts: which episodes are loaded, newest-first ordering, and the
 * gi/kg/bridge paths a resolved row contributes.
 */
import type { ParsedArtifact } from '../types/artifact'
import type { CorpusResolvedEpisodeArtifact } from '../api/corpusLibraryApi'

export function episodeIdsFromParsedArtifacts(parsed: ParsedArtifact[]): Set<string> {
  const s = new Set<string>()
  for (const p of parsed) {
    if (p.kind !== 'gi') {
      continue
    }
    const data = p.data as { episode_id?: unknown }
    const eid = data.episode_id
    if (typeof eid === 'string' && eid.trim()) {
      s.add(eid.trim())
    }
  }
  return s
}

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

export function artifactRelPathsForResolvedRow(r: CorpusResolvedEpisodeArtifact): string[] {
  const out: string[] = []
  if (r.gi_relative_path?.trim()) {
    out.push(r.gi_relative_path.trim())
  }
  if (r.kg_relative_path?.trim()) {
    out.push(r.kg_relative_path.trim())
  }
  if (r.bridge_relative_path?.trim()) {
    out.push(r.bridge_relative_path.trim())
  }
  return out
}
