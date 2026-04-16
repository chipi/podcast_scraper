/**
 * Topic-cluster sibling episode auto-load: candidates from ``topic_clusters.json``
 * vs loaded GI episode ids, catalog resolution order, cap.
 */
import type { ParsedArtifact } from '../types/artifact'
import type { TopicClustersCluster, TopicClustersDocument } from '../api/corpusTopicClustersApi'
import type { CorpusResolvedEpisodeArtifact } from '../api/corpusLibraryApi'

export function clusterSiblingEpisodeCap(): number {
  const raw = import.meta.env.VITE_CLUSTER_SIBLING_EPISODE_CAP
  const n = raw != null && String(raw).trim() !== '' ? Number.parseInt(String(raw), 10) : 10
  return Number.isFinite(n) && n >= 0 ? n : 10
}

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

/**
 * Union of episode ids listed on cluster members that share at least one episode id
 * with ``loadedEpisodeIds``, minus already-loaded ids (siblings to pull in).
 */
export function clusterSiblingEpisodeIdCandidates(
  doc: TopicClustersDocument | null | undefined,
  loadedEpisodeIds: Set<string>,
): { candidateIds: string[]; mTotal: number } {
  if (!doc?.clusters?.length) {
    return { candidateIds: [], mTotal: 0 }
  }
  const triggeredUnion = new Set<string>()
  for (const cl of doc.clusters) {
    if (!cl?.members?.length) {
      continue
    }
    const clusterEpisodes = new Set<string>()
    for (const m of cl.members) {
      const eps = m.episode_ids
      if (!Array.isArray(eps)) {
        continue
      }
      for (const e of eps) {
        const id = typeof e === 'string' ? e.trim() : ''
        if (id) {
          clusterEpisodes.add(id)
        }
      }
    }
    const touchesLoaded = [...clusterEpisodes].some((e) => loadedEpisodeIds.has(e))
    if (touchesLoaded) {
      for (const e of clusterEpisodes) {
        triggeredUnion.add(e)
      }
    }
  }
  const mTotal = triggeredUnion.size
  const candidateIds = [...triggeredUnion].filter((e) => !loadedEpisodeIds.has(e))
  return { candidateIds, mTotal }
}

/**
 * Episode ids listed for a single member topic in one cluster (from ``members[].episode_ids``).
 */
export function episodeIdsForClusterMember(
  cluster: TopicClustersCluster | null | undefined,
  topicId: string,
): string[] {
  const want = topicId.trim()
  if (!cluster?.members?.length || !want) {
    return []
  }
  for (const m of cluster.members) {
    const tid = m && typeof m.topic_id === 'string' ? m.topic_id.trim() : ''
    if (tid !== want) {
      continue
    }
    const eps = m.episode_ids
    if (!Array.isArray(eps)) {
      return []
    }
    const out: string[] = []
    for (const e of eps) {
      const id = typeof e === 'string' ? e.trim() : ''
      if (id) {
        out.push(id)
      }
    }
    return out
  }
  return []
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
