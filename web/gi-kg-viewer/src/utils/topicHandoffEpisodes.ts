import type { CorpusResolvedEpisodeArtifact } from '../api/corpusLibraryApi'
import type { TopicClustersDocument } from '../api/corpusTopicClustersApi'
import { artifactRelPathsForResolvedRow, sortResolvedArtifactsNewestFirst } from './clusterSiblingMerge'

/** How many of a topic's episodes a "View in graph" handoff loads when the topic is not on the canvas. */
export const TOPIC_HANDOFF_EPISODE_CAP = 3

/** The episodes `topic_clusters.json` records for a topic (its member row's `episode_ids`). */
export function topicEpisodeIdsFromClusters(
  doc: TopicClustersDocument | null | undefined,
  topicId: string,
): string[] {
  const out = new Set<string>()
  for (const cl of doc?.clusters ?? []) {
    for (const m of cl?.members ?? []) {
      if (m?.topic_id !== topicId) continue
      for (const e of m.episode_ids ?? []) {
        const id = typeof e === 'string' ? e.trim() : ''
        if (id) out.add(id)
      }
    }
  }
  return [...out]
}

/**
 * The gi/kg artifacts to append so a topic that is not on the canvas gets a node to select.
 *
 * Newest episodes first, capped: one episode is enough to draw the topic, and a big topic spans
 * hundreds.
 */
export function topicHandoffArtifactPaths(
  resolved: CorpusResolvedEpisodeArtifact[],
  cap = TOPIC_HANDOFF_EPISODE_CAP,
): string[] {
  const out: string[] = []
  let taken = 0
  for (const row of sortResolvedArtifactsNewestFirst(resolved)) {
    if (taken >= cap) break
    const rels = artifactRelPathsForResolvedRow(row)
    if (rels.length === 0) continue
    out.push(...rels)
    taken++
  }
  return out
}
