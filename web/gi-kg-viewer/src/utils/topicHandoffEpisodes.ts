import type { CorpusResolvedEpisodeArtifact } from '../api/corpusLibraryApi'
import { artifactRelPathsForResolvedRow, sortResolvedArtifactsNewestFirst } from './resolvedArtifacts'

/** How many of a topic's episodes a "View in graph" handoff loads when the topic is not on the canvas. */
export const TOPIC_HANDOFF_EPISODE_CAP = 3

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
