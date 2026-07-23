/**
 * Collapse foldable hits within a single episode-group's hit list so the search
 * results read as one summary row per (episode, source), not one row per hit
 * (#1261-3). Ported from the operator viewer's
 * ``collapseTranscriptHitsByEpisode`` (see gi-kg-viewer), tuned for the
 * consumer episode-first grouping.
 *
 * Foldable doc_types:
 *   - ``transcript`` — multiple matching chunks fold into a "Transcript ·
 *     N matches" summary row that expands to reveal the ordered excerpts.
 *   - ``episode_title`` / ``episode_description`` / ``summary_short`` —
 *     metadata-surface hits; each surface folds separately so a match in
 *     both title and description reads as two summary rows, not one.
 *
 * Insight / quote / kg_topic / kg_entity rows pass through untouched — each
 * carries a distinct claim worth surfacing individually.
 */

import type { SearchHit } from '../services/types'

export type FoldedKind = 'transcript' | 'episode_title' | 'episode_description' | 'summary_short'

export interface FoldedHitCluster {
  __kind: 'folded_cluster'
  foldedKind: FoldedKind
  members: SearchHit[]
  topScore: number
}

export type CollapsedRow = SearchHit | FoldedHitCluster

export function isFoldedCluster(row: CollapsedRow): row is FoldedHitCluster {
  return (row as FoldedHitCluster).__kind === 'folded_cluster'
}

const FOLDABLE: readonly FoldedKind[] = [
  'transcript',
  'episode_title',
  'episode_description',
  'summary_short',
]

function foldableKind(hit: SearchHit): FoldedKind | null {
  const md = hit.metadata as Record<string, unknown>
  const docType = typeof md.doc_type === 'string' ? md.doc_type : ''
  if (!(FOLDABLE as readonly string[]).includes(docType)) return null
  // Compound hits (transcript segment lifted into a GI insight) render as
  // their own valuable card — don't fold them into a "N matches" summary.
  if (docType === 'transcript' && hit.lifted != null && typeof hit.lifted === 'object') {
    return null
  }
  return docType as FoldedKind
}

/**
 * Given the ordered hits for one episode group, return a mixed list where
 * multiple hits of the same foldable kind collapse into a single cluster
 * placeholder occupying the slot of the highest-scoring member.
 */
export function collapseFoldableHits(hits: readonly SearchHit[]): CollapsedRow[] {
  const out: CollapsedRow[] = []
  const kindToIndex = new Map<FoldedKind, number>()
  for (const hit of hits) {
    const kind = foldableKind(hit)
    if (kind === null) {
      out.push(hit)
      continue
    }
    const existing = kindToIndex.get(kind)
    if (existing != null) {
      const cluster = out[existing] as FoldedHitCluster
      cluster.members.push(hit)
      if (hit.score > cluster.topScore) cluster.topScore = hit.score
      continue
    }
    const cluster: FoldedHitCluster = {
      __kind: 'folded_cluster',
      foldedKind: kind,
      members: [hit],
      topScore: hit.score,
    }
    kindToIndex.set(kind, out.length)
    out.push(cluster)
  }
  return out
}
