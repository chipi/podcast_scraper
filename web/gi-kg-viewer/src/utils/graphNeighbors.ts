/**
 * Neighbor rows for a node in a parsed graph artifact (GI/KG / merged view).
 */
import type { ParsedArtifact, RawGraphEdge } from '../types/artifact'
import { findRawNodeInArtifact, nodeLabel } from './parsing'
import { visualGroupForNode } from './visualGroup'

export interface GraphNeighborRow {
  id: string
  label: string
  type: string
  /** ``visualGroupForNode`` key — matches graph node colors. */
  visualType: string
  edgeType: string
  direction: 'in' | 'out'
  /**
   * When rows are merged from a TopicCluster, graph ids of member **Topic** nodes that had an
   * edge to this neighbor.
   */
  viaMemberTopicIds?: string[]
}

/**
 * Merge neighbor rows that point at the same graph id (e.g. two topics in one cluster both
 * link to the same episode).
 */
export function mergeNeighborRowsByNeighborId(rows: GraphNeighborRow[]): GraphNeighborRow[] {
  const map = new Map<string, GraphNeighborRow>()
  for (const r of rows) {
    const ex = map.get(r.id)
    if (!ex) {
      map.set(r.id, {
        ...r,
        viaMemberTopicIds: r.viaMemberTopicIds ? [...r.viaMemberTopicIds] : undefined,
      })
      continue
    }
    const a = String(ex.edgeType ?? '').trim()
    const b = String(r.edgeType ?? '').trim()
    if (a && b && a !== b) {
      ex.edgeType = `${a} · ${b}`
    } else if (!a && b) {
      ex.edgeType = b
    }
    const via = new Set<string>([
      ...(ex.viaMemberTopicIds ?? []),
      ...(r.viaMemberTopicIds ?? []),
    ])
    if (via.size > 0) {
      ex.viaMemberTopicIds = [...via]
    }
  }
  return [...map.values()]
}

/**
 * Union of incident edges for several topic (or other) nodes — used for TopicCluster rail detail.
 */
export function graphNeighborsForMemberGraphIds(
  art: ParsedArtifact | null,
  memberGraphNodeIds: string[],
): GraphNeighborRow[] {
  const acc: GraphNeighborRow[] = []
  for (const memberId of memberGraphNodeIds) {
    for (const row of graphNeighborsForNode(art, memberId)) {
      acc.push({ ...row, viaMemberTopicIds: [memberId] })
    }
  }
  return mergeNeighborRowsByNeighborId(acc)
}

export function graphNeighborsForNode(
  art: ParsedArtifact | null,
  nodeId: string | null,
): GraphNeighborRow[] {
  if (!art?.data || nodeId == null) return []
  const edges: RawGraphEdge[] = Array.isArray(art.data.edges) ? art.data.edges : []
  const sid = String(nodeId)
  const out: GraphNeighborRow[] = []
  const seen = new Set<string>()
  for (const e of edges) {
    if (!e) continue
    const from = e.from != null ? String(e.from) : ''
    const to = e.to != null ? String(e.to) : ''
    let neighborId: string | null = null
    let direction: 'in' | 'out' = 'out'
    if (from === sid && to && to !== sid) {
      neighborId = to
      direction = 'out'
    } else if (to === sid && from && from !== sid) {
      neighborId = from
      direction = 'in'
    }
    if (!neighborId) continue
    if (String(e.type ?? '') === '_tc_cohesion') continue
    const key = `${neighborId}:${direction}:${e.type ?? ''}`
    if (seen.has(key)) continue
    seen.add(key)
    const nNode = findRawNodeInArtifact(art, neighborId)
    out.push({
      id: neighborId,
      label: nNode ? nodeLabel(nNode) : neighborId,
      type: nNode ? String(nNode.type ?? '?') : '?',
      visualType: visualGroupForNode(nNode),
      edgeType: String(e.type ?? ''),
      direction,
    })
  }
  return out
}
