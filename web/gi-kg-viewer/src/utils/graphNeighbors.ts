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
