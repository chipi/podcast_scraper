import type { RawGraphNode } from '../types/artifact'

/**
 * vis-network group / Cytoscape data.type for styling. Entity split by entity_kind.
 */
export function visualGroupForNode(n: RawGraphNode | null | undefined): string {
  if (!n || typeof n !== 'object') return '?'
  const t = typeof n.type === 'string' ? n.type : '?'
  if (t !== 'Entity') return t
  const p = n.properties || {}
  const raw = p.entity_kind
  if (typeof raw !== 'string' || !raw.trim()) {
    return 'Entity_person'
  }
  const k = raw.trim().toLowerCase()
  const isOrg =
    k === 'organization' ||
    k === 'org' ||
    k === 'company' ||
    k === 'corporation' ||
    k === 'institution'
  return isOrg ? 'Entity_organization' : 'Entity_person'
}

export function visualNodeTypeCounts(nodes: RawGraphNode[]): Record<string, number> {
  const nt: Record<string, number> = {}
  const arr = Array.isArray(nodes) ? nodes : []
  for (const n of arr) {
    const g = visualGroupForNode(n)
    nt[g] = (nt[g] || 0) + 1
  }
  return nt
}
