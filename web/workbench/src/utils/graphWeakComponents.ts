import type { ParsedArtifact } from '../types/artifact'

/**
 * Weakly connected components on the filtered artifact (nodes + edges; undirected).
 */
export function weaklyConnectedComponentCount(art: ParsedArtifact | null): number {
  const nodes = art?.data?.nodes
  const edges = art?.data?.edges
  if (!Array.isArray(nodes) || nodes.length === 0) {
    return 0
  }
  const ids = new Set<string>()
  for (const n of nodes) {
    if (!n || n.id == null) continue
    const id = String(n.id).trim()
    if (id) ids.add(id)
  }
  if (ids.size === 0) {
    return 0
  }
  const adj = new Map<string, Set<string>>()
  for (const id of ids) {
    adj.set(id, new Set())
  }
  if (Array.isArray(edges)) {
    for (const e of edges) {
      if (!e) continue
      const a = String(e.from ?? '').trim()
      const b = String(e.to ?? '').trim()
      if (!a || !b || a === b) continue
      if (!ids.has(a) || !ids.has(b)) continue
      adj.get(a)!.add(b)
      adj.get(b)!.add(a)
    }
  }
  const seen = new Set<string>()
  let components = 0
  for (const id of ids) {
    if (seen.has(id)) continue
    components += 1
    const stack = [id]
    seen.add(id)
    while (stack.length) {
      const cur = stack.pop()!
      for (const nb of adj.get(cur) ?? []) {
        if (!seen.has(nb)) {
          seen.add(nb)
          stack.push(nb)
        }
      }
    }
  }
  return components
}
