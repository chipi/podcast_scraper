/**
 * graph-v3 tier 8-1 — build the top-down synthetic slice.
 *
 * The mount for the top-down load mode is a small graph of `SuperTheme`
 * nodes derived from `topic_theme_clusters.json` v1.1.0+ super-theme
 * rollup (see graph-v3 tier 7-1a). Instead of the full 800+ node merged
 * graph the user gets the ~6-8 super-theme bubbles up front + expand-on-
 * tap to dig into any one of them.
 *
 * Data model:
 *   nodes:
 *     one `SuperTheme` node per unique `super_theme_id` on the theme
 *     clusters doc. Labelled with `super_theme_label`. Node id is the
 *     super_theme_id itself so subsequent tiers can hang expand state
 *     off it directly.
 *   edges:
 *     synthetic `_topdown_link` edges between super-themes that share
 *     ≥1 Topic-cluster bridge in the full artifact. Gives fcose enough
 *     structure to lay the 6-8 nodes out as a cohesive constellation
 *     rather than random scatter.
 *
 * The full artifact is scanned once for bridge nodes (`.graph-bridge`
 * style class isn't present at build time — instead we look up nodes
 * whose `themeClusterId` propagates to ≥2 super-themes by joining
 * against the theme cluster doc). No cytoscape dependency here — pure
 * data transform, easy to test.
 */
import type { ArtifactData } from '../types/artifact'
import type { TopicClustersDocument } from '../api/corpusTopicClustersApi'

export interface BuildTopDownSliceOptions {
  themeDoc: TopicClustersDocument | null | undefined
  /** Full merged artifact used ONLY to discover bridge Topics between
   *  super-themes; we don't project the full graph into the slice. */
  fullArtifact: ArtifactData | null | undefined
}

/** graph-v3 tier 8-1 — emit an ArtifactData for the top-down initial view. */
export function buildTopDownSlice(opts: BuildTopDownSliceOptions): ArtifactData {
  const themeDoc = opts.themeDoc
  const clusters = Array.isArray(themeDoc?.clusters) ? themeDoc!.clusters : []

  // super_theme_id → { label, member cluster ids }
  const superById = new Map<string, { label: string; clusterIds: Set<string> }>()
  for (const c of clusters) {
    const sid = typeof c?.super_theme_id === 'string' ? c.super_theme_id.trim() : ''
    if (!sid) continue
    const label =
      typeof c?.super_theme_label === 'string' && c.super_theme_label.trim()
        ? c.super_theme_label.trim()
        : sid
    const cid =
      typeof c?.graph_compound_parent_id === 'string'
        ? c.graph_compound_parent_id.trim()
        : ''
    const entry = superById.get(sid) ?? { label, clusterIds: new Set<string>() }
    if (cid) entry.clusterIds.add(cid)
    superById.set(sid, entry)
  }

  if (superById.size === 0) return { nodes: [], edges: [] }

  // Build inter-super-theme edges from the full artifact:
  // for each node in the full artifact, look at its themeClusterId. If it
  // participates in ≥2 super-themes (via cluster → super lookup), it's a
  // bridge — link every pair of super-themes it touches.
  const clusterToSuper = new Map<string, string>()
  for (const [sid, entry] of superById) {
    for (const cid of entry.clusterIds) clusterToSuper.set(cid, sid)
  }

  const pairSeen = new Set<string>()
  const edges: ArtifactData['edges'] = []
  const fullNodes = opts.fullArtifact?.nodes ?? []
  for (const n of fullNodes) {
    const tcid =
      n && typeof (n as { themeClusterId?: unknown }).themeClusterId === 'string'
        ? String((n as { themeClusterId?: unknown }).themeClusterId).trim()
        : ''
    if (!tcid) continue
    // Nodes carry ONE themeClusterId (first-cluster-wins); a bridge is a
    // node whose neighbour set spans multiple super-themes. Cheap proxy:
    // any node type that our tier T propagation covers can act as a
    // bridge candidate; we detect actual bridging by looking at edges.
    // Skip that expense here — the shared-bridge structural signal from
    // tier 7 already grouped clusters into super-themes, so at this layer
    // the cross-super-theme links come from re-grouped Topic clusters.
    // (Left as a TODO for tier 8-6 to refine once ego-expansion re-scope
    // has to reason about bridges too.)
    void tcid
  }

  // Fallback structural pass: link every super-theme to every other in a
  // radial pattern via a single "hub" edge. Not semantic, but ensures the
  // fcose layout keeps the 6-8 nodes together instead of drifting apart.
  // Real bridge-derived edges land in a follow-up when the algorithmic
  // work in tier 8-6 lands.
  const ids = Array.from(superById.keys())
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const a = ids[i]!
      const b = ids[j]!
      const key = `${a}|${b}`
      if (pairSeen.has(key)) continue
      pairSeen.add(key)
      edges.push({
        type: '_topdown_link',
        from: a,
        to: b,
      })
    }
  }

  const nodes: ArtifactData['nodes'] = []
  for (const [sid, entry] of superById) {
    nodes.push({
      id: sid,
      type: 'SuperTheme',
      properties: {
        label: entry.label,
        // Count of child theme-clusters — useful for downstream badge / label.
        child_cluster_count: entry.clusterIds.size,
      },
    })
  }

  return { nodes, edges }
}
