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
import type { ArtifactData, RawGraphNode } from '../types/artifact'
import type { TopicClustersDocument } from '../api/corpusTopicClustersApi'

/* graph-v3 tier 8-1 viewer clamp (gap-4 harden). The enricher already caps
 * `super_theme_count` at 8 (_SUPER_THEME_MAX in
 * topic_theme_clusters.py), but a stale / hand-crafted / bad artifact
 * could still ship a doc with more. Belt-and-suspenders: the viewer
 * truncates to the top-N super-themes by `child_cluster_count` so a
 * pathological artifact can't overwhelm the layout. Mirrors the
 * enricher's own [MIN, MAX] band. */
export const VIEWER_SUPER_THEME_MAX = 8

export interface BuildTopDownSliceOptions {
  themeDoc: TopicClustersDocument | null | undefined
  /** Full merged artifact used to project expanded super-themes'
   *  children (TopicClusters + Topics + one-hop Insights/Persons)
   *  into the slice. Also used to discover bridge Topics. */
  fullArtifact: ArtifactData | null | undefined
  /** graph-v3 tier 8-2 — super-theme ids the user has tapped to expand.
   *  When a super_theme_id is in this set, `buildTopDownSlice` injects
   *  its child TopicClusters + tagged Topics + one-hop Insights /
   *  Persons under the SuperTheme node. Empty (or omitted) → the
   *  slice is the tier-8-1 static preview of super-theme bubbles. */
  expandedSuperThemeIds?: ReadonlySet<string> | null
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

  /* Viewer-side clamp (gap-4). Keep the largest N super-themes by child
   * count; drop the rest so a bad artifact can't blow up the layout.
   * Deterministic order: (-child_cluster_count, super_theme_id) so
   * repeated calls with the same input produce the same slice. */
  if (superById.size > VIEWER_SUPER_THEME_MAX) {
    const ranked = Array.from(superById.entries()).sort((a, b) => {
      const sizeDelta = b[1].clusterIds.size - a[1].clusterIds.size
      if (sizeDelta !== 0) return sizeDelta
      return a[0].localeCompare(b[0])
    })
    const kept = new Map(ranked.slice(0, VIEWER_SUPER_THEME_MAX))
    superById.clear()
    for (const [k, v] of kept) superById.set(k, v)
  }

  // cluster_id → super_theme_id lookup (drives expansion projection).
  const clusterToSuper = new Map<string, string>()
  for (const [sid, entry] of superById) {
    for (const cid of entry.clusterIds) clusterToSuper.set(cid, sid)
  }

  const expanded: ReadonlySet<string> = opts.expandedSuperThemeIds ?? new Set<string>()

  const nodes: ArtifactData['nodes'] = []
  const edges: ArtifactData['edges'] = []
  const emittedIds = new Set<string>()

  // Emit the SuperTheme bubbles first — these are always visible in
  // top-down mode regardless of expansion.
  for (const [sid, entry] of superById) {
    nodes.push({
      id: sid,
      type: 'SuperTheme',
      properties: {
        label: entry.label,
        child_cluster_count: entry.clusterIds.size,
        /* Expand state on the node itself so the stylesheet can differentiate
         * (a faint outline on expanded supers reads as "you've drilled here"). */
        top_down_expanded: expanded.has(sid),
      },
    })
    emittedIds.add(sid)
  }

  /* graph-v3 tier 8-1 → gap-3 harden — inter-super-theme edges.
   *
   * Replaces the scaffolding "all-pairs" edges with edges derived from
   * REAL bridge topics: for each edge in the full artifact that
   * connects two nodes whose themeClusterIds roll up to DIFFERENT
   * super-themes, we count that as one bridge signal between those two
   * super-themes. Deduped by unordered super-theme pair with a `weight`
   * = number of underlying bridge edges (a proxy for "how strongly
   * these two super-themes talk to each other").
   *
   * Fallback: if the artifact carries no bridge edges (small corpus /
   * theme propagation hasn't fanned out yet), we still emit a single
   * ring of edges so fcose has enough structure to lay the ~6-8
   * bubbles out cohesively instead of drifting apart. Ring beats
   * all-pairs — same layout cost, less visual noise, still connected.
   */
  const superIds = Array.from(superById.keys())
  const superIdSet = new Set(superIds)
  const bridgeSignal = new Map<string, number>()
  const fullNodesForBridges = opts.fullArtifact?.nodes ?? []
  const fullEdgesForBridges = opts.fullArtifact?.edges ?? []
  const nodeSuperById = new Map<string, string>()
  for (const n of fullNodesForBridges) {
    if (!n || n.id == null) continue
    const tcid =
      typeof (n as { themeClusterId?: unknown }).themeClusterId === 'string'
        ? String((n as { themeClusterId?: unknown }).themeClusterId).trim()
        : ''
    if (!tcid) continue
    const sid = clusterToSuper.get(tcid)
    if (sid && superIdSet.has(sid)) nodeSuperById.set(String(n.id), sid)
  }
  for (const e of fullEdgesForBridges) {
    if (!e || e.from == null || e.to == null) continue
    const sSuper = nodeSuperById.get(String(e.from))
    const tSuper = nodeSuperById.get(String(e.to))
    if (!sSuper || !tSuper || sSuper === tSuper) continue
    const [lo, hi] = sSuper < tSuper ? [sSuper, tSuper] : [tSuper, sSuper]
    const key = `${lo}|${hi}`
    bridgeSignal.set(key, (bridgeSignal.get(key) ?? 0) + 1)
  }
  if (bridgeSignal.size > 0) {
    for (const [key, weight] of bridgeSignal) {
      const [a, b] = key.split('|')
      edges.push({
        type: '_topdown_link',
        from: a!,
        to: b!,
        properties: { weight, source: 'bridge' },
      })
    }
  } else if (superIds.length > 1) {
    /* Fallback ring: N-1 edges chain (not N² all-pairs) — enough for
     * layout structure, minimal visual noise. Ordered path so no two
     * edges duplicate the same unordered pair. */
    for (let i = 0; i < superIds.length - 1; i++) {
      const a = superIds[i]!
      const b = superIds[i + 1]!
      edges.push({
        type: '_topdown_link',
        from: a,
        to: b,
        properties: { weight: 0, source: 'fallback_ring' },
      })
    }
  }

  if (expanded.size === 0) {
    return { nodes, edges }
  }

  // graph-v3 tier 8-2 — project every expanded super-theme's children
  // into the slice. Pipeline:
  //   1. For each expanded super_theme_id, collect its child cluster_ids
  //      from the theme doc.
  //   2. Pull every node in the full artifact whose themeClusterId is
  //      in that child set — these become the projected sub-graph.
  //   3. Also pull one-hop neighbours in the full artifact so Insights /
  //      Persons connected to the projected Topics come along.
  //   4. Attach every projected node to its super-theme via `parent`
  //      so Cytoscape renders them inside the SuperTheme compound.
  const fullNodes = opts.fullArtifact?.nodes ?? []
  const fullEdges = opts.fullArtifact?.edges ?? []
  const nodeById = new Map<string, RawGraphNode>()
  for (const n of fullNodes) {
    if (n?.id != null) nodeById.set(String(n.id), n)
  }

  // Adjacency for one-hop expansion.
  const adjacency = new Map<string, Set<string>>()
  for (const e of fullEdges) {
    if (!e || e.from == null || e.to == null) continue
    const f = String(e.from)
    const t = String(e.to)
    if (!adjacency.has(f)) adjacency.set(f, new Set())
    if (!adjacency.has(t)) adjacency.set(t, new Set())
    adjacency.get(f)!.add(t)
    adjacency.get(t)!.add(f)
  }

  // Node id → super_theme_id it belongs under (first-cluster-wins).
  const projected = new Map<string, string>()
  for (const n of fullNodes) {
    if (!n || n.id == null) continue
    const tcid =
      typeof (n as { themeClusterId?: unknown }).themeClusterId === 'string'
        ? String((n as { themeClusterId?: unknown }).themeClusterId).trim()
        : ''
    if (!tcid) continue
    const sid = clusterToSuper.get(tcid)
    if (!sid || !expanded.has(sid)) continue
    projected.set(String(n.id), sid)
  }

  // One-hop expansion — pull direct neighbours that either have no
  // themeClusterId of their own (Insights, Persons, Podcasts w/o a tag)
  // OR whose themeClusterId also rolls up to an expanded super-theme.
  // This respects the boundary: expanding sth:a doesn't leak into sth:b
  // via cross-super-theme edges.
  const seeded = new Set(projected.keys())
  for (const nodeId of seeded) {
    const sid = projected.get(nodeId)!
    const neighbours = adjacency.get(nodeId)
    if (!neighbours) continue
    for (const nb of neighbours) {
      if (projected.has(nb)) continue
      const nbNode = nodeById.get(nb)
      if (!nbNode) continue
      const nbTcid =
        typeof (nbNode as { themeClusterId?: unknown }).themeClusterId === 'string'
          ? String((nbNode as { themeClusterId?: unknown }).themeClusterId).trim()
          : ''
      if (nbTcid) {
        const nbSid = clusterToSuper.get(nbTcid)
        if (nbSid && !expanded.has(nbSid)) continue
      }
      projected.set(nb, sid)
    }
  }

  for (const [nodeId, sid] of projected) {
    if (emittedIds.has(nodeId)) continue
    const src = nodeById.get(nodeId)
    if (!src) continue
    /* Clone + attach `parent` so the projected node renders inside the
     * SuperTheme compound. Keep the original type intact so the existing
     * stylesheet rules (Topic / Insight / Person / …) all apply. */
    nodes.push({
      ...src,
      parent: sid,
    })
    emittedIds.add(nodeId)
  }

  // Include edges whose BOTH endpoints landed in the projected set. This
  // keeps the visible edges meaningful — no dangling half-edges pointing
  // at nodes hidden under a collapsed super-theme.
  for (const e of fullEdges) {
    if (!e || e.from == null || e.to == null) continue
    const f = String(e.from)
    const t = String(e.to)
    if (!emittedIds.has(f) || !emittedIds.has(t)) continue
    if (f === t) continue
    edges.push(e)
  }

  return { nodes, edges }
}
