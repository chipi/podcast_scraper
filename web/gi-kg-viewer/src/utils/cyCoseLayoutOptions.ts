/**
 * COSE layout tuning for RFC-075 topic clusters (`tc:…` compounds + member Topics).
 *
 * Cytoscape CoSE lays out each compound child graph separately; we shorten ideal edge
 * length for edges whose endpoints share the same `tc:` parent (tighter members), raise
 * repulsion on `TopicCluster` compounds (more clearance from unrelated root nodes), lower
 * repulsion for members (allow packing), and raise `nestingFactor` so cross-graph edges
 * prefer longer spans (keeps externals from sitting “inside” the cluster region).
 */
import type { EdgeSingular, NodeSingular } from 'cytoscape'

const MAIN = {
  padding: 36,
  fit: false as const,
  nodeRepulsionBase: 880_000,
  /** Stronger than root defaults so Episodes/Topics repel away from the compound node. */
  nodeRepulsionTopicClusterCompound: 1_450_000,
  /**
   * Much weaker than base so member Topics pack tightly inside the compound.
   * Previous 420k left 2–4 member clusters sprawled across ~1/4 of the canvas.
   */
  nodeRepulsionTopicClusterMember: 180_000,
  idealEdgeLengthBase: 96,
  /** Short intra-cluster edges pull connected members close together. */
  idealEdgeLengthIntraTopicCluster: 36,
  edgeElasticity: 100,
  gravity: 0.18,
  /** Default CoSE is 1.2; higher stretches cross-boundary edges vs intra-cluster edges. */
  nestingFactor: 1.52,
  nodeDimensionsIncludeLabels: true,
  numIter: 2500,
} as const

/** Scaled to match prior minimap baseline (nodeRepulsion 120k, idealEdge 52). */
const COMPACT = {
  padding: 14,
  fit: false as const,
  nodeRepulsionBase: 120_000,
  nodeRepulsionTopicClusterCompound: 198_000,
  nodeRepulsionTopicClusterMember: 24_000,
  idealEdgeLengthBase: 52,
  idealEdgeLengthIntraTopicCluster: 20,
  edgeElasticity: 80,
  gravity: 0.32,
  nestingFactor: 1.52,
  nodeDimensionsIncludeLabels: true,
} as const

function normalizedParent(node: NodeSingular): string | null {
  const p = node.data('parent')
  return typeof p === 'string' && p.trim() ? p.trim() : null
}

/** True when `parent` is a topic-cluster compound id (`tc:…`). */
export function isTopicClusterParentId(parent: string | null | undefined): boolean {
  return typeof parent === 'string' && parent.trim().startsWith('tc:')
}

/** Edge connects two Topics that share the same `tc:` compound parent. */
export function isIntraTopicClusterEdgeParents(
  sourceParent: string | null | undefined,
  targetParent: string | null | undefined,
): boolean {
  const a = typeof sourceParent === 'string' ? sourceParent.trim() : ''
  const b = typeof targetParent === 'string' ? targetParent.trim() : ''
  return Boolean(a && b && a === b && a.startsWith('tc:'))
}

/**
 * Unit-testable repulsion from node `type` / `parent` (mirrors `NodeSingular.data`).
 */
export function giKgCoseNodeRepulsionFromData(
  type: string | undefined,
  parent: string | null | undefined,
  profile: 'main' | 'compact',
): number {
  const t = String(type ?? '')
  if (t === 'TopicCluster') {
    return profile === 'main'
      ? MAIN.nodeRepulsionTopicClusterCompound
      : COMPACT.nodeRepulsionTopicClusterCompound
  }
  const pid = typeof parent === 'string' ? parent.trim() : ''
  if (pid.startsWith('tc:')) {
    return profile === 'main'
      ? MAIN.nodeRepulsionTopicClusterMember
      : COMPACT.nodeRepulsionTopicClusterMember
  }
  return profile === 'main' ? MAIN.nodeRepulsionBase : COMPACT.nodeRepulsionBase
}

export function giKgCoseNodeRepulsion(node: NodeSingular, profile: 'main' | 'compact'): number {
  return giKgCoseNodeRepulsionFromData(
    node.data('type') as string | undefined,
    normalizedParent(node),
    profile,
  )
}

export function giKgCoseIdealEdgeLength(edge: EdgeSingular, profile: 'main' | 'compact'): number {
  const s = edge.source()
  const t = edge.target()
  const base = profile === 'main' ? MAIN.idealEdgeLengthBase : COMPACT.idealEdgeLengthBase
  const intra =
    profile === 'main'
      ? MAIN.idealEdgeLengthIntraTopicCluster
      : COMPACT.idealEdgeLengthIntraTopicCluster
  if (isIntraTopicClusterEdgeParents(normalizedParent(s), normalizedParent(t))) {
    return intra
  }
  return base
}

export function giKgCoseLayoutOptionsMain(): Record<string, unknown> {
  return {
    name: 'cose',
    padding: MAIN.padding,
    fit: MAIN.fit,
    nodeRepulsion: (node: NodeSingular) => giKgCoseNodeRepulsion(node, 'main'),
    idealEdgeLength: (edge: EdgeSingular) => giKgCoseIdealEdgeLength(edge, 'main'),
    edgeElasticity: () => MAIN.edgeElasticity,
    gravity: MAIN.gravity,
    nestingFactor: MAIN.nestingFactor,
    numIter: MAIN.numIter,
    nodeDimensionsIncludeLabels: MAIN.nodeDimensionsIncludeLabels,
  }
}

/**
 * Static COSE options if `giKgCoseLayoutOptionsMain` fails at runtime (broken HMR chunk, etc.).
 * Keeps the graph usable; topic-cluster tuning is best-effort only.
 */
export function giKgCoseLayoutOptionsMainFallback(): Record<string, unknown> {
  return {
    name: 'cose',
    padding: MAIN.padding,
    fit: MAIN.fit,
    nodeRepulsion: () => MAIN.nodeRepulsionBase,
    idealEdgeLength: () => MAIN.idealEdgeLengthBase,
    edgeElasticity: () => MAIN.edgeElasticity,
    gravity: MAIN.gravity,
    nestingFactor: MAIN.nestingFactor,
    numIter: MAIN.numIter,
    nodeDimensionsIncludeLabels: MAIN.nodeDimensionsIncludeLabels,
  }
}

export function giKgCoseLayoutOptionsCompact(): Record<string, unknown> {
  return {
    name: 'cose',
    padding: COMPACT.padding,
    fit: COMPACT.fit,
    animate: false,
    nodeRepulsion: (node: NodeSingular) => giKgCoseNodeRepulsion(node, 'compact'),
    idealEdgeLength: (edge: EdgeSingular) => giKgCoseIdealEdgeLength(edge, 'compact'),
    edgeElasticity: () => COMPACT.edgeElasticity,
    gravity: COMPACT.gravity,
    nestingFactor: COMPACT.nestingFactor,
    nodeDimensionsIncludeLabels: COMPACT.nodeDimensionsIncludeLabels,
  }
}
