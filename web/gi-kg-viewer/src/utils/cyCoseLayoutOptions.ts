/**
 * COSE layout tuning for corpus topic clusters (`tc:…` compounds + member Topics).
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

/** Semantic ideal lengths (WIP §3.7); scaled down for compact profile vs main base. */
function semanticIdealEdgeLengthPx(edgeType: string, profile: 'main' | 'compact'): number {
  const mainBase = MAIN.idealEdgeLengthBase
  const compactBase = COMPACT.idealEdgeLengthBase
  const scale = profile === 'compact' ? compactBase / mainBase : 1
  const scaled = (mainPx: number) => Math.max(24, Math.round(mainPx * scale))
  switch (edgeType) {
    case 'HAS_INSIGHT':
      return scaled(60)
    case 'ABOUT':
      return scaled(80)
    case 'SUPPORTED_BY':
      return scaled(40)
    case 'RELATED_TO':
      return scaled(120)
    case 'SPOKE_IN':
      return scaled(100)
    case 'MENTIONS':
      return scaled(150)
    default:
      return profile === 'main' ? MAIN.idealEdgeLengthBase : COMPACT.idealEdgeLengthBase
  }
}

function semanticEdgeElasticity(edgeType: string, profile: 'main' | 'compact'): number {
  const mainMap: Record<string, number> = {
    HAS_INSIGHT: 180,
    ABOUT: 200,
    SUPPORTED_BY: 150,
    RELATED_TO: 100,
    SPOKE_IN: 120,
    MENTIONS: 60,
  }
  const v = mainMap[edgeType]
  if (v == null) {
    return profile === 'main' ? MAIN.edgeElasticity : COMPACT.edgeElasticity
  }
  if (profile === 'compact') {
    return Math.max(40, Math.round(v * 0.8))
  }
  return v
}

export function giKgCoseIdealEdgeLength(edge: EdgeSingular, profile: 'main' | 'compact'): number {
  const s = edge.source()
  const t = edge.target()
  const intra =
    profile === 'main'
      ? MAIN.idealEdgeLengthIntraTopicCluster
      : COMPACT.idealEdgeLengthIntraTopicCluster
  if (isIntraTopicClusterEdgeParents(normalizedParent(s), normalizedParent(t))) {
    return intra
  }
  const et = String(edge.data('edgeType') ?? '')
  return semanticIdealEdgeLengthPx(et, profile)
}

export function giKgCoseEdgeElasticity(edge: EdgeSingular, profile: 'main' | 'compact'): number {
  const s = edge.source()
  const t = edge.target()
  if (isIntraTopicClusterEdgeParents(normalizedParent(s), normalizedParent(t))) {
    return profile === 'main' ? MAIN.edgeElasticity : COMPACT.edgeElasticity
  }
  const et = String(edge.data('edgeType') ?? '')
  return semanticEdgeElasticity(et, profile)
}

export function giKgCoseLayoutOptionsMain(): Record<string, unknown> {
  return {
    name: 'cose',
    padding: MAIN.padding,
    fit: MAIN.fit,
    nodeRepulsion: (node: NodeSingular) => giKgCoseNodeRepulsion(node, 'main'),
    idealEdgeLength: (edge: EdgeSingular) => giKgCoseIdealEdgeLength(edge, 'main'),
    edgeElasticity: (edge: EdgeSingular) => giKgCoseEdgeElasticity(edge, 'main'),
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
    edgeElasticity: (edge: EdgeSingular) => giKgCoseEdgeElasticity(edge, 'compact'),
    gravity: COMPACT.gravity,
    nestingFactor: COMPACT.nestingFactor,
    nodeDimensionsIncludeLabels: COMPACT.nodeDimensionsIncludeLabels,
  }
}
