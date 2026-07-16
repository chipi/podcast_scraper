/**
 * COSE layout tuning for corpus topic clusters (`tc:…` compounds + member Topics).
 *
 * Cytoscape CoSE lays out each compound child graph separately; we shorten ideal edge
 * length for edges whose endpoints share the same `tc:` parent (tighter members), raise
 * repulsion on `TopicCluster` compounds (more clearance from unrelated root nodes), lower
 * repulsion for members (allow packing), and raise `nestingFactor` so cross-graph edges
 * prefer longer spans (keeps externals from sitting “inside” the cluster region).
 */
import cytoscape, { type EdgeSingular, type NodeSingular } from 'cytoscape'
import fcose from 'cytoscape-fcose'

// #967 — register fcose (spectral pre-placement → seconds at thousands of nodes,
// vs cose's ~O(n²) that froze the canvas for 2+ min at ~2.9k nodes). Registered in
// this layout-options module (imported by GraphCanvas + the minimap before any
// layout runs); the ES-module cache guarantees ``cytoscape.use`` fires exactly once.
cytoscape.use(fcose)

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
  /** graph-v3 M — 0.18 → 0.12. Lower gravity lets natural communities
   *  drift apart into visible clusters (docs' "galaxy" effect). Not the
   *  full 0.08 the docs proposed — our graph is bipartite (Insight in
   *  the middle) so aggressive cluster-drift can leave orphan
   *  Insights stranded far from their Topic + Episode neighbours.
   *  0.12 keeps hubs anchored while opening the cluster gaps. */
  gravity: 0.12,
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
    case 'MENTIONS_PERSON':
    case 'MENTIONS_ORG':
      // RFC-097 v3.0 typed MENTIONS family — same layout treatment as legacy.
      return scaled(150)
    default:
      return profile === 'main' ? MAIN.idealEdgeLengthBase : COMPACT.idealEdgeLengthBase
  }
}

function semanticEdgeElasticity(edgeType: string, profile: 'main' | 'compact'): number {
  // RFC-097 v3.0: ``MENTIONS_PERSON`` / ``MENTIONS_ORG`` share the legacy
  // ``MENTIONS`` elasticity so the typed split doesn't change graph layout
  // for users on the same corpus shape.
  const mainMap: Record<string, number> = {
    HAS_INSIGHT: 180,
    ABOUT: 200,
    SUPPORTED_BY: 150,
    RELATED_TO: 100,
    SPOKE_IN: 120,
    MENTIONS: 60,
    MENTIONS_PERSON: 60,
    MENTIONS_ORG: 60,
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

/**
 * #767-B — redraw debounce window.
 *
 *   - 150 ms catches Vue ``nextTick`` cascades on internal flows; still
 *     feels instant to the user.
 *   - 0 ms when an FSM envelope is pending — the FSM is in
 *     ``loading_*`` waiting for the redraw to drive it forward; the
 *     debounce slack stacks against the stuck-handoff timeout.
 *
 * Behavior contract pinned by ``redrawDebounceMs.test.ts``:
 *
 *   before this rule (always 150 ms):  redrawDebounceMs(true)  → 150
 *   after  this rule:                  redrawDebounceMs(true)  →   0
 *                                      redrawDebounceMs(false) → 150
 *
 * Saving: 150 ms per cross-surface handoff click.
 */
export const REDRAW_DEBOUNCE_INTERNAL_MS = 150
export function redrawDebounceMs(hasPendingHandoff: boolean): number {
  return hasPendingHandoff ? 0 : REDRAW_DEBOUNCE_INTERNAL_MS
}

/**
 * #767-C — post-animation safety-net recenter timings.
 *
 * ``animateCameraToFocusedNode`` animates the camera over 320 ms then
 * schedules ``recenterIfPending`` at the timings below. Each timer fires
 * a best-effort ``core.center(targetNode)`` so the camera converges on
 * the focus target even if the canvas resized mid-animation.
 *
 * The schedule is ``[400, 900, 1800]``. #787 originally trimmed this to
 * ``[400]`` based on linux-CI observation, but the Tier-2 production-shaped
 * matrix (``e2e/handoff-production/``) ran red on firefox-mac: cytoscape
 * canvas resize on first graph mount settles past 400 ms locally, so the
 * 900 / 1800 ms timers were catching the late recenter that the trimmed
 * schedule no longer fires. Restored to keep cross-platform stability;
 * each timer is a no-op when the pending recenter is already consumed,
 * so the cost on linux is negligible.
 */
export const RECENTER_SAFETY_TAIL_TIMINGS_MS: readonly number[] = [400, 900, 1800]

/**
 * #767-A — derive cose `numIter` from node count for external-nav redraws.
 *
 * Default is `MAIN.numIter` = 2500 iterations regardless of graph size.
 * For external-nav redraws (Library / Digest "Open in graph") on graphs
 * already past 50 nodes, the prior layout is a good warm start — cose
 * converges well below 2500 iterations and the extra work is pure tail
 * latency. Cap at `Math.min(2500, 200 + 8 × nodeCount)`:
 *
 *   - 50 nodes  → 600 iter (cose still has room to settle from scratch)
 *   - 200 nodes → 1800 iter
 *   - 270 nodes → 2360 iter (production-shaped fixture)
 *   - 300+ nodes → 2500 (cap)
 *
 * First paint (no warm start) still gets the full 2500 — caller decides
 * which mode applies.
 */
export function giKgCoseNumIterCapped(nodeCount: number): number {
  if (!Number.isFinite(nodeCount) || nodeCount <= 0) return MAIN.numIter
  return Math.min(MAIN.numIter, 200 + Math.floor(nodeCount) * 8)
}

export function giKgCoseLayoutOptionsMain(numIterOverride?: number): Record<string, unknown> {
  return {
    name: 'fcose',
    // fcose speed levers: spectral seeding ('default' quality) + no per-iteration
    // animation + component packing. The tc:-compound tuning (repulsion / edge
    // length / nesting) below is fcose-compatible (same fn-valued options as cose).
    quality: 'default',
    randomize: true,
    animate: false,
    packComponents: true,
    padding: MAIN.padding,
    fit: MAIN.fit,
    nodeRepulsion: (node: NodeSingular) => giKgCoseNodeRepulsion(node, 'main'),
    idealEdgeLength: (edge: EdgeSingular) => giKgCoseIdealEdgeLength(edge, 'main'),
    edgeElasticity: (edge: EdgeSingular) => giKgCoseEdgeElasticity(edge, 'main'),
    gravity: MAIN.gravity,
    nestingFactor: MAIN.nestingFactor,
    numIter: numIterOverride ?? MAIN.numIter,
    nodeDimensionsIncludeLabels: MAIN.nodeDimensionsIncludeLabels,
  }
}

/**
 * Static COSE options if `giKgCoseLayoutOptionsMain` fails at runtime (broken HMR chunk, etc.).
 * Keeps the graph usable; topic-cluster tuning is best-effort only.
 */
export function giKgCoseLayoutOptionsMainFallback(
  numIterOverride?: number,
): Record<string, unknown> {
  return {
    name: 'fcose',
    quality: 'default',
    randomize: true,
    animate: false,
    packComponents: true,
    padding: MAIN.padding,
    fit: MAIN.fit,
    nodeRepulsion: () => MAIN.nodeRepulsionBase,
    idealEdgeLength: () => MAIN.idealEdgeLengthBase,
    edgeElasticity: () => MAIN.edgeElasticity,
    gravity: MAIN.gravity,
    nestingFactor: MAIN.nestingFactor,
    numIter: numIterOverride ?? MAIN.numIter,
    nodeDimensionsIncludeLabels: MAIN.nodeDimensionsIncludeLabels,
  }
}

export function giKgCoseLayoutOptionsCompact(): Record<string, unknown> {
  return {
    name: 'fcose',
    quality: 'default',
    randomize: true,
    packComponents: true,
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
