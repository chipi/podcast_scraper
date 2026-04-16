/**
 * RFC-075: apply ``topic_clusters.json`` as Cytoscape compound parents on Topic nodes.
 */
import type { ArtifactData, ParsedArtifact, RawGraphNode } from '../types/artifact'
import type { TopicClustersCluster, TopicClustersDocument } from '../api/corpusTopicClustersApi'
import { visualNodeTypeCounts } from './visualGroup'
import { stripLayerPrefixesForCil } from './mergeGiKg'
import {
  filterArtifactEgoAroundTopicCluster,
  filterArtifactEgoOneHop,
  findRawNodeInArtifact,
  fullPrimaryNodeLabel,
  unionParsedArtifacts,
} from './parsing'

/** v2 `graph_compound_parent_id` vs v1 `cluster_id` — graph compound only, never a CIL merge target. */
export function graphCompoundParentIdFromCluster(cl: TopicClustersCluster): string {
  const o = cl as Record<string, unknown>
  const v = o.graph_compound_parent_id ?? o.cluster_id
  return typeof v === 'string' ? v.trim() : ''
}

export type TopicClusterMemberDetailRow = {
  topicId: string
  /** Present when this corpus topic appears in the merged graph. */
  graphNodeId: string | null
  label: string
}

function bareIdMatchesTopic(memberTopicId: string, graphNodeId: string): boolean {
  const want = memberTopicId.trim()
  if (!want) {
    return false
  }
  const stripped = stripLayerPrefixesForCil(graphNodeId)
  return stripped === want
}

/**
 * Resolve cluster metadata from ``topic_clusters.json`` by compound id (``tc:…``).
 */
export function findClusterByCompoundId(
  doc: TopicClustersDocument | null | undefined,
  compoundId: string | null | undefined,
): TopicClustersCluster | null {
  if (!doc?.clusters?.length || !compoundId?.trim()) {
    return null
  }
  const want = compoundId.trim()
  for (const cl of doc.clusters) {
    if (!cl || typeof cl !== 'object') {
      continue
    }
    const cid = graphCompoundParentIdFromCluster(cl)
    if (cid === want) {
      return cl
    }
  }
  return null
}

/**
 * When ``topic_clusters.json`` has no member rows for this compound but the merged graph has
 * Topic nodes with ``parent`` set to the TopicCluster compound id (overlay), derive CIL topic ids
 * for merged cluster timeline.
 */
/**
 * Topic ids to send to ``POST /api/topics/timeline`` (CIL merged arc).
 *
 * Prefer each member's **graph** node id when present (matches GI/bridge on disk). The corpus
 * ``topic_clusters.json`` ``topic_id`` strings can drift from merged-graph / GI ids; using the
 * loaded Topic node id fixes empty timelines when the JSON slug does not appear in GI ABOUT edges.
 */
export function clusterTimelineCilTopicIdsFromMemberRows(
  rows: TopicClusterMemberDetailRow[],
): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const r of rows) {
    const gid = r.graphNodeId?.trim()
    const raw = gid ? stripLayerPrefixesForCil(gid) : r.topicId.trim()
    if (!raw || seen.has(raw)) {
      continue
    }
    seen.add(raw)
    out.push(raw)
  }
  return out
}

/**
 * CIL topic ids for ``POST /api/topics/timeline`` for a TopicCluster compound.
 *
 * Prefer Topic children under the compound in the loaded graph (GI-aligned ids from the overlay).
 * If the graph has no such children in this view, fall back to member rows (per-row graph id or
 * JSON ``topic_id``). This avoids empty timelines when ``topic_clusters.json`` slugs drift from
 * GI topic node ids but the merged graph still has the real Topic nodes under ``tc:…``.
 */
export function clusterTimelineCilTopicIdsForCluster(
  art: ParsedArtifact | null,
  compoundId: string | null | undefined,
  memberRows: TopicClusterMemberDetailRow[],
): string[] {
  const cid = compoundId?.trim()
  if (cid) {
    const fromCompound = topicIdsFromGraphClusterCompound(art, cid)
    if (fromCompound.length > 0) {
      return fromCompound
    }
  }
  const fromMembers = clusterTimelineCilTopicIdsFromMemberRows(memberRows)
  if (fromMembers.length > 0) {
    return fromMembers
  }
  return cid ? topicIdsFromGraphClusterCompound(art, cid) : []
}

export function topicIdsFromGraphClusterCompound(
  art: ParsedArtifact | null,
  compoundId: string | null | undefined,
): string[] {
  const cid = compoundId?.trim()
  if (!cid || !art?.data?.nodes) {
    return []
  }
  const out: string[] = []
  for (const n of art.data.nodes) {
    if (!n || String(n.type ?? '').trim().toLowerCase() !== 'topic') {
      continue
    }
    const p = typeof n.parent === 'string' ? n.parent.trim() : ''
    if (p !== cid) {
      continue
    }
    if (n.id == null) {
      continue
    }
    const tid = stripLayerPrefixesForCil(String(n.id)).trim()
    if (tid) {
      out.push(tid)
    }
  }
  return [...new Set(out)].sort()
}

/**
 * Member rows for the node rail: one row per corpus member, with graph id when that Topic exists
 * in the merged artifact.
 */
export function topicClusterMemberRowsForDetail(
  art: ParsedArtifact | null,
  cl: TopicClustersCluster | null,
): TopicClusterMemberDetailRow[] {
  if (!cl?.members?.length) {
    return []
  }
  const nodes = art?.data?.nodes
  if (!Array.isArray(nodes)) {
    return []
  }
  const out: TopicClusterMemberDetailRow[] = []
  for (const m of cl.members) {
    const tid = m && typeof m.topic_id === 'string' ? m.topic_id.trim() : ''
    if (!tid) {
      continue
    }
    let graphNodeId: string | null = null
    for (const n of nodes) {
      if (!n || n.type !== 'Topic' || n.id == null) {
        continue
      }
      const gid = String(n.id)
      if (bareIdMatchesTopic(tid, gid)) {
        graphNodeId = gid
        break
      }
    }
    const rawMemberLabel = m && typeof m.label === 'string' ? m.label.trim() : ''
    const n = graphNodeId ? findRawNodeInArtifact(art!, graphNodeId) : null
    const label = n
      ? fullPrimaryNodeLabel(n)
      : rawMemberLabel || tid
    out.push({ topicId: tid, graphNodeId, label })
  }
  return out
}

export function findTopicClusterContextForGraphNode(
  graphNodeId: string | null,
  doc: TopicClustersDocument | null | undefined,
): { compoundParentId: string; canonicalLabel: string } | null {
  if (!graphNodeId || !doc?.clusters?.length) {
    return null
  }
  for (const cl of doc.clusters) {
    if (!cl || typeof cl !== 'object') {
      continue
    }
    const compoundParentId = graphCompoundParentIdFromCluster(cl)
    if (!compoundParentId) {
      continue
    }
    const members = Array.isArray(cl.members) ? cl.members : []
    for (const m of members) {
      const tid = m && typeof m.topic_id === 'string' ? m.topic_id.trim() : ''
      if (tid && bareIdMatchesTopic(tid, graphNodeId)) {
        const raw = cl.canonical_label
        const canonicalLabel =
          typeof raw === 'string' && raw.trim() ? raw.trim() : compoundParentId
        return { compoundParentId, canonicalLabel }
      }
    }
  }
  return null
}

/**
 * After a one-hop ego slice around ``focusId``, add every **TopicCluster** that intersects that
 * slice: compound, all member topics present in the merged graph, and one hop from each member
 * (same rule as the cluster minimap). Keeps connectivity and cluster detail consistent when
 * **Shift+ego** would otherwise drop non-edge-linked members.
 */
export function expandFilteredArtifactEgoWithTopicClusterNeighbors(
  fullFilteredArt: ParsedArtifact,
  focusId: string | null | undefined,
  doc: TopicClustersDocument | null | undefined,
): ParsedArtifact {
  const ego = filterArtifactEgoOneHop(fullFilteredArt, focusId)
  if (!doc?.clusters?.length) {
    return ego
  }
  if (focusId == null || focusId === '') {
    return ego
  }
  const egoNodes = ego.data.nodes
  const egoIds = new Set(
    Array.isArray(egoNodes)
      ? egoNodes.map((n) => String(n?.id ?? '')).filter(Boolean)
      : [],
  )
  let merged: ParsedArtifact = ego
  for (const cl of doc.clusters) {
    if (!cl || typeof cl !== 'object') {
      continue
    }
    const compoundId = graphCompoundParentIdFromCluster(cl)
    if (!compoundId) {
      continue
    }
    const rows = topicClusterMemberRowsForDetail(fullFilteredArt, cl)
    const memberGraphIds = rows
      .map((r) => r.graphNodeId)
      .filter((x): x is string => x != null)
    const touches =
      egoIds.has(compoundId) || memberGraphIds.some((id) => egoIds.has(id))
    if (!touches) {
      continue
    }
    const slice = filterArtifactEgoAroundTopicCluster(
      fullFilteredArt,
      compoundId,
      memberGraphIds,
    )
    merged = unionParsedArtifacts(merged, slice)
  }
  return merged
}

/**
 * Clone artifact data: add ``TopicCluster`` parent nodes and set ``parent`` on member Topics.
 */
export function applyTopicClustersOverlay(
  data: ArtifactData,
  doc: TopicClustersDocument | null | undefined,
): ArtifactData {
  if (!doc || !Array.isArray(doc.clusters) || doc.clusters.length === 0) {
    return data
  }

  const nodes = Array.isArray(data.nodes) ? data.nodes.map((n) => ({ ...n })) : []
  const edges = Array.isArray(data.edges) ? data.edges.map((e) => ({ ...e })) : []

  const existingIds = new Set<string>()
  for (const n of nodes) {
    if (n?.id != null) {
      existingIds.add(String(n.id))
    }
  }

  const parents: RawGraphNode[] = []
  /** Invisible layout-only edges that pull cluster members together (COSE cohesion). */
  const cohesionEdges: { type: string; from: string; to: string }[] = []

  for (const cl of doc.clusters) {
    if (!cl || typeof cl !== 'object') {
      continue
    }
    const clusterId = graphCompoundParentIdFromCluster(cl)
    const label =
      typeof cl.canonical_label === 'string' && cl.canonical_label.trim()
        ? cl.canonical_label.trim()
        : clusterId
    if (!clusterId || existingIds.has(clusterId)) {
      continue
    }
    const members = Array.isArray(cl.members) ? cl.members : []
    const attachedIds: string[] = []
    for (const n of nodes) {
      if (!n || n.type !== 'Topic' || n.id == null) {
        continue
      }
      const gid = String(n.id)
      for (const m of members) {
        const tid = m && typeof m.topic_id === 'string' ? m.topic_id.trim() : ''
        if (tid && bareIdMatchesTopic(tid, gid)) {
          ;(n as RawGraphNode & { parent?: string }).parent = clusterId
          attachedIds.push(gid)
          break
        }
      }
    }
    if (attachedIds.length === 0) {
      continue
    }
    parents.push({
      id: clusterId,
      type: 'TopicCluster',
      properties: { label },
    })
    existingIds.add(clusterId)

    for (let i = 0; i < attachedIds.length; i++) {
      for (let j = i + 1; j < attachedIds.length; j++) {
        cohesionEdges.push({
          type: '_tc_cohesion',
          from: attachedIds[i],
          to: attachedIds[j],
        })
      }
    }
  }

  if (parents.length === 0) {
    return { ...data, nodes, edges }
  }

  return {
    ...data,
    nodes: parents.concat(nodes),
    edges: edges.concat(cohesionEdges),
  }
}

/** Wrap ``buildDisplayArtifact`` result with topic cluster parents when *doc* is set. */
export function withTopicClustersOnDisplay(
  art: ParsedArtifact | null,
  doc: TopicClustersDocument | null | undefined,
): ParsedArtifact | null {
  if (!art || !doc) {
    return art
  }
  const nextData = applyTopicClustersOverlay(art.data, doc)
  const rawNodes = Array.isArray(nextData.nodes) ? nextData.nodes : []
  const nt = visualNodeTypeCounts(rawNodes)
  return {
    ...art,
    nodes: rawNodes.length,
    edges: Array.isArray(nextData.edges) ? nextData.edges.length : art.edges,
    nodeTypes: nt,
    data: nextData,
  }
}
