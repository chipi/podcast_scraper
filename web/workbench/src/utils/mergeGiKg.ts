/**
 * Same-layer merge of the drawn artifacts (ported from web/gi-kg-viz/shared.js), entity dedup and
 * id helpers. A downstream build that draws another layer beside KG combines it through
 * `artifactCombiners` and reuses the merge helpers exported here.
 */
import type { ArtifactData, ParsedArtifact, RawGraphEdge, RawGraphNode } from "../types/artifact"

/** Per-episode corpus paths of the artifacts a merged graph was built from. */
export function mergeEpisodeSourcePathsFromParsed(arts: ParsedArtifact[]): Record<string, string> | null {
  const out: Record<string, string> = {}
  for (const a of arts) {
    if (a.sourceCorpusRelPathByEpisodeId) {
      Object.assign(out, a.sourceCorpusRelPathByEpisodeId)
    }
    const rel = a.sourceCorpusRelPath
    const eid = a.episodeId
    if (rel && eid) {
      const k = String(eid)
      if (!out[k]) {
        out[k] = rel
      }
    }
  }
  return Object.keys(out).length ? out : null
}

function deepClone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T
}

// RFC-097 v3.0: Organization is a first-class typed node alongside Person —
// both dedupe by lowercased name when the same brand / individual recurs
// across episodes. Without Organization here, the same "Acme" referenced
// in 50 episodes would render as 50 separate nodes in the merged graph
// instead of one canonical Organization with 50 incident MENTIONS_ORG edges.
// Object joined in KG schema 2.1 (#2057) for the same reason Organization is here: a named
// thing discussed across 50 episodes is ONE referent. Without it "The Norman Conquest" would
// render as 50 separate nodes in the merged graph.
const DEDUP_TYPES = new Set(["Entity", "Topic", "Person", "Organization", "Object"])

/**
 * Canonical identity key for a node that should be deduplicated across episodes.
 * Entity / Person / Organization / Object → lowercased `name` (or `label` / `title` when
 * present), Topic → lowercased `label`. Returns null for types we don't dedup
 * (Episode, Insight, Quote, Podcast, etc.).
 */
function entityCanonicalKey(node: RawGraphNode): string | null {
  const t = node.type
  if (!t || !DEDUP_TYPES.has(t)) return null
  const props = node.properties || {}
  const raw = (props.name ?? props.label ?? props.title) as string | undefined
  if (!raw || typeof raw !== "string") return null
  return `${t}\0${raw.toLowerCase().trim()}`
}

export type EntityDedupMode = "name-based" | "cil-first"

/** Strip ``g:`` / ``k:`` / ``kg:`` layer prefixes for CIL id comparison (exported for topic cluster overlay). */
export function stripLayerPrefixesForCil(rawId: string): string {
  let s = String(rawId).trim()
  let prev = ""
  while (s !== prev) {
    prev = s
    if (s.startsWith("g:")) {
      s = s.slice(2)
    } else if (s.startsWith("k:") && !s.startsWith("kg:")) {
      s = s.slice(2)
    } else if (s.startsWith("kg:")) {
      s = s.slice(3)
    }
  }
  return s
}

/**
 * When GI+KG are combined, the same CIL id (e.g. ``person:alice``) may appear
 * as a GI ``Person`` and a KG ``Entity`` with different graph types; merge on CIL id.
 */
function cilMergeKey(node: RawGraphNode): string | null {
  if (!node?.id) return null
  const bare = stripLayerPrefixesForCil(String(node.id))
  if (/^(person|org|topic):/.test(bare)) {
    return `cil\0${bare}`
  }
  return null
}

function deduplicationKey(node: RawGraphNode, mode: EntityDedupMode): string | null {
  if (mode === "cil-first") {
    const c = cilMergeKey(node)
    if (c) return c
  }
  return entityCanonicalKey(node)
}

/**
 * Deduplicate Entity, Topic, and Person nodes that share the same canonical name
 * (or the same CIL id when ``mode`` is ``cil-first``).
 * Keeps the first occurrence, merges properties from duplicates,
 * and rewrites all edges to point to the surviving node ID.
 */
export function deduplicateEntities(
  nodes: RawGraphNode[],
  edges: RawGraphEdge[],
  mode: EntityDedupMode = "name-based"
): { nodes: RawGraphNode[]; edges: RawGraphEdge[] } {
  const canonToWinner = new Map<string, RawGraphNode>()
  const idReplace = new Map<string, string>()

  for (const n of nodes) {
    if (!n || n.id == null) continue
    const key = deduplicationKey(n, mode)
    if (!key) continue
    const existing = canonToWinner.get(key)
    if (!existing) {
      canonToWinner.set(key, n)
    } else {
      existing.properties = Object.assign({}, existing.properties || {}, n.properties || {})
      idReplace.set(String(n.id), String(existing.id))
    }
  }

  if (idReplace.size === 0) return { nodes, edges }

  const removedIds = new Set(idReplace.keys())
  const dedupedNodes = nodes.filter((n) => n && n.id != null && !removedIds.has(String(n.id)))

  const edgeSeen = new Set<string>()
  const dedupedEdges: RawGraphEdge[] = []
  for (const e of edges) {
    if (!e) continue
    const from = e.from != null ? String(e.from) : ""
    const to = e.to != null ? String(e.to) : ""
    const newFrom = idReplace.get(from) ?? from
    const newTo = idReplace.get(to) ?? to
    const ek = `${newFrom}\0${newTo}\0${String(e.type || "")}`
    if (edgeSeen.has(ek)) continue
    edgeSeen.add(ek)
    dedupedEdges.push({ ...e, from: newFrom, to: newTo })
  }

  return { nodes: dedupedNodes, edges: dedupedEdges }
}

export function nodeTypesFromNodesLocal(nodes: RawGraphNode[]): Record<string, number> {
  const nt: Record<string, number> = {}
  for (const n of nodes) {
    const t = n && typeof n.type === "string" ? n.type : "?"
    nt[t] = (nt[t] || 0) + 1
  }
  return nt
}

export function mergeParsedArtifacts(arts: ParsedArtifact[]): ParsedArtifact | null {
  if (!arts || arts.length < 2) return null
  const kind = arts[0].kind
  if (kind !== "gi" && kind !== "kg") return null
  for (let i = 1; i < arts.length; i++) {
    if (arts[i].kind !== kind) return null
  }
  const nodeById = new Map<string, RawGraphNode>()
  for (const a of arts) {
    const rawNodes = Array.isArray(a.data.nodes) ? a.data.nodes : []
    for (const n of rawNodes) {
      if (!n || n.id == null) continue
      const id = String(n.id)
      if (!nodeById.has(id)) {
        nodeById.set(id, deepClone(n))
      }
    }
  }
  const idSet = new Set(nodeById.keys())
  const edgeList: RawGraphEdge[] = []
  const edgeSeen = new Set<string>()
  for (const a of arts) {
    const rawEdges = Array.isArray(a.data.edges) ? a.data.edges : []
    for (const e of rawEdges) {
      if (!e || e.from == null || e.to == null) continue
      const from = String(e.from)
      const to = String(e.to)
      if (!idSet.has(from) || !idSet.has(to)) continue
      const ek = `${from}\0${to}\0${String(e.type || "")}`
      if (edgeSeen.has(ek)) continue
      edgeSeen.add(ek)
      edgeList.push(deepClone(e))
    }
  }
  const mergedData = deepClone(arts[0].data) as ArtifactData
  const deduped = deduplicateEntities(Array.from(nodeById.values()), edgeList, "name-based")
  mergedData.nodes = deduped.nodes
  mergedData.edges = deduped.edges
  mergedData.episode_id = `merged:${String(arts.length)}-artifacts`
  const nodeTypes = nodeTypesFromNodesLocal(mergedData.nodes || [])
  return {
    name: `Merged ${kind.toUpperCase()} (${arts.length} files)`,
    kind,
    episodeId: mergedData.episode_id || null,
    nodes: mergedData.nodes!.length,
    edges: mergedData.edges!.length,
    nodeTypes,
    data: mergedData,
    sourceCorpusRelPath: null,
    sourceCorpusRelPathByEpisodeId: mergeEpisodeSourcePathsFromParsed(arts),
  }
}

/** The graph the viewer draws from the loaded KG artifacts: one as is, several merged. */
export function buildDisplayArtifact(kgArts: ParsedArtifact[]): ParsedArtifact | null {
  if (kgArts.length >= 2) return mergeParsedArtifacts(kgArts)
  return kgArts[0] ?? null
}
