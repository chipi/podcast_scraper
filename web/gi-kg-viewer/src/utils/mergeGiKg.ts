/**
 * Same-layer merge and GI+KG combine (ported from web/gi-kg-viz/shared.js).
 */
import type { ArtifactData, ParsedArtifact, RawGraphEdge, RawGraphNode } from '../types/artifact'
import { ensureEpisodeToInsightEdges } from './parsing'

/**
 * Build per-episode ``.gi.json`` corpus paths for merged graphs (multi-file GI or GI+KG).
 */
function mergeEpisodeSourcePathsFromParsed(arts: ParsedArtifact[]): Record<string, string> | null {
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

const DEDUP_TYPES = new Set(['Entity', 'Topic', 'Person'])

/**
 * Canonical identity key for a node that should be deduplicated across episodes.
 * Entity / Person → lowercased `name` (or `label` / `title` when present), Topic → lowercased `label`.
 * Returns null for types we don't dedup (Episode, Insight, Quote, etc.).
 */
function entityCanonicalKey(node: RawGraphNode): string | null {
  const t = node.type
  if (!t || !DEDUP_TYPES.has(t)) return null
  const props = node.properties || {}
  const raw = (props.name ?? props.label ?? props.title) as string | undefined
  if (!raw || typeof raw !== 'string') return null
  return `${t}\0${raw.toLowerCase().trim()}`
}

export type EntityDedupMode = 'name-based' | 'cil-first'

/** Strip ``g:`` / ``k:`` / ``kg:`` layer prefixes for RFC-072 id comparison (exported for topic cluster overlay). */
export function stripLayerPrefixesForCil(rawId: string): string {
  let s = String(rawId).trim()
  let prev = ''
  while (s !== prev) {
    prev = s
    if (s.startsWith('g:')) {
      s = s.slice(2)
    } else if (s.startsWith('k:') && !s.startsWith('kg:')) {
      s = s.slice(2)
    } else if (s.startsWith('kg:')) {
      s = s.slice(3)
    }
  }
  return s
}

/**
 * When GI+KG are combined, the same RFC-072 id (e.g. ``person:alice``) may appear
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
  if (mode === 'cil-first') {
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
function deduplicateEntities(
  nodes: RawGraphNode[],
  edges: RawGraphEdge[],
  mode: EntityDedupMode = 'name-based',
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
      existing.properties = Object.assign(
        {},
        existing.properties || {},
        n.properties || {},
      )
      idReplace.set(String(n.id), String(existing.id))
    }
  }

  if (idReplace.size === 0) return { nodes, edges }

  const removedIds = new Set(idReplace.keys())
  const dedupedNodes = nodes.filter(
    (n) => n && n.id != null && !removedIds.has(String(n.id)),
  )

  const edgeSeen = new Set<string>()
  const dedupedEdges: RawGraphEdge[] = []
  for (const e of edges) {
    if (!e) continue
    const from = e.from != null ? String(e.from) : ''
    const to = e.to != null ? String(e.to) : ''
    const newFrom = idReplace.get(from) ?? from
    const newTo = idReplace.get(to) ?? to
    const ek = `${newFrom}\0${newTo}\0${String(e.type || '')}`
    if (edgeSeen.has(ek)) continue
    edgeSeen.add(ek)
    dedupedEdges.push({ ...e, from: newFrom, to: newTo })
  }

  return { nodes: dedupedNodes, edges: dedupedEdges }
}

function nodeTypesFromNodesLocal(nodes: RawGraphNode[]): Record<string, number> {
  const nt: Record<string, number> = {}
  for (const n of nodes) {
    const t = n && typeof n.type === 'string' ? n.type : '?'
    nt[t] = (nt[t] || 0) + 1
  }
  return nt
}

export function mergeParsedArtifacts(arts: ParsedArtifact[]): ParsedArtifact | null {
  if (!arts || arts.length < 2) return null
  const kind = arts[0].kind
  if (kind !== 'gi' && kind !== 'kg') return null
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
      const ek = `${from}\0${to}\0${String(e.type || '')}`
      if (edgeSeen.has(ek)) continue
      edgeSeen.add(ek)
      edgeList.push(deepClone(e))
    }
  }
  const mergedData = deepClone(arts[0].data) as ArtifactData
  const deduped = deduplicateEntities(Array.from(nodeById.values()), edgeList, 'name-based')
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

export function combineGiKgParsedArtifacts(
  giArt: ParsedArtifact,
  kgArt: ParsedArtifact,
): ParsedArtifact | null {
  if (!giArt || !kgArt || giArt.kind !== 'gi' || kgArt.kind !== 'kg') {
    return null
  }

  function remapData(artData: ArtifactData, pfx: string): ArtifactData {
    const data = deepClone(artData)
    const nodes = Array.isArray(data.nodes) ? data.nodes : []
    const idMap = new Map<string, string>()
    for (const n of nodes) {
      if (!n || n.id == null) continue
      const old = String(n.id)
      const neu = pfx + old
      idMap.set(old, neu)
      n.id = neu
    }
    const edges = Array.isArray(data.edges) ? data.edges : []
    for (const ed of edges) {
      if (!ed) continue
      if (ed.from != null) {
        const f = String(ed.from)
        ed.from = idMap.has(f) ? idMap.get(f)! : pfx + f
      }
      if (ed.to != null) {
        const t = String(ed.to)
        ed.to = idMap.has(t) ? idMap.get(t)! : pfx + t
      }
    }
    return data
  }

  function repairStalePrefixedEpisodeRefs(
    nodes: RawGraphNode[],
    edges: RawGraphEdge[] | undefined,
  ): RawGraphEdge[] {
    const unified = new Map<string, string>()
    const nArr = Array.isArray(nodes) ? nodes : []
    for (const n of nArr) {
      if (!n || n.type !== 'Episode' || n.id == null) continue
      const id = String(n.id)
      const p = '__unified_ep__:'
      if (id.startsWith(p)) {
        const key = id.slice(p.length)
        if (key) unified.set(key, id)
      }
    }
    if (unified.size === 0) return Array.isArray(edges) ? edges : []
    const eArr = Array.isArray(edges) ? edges : []
    return eArr.map((ed) => {
      if (!ed || typeof ed !== 'object') return ed
      const o = { ...ed }
      function fix(v: unknown): unknown {
        if (v == null) return v
        const s = String(v)
        const key = episodeKey(s)
        if (key && unified.has(key)) return unified.get(key)
        return v
      }
      o.from = fix(o.from) as string | number | undefined
      o.to = fix(o.to) as string | number | undefined
      return o
    })
  }

  /**
   * Extract the bare UUID from a prefixed episode node ID.
   * Handles: g:ep:<uuid>, g:episode:<uuid>, k:kg:episode:<uuid>, k:episode:<uuid>, etc.
   */
  function episodeKey(prefixedId: string): string | null {
    const m = prefixedId.match(
      /^[gk]:(?:kg:)?(?:ep(?:isode)?):(.+)$/,
    )
    return m ? m[1] : null
  }

  function unifyGiKgEpisodeAnchors(
    gdIn: ArtifactData,
    kdIn: ArtifactData,
  ): { nodes: RawGraphNode[]; edges: RawGraphEdge[]; episode_id: string } | null {
    const nodesGi = (gdIn.nodes || []).slice()
    const nodesKg = (kdIn.nodes || []).slice()
    const mapGi = new Map<string, string>()
    const mapKg = new Map<string, string>()
    for (const n of nodesGi) {
      if (!n || n.type !== 'Episode' || n.id == null) continue
      const sid = String(n.id)
      const key = episodeKey(sid)
      if (key) mapGi.set(key, sid)
    }
    for (const n of nodesKg) {
      if (!n || n.type !== 'Episode' || n.id == null) continue
      const sid = String(n.id)
      const key = episodeKey(sid)
      if (key) mapKg.set(key, sid)
    }
    const keys: string[] = []
    mapGi.forEach((_gid, key) => {
      if (mapKg.has(key)) keys.push(key)
    })
    keys.sort()
    if (keys.length === 0) return null

    const giRemove = new Set<string>()
    const kgRemove = new Set<string>()
    const unifiedList: RawGraphNode[] = []
    for (const key of keys) {
      const giEpId = mapGi.get(key)!
      const kgEpId = mapKg.get(key)!
      giRemove.add(giEpId)
      kgRemove.add(kgEpId)
      let giNode: RawGraphNode | null = null
      let kgNode: RawGraphNode | null = null
      for (const n of nodesGi) {
        if (n && String(n.id) === giEpId) {
          giNode = n
          break
        }
      }
      for (const n of nodesKg) {
        if (n && String(n.id) === kgEpId) {
          kgNode = n
          break
        }
      }
      const unifiedId = `__unified_ep__:${key}`
      unifiedList.push({
        id: unifiedId,
        type: 'Episode',
        properties: Object.assign(
          {},
          giNode?.properties || {},
          kgNode?.properties || {},
        ),
      })
    }
    const restGi = nodesGi.filter((n) => n && !giRemove.has(String(n.id)))
    const restKg = nodesKg.filter((n) => n && !kgRemove.has(String(n.id)))
    const repl = new Map<string, string>()
    for (const key of keys) {
      const u = `__unified_ep__:${key}`
      const giId = mapGi.get(key)
      const kgId = mapKg.get(key)
      if (giId) repl.set(giId, u)
      if (kgId) repl.set(kgId, u)
    }
    function rewriteEdges(edges: RawGraphEdge[] | undefined): RawGraphEdge[] {
      const arr = Array.isArray(edges) ? edges : []
      return arr.map((ed) => {
        if (!ed || typeof ed !== 'object') return ed
        const o = { ...ed }
        const from = o.from != null ? String(o.from) : ''
        const to = o.to != null ? String(o.to) : ''
        if (from && repl.has(from)) o.from = repl.get(from)!
        if (to && repl.has(to)) o.to = repl.get(to)!
        return o
      })
    }
    const edgesGi = rewriteEdges(gdIn.edges)
    const edgesKg = rewriteEdges(kdIn.edges)
    const nodesOut = restGi.concat(restKg).concat(unifiedList)
    const edgesOut = repairStalePrefixedEpisodeRefs(nodesOut, edgesGi.concat(edgesKg))
    const epRoot =
      keys.length === 1 ? `merged:gi+kg:${keys[0]}` : 'merged:gi+kg:multi'
    return { nodes: nodesOut, edges: edgesOut, episode_id: epRoot }
  }

  const gd = remapData(giArt.data, 'g:')
  const kd = remapData(kgArt.data, 'k:')

  const unified = unifyGiKgEpisodeAnchors(gd, kd)
  let mergedData: ArtifactData
  if (unified) {
    mergedData = Object.assign({}, gd, {
      nodes: unified.nodes,
      edges: unified.edges,
      episode_id: unified.episode_id,
    })
  } else {
    mergedData = Object.assign({}, gd, {
      nodes: (gd.nodes || []).concat(kd.nodes || []),
      edges: (gd.edges || []).concat(kd.edges || []),
      episode_id: `merged:gi+kg:${String(giArt.episodeId || giArt.name)}+${String(kgArt.episodeId || kgArt.name)}`,
    })
  }
  if (kd.extraction && typeof kd.extraction === 'object') {
    mergedData.extraction = kd.extraction
  }
  const epAug = ensureEpisodeToInsightEdges(
    mergedData.nodes || [],
    mergedData.edges || [],
  )
  const deduped = deduplicateEntities(epAug.nodes, epAug.edges, 'cil-first')
  mergedData.nodes = deduped.nodes
  mergedData.edges = deduped.edges
  const nodeTypes = nodeTypesFromNodesLocal(mergedData.nodes || [])
  return {
    name: 'Merged GI + KG',
    kind: 'both',
    episodeId: mergedData.episode_id || null,
    nodes: mergedData.nodes!.length,
    edges: mergedData.edges!.length,
    nodeTypes,
    data: mergedData,
    sourceCorpusRelPath: giArt.sourceCorpusRelPath ?? null,
    sourceCorpusRelPathByEpisodeId: mergeEpisodeSourcePathsFromParsed([giArt]),
  }
}

export function mergeGiKgFromArtifactArrays(
  giArts: ParsedArtifact[],
  kgArts: ParsedArtifact[],
): ParsedArtifact | null {
  if (!giArts?.length || !kgArts?.length) return null
  const giMerged = giArts.length >= 2 ? mergeParsedArtifacts(giArts) : giArts[0]
  const kgMerged = kgArts.length >= 2 ? mergeParsedArtifacts(kgArts) : kgArts[0]
  if (!giMerged || !kgMerged) return null
  const combined = combineGiKgParsedArtifacts(giMerged, kgMerged)
  if (!combined) return null
  combined.name = `Merged GI + KG (${String(giArts.length)} GI · ${String(kgArts.length)} KG)`
  return combined
}

/** Build one graph to display from selected GI and KG parsed artifacts. */
export function buildDisplayArtifact(
  giArts: ParsedArtifact[],
  kgArts: ParsedArtifact[],
): ParsedArtifact | null {
  if (giArts.length >= 1 && kgArts.length >= 1) {
    return mergeGiKgFromArtifactArrays(giArts, kgArts)
  }
  if (giArts.length >= 2) {
    return mergeParsedArtifacts(giArts)
  }
  if (giArts.length === 1) return giArts[0]
  if (kgArts.length >= 2) {
    return mergeParsedArtifacts(kgArts)
  }
  if (kgArts.length === 1) return kgArts[0]
  return null
}
