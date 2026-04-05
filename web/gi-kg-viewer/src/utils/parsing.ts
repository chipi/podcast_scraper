/**
 * Artifact parsing, filters, and Cytoscape element building (ported from web/gi-kg-viz/shared.js).
 */
import type {
  ArtifactData,
  ArtifactKind,
  GraphFilterState,
  ParsedArtifact,
  RawGraphEdge,
  RawGraphNode,
} from '../types/artifact'
import { humanizeSlug, truncate } from './formatting'
import { visualGroupForNode } from './visualGroup'

export { visualGroupForNode, visualNodeTypeCounts } from './visualGroup'

export function ensureEpisodeToInsightEdges(
  nodes: RawGraphNode[],
  edges: RawGraphEdge[],
): { nodes: RawGraphNode[]; edges: RawGraphEdge[] } {
  const nList = Array.isArray(nodes) ? nodes.slice() : []
  const eList: RawGraphEdge[] = Array.isArray(edges)
    ? edges.map((e) => ({ ...e }))
    : []
  const episodes = nList.filter((n) => n && n.type === 'Episode')
  const insights = nList.filter((n) => n && n.type === 'Insight')
  if (episodes.length === 0 || insights.length === 0) {
    return { nodes: nList, edges: eList }
  }

  function episodeKeyFromNode(ep: RawGraphNode): string | null {
    const id = String(ep.id)
    if (id.startsWith('g:episode:')) return id.slice('g:episode:'.length)
    if (id.startsWith('k:episode:')) return id.slice('k:episode:'.length)
    if (id.startsWith('episode:')) return id.slice('episode:'.length)
    if (id.startsWith('__unified_ep__:')) return id.slice('__unified_ep__:'.length)
    return null
  }

  const seen = new Set<string>()
  for (const e of eList) {
    if (!e) continue
    seen.add(`${String(e.from)}\0${String(e.to)}\0${String(e.type || '')}`)
  }

  for (const ep of episodes) {
    const eid = episodeKeyFromNode(ep)
    if (eid == null) continue
    const epId = String(ep.id)
    for (const ins of insights) {
      const p = ins.properties || {}
      if (String(p.episode_id || '') !== eid) continue
      const iid = String(ins.id)
      const k = `${epId}\0${iid}\0HAS_INSIGHT`
      if (seen.has(k)) continue
      eList.push({ type: 'HAS_INSIGHT', from: epId, to: iid })
      seen.add(k)
    }
  }

  if (episodes.length === 1) {
    const soleEp = episodes[0]
    const epId = String(soleEp.id)
    for (const ins of insights) {
      const iid = String(ins.id)
      let hasIncoming = false
      for (const e of eList) {
        if (
          e &&
          String(e.type || '') === 'HAS_INSIGHT' &&
          String(e.to) === iid
        ) {
          hasIncoming = true
          break
        }
      }
      if (hasIncoming) continue
      const k = `${epId}\0${iid}\0HAS_INSIGHT`
      if (seen.has(k)) continue
      eList.push({ type: 'HAS_INSIGHT', from: epId, to: iid })
      seen.add(k)
    }
  }
  return { nodes: nList, edges: eList }
}

export function parseArtifact(filename: string, data: ArtifactData): ParsedArtifact {
  let nodes = Array.isArray(data.nodes) ? data.nodes.slice() : []
  let edges = Array.isArray(data.edges) ? data.edges.slice() : []
  const episodeId =
    typeof data.episode_id === 'string' ? data.episode_id : null

  let kind: ArtifactKind = 'unknown'
  const lower = filename.toLowerCase()
  if (lower.endsWith('.gi.json')) {
    kind = 'gi'
  } else if (lower.endsWith('.kg.json')) {
    kind = 'kg'
  } else if (
    data.extraction &&
    typeof data.extraction === 'object' &&
    !Object.prototype.hasOwnProperty.call(data, 'prompt_version')
  ) {
    kind = 'kg'
  } else if (
    typeof data.model_version === 'string' &&
    typeof data.prompt_version === 'string'
  ) {
    kind = 'gi'
  }

  if (kind === 'gi') {
    const aug = ensureEpisodeToInsightEdges(nodes, edges)
    nodes = aug.nodes
    edges = aug.edges
  }

  const nodeTypes: Record<string, number> = {}
  for (const n of nodes) {
    const t = n && typeof n.type === 'string' ? n.type : '?'
    nodeTypes[t] = (nodeTypes[t] || 0) + 1
  }

  const dataOut =
    kind === 'gi' ? { ...data, nodes, edges } : { ...data }

  return {
    name: filename,
    kind,
    episodeId,
    nodes: nodes.length,
    edges: edges.length,
    nodeTypes,
    data: dataOut,
  }
}

export function entityDisplayNameFromId(idStr: string): string {
  let s = String(idStr)
  if (s.startsWith('k:') || s.startsWith('g:')) {
    s = s.slice(2)
  }
  const m = s.match(/^entity:(?:person|organization):(.+)$/)
  if (!m?.[1]) return ''
  return humanizeSlug(m[1])
}

export function nodeLabel(n: RawGraphNode): string {
  const tRaw = n.type != null ? String(n.type) : '?'
  const tLower = tRaw.toLowerCase()
  const p = n.properties || {}
  let idShort = n.id != null ? String(n.id) : ''
  if (idShort.length > 24) {
    idShort = `${idShort.slice(0, 22)}…`
  }
  if (tLower === 'insight' && p.text) {
    return truncate(String(p.text), 36)
  }
  if (tLower === 'quote' && p.text) {
    return truncate(String(p.text), 36)
  }
  if (tLower === 'topic' && p.label) {
    return truncate(String(p.label), 80)
  }
  if (tLower === 'entity') {
    const fromLabel = p.label != null ? String(p.label).trim() : ''
    const fromName = p.name != null ? String(p.name).trim() : ''
    const nm =
      fromLabel ||
      fromName ||
      entityDisplayNameFromId(String(n.id ?? '')) ||
      ''
    if (nm) {
      let out = truncate(nm, 52)
      const role = p.role != null ? String(p.role).trim() : ''
      if (role !== '' && role !== 'mentioned') {
        out = truncate(`${out} (${role})`, 58)
      }
      return out
    }
  }
  if (
    tLower === 'speaker' &&
    ((p.name && String(p.name).trim()) || (p.label && String(p.label).trim()))
  ) {
    return truncate(String(p.name || p.label).trim(), 48)
  }
  if (tLower === 'episode' && p.title) {
    return truncate(String(p.title), 40)
  }
  return tRaw + (idShort ? `: ${idShort}` : '')
}

export function buildNodeTitle(n: RawGraphNode): string {
  try {
    const disp = nodeLabel(n)
    const head = `${n.type || '?'}\n${disp}\nid: ${String(n.id || '')}\n\nproperties:\n${JSON.stringify(n.properties || {}, null, 2)}`
    return truncate(head, 900)
  } catch {
    return String(n.id || '')
  }
}

export function findRawNodeInArtifact(
  art: ParsedArtifact | null,
  nodeId: string | number,
): RawGraphNode | null {
  if (!art?.data) return null
  const nodes = Array.isArray(art.data.nodes) ? art.data.nodes : []
  const sid = String(nodeId)
  for (const n of nodes) {
    if (n && n.id != null && String(n.id) === sid) {
      return n
    }
  }
  return null
}

export function defaultFilterState(art: ParsedArtifact | null): GraphFilterState | null {
  if (!art) return null
  const allowedTypes: Record<string, boolean> = {}
  const rawNodes = Array.isArray(art.data.nodes) ? art.data.nodes : []
  const seen = new Set<string>()
  for (const n of rawNodes) {
    if (!n || typeof n !== 'object') continue
    const t = n.type || '?'
    if (!seen.has(t)) {
      seen.add(t)
      allowedTypes[t] = true
    }
  }
  return {
    allowedTypes,
    hideUngroundedInsights: false,
    legendSoloVisual: null,
    showGiLayer: true,
    showKgLayer: true,
  }
}

export function filtersActive(
  fullArt: ParsedArtifact | null,
  state: GraphFilterState | null,
): boolean {
  if (!fullArt || !state) return false
  const keys = Object.keys(state.allowedTypes)
  for (const k of keys) {
    if (state.allowedTypes[k] === false) return true
  }
  if (
    (fullArt.kind === 'gi' || fullArt.kind === 'both') &&
    state.hideUngroundedInsights
  ) {
    return true
  }
  if (
    typeof state.legendSoloVisual === 'string' &&
    state.legendSoloVisual.length > 0
  ) {
    return true
  }
  if (
    fullArt.kind === 'both' &&
    (state.showGiLayer === false || state.showKgLayer === false)
  ) {
    return true
  }
  return false
}

export function applyGraphFilters(
  fullArt: ParsedArtifact,
  state: GraphFilterState,
): ParsedArtifact {
  let nodes = (fullArt.data.nodes || []).slice()
  if (fullArt.kind === 'both' && (!state.showGiLayer || !state.showKgLayer)) {
    const gi = state.showGiLayer
    const kg = state.showKgLayer
    nodes = nodes.filter((n) => {
      if (!n || typeof n !== 'object') return false
      const id = String(n.id ?? '')
      if (id.startsWith('__unified_ep__:')) {
        return gi || kg
      }
      if (id.startsWith('g:')) {
        return gi
      }
      if (id.startsWith('k:')) {
        return kg
      }
      return true
    })
  }
  if (
    (fullArt.kind === 'gi' || fullArt.kind === 'both') &&
    state.hideUngroundedInsights
  ) {
    nodes = nodes.filter((n) => {
      if (!n || typeof n !== 'object') return false
      if (n.type !== 'Insight') return true
      return Boolean(n.properties && n.properties.grounded === true)
    })
  }
  const { allowedTypes } = state
  nodes = nodes.filter((n) => {
    if (!n || typeof n !== 'object') return false
    const t = n.type || '?'
    return allowedTypes[t] !== false
  })
  const soloV = state.legendSoloVisual
  if (typeof soloV === 'string' && soloV.length > 0) {
    nodes = nodes.filter((n) => visualGroupForNode(n) === soloV)
  }
  const ids = new Set<string>()
  for (const n of nodes) {
    if (n.id != null) ids.add(String(n.id))
  }
  const edges = (fullArt.data.edges || []).filter((e) => {
    return ids.has(String(e.from)) && ids.has(String(e.to))
  })
  const nodeTypes: Record<string, number> = {}
  for (const n of nodes) {
    const t = n.type || '?'
    nodeTypes[t] = (nodeTypes[t] || 0) + 1
  }
  return {
    name: fullArt.name,
    kind: fullArt.kind,
    episodeId: fullArt.episodeId,
    nodes: nodes.length,
    edges: edges.length,
    nodeTypes,
    data: { ...fullArt.data, nodes, edges },
  }
}

export function filterArtifactEgoOneHop(
  art: ParsedArtifact,
  focusId: string | null | undefined,
): ParsedArtifact {
  if (focusId == null || focusId === '') return art
  const f = String(focusId)
  const nodes = Array.isArray(art.data.nodes) ? art.data.nodes : []
  const edges = Array.isArray(art.data.edges) ? art.data.edges : []
  const idSet = new Set<string>()
  for (const n of nodes) {
    if (n?.id != null) idSet.add(String(n.id))
  }
  if (!idSet.has(f)) return art

  const keep = new Set<string>([f])
  for (const e of edges) {
    if (!e) continue
    const fr = String(e.from)
    const to = String(e.to)
    if (fr === f && idSet.has(to)) keep.add(to)
    else if (to === f && idSet.has(fr)) keep.add(fr)
  }
  const nodesOut = nodes.filter((n) => n?.id != null && keep.has(String(n.id)))
  const outIds = new Set(nodesOut.map((n) => String(n!.id)))
  const edgesOut = edges.filter(
    (e) => e && outIds.has(String(e.from)) && outIds.has(String(e.to)),
  )
  const nodeTypes: Record<string, number> = {}
  for (const n of nodesOut) {
    const t = n!.type || '?'
    nodeTypes[t] = (nodeTypes[t] || 0) + 1
  }
  return {
    name: art.name,
    kind: art.kind,
    episodeId: art.episodeId,
    nodes: nodesOut.length,
    edges: edgesOut.length,
    nodeTypes,
    data: { ...art.data, nodes: nodesOut, edges: edgesOut },
  }
}

export function toGraphElements(art: ParsedArtifact): {
  visNodes: { id: string; label: string; group: string; title: string }[]
  visEdges: { id: string; from: string; to: string; label: string }[]
  idSet: Set<string>
} {
  const d = art.data
  let rawNodes = Array.isArray(d.nodes) ? d.nodes.slice() : []
  let rawEdges = Array.isArray(d.edges) ? d.edges.map((e) => ({ ...e })) : []
  if (art.kind === 'gi' || art.kind === 'both') {
    const aug = ensureEpisodeToInsightEdges(rawNodes, rawEdges)
    rawNodes = aug.nodes
    rawEdges = aug.edges
  }
  const visNodes = rawNodes.map((n, i) => {
    if (!n || typeof n !== 'object') {
      return { id: `n${i}`, label: '?', group: '?', title: '' }
    }
    const id = n.id != null ? String(n.id) : `n${i}`
    return {
      id,
      label: nodeLabel(n),
      group: visualGroupForNode(n),
      title: buildNodeTitle(n),
    }
  })
  const idSet = new Set(visNodes.map((x) => x.id))
  const visEdges = rawEdges.map((e, i) => {
    if (!e || typeof e !== 'object') {
      return { id: `e${i}`, from: '', to: '', label: '' }
    }
    return {
      id: `e${i}`,
      from: e.from != null ? String(e.from) : '',
      to: e.to != null ? String(e.to) : '',
      label: e.type ? String(e.type) : '',
    }
  })
  return { visNodes, visEdges, idSet }
}

/** Cytoscape element list (nodes + edges). */
export function toCytoElements(art: ParsedArtifact): import('cytoscape').ElementDefinition[] {
  const g = toGraphElements(art)
  const nodes: import('cytoscape').ElementDefinition[] = g.visNodes.map((n) => ({
    data: { id: n.id, label: n.label, type: n.group },
  }))
  const edges: import('cytoscape').ElementDefinition[] = g.visEdges
    .filter((e) => g.idSet.has(e.from) && g.idSet.has(e.to))
    .map((e) => ({
      data: {
        id: e.id,
        source: e.from,
        target: e.to,
        label: e.label,
      },
    }))
  return [...nodes, ...edges]
}
