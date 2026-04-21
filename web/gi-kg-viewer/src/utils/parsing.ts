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
import { humanizeSlug, shortPhrase, truncate } from './formatting'
import { logicalEpisodeIdFromGraphNodeId } from './graphEpisodeMetadata'
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
    return logicalEpisodeIdFromGraphNodeId(String(ep.id ?? ''))
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

/**
 * Episode graph node id in ``art`` whose logical id matches ``episodeKey`` (GI ``properties.episode_id``).
 */
export function findEpisodeGraphNodeIdForEpisodeKey(
  art: ParsedArtifact | null,
  episodeKey: string,
): string | null {
  const want = episodeKey.trim()
  if (!want || !art?.data?.nodes) return null
  for (const n of art.data.nodes) {
    if (!n || n.type !== 'Episode' || n.id == null) continue
    const logical = logicalEpisodeIdFromGraphNodeId(String(n.id))
    if (logical === want) return String(n.id)
  }
  return null
}

export function parseArtifact(
  filename: string,
  data: ArtifactData,
  sourceCorpusRelPath?: string | null,
): ParsedArtifact {
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

  const rel = sourceCorpusRelPath != null && String(sourceCorpusRelPath).trim()
    ? String(sourceCorpusRelPath).trim().replace(/\\/g, '/').replace(/^\/+/, '')
    : null

  const sourceCorpusRelPathByEpisodeId =
    rel && episodeId
      ? { [episodeId]: rel }
      : null

  return {
    name: filename,
    kind,
    episodeId,
    nodes: nodes.length,
    edges: edges.length,
    nodeTypes,
    data: dataOut,
    sourceCorpusRelPath: rel,
    sourceCorpusRelPathByEpisodeId,
  }
}

export function entityDisplayNameFromId(idStr: string): string {
  let s = String(idStr)
  if (s.startsWith('k:') || s.startsWith('g:')) {
    s = s.slice(2)
  }
  if (s.startsWith('kg:')) {
    s = s.slice(3)
  }
  let m = s.match(/^entity:(?:person|organization):(.+)$/)
  if (m?.[1]) return humanizeSlug(m[1])
  m = s.match(/^person:(.+)$/)
  if (m?.[1]) return humanizeSlug(m[1])
  m = s.match(/^org:(.+)$/)
  if (m?.[1]) return humanizeSlug(m[1])
  return ''
}

/**
 * Label for quote attribution when only an id is available (search hits, explore rows).
 * Uses {@link entityDisplayNameFromId} for `person:` / `org:` / legacy `entity:…` ids;
 * humanises legacy `speaker:` slugs; otherwise returns the trimmed string.
 */
export function quoteAttributionDisplayFromId(idStr: string | null | undefined): string {
  if (idStr == null) return ''
  const raw = String(idStr).trim()
  if (!raw) return ''
  const fromEntity = entityDisplayNameFromId(raw)
  if (fromEntity) return fromEntity
  if (raw.startsWith('speaker:')) {
    return humanizeSlug(raw.slice('speaker:'.length))
  }
  return raw
}

const MAX_GRAPH_LABEL = 40

/**
 * Produce a short label suitable for graph display.  Works for any node type
 * by checking common property names in priority order, then falling back to
 * humanising the node id.  All results are capped at ~40 chars via
 * `shortPhrase` (prefers natural break points like commas).
 *
 * For **node detail** / right-rail copy, use {@link fullPrimaryNodeLabel} instead so long quotes are not pre-truncated.
 */
export function nodeLabel(n: RawGraphNode): string {
  const p = n.properties || {}
  const typeStr = n.type != null ? String(n.type) : '?'

  const raw =
    str(p.name) ||
    str(p.title) ||
    str(p.label) ||
    str(p.text) ||
    entityDisplayNameFromId(String(n.id ?? '')) ||
    ''

  if (raw) return shortPhrase(raw, MAX_GRAPH_LABEL)

  let idShort = n.id != null ? String(n.id) : ''
  if (idShort.length > 24) idShort = `${idShort.slice(0, 22)}…`
  return typeStr + (idShort ? `: ${idShort}` : '')
}

/**
 * Same primary text as ``nodeLabel`` but without the short-phrase cap — for tooltips and deduping body copy.
 */
export function fullPrimaryNodeLabel(n: RawGraphNode): string {
  const p = n.properties || {}
  const typeStr = n.type != null ? String(n.type) : '?'
  const raw =
    str(p.name) ||
    str(p.title) ||
    str(p.label) ||
    str(p.text) ||
    entityDisplayNameFromId(String(n.id ?? '')) ||
    ''
  if (raw) return raw
  let idShort = n.id != null ? String(n.id) : ''
  if (idShort.length > 24) idShort = `${idShort.slice(0, 22)}…`
  return typeStr + (idShort ? `: ${idShort}` : '')
}

/**
 * Primary display string for a GI node object from JSON (e.g. CIL timeline ``insights[]``).
 * Uses the same field order as ``fullPrimaryNodeLabel``.
 */
export function primaryTextFromLooseGiNode(node: Record<string, unknown>): string {
  const rawProps = node.properties
  const props =
    rawProps != null &&
    typeof rawProps === 'object' &&
    !Array.isArray(rawProps)
      ? (rawProps as Record<string, unknown>)
      : {}
  return fullPrimaryNodeLabel({
    id: node.id,
    type: node.type,
    properties: props,
  } as RawGraphNode)
}

function str(v: unknown): string {
  if (v == null) return ''
  const s = String(v).trim()
  return s.length > 0 ? s : ''
}

/** Normalise raw GI/KG edge `type` strings for Cytoscape `edgeType` (case + hyphen). */
export function canonicalArtifactEdgeType(raw: string | undefined): string {
  const t = String(raw ?? '').trim()
  if (!t) {
    return '(unknown)'
  }
  return t.toUpperCase().replace(/-/g, '_')
}

/** WIP §3.5 — graph label truncation for the medium-zoom tier. */
export function graphShortLabelFromDisplayLabel(label: string): string {
  const s = String(label ?? '')
  return s.length > 18 ? `${s.slice(0, 16)}…` : s
}

function parsePublishDateMs(value: unknown): number | null {
  if (value == null) {
    return null
  }
  const parsed = Date.parse(String(value).trim())
  return Number.isFinite(parsed) ? parsed : null
}

function publishTimeMsForRawNode(
  n: RawGraphNode | undefined,
  nodesById: Map<string, RawGraphNode>,
): number | null {
  if (!n) {
    return null
  }
  if (String(n.type) === 'Episode') {
    return parsePublishDateMs(n.properties?.publish_date)
  }
  const eid = str((n.properties || {}).episode_id)
  if (!eid) {
    return null
  }
  for (const cand of nodesById.values()) {
    if (!cand || String(cand.type) !== 'Episode') {
      continue
    }
    if (logicalEpisodeIdFromGraphNodeId(String(cand.id ?? '')) === eid) {
      return parsePublishDateMs(cand.properties?.publish_date)
    }
  }
  return null
}

function recencyWeightFromPublishMs(ms: number | null): number {
  if (ms == null || !Number.isFinite(ms)) {
    return 1
  }
  const daysSince = (Date.now() - ms) / 86_400_000
  return Math.max(0.4, 1.0 - (daysSince / 90) * 0.6)
}

function confidenceOpacityFromInsightProperties(
  properties: Record<string, unknown> | undefined,
): number {
  const raw = properties?.confidence
  let c = Number.NaN
  if (typeof raw === 'number') {
    c = raw
  } else if (typeof raw === 'string' && raw.trim()) {
    c = Number(raw)
  }
  const conf = Number.isFinite(c) ? Math.max(0, Math.min(1, c)) : 0.7
  return 0.5 + conf * 0.5
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

/**
 * Count GI-style incident edges for Person / Entity / Speaker nodes in the loaded graph slice.
 * ``SPOKEN_BY`` (Quote → node): quotes attributed to this identity.
 * ``SPOKE_IN`` (node → Episode): episode participation links.
 */
export function countPersonEntityIncidentEdges(
  art: ParsedArtifact | null,
  nodeId: string | null,
): { spokenByQuotes: number; spokeInEpisodes: number } {
  if (!art?.data?.edges || nodeId == null) {
    return { spokenByQuotes: 0, spokeInEpisodes: 0 }
  }
  const id = String(nodeId).trim()
  if (!id) return { spokenByQuotes: 0, spokeInEpisodes: 0 }
  let spokenByQuotes = 0
  let spokeInEpisodes = 0
  for (const e of art.data.edges) {
    if (!e || typeof e !== 'object') continue
    const ty = normalizeGiEdgeType(e.type)
    const from = e.from != null ? String(e.from).trim() : ''
    const to = e.to != null ? String(e.to).trim() : ''
    if (ty === 'spoken_by' && to === id) spokenByQuotes += 1
    if (ty === 'spoke_in' && from === id) spokeInEpisodes += 1
  }
  return { spokenByQuotes, spokeInEpisodes }
}

export interface InsightSupportingQuoteRow {
  id: string
  preview: string
  /** Sort key from quote ``char_start`` when finite. */
  charStart: number | null
  /** Fallback sort key from ``timestamp_start_ms`` when ``char_start`` ties or is absent. */
  timestampStartMs: number | null
}

export function normalizeGiEdgeType(type: string | undefined | null): string {
  return String(type ?? '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_')
}

function quoteSortKeysFromNode(n: RawGraphNode | null): {
  charStart: number | null
  timestampStartMs: number | null
} {
  if (!n?.properties || typeof n.properties !== 'object') {
    return { charStart: null, timestampStartMs: null }
  }
  const p = n.properties as Record<string, unknown>
  const cs = p.char_start
  const charStart =
    typeof cs === 'number' && Number.isFinite(cs) ? cs : null
  const ts = p.timestamp_start_ms
  const timestampStartMs =
    typeof ts === 'number' && Number.isFinite(ts) ? ts : null
  return { charStart, timestampStartMs }
}

/** Quote targets of ``SUPPORTED_BY`` out-edges from this insight (GI). Sorted by ``char_start``, then ``timestamp_start_ms``. */
export function insightSupportingQuoteRows(
  art: ParsedArtifact | null,
  insightGraphId: string | null,
): InsightSupportingQuoteRow[] {
  if (!art?.data?.edges || insightGraphId == null) return []
  const sid = String(insightGraphId).trim()
  if (!sid) return []
  const toIds: string[] = []
  for (const e of art.data.edges) {
    if (!e) continue
    const etNorm = normalizeGiEdgeType(e.type).replace(/_/g, '')
    if (etNorm !== 'supportedby') continue
    if (String(e.from) !== sid) continue
    const to = String(e.to ?? '').trim()
    if (!to) continue
    const qn = findRawNodeInArtifact(art, to)
    if (!qn || String(qn.type) !== 'Quote') continue
    toIds.push(to)
  }
  const seen = new Set<string>()
  const out: InsightSupportingQuoteRow[] = []
  for (const id of toIds) {
    if (seen.has(id)) continue
    seen.add(id)
    const n = findRawNodeInArtifact(art, id)
    const raw =
      n?.properties && typeof (n.properties as Record<string, unknown>).text === 'string'
        ? (n.properties as Record<string, unknown>).text
        : ''
    const preview = truncate(String(raw).trim(), 120)
    const { charStart, timestampStartMs } = quoteSortKeysFromNode(n)
    out.push({
      id,
      preview: preview || id,
      charStart,
      timestampStartMs,
    })
  }
  out.sort((a, b) => {
    const ca = a.charStart ?? Number.POSITIVE_INFINITY
    const cb = b.charStart ?? Number.POSITIVE_INFINITY
    if (ca !== cb) return ca - cb
    const ta = a.timestampStartMs ?? Number.POSITIVE_INFINITY
    const tb = b.timestampStartMs ?? Number.POSITIVE_INFINITY
    return ta - tb
  })
  return out
}

/** Resolved transcript + char ranges when every supporting quote shares one ``transcript_ref`` and finite GI offsets. */
export type InsightSupportingTranscriptAggregate = {
  transcriptRef: string
  episodeId: string | null
  charRanges: Array<{ charStart: unknown; charEnd: unknown }>
}

/**
 * Collect ``SUPPORTED_BY`` quotes for an insight for a single-file transcript open.
 * Returns null when there are no quotes, mixed ``transcript_ref`` values, or any supporting quote
 * lacks a non-empty ``transcript_ref`` with both finite ``char_start`` and ``char_end``.
 */
export function insightSupportingTranscriptAggregate(
  art: ParsedArtifact | null,
  insightGraphId: string | null,
): InsightSupportingTranscriptAggregate | null {
  const rows = insightSupportingQuoteRows(art, insightGraphId)
  if (rows.length === 0) {
    return null
  }
  const insight = findRawNodeInArtifact(art, insightGraphId ?? '')
  const iep = insight?.properties as Record<string, unknown> | undefined
  const rawEp = iep?.episode_id
  let episodeId =
    typeof rawEp === 'string' && rawEp.trim()
      ? rawEp.trim()
      : typeof rawEp === 'number' && Number.isFinite(rawEp)
        ? String(rawEp)
        : null

  const refs = new Set<string>()
  const charRanges: Array<{ charStart: unknown; charEnd: unknown }> = []
  for (const row of rows) {
    const n = findRawNodeInArtifact(art, row.id)
    const p = n?.properties as Record<string, unknown> | undefined
    if (!p) {
      continue
    }
    const ref = typeof p.transcript_ref === 'string' ? p.transcript_ref.trim() : ''
    const cs = p.char_start
    const ce = p.char_end
    if (!ref) {
      continue
    }
    if (typeof cs !== 'number' || !Number.isFinite(cs)) {
      continue
    }
    if (typeof ce !== 'number' || !Number.isFinite(ce)) {
      continue
    }
    refs.add(ref)
    charRanges.push({ charStart: cs, charEnd: ce })
  }
  if (charRanges.length === 0 || refs.size !== 1 || charRanges.length !== rows.length) {
    return null
  }
  if (!episodeId && rows.length > 0) {
    const qn = findRawNodeInArtifact(art, rows[0]!.id)
    const qp = qn?.properties as Record<string, unknown> | undefined
    const qe = qp?.episode_id
    if (typeof qe === 'string' && qe.trim()) {
      episodeId = qe.trim()
    } else if (typeof qe === 'number' && Number.isFinite(qe)) {
      episodeId = String(qe)
    }
  }
  return {
    transcriptRef: [...refs][0],
    episodeId,
    charRanges,
  }
}

export interface InsightRelatedTopicRow {
  topicId: string
  label: string
}

const INSIGHT_TOPIC_EDGE_TYPES = new Set(['about', 'related_to'])

/**
 * Topic neighbors of an insight linked by ``ABOUT`` / ``RELATED_TO`` (either direction).
 */
export function insightRelatedTopicRows(
  art: ParsedArtifact | null,
  insightGraphId: string | null,
): InsightRelatedTopicRow[] {
  if (!art?.data?.edges || insightGraphId == null) return []
  const sid = String(insightGraphId).trim()
  if (!sid) return []
  const nodes = Array.isArray(art.data.nodes) ? art.data.nodes : []
  const idSet = new Set(
    nodes.map((n) => (n?.id != null ? String(n.id) : '')).filter(Boolean),
  )
  const topicIds: string[] = []
  for (const e of art.data.edges) {
    if (!e) continue
    const t = normalizeGiEdgeType(e.type)
    if (!INSIGHT_TOPIC_EDGE_TYPES.has(t)) continue
    const from = String(e.from ?? '').trim()
    const to = String(e.to ?? '').trim()
    let topicId: string | null = null
    if (from === sid && idSet.has(to)) {
      topicId = to
    } else if (to === sid && idSet.has(from)) {
      topicId = from
    } else {
      continue
    }
    const tn = findRawNodeInArtifact(art, topicId)
    if (!tn || String(tn.type) !== 'Topic') continue
    topicIds.push(topicId)
  }
  const seen = new Set<string>()
  const out: InsightRelatedTopicRow[] = []
  for (const topicId of topicIds) {
    if (seen.has(topicId)) continue
    seen.add(topicId)
    const n = findRawNodeInArtifact(art, topicId)
    const p = n?.properties as Record<string, unknown> | undefined
    const labelRaw =
      (typeof p?.label === 'string' && p.label.trim()) ||
      (typeof p?.name === 'string' && p.name.trim()) ||
      ''
    const label = labelRaw || topicId
    out.push({ topicId, label })
  }
  return out
}

/** Compact provenance line from GI artifact root + optional ``extraction`` (viewer-only; may be absent in strict schema files). */
export function insightProvenanceLine(art: ParsedArtifact | null): string | null {
  if (!art?.data) return null
  const d = art.data as Record<string, unknown>
  const parts: string[] = []
  const mv = d.model_version
  if (typeof mv === 'string' && mv.trim()) {
    parts.push(`model ${mv.trim()}`)
  }
  const pv = d.prompt_version
  if (typeof pv === 'string' && pv.trim()) {
    parts.push(`prompt ${pv.trim()}`)
  }
  const ex = d.extraction
  if (ex && typeof ex === 'object' && !Array.isArray(ex)) {
    const ext = (ex as Record<string, unknown>).extracted_at
    if (typeof ext === 'string' && ext.trim()) {
      parts.push(`extracted ${ext.trim()}`)
    }
  }
  const nm = art.name?.trim()
  if (nm) {
    parts.push(`from ${nm}`)
  }
  if (parts.length === 0) return null
  return parts.join(' · ')
}

function collectEdgeTypeKeys(art: ParsedArtifact): Record<string, boolean> {
  const allowedEdgeTypes: Record<string, boolean> = {}
  let rawNodes = Array.isArray(art.data.nodes) ? art.data.nodes.slice() : []
  let rawEdges = Array.isArray(art.data.edges) ? art.data.edges.map((e) => ({ ...e })) : []
  if (art.kind === 'gi' || art.kind === 'both') {
    const aug = ensureEpisodeToInsightEdges(rawNodes, rawEdges)
    rawNodes = aug.nodes
    rawEdges = aug.edges
  }
  const seenE = new Set<string>()
  for (const e of rawEdges) {
    if (!e || typeof e !== 'object') continue
    const k = e.type != null && String(e.type).trim() !== '' ? String(e.type) : '(unknown)'
    if (k === '_tc_cohesion') continue
    if (!seenE.has(k)) {
      seenE.add(k)
      allowedEdgeTypes[k] = true
    }
  }
  return allowedEdgeTypes
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
    allowedEdgeTypes: collectEdgeTypeKeys(art),
    hideUngroundedInsights: false,
    showGiLayer: true,
    showKgLayer: true,
  }
}

function pruneOrphanTopicClusterParents(nodes: RawGraphNode[]): RawGraphNode[] {
  const clusterIds = new Set(
    nodes
      .filter((n) => n?.type === 'TopicCluster' && n.id != null)
      .map((n) => String(n!.id)),
  )
  if (clusterIds.size === 0) {
    return nodes
  }

  const childCount = new Map<string, number>()
  for (const n of nodes) {
    const p = typeof n.parent === 'string' ? n.parent.trim() : ''
    if (p && clusterIds.has(p)) {
      childCount.set(p, (childCount.get(p) || 0) + 1)
    }
  }

  const keep = new Set<string>()
  for (const cid of clusterIds) {
    if ((childCount.get(cid) || 0) > 0) {
      keep.add(cid)
    }
  }

  let out = nodes.filter((n) => {
    if (n?.type !== 'TopicCluster') {
      return true
    }
    const id = String(n.id ?? '')
    return keep.has(id)
  })

  out = out.map((n) => {
    const p = typeof n.parent === 'string' ? n.parent.trim() : ''
    if (!p || keep.has(p)) {
      return n
    }
    const { parent: _removed, ...rest } = n
    return rest
  })

  return out
}

const GRAPH_TYPES_OFF_BY_DEFAULT = new Set(['Quote', 'Speaker'])

/** Graph tab: hide noisy node types on first paint (see docs/architecture/VIEWER_GRAPH_SPEC.md). */
export function applyGraphDefaultNodeTypeVisibility(state: GraphFilterState): void {
  const next: Record<string, boolean> = { ...state.allowedTypes }
  for (const k of Object.keys(next)) {
    next[k] = !GRAPH_TYPES_OFF_BY_DEFAULT.has(k)
  }
  state.allowedTypes = next
}

/** True when any per-type checkbox differs from graph default visibility. */
export function graphTypesDeviateFromGraphSpec(state: GraphFilterState | null): boolean {
  if (!state) return false
  for (const [k, on] of Object.entries(state.allowedTypes)) {
    const expected = !GRAPH_TYPES_OFF_BY_DEFAULT.has(k)
    if (Boolean(on) !== expected) return true
  }
  return false
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
    fullArt.kind === 'both' &&
    (state.showGiLayer === false || state.showKgLayer === false)
  ) {
    return true
  }
  const aet = state.allowedEdgeTypes
  if (aet && typeof aet === 'object') {
    for (const k of Object.keys(aet)) {
      if (aet[k] === false) {
        return true
      }
    }
  }
  return false
}

/** Like ``filtersActive`` but ignores per-node-type visibility toggles (used for graph “more filters” popover indicator). */
export function filtersActiveExcludingNodeTypes(
  fullArt: ParsedArtifact | null,
  state: GraphFilterState | null,
): boolean {
  if (!fullArt || !state) return false
  if (
    (fullArt.kind === 'gi' || fullArt.kind === 'both') &&
    state.hideUngroundedInsights
  ) {
    return true
  }
  if (
    fullArt.kind === 'both' &&
    (state.showGiLayer === false || state.showKgLayer === false)
  ) {
    return true
  }
  const aet = state.allowedEdgeTypes
  if (aet && typeof aet === 'object') {
    for (const k of Object.keys(aet)) {
      if (aet[k] === false) {
        return true
      }
    }
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
  nodes = pruneOrphanTopicClusterParents(nodes)
  const ids = new Set<string>()
  for (const n of nodes) {
    if (n.id != null) ids.add(String(n.id))
  }
  const aet = state.allowedEdgeTypes || {}
  const edges = (fullArt.data.edges || []).filter((e) => {
    if (!e || !ids.has(String(e.from)) || !ids.has(String(e.to))) {
      return false
    }
    const et =
      e.type != null && String(e.type).trim() !== '' ? String(e.type) : '(unknown)'
    if (et === '_tc_cohesion') return true
    return aet[et] !== false
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
    sourceCorpusRelPath: fullArt.sourceCorpusRelPath,
    sourceCorpusRelPathByEpisodeId: fullArt.sourceCorpusRelPathByEpisodeId,
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
    sourceCorpusRelPath: art.sourceCorpusRelPath,
    sourceCorpusRelPathByEpisodeId: art.sourceCorpusRelPathByEpisodeId,
  }
}

/**
 * Subgraph for the neighborhood minimap when a **TopicCluster** compound is selected: the compound
 * node, all member Topic nodes, every node adjacent to any member (one hop from the cluster), and
 * edges between kept nodes. Matches treating the cluster as one collapsed logical unit.
 */
export function filterArtifactEgoAroundTopicCluster(
  art: ParsedArtifact,
  compoundId: string,
  memberGraphIds: string[],
): ParsedArtifact {
  const cid = compoundId.trim()
  const members = memberGraphIds.map((x) => String(x).trim()).filter(Boolean)
  const nodes = Array.isArray(art.data.nodes) ? art.data.nodes : []
  const edges = Array.isArray(art.data.edges) ? art.data.edges : []
  const idSet = new Set<string>()
  for (const n of nodes) {
    if (n?.id != null) {
      idSet.add(String(n.id))
    }
  }
  const memberSet = new Set(members)
  if (!memberSet.size && !idSet.has(cid)) {
    return art
  }

  const keep = new Set<string>()
  if (idSet.has(cid)) {
    keep.add(cid)
  }
  for (const m of members) {
    if (idSet.has(m)) {
      keep.add(m)
    }
  }
  for (const e of edges) {
    if (!e) {
      continue
    }
    const fr = String(e.from)
    const to = String(e.to)
    if (memberSet.has(fr) && idSet.has(to)) {
      keep.add(to)
    }
    if (memberSet.has(to) && idSet.has(fr)) {
      keep.add(fr)
    }
    if (fr === cid && idSet.has(to)) {
      keep.add(to)
    }
    if (to === cid && idSet.has(fr)) {
      keep.add(fr)
    }
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
    sourceCorpusRelPath: art.sourceCorpusRelPath,
    sourceCorpusRelPathByEpisodeId: art.sourceCorpusRelPathByEpisodeId,
  }
}

/**
 * Union two graph views (dedupe nodes by id, edges by from+to+type). Used to merge ego slices.
 */
export function unionParsedArtifacts(a: ParsedArtifact, b: ParsedArtifact): ParsedArtifact {
  const nodeById = new Map<string, RawGraphNode>()
  for (const n of [...(a.data.nodes || []), ...(b.data.nodes || [])]) {
    if (n && n.id != null) {
      nodeById.set(String(n.id), n)
    }
  }
  const nodes = Array.from(nodeById.values())
  const edgeSeen = new Set<string>()
  const edgesOut: RawGraphEdge[] = []
  for (const e of [...(a.data.edges || []), ...(b.data.edges || [])]) {
    if (!e || e.from == null || e.to == null) {
      continue
    }
    const k = `${String(e.from)}\0${String(e.to)}\0${String(e.type || '')}`
    if (edgeSeen.has(k)) {
      continue
    }
    edgeSeen.add(k)
    edgesOut.push(e)
  }
  const nodeTypes: Record<string, number> = {}
  for (const n of nodes) {
    const t = n.type || '?'
    nodeTypes[t] = (nodeTypes[t] || 0) + 1
  }
  return {
    name: a.name,
    kind: a.kind,
    episodeId: a.episodeId,
    nodes: nodes.length,
    edges: edgesOut.length,
    nodeTypes,
    data: { ...a.data, nodes, edges: edgesOut },
    sourceCorpusRelPath: a.sourceCorpusRelPath,
    sourceCorpusRelPathByEpisodeId: a.sourceCorpusRelPathByEpisodeId,
  }
}

export function toGraphElements(art: ParsedArtifact): {
  visNodes: {
    id: string
    label: string
    group: string
    title: string
    parent?: string
  }[]
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
  const nodesById = new Map<string, RawGraphNode>()
  for (const n of rawNodes) {
    if (n && n.id != null) {
      nodesById.set(String(n.id), n)
    }
  }
  const visNodes = rawNodes.map((n, i) => {
    if (!n || typeof n !== 'object') {
      return { id: `n${i}`, label: '?', group: '?', title: '' }
    }
    const id = n.id != null ? String(n.id) : `n${i}`
    const parentId =
      typeof n.parent === 'string' && n.parent.trim() ? n.parent.trim() : undefined
    let label = nodeLabel(n)
    if (parentId && String(n.type) === 'Topic') {
      const par = nodesById.get(parentId)
      if (par && String(par.type) === 'TopicCluster' && label === nodeLabel(par)) {
        label = ''
      }
    }
    return {
      id,
      label,
      group: visualGroupForNode(n),
      title: buildNodeTitle(n),
      ...(parentId ? { parent: parentId } : {}),
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
      label: canonicalArtifactEdgeType(e.type ? String(e.type) : undefined),
    }
  })
  return { visNodes, visEdges, idSet }
}

/** Cytoscape element list (nodes + edges). */
export function toCytoElements(art: ParsedArtifact): import('cytoscape').ElementDefinition[] {
  const g = toGraphElements(art)
  const nodesById = new Map<string, RawGraphNode>()
  for (const n of art.data.nodes ?? []) {
    if (n && n.id != null) {
      nodesById.set(String(n.id), n)
    }
  }
  const nodes: import('cytoscape').ElementDefinition[] = g.visNodes.map((n) => {
    const raw = nodesById.get(n.id)
    const publishMs = publishTimeMsForRawNode(raw, nodesById)
    const data: Record<string, unknown> = {
      id: n.id,
      label: n.label,
      shortLabel: graphShortLabelFromDisplayLabel(n.label),
      type: n.group,
      recencyWeight: recencyWeightFromPublishMs(publishMs),
    }
    if (n.parent) {
      data.parent = n.parent
    }
    if (String(raw?.type) === 'Insight') {
      data.confidenceOpacity = confidenceOpacityFromInsightProperties(raw?.properties)
    }
    return { data }
  })
  const edges: import('cytoscape').ElementDefinition[] = g.visEdges
    .filter((e) => g.idSet.has(e.from) && g.idSet.has(e.to))
    .map((e) => ({
      data: {
        id: e.id,
        source: e.from,
        target: e.to,
        label: e.label,
        edgeType: e.label || '(unknown)',
      },
    }))
  return [...nodes, ...edges]
}
