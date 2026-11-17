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
import { graphElementDecorators, graphNodeFilters, graphTypeTraits } from '../extensions'
import { humanizeSlug, shortPhrase, truncate } from './formatting'
import { logicalEpisodeIdFromGraphNodeId } from './graphEpisodeMetadata'
import { visualGroupForNode } from './visualGroup'

export { visualGroupForNode, visualNodeTypeCounts } from './visualGroup'

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
  const nodes = Array.isArray(data.nodes) ? data.nodes.slice() : []
  const edges = Array.isArray(data.edges) ? data.edges.slice() : []
  const episodeId =
    typeof data.episode_id === 'string' ? data.episode_id : null

  let kind: ArtifactKind = 'unknown'
  const lower = filename.toLowerCase()
  if (lower.endsWith('.kg.json')) {
    kind = 'kg'
  } else if (
    data.extraction &&
    typeof data.extraction === 'object' &&
    !Object.prototype.hasOwnProperty.call(data, 'prompt_version')
  ) {
    kind = 'kg'
  }

  const nodeTypes: Record<string, number> = {}
  for (const n of nodes) {
    const t = n && typeof n.type === 'string' ? n.type : '?'
    nodeTypes[t] = (nodeTypes[t] || 0) + 1
  }

  const dataOut = { ...data }

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
 * Like {@link findRawNodeInArtifact} but tolerates the GI/KG layer prefixes
 * ``g:`` / ``k:`` / ``kg:`` that appear after merge.
 *
 * Search hits and explore rollups carry ``speaker_id`` in the bare GI form
 * (e.g. ``person:ada``); after the GI/KG merge in ``mergeGiKg.ts`` the
 * Person / Entity node id may be prefixed (``g:person:ada`` /
 * ``k:kg:entity:person:ada`` / etc.). ``findRawNodeInArtifact`` is exact-match
 * only, so a focusPerson from search would silently miss otherwise-loaded
 * graph data. This helper falls back through the common prefix variants
 * before giving up.
 */
export function findRawNodeInArtifactByIdOrPrefixed(
  art: ParsedArtifact | null,
  nodeId: string | number,
): RawGraphNode | null {
  const exact = findRawNodeInArtifact(art, nodeId)
  if (exact) return exact
  if (!art?.data) return null
  const sid = String(nodeId).trim()
  if (!sid) return null
  const prefixCandidates = ['g:', 'k:', 'kg:', 'g:kg:', 'k:kg:']
  for (const p of prefixCandidates) {
    const hit = findRawNodeInArtifact(art, `${p}${sid}`)
    if (hit) return hit
  }
  // Reverse direction: graph node may be the bare form when the lookup
  // arrives prefixed (e.g. handoff carried ``g:person:ada``).
  const stripped = sid.replace(/^(?:g:|k:|kg:|g:kg:|k:kg:)/, '')
  if (stripped !== sid) {
    return findRawNodeInArtifact(art, stripped)
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

/**
 * #1076 chunk 2 — viewer-side near-duplicate Topic suppression.
 *
 * Real corpora frequently emit near-duplicate Topic names ("AI ethics",
 * "AI Ethics", "AI-ethics") which scatter the user across multiple
 * rows for what is one conceptual topic. Until full corpus-wide
 * clustering ships, the viewer dedupes on the
 * read side via simple normalization: lowercase, strip ASCII
 * punctuation, collapse whitespace. Groups produced by this key
 * surface under the canonical (= first-seen-after-sort) Topic id +
 * its name, so clicks through to Position Tracker resolve to a real
 * graph node.
 *
 * Returns the empty string for non-string input — callers should not
 * group on the empty string (that's a "no name" sentinel and would
 * fold every nameless Topic into one row).
 */
export function canonicalTopicKey(name: string | null | undefined): string {
  if (typeof name !== 'string') return ''
  return name
    .toLowerCase()
    .replace(/[\p{P}\p{S}]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * Ranked Topic mentions a Person discusses, via the ``ABOUT(Insight→Topic)``
 * chain whose Insight has a ``MENTIONS_PERSON(Insight→Person)`` edge to the
 * given person. Counts collapse to unique (insight, topic) pairs so the
 * same Insight pointing at the same Topic via two edges still scores 1.
 * Topics with no graph-attested name fall back to the id.
 *
 * #1076 chunk 2: Topic rows whose names normalize to the same canonical
 * key (lowercase, strip punctuation, collapse whitespace) are folded
 * into one row. The surviving row uses the alphabetically-smallest
 * topic id + its name; counts are summed across the duplicates.
 * Topics with no extractable name (callback returns empty string) are
 * NOT grouped — every nameless Topic gets its own row.
 *
 * Returns Top-N by count, then alphabetically.
 */
export interface PersonTopicMention {
  id: string
  name: string
  count: number
}
export function rankedPersonTopicMentions(
  art: ParsedArtifact | null,
  personGraphId: string | null,
  topN = 10,
): PersonTopicMention[] {
  if (!art?.data?.edges || personGraphId == null) return []
  const pid = String(personGraphId).trim()
  if (!pid) return []
  const mentioningInsights = new Set<string>()
  for (const e of art.data.edges) {
    if (!e) continue
    if (normalizeGiEdgeType(e.type) !== 'mentions_person') continue
    if (String(e.to ?? '').trim() !== pid) continue
    const from = String(e.from ?? '').trim()
    if (from) mentioningInsights.add(from)
  }
  if (mentioningInsights.size === 0) return []
  const counts = new Map<string, number>()
  const seenPair = new Set<string>()
  for (const e of art.data.edges) {
    if (!e) continue
    if (normalizeGiEdgeType(e.type) !== 'about') continue
    const from = String(e.from ?? '').trim()
    if (!mentioningInsights.has(from)) continue
    const to = String(e.to ?? '').trim()
    if (!to) continue
    const key = `${from}\0${to}`
    if (seenPair.has(key)) continue
    seenPair.add(key)
    counts.set(to, (counts.get(to) ?? 0) + 1)
  }
  const rows: PersonTopicMention[] = []
  for (const [id, count] of counts) {
    const n = findRawNodeInArtifact(art, id)
    const p = n?.properties as Record<string, unknown> | undefined
    const rawName = typeof p?.name === 'string' ? p.name.trim() : ''
    const label = typeof p?.label === 'string' ? p.label.trim() : ''
    rows.push({ id, name: rawName || label || id, count })
  }
  // #1076 chunk 2 — fold near-duplicates by canonical key (lowercase +
  // strip punctuation + collapse whitespace). Rows whose names produce
  // the same canonical key sum their counts under the alphabetically-
  // smallest topic id. Empty canonical key = no extractable name; those
  // rows pass through ungrouped.
  const grouped = new Map<string, PersonTopicMention>()
  const ungrouped: PersonTopicMention[] = []
  for (const row of rows) {
    const key = canonicalTopicKey(row.name)
    if (!key) {
      ungrouped.push(row)
      continue
    }
    const existing = grouped.get(key)
    if (!existing) {
      grouped.set(key, { ...row })
    } else {
      existing.count += row.count
      // Prefer the alphabetically-smallest topic id so navigation is
      // deterministic across runs and across viewer sessions.
      if (row.id < existing.id) {
        existing.id = row.id
        existing.name = row.name
      }
    }
  }
  const deduped = [...grouped.values(), ...ungrouped]
  deduped.sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
  return deduped.slice(0, topN)
}

/**
 * Organizations co-mentioned with a Person, via Insights that carry both a
 * ``MENTIONS_PERSON`` edge to the Person and a ``MENTIONS_ORG`` edge to the
 * Organization. Counts collapse to unique (insight, org) pairs.
 *
 * Returns Top-N by count, then alphabetically. Organization names fall back
 * to the id when the node has no name/label property.
 */
export interface PersonOrganizationMention {
  id: string
  name: string
  count: number
}
export function rankedPersonOrganizations(
  art: ParsedArtifact | null,
  personGraphId: string | null,
  topN = 10,
): PersonOrganizationMention[] {
  if (!art?.data?.edges || personGraphId == null) return []
  const pid = String(personGraphId).trim()
  if (!pid) return []
  const mentioningInsights = new Set<string>()
  for (const e of art.data.edges) {
    if (!e) continue
    if (normalizeGiEdgeType(e.type) !== 'mentions_person') continue
    if (String(e.to ?? '').trim() !== pid) continue
    const from = String(e.from ?? '').trim()
    if (from) mentioningInsights.add(from)
  }
  if (mentioningInsights.size === 0) return []
  const counts = new Map<string, number>()
  const seenPair = new Set<string>()
  for (const e of art.data.edges) {
    if (!e) continue
    if (normalizeGiEdgeType(e.type) !== 'mentions_org') continue
    const from = String(e.from ?? '').trim()
    if (!mentioningInsights.has(from)) continue
    const to = String(e.to ?? '').trim()
    if (!to) continue
    const key = `${from}\0${to}`
    if (seenPair.has(key)) continue
    seenPair.add(key)
    counts.set(to, (counts.get(to) ?? 0) + 1)
  }
  const rows: PersonOrganizationMention[] = []
  for (const [id, count] of counts) {
    const n = findRawNodeInArtifact(art, id)
    const p = n?.properties as Record<string, unknown> | undefined
    const rawName = typeof p?.name === 'string' ? p.name.trim() : ''
    const label = typeof p?.label === 'string' ? p.label.trim() : ''
    rows.push({ id, name: rawName || label || id, count })
  }
  rows.sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
  return rows.slice(0, topN)
}

/**
 * Episodes a Person appeared in, derived from `SPOKE_IN` out-edges
 * (Person → Episode) in the loaded graph slice. PRD-029 § "Episodes
 * appeared in" — list, not just a count.
 *
 * Each row carries the episode id, the display title (with fallbacks),
 * and the publish date (10-char YYYY-MM-DD slice). Sorted by
 * publish_date desc — newest first — with undated rows clustering at
 * the bottom. Duplicates collapsed.
 */
export interface PersonEpisodeAppearance {
  episodeId: string
  title: string | null
  publishDate: string | null
}
export function personEpisodeAppearances(
  art: ParsedArtifact | null,
  personGraphId: string | null,
): PersonEpisodeAppearance[] {
  if (!art?.data?.edges || personGraphId == null) return []
  const pid = String(personGraphId).trim()
  if (!pid) return []
  const epIds: string[] = []
  const seen = new Set<string>()
  for (const e of art.data.edges) {
    if (!e) continue
    if (normalizeGiEdgeType(e.type) !== 'spoke_in') continue
    if (String(e.from ?? '').trim() !== pid) continue
    const to = String(e.to ?? '').trim()
    if (!to || seen.has(to)) continue
    seen.add(to)
    epIds.push(to)
  }
  const rows: PersonEpisodeAppearance[] = []
  for (const epId of epIds) {
    const n = findRawNodeInArtifact(art, epId)
    const p = n?.properties as Record<string, unknown> | undefined
    const rawTitle = typeof p?.episode_title === 'string' ? p.episode_title.trim() : ''
    const altTitle = typeof p?.title === 'string' ? p.title.trim() : ''
    const title = rawTitle || altTitle || null
    const pdRaw = typeof p?.publish_date === 'string' ? p.publish_date.trim() : ''
    const publishDate = pdRaw ? pdRaw.slice(0, 10) : null
    rows.push({ episodeId: epId, title, publishDate })
  }
  rows.sort((a, b) => {
    // Desc by date; undated rows sink to the bottom.
    if (a.publishDate && b.publishDate) {
      if (a.publishDate > b.publishDate) return -1
      if (a.publishDate < b.publishDate) return 1
    } else if (a.publishDate) return -1
    else if (b.publishDate) return 1
    return a.episodeId < b.episodeId ? -1 : a.episodeId > b.episodeId ? 1 : 0
  })
  return rows
}

/**
 * Read Person.role from the node's ``properties.role`` when present and
 * normalize to lowercase trimmed text. Returns null for unknown / missing.
 * The role value is one of ``"host" | "guest" | "mention"`` by RFC-097
 * convention; callers should not assume any other vocabulary.
 */
export function personRoleFromNode(node: RawGraphNode | null): string | null {
  if (!node) return null
  const p = node.properties as Record<string, unknown> | undefined
  const role = p?.role
  if (typeof role !== 'string') return null
  const trimmed = role.trim().toLowerCase()
  return trimmed || null
}

export function normalizeGiEdgeType(type: string | undefined | null): string {
  return String(type ?? '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_')
}

function collectEdgeTypeKeys(art: ParsedArtifact): Record<string, boolean> {
  const allowedEdgeTypes: Record<string, boolean> = {}
  const rawEdges = Array.isArray(art.data.edges) ? art.data.edges.map((e) => ({ ...e })) : []
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

/** graph-v3 tier 6-3 — types whose filter defaults to OFF so the initial
 *  canvas isn't drowned by evidence / diarization plumbing. Users can
 *  re-enable via the Types chip. Deliberately conservative: only the
 *  highest-count evidence types default off; anything a viewer might
 *  reasonably expect to see on first load stays on. */
/** Off by default: the types a build declares in `graphTypeTraits.defaultHidden`. */
function typeHiddenByDefault(type: string): boolean {
  return graphTypeTraits.list().some((t) => t.defaultHidden?.includes(type))
}

/** A registered node filter with state of its own is set away from its default. */
function nodeFiltersActive(art: ParsedArtifact): boolean {
  return graphNodeFilters.list().some((f) => f.active?.(art) === true)
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
      allowedTypes[t] = !typeHiddenByDefault(t)
    }
  }
  return {
    allowedTypes,
    allowedEdgeTypes: collectEdgeTypeKeys(art),
    graphFeedFilterId: null,
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

/** Graph tab: hide noisy node types on first paint (see docs/architecture/VIEWER_GRAPH_SPEC.md). */
export function applyGraphDefaultNodeTypeVisibility(state: GraphFilterState): void {
  const next: Record<string, boolean> = { ...state.allowedTypes }
  for (const k of Object.keys(next)) {
    next[k] = !typeHiddenByDefault(k)
  }
  state.allowedTypes = next
}

/** True when any per-type checkbox differs from graph default visibility. */
export function graphTypesDeviateFromGraphSpec(state: GraphFilterState | null): boolean {
  if (!state) return false
  for (const [k, on] of Object.entries(state.allowedTypes)) {
    const expected = !typeHiddenByDefault(k)
    if (Boolean(on) !== expected) return true
  }
  return false
}

export function filtersActive(
  fullArt: ParsedArtifact | null,
  state: GraphFilterState | null,
): boolean {
  if (!fullArt || !state) return false
  /* Tier 6-3: Quote / Speaker default to hidden. A user who hasn't touched
   * anything shouldn't see the "filters active" indicator light up just
   * because these defaults are false — so only count a type as "active"
   * when it deviates from its default-visibility. */
  const keys = Object.keys(state.allowedTypes)
  for (const k of keys) {
    const defaultVisible = !typeHiddenByDefault(k)
    if (state.allowedTypes[k] !== defaultVisible) return true
  }
  const aet = state.allowedEdgeTypes
  if (aet && typeof aet === 'object') {
    for (const k of Object.keys(aet)) {
      if (aet[k] === false) {
        return true
      }
    }
  }
  if (state.graphFeedFilterId !== null) return true
  return nodeFiltersActive(fullArt)
}

/** Like ``filtersActive`` but ignores per-node-type visibility toggles (used for graph “more filters” popover indicator). */
export function filtersActiveExcludingNodeTypes(
  fullArt: ParsedArtifact | null,
  state: GraphFilterState | null,
): boolean {
  if (!fullArt || !state) return false
  const aet = state.allowedEdgeTypes
  if (aet && typeof aet === 'object') {
    for (const k of Object.keys(aet)) {
      if (aet[k] === false) {
        return true
      }
    }
  }
  if (state.graphFeedFilterId !== null) return true
  return nodeFiltersActive(fullArt)
}

export function applyGraphFilters(
  fullArt: ParsedArtifact,
  state: GraphFilterState,
): ParsedArtifact {
  let nodes = (fullArt.data.nodes || []).slice()
  const nodeFilters = graphNodeFilters.list()
  if (nodeFilters.length) {
    nodes = nodes.filter((n) => Boolean(n) && nodeFilters.every((f) => f.keep(n, state, fullArt)))
  }
  const { allowedTypes } = state
  nodes = nodes.filter((n) => {
    if (!n || typeof n !== 'object') return false
    const t = n.type || '?'
    return allowedTypes[t] !== false
  })
  // Feed filter (#658): prune Episode nodes that don't match the
  // selected ``properties.feed_id``, then prune any non-Episode node
  // that has no path (via the pre-feed-prune edge graph) to a surviving
  // Episode. BFS from kept Episodes marks the reachable set; everything
  // else drops so the result has no visually orphaned Insights /
  // Quotes / Topics floating after their only Episode is removed.
  if (state.graphFeedFilterId !== null && state.graphFeedFilterId !== '') {
    const wantFeed = state.graphFeedFilterId
    const isEpisodeNode = (n: RawGraphNode): boolean => {
      if (n.type === 'Episode') return true
      // Merged GI+KG views use ``__unified_ep__:`` synthetic Episode ids.
      const id = n.id != null ? String(n.id) : ''
      return id.startsWith('__unified_ep__:')
    }
    const keptEpisodeIds = new Set<string>()
    nodes = nodes.filter((n) => {
      if (!isEpisodeNode(n)) return true
      const props = (n.properties || {}) as Record<string, unknown>
      const f = typeof props.feed_id === 'string' ? props.feed_id.trim() : ''
      if (f === wantFeed) {
        if (n.id != null) keptEpisodeIds.add(String(n.id))
        return true
      }
      return false
    })
    // Build the undirected adjacency from the *original* edge list so
    // BFS sees the full graph topology (subsequent edge-prune happens
    // after node-prune below).
    const allEdges = (fullArt.data.edges || []) as RawGraphEdge[]
    const adj = new Map<string, Set<string>>()
    for (const e of allEdges) {
      if (!e) continue
      const a = e.from != null ? String(e.from) : ''
      const b = e.to != null ? String(e.to) : ''
      if (!a || !b) continue
      if (!adj.has(a)) adj.set(a, new Set())
      if (!adj.has(b)) adj.set(b, new Set())
      adj.get(a)!.add(b)
      adj.get(b)!.add(a)
    }
    const reachable = new Set<string>(keptEpisodeIds)
    const queue: string[] = Array.from(keptEpisodeIds)
    while (queue.length > 0) {
      const cur = queue.shift()!
      const neighbours = adj.get(cur)
      if (!neighbours) continue
      for (const nb of neighbours) {
        if (reachable.has(nb)) continue
        reachable.add(nb)
        queue.push(nb)
      }
    }
    nodes = nodes.filter((n) => {
      if (isEpisodeNode(n)) return true
      const id = n.id != null ? String(n.id) : ''
      if (!id) return true
      return reachable.has(id)
    })
  }
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
  const rawNodes = Array.isArray(d.nodes) ? d.nodes.slice() : []
  const rawEdges = Array.isArray(d.edges) ? d.edges.map((e) => ({ ...e })) : []
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
export function toCytoElements(
  art: ParsedArtifact,
  options?: {
    /** RFC-080 V1 — the aggregated-edges lens, passed to `graphElementDecorators.edges` so a
     *  decorator can add render-only aggregated edges. */
    enableAggregatedEdges?: boolean
  },
): import('cytoscape').ElementDefinition[] {
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
    // RFC-080 V3 — surface publishDate (Unix ms) on Episode nodes so
    // cyTimelineLayout can read `n.data('publishDate')` directly. Set
    // only when finite to keep the missing-date branch in the timeline
    // helper deterministic (typeof check there is `=== 'number'`).
    if (String(raw?.type) === 'Episode' && typeof publishMs === 'number' && Number.isFinite(publishMs)) {
      data.publishDate = publishMs
    }
    if (n.parent) {
      data.parent = n.parent
    }
    // Theme-cluster membership (co-occurrence "Theme") is a node decoration — a
    // teal ring — not a compound parent, so it coexists with the semantic boxes.
    // graph-v3 Tier 5A-2 bugfix: read from `raw` (the raw graph node populated
    // by applyStorylinesOverlay), not `n` (the visNode from toGraphElements
    // which only carries {id, label, group, title, parent}). Prior code read
    // from `n` and always missed → `.theme-member` class never actually fired.
    const classes: string[] = []
    const storylineId = (raw as { storylineId?: unknown } | undefined)?.storylineId
    if (typeof storylineId === 'string' && storylineId) {
      data.storylineId = storylineId
      classes.push('theme-member')
    }
    const elem: import('cytoscape').ElementDefinition = { data }
    if (raw) {
      for (const d of graphElementDecorators.list()) {
        const add = d.node?.(raw)
        if (!add) continue
        Object.assign(data, add.data ?? {})
        classes.push(...(add.classes ?? []))
      }
    }
    if (classes.length) elem.classes = classes.join(' ')
    return elem
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
  const drawn = (id: string) => g.idSet.has(id)
  const decoratorEdges = graphElementDecorators
    .list()
    .flatMap((d) => d.edges?.(art, drawn, { aggregated: Boolean(options?.enableAggregatedEdges) }) ?? [])
  return [...nodes, ...edges, ...decoratorEdges]
}

/**
 * Subgraph for the neighborhood minimap around a compound (group) node: the compound, its member
 * nodes, every node adjacent to any member (one hop from the group), and edges between kept
 * nodes, treating the group as one logical unit.
 */
/** A compound (group) node and its member node ids, for neighbourhood views. */
export interface CompoundNeighborhood {
  compoundId: string
  memberIds: string[]
  /** Minimap heading and aria label; defaults name a group. */
  label?: string
  ariaLabel?: string
}

export function filterArtifactEgoAroundCompound(
  art: ParsedArtifact,
  compoundId: string,
  memberIds: string[],
): ParsedArtifact {
  const cid = compoundId.trim()
  const members = memberIds.map((x) => String(x).trim()).filter(Boolean)
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
