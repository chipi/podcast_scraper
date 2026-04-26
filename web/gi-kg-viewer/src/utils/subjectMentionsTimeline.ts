/**
 * #672 — Mentions timeline for a focused Topic / Entity / Person from the
 * merged GI/KG slice. Walks edges incident to the subject, hops one step
 * to the linked Insight / Quote nodes, then resolves each item's
 * ``properties.episode_id`` to its Episode node ``properties.publish_date``
 * to produce a YYYY-MM bucket.
 *
 * The merged graph carries no per-Insight timestamps, so dropping back to
 * the linked Episode is the only client-side path to a timeline without a
 * new API call.
 */
import type { ParsedArtifact, RawGraphNode } from '../types/artifact'
import {
  findRawNodeInArtifact,
  findRawNodeInArtifactByIdOrPrefixed,
  normalizeGiEdgeType,
} from './parsing'
import { logicalEpisodeIdFromGraphNodeId } from './graphEpisodeMetadata'

/** Edge types that link a subject node to an Insight or Quote item. */
const SUBJECT_INCIDENT_EDGE_TYPES = new Set([
  'about',
  'related_to',
  'mentions',
  'spoken_by',
])

export interface SubjectMentionsMonth {
  /** ``YYYY-MM`` (UTC). */
  ymd: string
  /** Insight + quote count for this month. */
  count: number
}

export interface SubjectMentionsTimeline {
  months: SubjectMentionsMonth[]
  /** Total dated mentions across all months. */
  total: number
  /** Mentions whose linked episode lacks a publish_date. */
  undated: number
  /** Distinct linked Episode ids covered. */
  episodeCount: number
  /** Distinct linked Insight ids. */
  insightIds: string[]
  /** Distinct linked Quote ids. */
  quoteIds: string[]
}

const EMPTY_TIMELINE: SubjectMentionsTimeline = {
  months: [],
  total: 0,
  undated: 0,
  episodeCount: 0,
  insightIds: [],
  quoteIds: [],
}

function ymdMonthFromMs(ms: number): string | null {
  if (!Number.isFinite(ms)) return null
  const d = new Date(ms)
  if (Number.isNaN(d.getTime())) return null
  const y = d.getUTCFullYear()
  const m = d.getUTCMonth() + 1
  return `${y}-${String(m).padStart(2, '0')}`
}

function publishMsForLinkedItem(
  item: RawGraphNode,
  episodesByLogicalId: Map<string, RawGraphNode>,
): number | null {
  const eid =
    typeof item.properties?.episode_id === 'string'
      ? item.properties.episode_id.trim()
      : ''
  if (!eid) return null
  const ep = episodesByLogicalId.get(eid)
  if (!ep) return null
  const pd = ep.properties?.publish_date
  if (typeof pd !== 'string') return null
  const parsed = Date.parse(pd.trim())
  return Number.isFinite(parsed) ? parsed : null
}

/**
 * Build a month-bucketed timeline of Insight + Quote mentions for a focused
 * subject node. Returns an empty timeline when no mentions resolve.
 */
export function buildSubjectMentionsTimeline(
  art: ParsedArtifact | null,
  subjectNodeId: string | null | undefined,
): SubjectMentionsTimeline {
  if (!art?.data || !subjectNodeId?.trim()) {
    return EMPTY_TIMELINE
  }
  const rawSid = subjectNodeId.trim()
  const subject = findRawNodeInArtifactByIdOrPrefixed(art, rawSid)
  if (!subject) {
    return EMPTY_TIMELINE
  }
  /** Edge endpoints carry the post-merge prefixed form; align ``sid`` with
   * the actual matched node id so the from/to comparisons below hit. */
  const sid = subject.id != null ? String(subject.id) : rawSid

  const nodes = Array.isArray(art.data.nodes) ? art.data.nodes : []
  const edges = Array.isArray(art.data.edges) ? art.data.edges : []

  const episodesByLogicalId = new Map<string, RawGraphNode>()
  for (const n of nodes) {
    if (!n || String(n.type) !== 'Episode') continue
    const lid = logicalEpisodeIdFromGraphNodeId(String(n.id ?? ''))
    if (lid) episodesByLogicalId.set(lid, n)
  }

  const insightIdSet = new Set<string>()
  const quoteIdSet = new Set<string>()
  const linkedEpisodes = new Set<string>()
  const monthCounts = new Map<string, number>()
  let total = 0
  let undated = 0

  for (const e of edges) {
    if (!e) continue
    const ty = normalizeGiEdgeType(e.type)
    if (!SUBJECT_INCIDENT_EDGE_TYPES.has(ty)) continue
    const from = String(e.from ?? '').trim()
    const to = String(e.to ?? '').trim()
    let otherId: string | null = null
    if (from === sid) otherId = to
    else if (to === sid) otherId = from
    else continue
    if (!otherId) continue
    const item = findRawNodeInArtifact(art, otherId)
    if (!item) continue
    const itemType = String(item.type)
    if (itemType !== 'Insight' && itemType !== 'Quote') continue
    if (itemType === 'Insight') insightIdSet.add(otherId)
    else quoteIdSet.add(otherId)
    const eid =
      typeof item.properties?.episode_id === 'string'
        ? item.properties.episode_id.trim()
        : ''
    if (eid) linkedEpisodes.add(eid)
    const ms = publishMsForLinkedItem(item, episodesByLogicalId)
    if (ms == null) {
      undated += 1
      continue
    }
    const month = ymdMonthFromMs(ms)
    if (!month) {
      undated += 1
      continue
    }
    monthCounts.set(month, (monthCounts.get(month) ?? 0) + 1)
    total += 1
  }

  const months: SubjectMentionsMonth[] = Array.from(monthCounts.entries())
    .map(([ymd, count]) => ({ ymd, count }))
    .sort((a, b) => (a.ymd < b.ymd ? -1 : a.ymd > b.ymd ? 1 : 0))

  return {
    months,
    total,
    undated,
    episodeCount: linkedEpisodes.size,
    insightIds: Array.from(insightIdSet),
    quoteIds: Array.from(quoteIdSet),
  }
}
