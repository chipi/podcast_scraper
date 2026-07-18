/**
 * graph-v3 Tier 5C/5D — enricher-driven graph lens overlays.
 *
 * Each apply function takes the Cytoscape core + the corpus-scope
 * enrichment envelope (already fetched + cached by GraphCanvas via
 * `fetchCachedCorpusEnvelope`) and paints classes / adds virtual edges.
 * Every function is a no-op when the envelope is null so callers can
 * hand it the fetch promise's resolved value without conditional
 * branches at the call site.
 *
 * Kept in a plain util (out of GraphCanvas.vue's 4400-line component)
 * so each lens's data-model contract is easy to inspect + test in
 * isolation.
 */

import type { Core, EdgeSingular, NodeSingular } from 'cytoscape'
import { trendDirection } from './trend'

/** ---------------------------------------------------------------- */
/** Class names + edge type tags — kept in constants so the           */
/** stylesheet + the apply functions share one source of truth.       */
/** ---------------------------------------------------------------- */
export const VELOCITY_CLASSES = ['velocity-up', 'velocity-down', 'velocity-steady'] as const
export const CREDIBILITY_CLASSES = ['credibility-high', 'credibility-med', 'credibility-low'] as const
export const CONSENSUS_EDGE_CLASS = 'lens-consensus-edge'
export const COGUEST_EDGE_CLASS = 'lens-coguest-edge'
export const CONSENSUS_EDGE_TYPE = '_lens_consensus'
export const COGUEST_EDGE_TYPE = '_lens_coguest'

/** ---------------------------------------------------------------- */
/** Envelope shapes — narrow subsets so callers can type payloads.   */
/** ---------------------------------------------------------------- */

export interface VelocityTopicRow {
  topic_id?: string
  velocity_last_over_6mo?: number
}
export interface VelocityEnvelopeData {
  topics?: VelocityTopicRow[]
}

export interface GroundingPersonRow {
  person_id?: string
  rate?: number
}
export interface GroundingEnvelopeData {
  persons?: GroundingPersonRow[]
}

export interface CoGuestPairRow {
  person_a_id?: string
  person_b_id?: string
  episode_count?: number
}
export interface CoGuestCommunityRow {
  community_id?: string
  community_label?: string
  member_ids?: string[]
  member_count?: number
}
export interface CoGuestEnvelopeData {
  pairs?: CoGuestPairRow[]
  /* graph-v3 tier 7-4 — enricher v1.1.0+ optional field. */
  communities?: CoGuestCommunityRow[]
}

export interface ConsensusRow {
  topic_id?: string
  person_a_id?: string
  person_b_id?: string
}
export interface ConsensusEnvelopeData {
  consensus?: ConsensusRow[]
}

/** ---------------------------------------------------------------- */
/** Helper: derive the "bare" node id suffix (strip `g:` / `k:`).    */
/** Same rule as stripLayerPrefixesForCil but inline to avoid a      */
/** circular utils import + keep this file self-contained.           */
/** ---------------------------------------------------------------- */
function bareSuffix(id: string): string {
  if (!id) return ''
  const parts = id.split(':')
  // First segment is the layer tag when it's `g` or `k`; otherwise it's
  // part of the real id (`topic:x`, `person:y`).
  if (parts.length > 1 && (parts[0] === 'g' || parts[0] === 'k')) {
    return parts.slice(1).join(':')
  }
  return id
}

function findNodeByBareId(core: Core, bareId: string): NodeSingular | null {
  const wanted = bareId.trim()
  if (!wanted) return null
  const direct = core.$id(wanted)
  if (!direct.empty()) return direct as NodeSingular
  const prefixed = core.$id(`g:${wanted}`)
  if (!prefixed.empty()) return prefixed as NodeSingular
  const kg = core.$id(`k:${wanted}`)
  if (!kg.empty()) return kg as NodeSingular
  // Slow-path fallback: scan by suffix match (handles `person:...` bare ids
  // where the graph carries a longer merged variant).
  let found: NodeSingular | null = null
  core.nodes().forEach((n) => {
    if (found) return
    if (bareSuffix(String(n.id())) === wanted) found = n as NodeSingular
  })
  return found
}

/** ---------------------------------------------------------------- */
/** Tier 5C-1 — Velocity halo (temporal_velocity enricher).          */
/** ---------------------------------------------------------------- */

export function applyVelocityHalo(
  core: Core,
  envelope: VelocityEnvelopeData | null | undefined,
): void {
  clearVelocityHalo(core)
  if (!envelope?.topics?.length) return
  core.batch(() => {
    for (const row of envelope.topics ?? []) {
      const tid = typeof row?.topic_id === 'string' ? row.topic_id.trim() : ''
      if (!tid) continue
      const v = typeof row?.velocity_last_over_6mo === 'number' ? row.velocity_last_over_6mo : NaN
      if (!Number.isFinite(v)) continue
      const node = findNodeByBareId(core, tid)
      if (!node || node.empty()) continue
      const dir = trendDirection(v)
      const cls =
        dir === 'up' ? 'velocity-up' : dir === 'down' ? 'velocity-down' : 'velocity-steady'
      node.addClass(cls)
    }
  })
}

export function clearVelocityHalo(core: Core): void {
  core.batch(() => {
    for (const c of VELOCITY_CLASSES) core.nodes().removeClass(c)
  })
}

/** ---------------------------------------------------------------- */
/** Tier 5C-2 — Person credibility border (grounding_rate enricher). */
/** ---------------------------------------------------------------- */

/** Rate ≥ 0.7 → high (solid green); ≥ 0.4 → med (solid amber); < 0.4
 *  → low (dashed red). Absent rate → no class (default node border). */
function credibilityClass(rate: number): (typeof CREDIBILITY_CLASSES)[number] | null {
  if (!Number.isFinite(rate)) return null
  if (rate >= 0.7) return 'credibility-high'
  if (rate >= 0.4) return 'credibility-med'
  return 'credibility-low'
}

export function applyCredibilityBorder(
  core: Core,
  envelope: GroundingEnvelopeData | null | undefined,
): void {
  clearCredibilityBorder(core)
  if (!envelope?.persons?.length) return
  core.batch(() => {
    for (const row of envelope.persons ?? []) {
      const pid = typeof row?.person_id === 'string' ? row.person_id.trim() : ''
      const rate = typeof row?.rate === 'number' ? row.rate : NaN
      const cls = credibilityClass(rate)
      if (!pid || !cls) continue
      const node = findNodeByBareId(core, pid)
      if (!node || node.empty()) continue
      node.addClass(cls)
    }
  })
}

export function clearCredibilityBorder(core: Core): void {
  core.batch(() => {
    for (const c of CREDIBILITY_CLASSES) core.nodes().removeClass(c)
  })
}

/** ---------------------------------------------------------------- */
/** Tier 5D-1 — Consensus edges (topic_consensus enricher).          */
/** Adds a virtual Person↔Person edge per consensus row. Deduped by  */
/** unordered pair + topic so a pair agreeing on the same topic in   */
/** multiple insights only paints one edge.                           */
/** ---------------------------------------------------------------- */

export function applyConsensusEdges(
  core: Core,
  envelope: ConsensusEnvelopeData | null | undefined,
): void {
  clearConsensusEdges(core)
  if (!envelope?.consensus?.length) return
  const seen = new Set<string>()
  const toAdd: Array<{ id: string; source: string; target: string; topic: string }> = []
  for (const row of envelope.consensus ?? []) {
    const aRaw = typeof row?.person_a_id === 'string' ? row.person_a_id.trim() : ''
    const bRaw = typeof row?.person_b_id === 'string' ? row.person_b_id.trim() : ''
    const topic = typeof row?.topic_id === 'string' ? row.topic_id.trim() : ''
    if (!aRaw || !bRaw || !topic) continue
    const aNode = findNodeByBareId(core, aRaw)
    const bNode = findNodeByBareId(core, bRaw)
    if (!aNode || aNode.empty() || !bNode || bNode.empty()) continue
    const aId = aNode.id()
    const bId = bNode.id()
    if (aId === bId) continue
    const [lo, hi] = aId < bId ? [aId, bId] : [bId, aId]
    const dedupeKey = `${lo}|${hi}|${topic}`
    if (seen.has(dedupeKey)) continue
    seen.add(dedupeKey)
    toAdd.push({
      id: `${CONSENSUS_EDGE_TYPE}::${lo}::${hi}::${topic}`,
      source: lo,
      target: hi,
      topic,
    })
  }
  core.batch(() => {
    for (const e of toAdd) {
      core.add({
        group: 'edges',
        data: {
          id: e.id,
          source: e.source,
          target: e.target,
          edgeType: CONSENSUS_EDGE_TYPE,
          topic: e.topic,
        },
        classes: CONSENSUS_EDGE_CLASS,
      })
    }
  })
}

export function clearConsensusEdges(core: Core): void {
  core.batch(() => {
    core
      .edges(`.${CONSENSUS_EDGE_CLASS}`)
      .forEach((e: EdgeSingular) => {
        core.remove(e)
      })
  })
}

/** ---------------------------------------------------------------- */
/** Tier 5D-2 — Co-guest edges (guest_coappearance enricher).        */
/** Adds a virtual Person↔Person edge per pair with episode_count    */
/** >= minEpisodeCount (default 2). Deduped by unordered pair.       */
/** ---------------------------------------------------------------- */

export function applyCoGuestEdges(
  core: Core,
  envelope: CoGuestEnvelopeData | null | undefined,
  minEpisodeCount = 2,
): void {
  clearCoGuestEdges(core)
  if (!envelope?.pairs?.length) return
  const seen = new Set<string>()
  const toAdd: Array<{ id: string; source: string; target: string; weight: number }> = []
  for (const row of envelope.pairs ?? []) {
    const count = typeof row?.episode_count === 'number' ? row.episode_count : 0
    if (count < minEpisodeCount) continue
    const aRaw = typeof row?.person_a_id === 'string' ? row.person_a_id.trim() : ''
    const bRaw = typeof row?.person_b_id === 'string' ? row.person_b_id.trim() : ''
    if (!aRaw || !bRaw) continue
    const aNode = findNodeByBareId(core, aRaw)
    const bNode = findNodeByBareId(core, bRaw)
    if (!aNode || aNode.empty() || !bNode || bNode.empty()) continue
    const aId = aNode.id()
    const bId = bNode.id()
    if (aId === bId) continue
    const [lo, hi] = aId < bId ? [aId, bId] : [bId, aId]
    const dedupeKey = `${lo}|${hi}`
    if (seen.has(dedupeKey)) continue
    seen.add(dedupeKey)
    toAdd.push({
      id: `${COGUEST_EDGE_TYPE}::${lo}::${hi}`,
      source: lo,
      target: hi,
      weight: count,
    })
  }
  core.batch(() => {
    for (const e of toAdd) {
      core.add({
        group: 'edges',
        data: {
          id: e.id,
          source: e.source,
          target: e.target,
          edgeType: COGUEST_EDGE_TYPE,
          weight: e.weight,
        },
        classes: COGUEST_EDGE_CLASS,
      })
    }
  })
}

export function clearCoGuestEdges(core: Core): void {
  core.batch(() => {
    core
      .edges(`.${COGUEST_EDGE_CLASS}`)
      .forEach((e: EdgeSingular) => {
        core.remove(e)
      })
  })
}

/** ---------------------------------------------------------------- */
/** Tier 7-4 — Person community underlay (guest_coappearance v1.1.0). */
/** Reads the enricher's `communities[]` and paints a soft underlay   */
/** tint on every Person node that belongs to a community. Palette    */
/** is the shared 8-hue theme palette (see themeRegionPalette.ts) so  */
/** person communities visually rhyme with theme regions without      */
/** colliding on the same axis (Person nodes are almost never the     */
/** same node as a Topic that owns a theme region).                   */
/** ---------------------------------------------------------------- */

export const PERSON_COMMUNITY_PALETTE_SIZE = 8
const PERSON_COMMUNITY_CLASSES: string[] = Array.from(
  { length: PERSON_COMMUNITY_PALETTE_SIZE },
  (_, i) => `person-region-${i}`,
)

function personCommunityHashIndex(cid: string): number {
  // djb2 hash (different from themeRegionIndex which uses multiplier-31).
  // Inlined to keep this file dep-free from the theme palette module.
  // Person-community palette slots are independent of theme-region palette slots.
  let h = 5381
  for (let i = 0; i < cid.length; i++) {
    h = ((h << 5) + h + cid.charCodeAt(i)) | 0
  }
  const idx = ((h % PERSON_COMMUNITY_PALETTE_SIZE) + PERSON_COMMUNITY_PALETTE_SIZE) %
    PERSON_COMMUNITY_PALETTE_SIZE
  return idx
}

export function applyPersonCommunityRegions(
  core: Core,
  envelope: CoGuestEnvelopeData | null | undefined,
): void {
  clearPersonCommunityRegions(core)
  if (!envelope?.communities?.length) return
  core.batch(() => {
    for (const c of envelope.communities ?? []) {
      const cid = typeof c?.community_id === 'string' ? c.community_id.trim() : ''
      if (!cid) continue
      const cls = `person-region-${personCommunityHashIndex(cid)}`
      for (const raw of c.member_ids ?? []) {
        const pid = typeof raw === 'string' ? raw.trim() : ''
        if (!pid) continue
        const node = findNodeByBareId(core, pid)
        if (!node || node.empty()) continue
        node.addClass(cls)
        node.data('personCommunityId', cid)
      }
    }
  })
}

export function clearPersonCommunityRegions(core: Core): void {
  core.batch(() => {
    for (const c of PERSON_COMMUNITY_CLASSES) core.nodes().removeClass(c)
    core.nodes().forEach((n) => {
      try {
        n.removeData('personCommunityId')
      } catch {
        /* removeData missing on some cy versions */
      }
    })
  })
}
