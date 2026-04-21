/**
 * Graph initial load: filter artifact list rows by graph time lens, score episodes, then cap.
 */
import type { TopicClustersDocument } from '../api/corpusTopicClustersApi'
import { formatLocalYmd } from './localCalendarDate'
import type { ParsedArtifact } from '../types/artifact'

/** Tunable: max episodes merged on corpus graph auto-load (ceiling on episodes, not nodes). */
export const GRAPH_DEFAULT_EPISODE_CAP = 15

/** Tunable: recency component for the oldest episode in a dated lens window (linear floor). */
export const GRAPH_SCORE_RECENCY_MIN = 0.2
/** Tunable: recency component for the newest episode in the active decay window. */
export const GRAPH_SCORE_RECENCY_MAX = 1.0
/** Tunable: additive score when the episode appears in ≥1 topic cluster (cross-episode connectivity). */
export const GRAPH_SCORE_TOPIC_CLUSTER_BONUS = 0.4
/**
 * Tunable: for **all time** graph lens, recency decays linearly over this many calendar days
 * ending at the newest corpus publish date; older episodes receive {@link GRAPH_SCORE_RECENCY_MIN}.
 */
export const GRAPH_SCORE_ALL_TIME_DECAY_DAYS = 90
/**
 * Tunable (future): max weight for GI Insight density once `/api/corpus/coverage` exposes per-episode counts.
 * Not applied until then; documented in UXS-001.
 */
export const GRAPH_SCORE_GI_DENSITY_MAX = 0.4

export type ArtifactListRowForGraph = {
  relative_path: string
  kind: string
  publish_date: string
}

const YMD_RE = /^\d{4}-\d{2}-\d{2}$/

const MS_PER_DAY = 86400000

export function isValidPublishYmd(s: string): boolean {
  return YMD_RE.test(s.trim())
}

/** Corpus-relative stem shared by sibling ``*.gi.json`` / ``*.kg.json`` / ``*.bridge.json``. */
export function episodeStemFromArtifactRelPath(rel: string): string {
  const n = rel.replace(/\\/g, '/').trim()
  if (n.endsWith('.gi.json')) return n.slice(0, -'.gi.json'.length)
  if (n.endsWith('.kg.json')) return n.slice(0, -'.kg.json'.length)
  if (n.endsWith('.bridge.json')) return n.slice(0, -'.bridge.json'.length)
  return n
}

function sortKeyForYmd(ymd: string): string {
  return isValidPublishYmd(ymd) ? ymd : '0000-00-00'
}

function publishDayMs(ymd: string): number {
  const t = ymd.trim().slice(0, 10)
  if (!YMD_RE.test(t)) {
    return NaN
  }
  return Date.parse(`${t}T12:00:00`)
}

/**
 * Every ``episode_id`` listed on a topic-cluster member (``topic_clusters.json``), for graph-load scoring.
 */
export function episodeIdsInTopicClustersForGraphScoring(
  doc: TopicClustersDocument | null | undefined,
): Set<string> {
  const out = new Set<string>()
  const clusters = doc?.clusters
  if (!Array.isArray(clusters)) {
    return out
  }
  for (const cl of clusters) {
    const members = cl?.members
    if (!Array.isArray(members)) {
      continue
    }
    for (const m of members) {
      const eps = m?.episode_ids
      if (!Array.isArray(eps)) {
        continue
      }
      for (const e of eps) {
        const id = typeof e === 'string' ? e.trim() : ''
        if (id) {
          out.add(id)
        }
      }
    }
  }
  return out
}

/**
 * Whether ``stem`` (GI/KG/bridge path stem) likely matches a cluster ``episode_id``.
 * Conservative: full stem, ``…/id`` suffix, or basename equals id (covers common corpus layouts).
 */
export function stemMatchesTopicClusterEpisodeId(stem: string, clusterEpisodeIds: Set<string>): boolean {
  const n = stem.replace(/\\/g, '/').trim()
  if (!n || clusterEpisodeIds.size === 0) {
    return false
  }
  for (const raw of clusterEpisodeIds) {
    const e = raw.trim()
    if (!e) {
      continue
    }
    if (n === e || n.endsWith(`/${e}`)) {
      return true
    }
    const slash = n.lastIndexOf('/')
    const base = slash >= 0 ? n.slice(slash + 1) : n
    if (base === e) {
      return true
    }
  }
  return false
}

function recencyWeightLinear(
  publishMs: number,
  poolMinMs: number,
  poolMaxMs: number,
  lensMode: 'dated' | 'all_time',
): number {
  if (!Number.isFinite(publishMs)) {
    return GRAPH_SCORE_RECENCY_MIN
  }
  if (!Number.isFinite(poolMinMs) || !Number.isFinite(poolMaxMs)) {
    return GRAPH_SCORE_RECENCY_MAX
  }
  const lo = GRAPH_SCORE_RECENCY_MIN
  const hi = GRAPH_SCORE_RECENCY_MAX
  const spanBase = poolMaxMs - poolMinMs

  if (lensMode === 'dated') {
    if (spanBase <= 0) {
      return hi
    }
    const clamped = Math.min(Math.max(publishMs, poolMinMs), poolMaxMs)
    const t = (clamped - poolMinMs) / spanBase
    return lo + (hi - lo) * t
  }

  // All time: decay over GRAPH_SCORE_ALL_TIME_DECAY_DAYS ending at poolMaxMs
  const cutoff = poolMaxMs - GRAPH_SCORE_ALL_TIME_DECAY_DAYS * MS_PER_DAY
  if (publishMs <= cutoff) {
    return lo
  }
  const span = poolMaxMs - cutoff
  if (span <= 0) {
    return hi
  }
  const t = (publishMs - cutoff) / span
  return lo + (hi - lo) * t
}

type StemBlock = { publishYmd: string; paths: Set<string>; publishMs: number }

/**
 * ``sinceYmd`` empty = all time (score entire corpus; recency uses a trailing ``GRAPH_SCORE_ALL_TIME_DECAY_DAYS``
 * window from the newest publish date). Non-empty = publish_date >= since (local YYYY-MM-DD).
 * Returns relative paths for GI, KG, and bridge rows belonging to selected episode stems.
 *
 * Selection: ``score = recency_weight + cluster_bonus`` (see UXS-001 tunables). Tie-break: newer publish,
 * then stem id ascending.
 */
export function selectRelPathsForGraphLoad(
  artifactRows: ArtifactListRowForGraph[],
  sinceYmd: string,
  cap: number,
  topicClustersDoc?: TopicClustersDocument | null,
): { selectedRelPaths: string[]; wasCapped: boolean; episodeCount: number } {
  const capN = Math.max(1, cap)
  const since = sinceYmd.trim()
  const datedWindow = since.length > 0
  const lensMode: 'dated' | 'all_time' = datedWindow ? 'dated' : 'all_time'
  const clusterIds = episodeIdsInTopicClustersForGraphScoring(topicClustersDoc ?? null)

  const byStem = new Map<string, StemBlock>()

  for (const row of artifactRows) {
    const k = row.kind
    if (k !== 'gi' && k !== 'kg' && k !== 'bridge') continue
    const rel = row.relative_path.replace(/\\/g, '/').trim()
    if (!rel) continue
    const stem = episodeStemFromArtifactRelPath(rel)
    if (!stem) continue
    const rawPd = String(row.publish_date || '').trim()
    const pd = isValidPublishYmd(rawPd) ? rawPd.slice(0, 10) : ''
    const block = byStem.get(stem) ?? { publishYmd: '', paths: new Set<string>(), publishMs: NaN }
    if (pd && (!block.publishYmd || pd > block.publishYmd)) {
      block.publishYmd = pd
      block.publishMs = publishDayMs(pd)
    }
    block.paths.add(rel)
    byStem.set(stem, block)
  }

  let stems = Array.from(byStem.entries())

  if (datedWindow) {
    stems = stems.filter(([, v]) => {
      if (!isValidPublishYmd(v.publishYmd)) return false
      return v.publishYmd >= since
    })
  }

  if (stems.length === 0) {
    return { selectedRelPaths: [], wasCapped: false, episodeCount: 0 }
  }

  const finiteMs = stems
    .map(([, v]) => v.publishMs)
    .filter((ms) => Number.isFinite(ms)) as number[]
  const poolMinMs = finiteMs.length ? Math.min(...finiteMs) : NaN
  const poolMaxMs = finiteMs.length ? Math.max(...finiteMs) : NaN

  type Scored = { stem: string; block: StemBlock; score: number }
  const scored: Scored[] = stems.map(([stem, block]) => {
    const rec = recencyWeightLinear(block.publishMs, poolMinMs, poolMaxMs, lensMode)
    const bonus = stemMatchesTopicClusterEpisodeId(stem, clusterIds)
      ? GRAPH_SCORE_TOPIC_CLUSTER_BONUS
      : 0
    return { stem, block, score: rec + bonus }
  })

  scored.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score
    }
    const ka = sortKeyForYmd(a.block.publishYmd)
    const kb = sortKeyForYmd(b.block.publishYmd)
    if (ka !== kb) {
      return kb.localeCompare(ka)
    }
    return a.stem.localeCompare(b.stem)
  })

  const inWindow = scored.length
  const chosen = scored.slice(0, capN)
  const wasCapped = inWindow > capN

  const selected = new Set<string>()
  for (const { block } of chosen) {
    for (const p of block.paths) selected.add(p)
  }

  return {
    selectedRelPaths: [...selected].sort((a, b) => a.localeCompare(b)),
    wasCapped,
    episodeCount: chosen.length,
  }
}

function publishMsFromEpisodeNode(art: ParsedArtifact): number | null {
  const nodes = Array.isArray(art.data.nodes) ? art.data.nodes : []
  for (const n of nodes) {
    if (!n || String(n.type) !== 'Episode') continue
    const raw = (n.properties as { publish_date?: unknown } | undefined)?.publish_date
    if (raw == null) continue
    const ms = Date.parse(String(raw).trim())
    if (Number.isFinite(ms)) return ms
  }
  return null
}

/** Calendar day for a parsed GI/KG artifact (local) with ingested fallback from file mtime. */
export function calendarPublishYmdFromParsedArtifact(
  art: ParsedArtifact,
  fileLastModifiedMs: number,
): string {
  const ms = publishMsFromEpisodeNode(art)
  if (ms != null) {
    return formatLocalYmd(new Date(ms))
  }
  return formatLocalYmd(new Date(fileLastModifiedMs))
}

export type LocalArtifactCandidate = {
  art: ParsedArtifact
  relKey: string
  publishYmd: string
  fileLastModifiedMs: number
}

type LocalStemBlock = { publishYmd: string; arts: ParsedArtifact[]; maxMtime: number; publishMs: number }

/**
 * After parsing local files, cap by graph lens (``sinceYmd`` empty = all) and ``cap`` episodes.
 * Uses the same scoring as {@link selectRelPathsForGraphLoad} when ``topicClustersDoc`` is provided.
 */
export function selectParsedArtifactsForGraphLoad(
  candidates: LocalArtifactCandidate[],
  sinceYmd: string,
  cap: number,
  topicClustersDoc?: TopicClustersDocument | null,
): { kept: ParsedArtifact[]; wasCapped: boolean; episodeCount: number } {
  const capN = Math.max(1, cap)
  const since = sinceYmd.trim()
  const datedWindow = since.length > 0
  const lensMode: 'dated' | 'all_time' = datedWindow ? 'dated' : 'all_time'
  const clusterIds = episodeIdsInTopicClustersForGraphScoring(topicClustersDoc ?? null)

  const byStem = new Map<string, LocalStemBlock>()

  for (const c of candidates) {
    const name = c.art.name.replace(/\\/g, '/')
    const stem = episodeStemFromArtifactRelPath(name) || name
    const block = byStem.get(stem) ?? {
      publishYmd: '',
      arts: [],
      maxMtime: 0,
      publishMs: NaN,
    }
    block.arts.push(c.art)
    block.maxMtime = Math.max(block.maxMtime, c.fileLastModifiedMs)
    const ymd = c.publishYmd
    if (isValidPublishYmd(ymd) && (!block.publishYmd || ymd > block.publishYmd)) {
      block.publishYmd = ymd
      block.publishMs = publishDayMs(ymd)
    }
    byStem.set(stem, block)
  }

  for (const [, v] of byStem) {
    if (!v.publishYmd) {
      v.publishYmd = formatLocalYmd(new Date(v.maxMtime))
      v.publishMs = publishDayMs(v.publishYmd)
    }
  }

  let stems = Array.from(byStem.entries())
  if (datedWindow) {
    stems = stems.filter(([, v]) => isValidPublishYmd(v.publishYmd) && v.publishYmd >= since)
  }

  if (stems.length === 0) {
    return { kept: [], wasCapped: false, episodeCount: 0 }
  }

  const finiteMs = stems
    .map(([, v]) => v.publishMs)
    .filter((ms) => Number.isFinite(ms)) as number[]
  const poolMinMs = finiteMs.length ? Math.min(...finiteMs) : NaN
  const poolMaxMs = finiteMs.length ? Math.max(...finiteMs) : NaN

  type Scored = { stem: string; block: LocalStemBlock; score: number }
  const scored: Scored[] = stems.map(([stem, block]) => {
    const rec = recencyWeightLinear(block.publishMs, poolMinMs, poolMaxMs, lensMode)
    const bonus = stemMatchesTopicClusterEpisodeId(stem, clusterIds) ? GRAPH_SCORE_TOPIC_CLUSTER_BONUS : 0
    return { stem, block, score: rec + bonus }
  })

  scored.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score
    }
    const ka = sortKeyForYmd(a.block.publishYmd)
    const kb = sortKeyForYmd(b.block.publishYmd)
    if (ka !== kb) {
      return kb.localeCompare(ka)
    }
    return a.stem.localeCompare(b.stem)
  })

  const inWindow = scored.length
  const chosen = scored.slice(0, capN)
  const wasCapped = inWindow > capN
  const kept: ParsedArtifact[] = []
  for (const { block } of chosen) {
    kept.push(...block.arts)
  }
  return { kept, wasCapped, episodeCount: chosen.length }
}
