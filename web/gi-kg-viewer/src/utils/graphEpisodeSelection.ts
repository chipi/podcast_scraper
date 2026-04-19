/**
 * Graph initial load: filter artifact list rows by graph time lens and episode cap.
 */
import { formatLocalYmd } from './localCalendarDate'
import type { ParsedArtifact } from '../types/artifact'

export const GRAPH_DEFAULT_EPISODE_CAP = 15

export type ArtifactListRowForGraph = {
  relative_path: string
  kind: string
  publish_date: string
}

const YMD_RE = /^\d{4}-\d{2}-\d{2}$/

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

/**
 * ``sinceYmd`` empty = all time (most recent ``cap`` episodes). Non-empty = publish_date >= since (local YYYY-MM-DD).
 * Returns relative paths for GI, KG, and bridge rows belonging to selected episode stems.
 */
export function selectRelPathsForGraphLoad(
  artifactRows: ArtifactListRowForGraph[],
  sinceYmd: string,
  cap: number,
): { selectedRelPaths: string[]; wasCapped: boolean; episodeCount: number } {
  const capN = Math.max(1, cap)
  const since = sinceYmd.trim()
  const datedWindow = since.length > 0

  const byStem = new Map<
    string,
    { publishYmd: string; paths: Set<string> }
  >()

  for (const row of artifactRows) {
    const k = row.kind
    if (k !== 'gi' && k !== 'kg' && k !== 'bridge') continue
    const rel = row.relative_path.replace(/\\/g, '/').trim()
    if (!rel) continue
    const stem = episodeStemFromArtifactRelPath(rel)
    if (!stem) continue
    const rawPd = String(row.publish_date || '').trim()
    const pd = isValidPublishYmd(rawPd) ? rawPd.slice(0, 10) : ''
    const block = byStem.get(stem) ?? { publishYmd: '', paths: new Set<string>() }
    if (pd && (!block.publishYmd || pd > block.publishYmd)) {
      block.publishYmd = pd
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

  stems.sort((a, b) => {
    const ka = sortKeyForYmd(a[1].publishYmd)
    const kb = sortKeyForYmd(b[1].publishYmd)
    if (ka !== kb) return kb.localeCompare(ka)
    return a[0].localeCompare(b[0])
  })

  const inWindow = stems.length
  const chosen = stems.slice(0, capN)
  const wasCapped = inWindow > capN

  const selected = new Set<string>()
  for (const [, v] of chosen) {
    for (const p of v.paths) selected.add(p)
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

/**
 * After parsing local files, cap by graph lens (``sinceYmd`` empty = all) and ``cap`` episodes.
 * ``relKey`` groups GI/KG/bridge from the same basename (file name without suffix).
 */
export function selectParsedArtifactsForGraphLoad(
  candidates: LocalArtifactCandidate[],
  sinceYmd: string,
  cap: number,
): { kept: ParsedArtifact[]; wasCapped: boolean; episodeCount: number } {
  const capN = Math.max(1, cap)
  const since = sinceYmd.trim()
  const datedWindow = since.length > 0

  const byStem = new Map<
    string,
    { publishYmd: string; arts: ParsedArtifact[]; maxMtime: number }
  >()

  for (const c of candidates) {
    const name = c.art.name.replace(/\\/g, '/')
    const stem = episodeStemFromArtifactRelPath(name) || name
    const block = byStem.get(stem) ?? { publishYmd: '', arts: [], maxMtime: 0 }
    block.arts.push(c.art)
    block.maxMtime = Math.max(block.maxMtime, c.fileLastModifiedMs)
    const ymd = c.publishYmd
    if (isValidPublishYmd(ymd) && (!block.publishYmd || ymd > block.publishYmd)) {
      block.publishYmd = ymd
    }
    byStem.set(stem, block)
  }

  for (const [, v] of byStem) {
    if (!v.publishYmd) {
      v.publishYmd = formatLocalYmd(new Date(v.maxMtime))
    }
  }

  let stems = Array.from(byStem.entries())
  if (datedWindow) {
    stems = stems.filter(([, v]) => isValidPublishYmd(v.publishYmd) && v.publishYmd >= since)
  }
  stems.sort((a, b) => {
    const ka = sortKeyForYmd(a[1].publishYmd)
    const kb = sortKeyForYmd(b[1].publishYmd)
    if (ka !== kb) return kb.localeCompare(ka)
    return a[0].localeCompare(b[0])
  })

  const inWindow = stems.length
  const chosen = stems.slice(0, capN)
  const wasCapped = inWindow > capN
  const kept: ParsedArtifact[] = []
  for (const [, v] of chosen) {
    kept.push(...v.arts)
  }
  return { kept, wasCapped, episodeCount: chosen.length }
}
