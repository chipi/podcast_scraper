/**
 * The Brief's search results, shaped for what a listener remembers (operator 2026-10-10).
 *
 * Measured on 94 real episodes (docs/wip/BRIEF-SEARCH-EVAL-2026-10-10.md): one mixed list of ten
 * gave the transcript a third of the slots — insights and their own supporting quotes took the rest
 * — and showed a transcript hit as a ~300-word block. So:
 *
 * * Two groups (C): **In the transcript** — verbatim passages first, then quotes and passages — and
 *   **Insights**. The episode's own title / summary / description hits are left out: the Brief
 *   already shows them.
 * * The sentence, not the passage (D): a transcript result is cut to the segment that holds the
 *   remembered words (plus the next one when that is very short), timed from it, with the words
 *   marked — and saved as a highlight from exactly those segments.
 */
import type { SearchHit, Segment } from '../services/types'

/** Words a remembered query drops when it has others ("the worst week" → worst, week). */
const STOP = new Set(
  'a an and are as at be but by for from had has have he her his i if in is it its me my no not of on or our she so that the their them they this to too us was we were what when which who will with you your'.split(
    ' ',
  ),
)
const WORD = /[\p{L}\p{N}]+(?:'[\p{L}]+)?/gu
/** A matched segment this short gets the next one too, so the piece reads as a sentence. */
const SHORT_WORDS = 12
/** Words of context either side of the first remembered word, when no timed segment is known. */
const CONTEXT = 15

export interface BriefPiece {
  key: string
  text: string
  /** Seconds into the episode, or null when the result carries no time. */
  start: number | null
  /** The transcript segments the piece is made of (empty when cut from the result's own text). */
  segments: Segment[]
  /** The words exactly as typed (``phrase``), all of them in any order (``words``), or neither. */
  match: 'phrase' | 'words' | null
}

function words(text: string): string[] {
  return text.toLowerCase().match(WORD) ?? []
}

/** The remembered words to find and mark: content words, or every word when there are none. */
export function searchTerms(query: string): string[] {
  const all = words(query)
  const content = all.filter((w) => !STOP.has(w))
  return [...new Set(content.length ? content : all)]
}

function startMs(hit: SearchHit): number | null {
  const md = (hit.metadata ?? {}) as Record<string, unknown>
  const lifted = (hit.lifted as { quote?: Record<string, unknown> } | null | undefined)?.quote
  for (const v of [md.timestamp_start_ms, lifted?.timestamp_start_ms, md.start_ms]) {
    if (typeof v === 'number' && Number.isFinite(v)) return v
  }
  return null
}

function endMs(hit: SearchHit, start: number): number {
  const md = (hit.metadata ?? {}) as Record<string, unknown>
  const lifted = (hit.lifted as { quote?: Record<string, unknown> } | null | undefined)?.quote
  for (const v of [md.timestamp_end_ms, lifted?.timestamp_end_ms, md.end_ms]) {
    if (typeof v === 'number' && Number.isFinite(v) && v > start) return v
  }
  return start + 1
}

/** ~``CONTEXT`` words either side of the first remembered word in ``text`` (the whole when short). */
export function snippetAround(text: string, terms: string[]): string {
  const tokens = text.split(/\s+/).filter(Boolean)
  if (tokens.length <= CONTEXT * 2 + 1) return tokens.join(' ')
  const at = tokens.findIndex((t) => terms.some((term) => words(t).includes(term)))
  const centre = at < 0 ? 0 : at
  const lo = Math.max(0, centre - CONTEXT)
  const hi = Math.min(tokens.length, centre + CONTEXT + 1)
  return `${lo > 0 ? '… ' : ''}${tokens.slice(lo, hi).join(' ')}${hi < tokens.length ? ' …' : ''}`
}

function piece(hit: SearchHit, segments: Segment[], terms: string[]): BriefPiece {
  const md = (hit.metadata ?? {}) as Record<string, unknown>
  const match = md.match === 'phrase' || md.match === 'words' ? md.match : null
  const s = startMs(hit)
  if (s != null && segments.length) {
    const e = endMs(hit, s)
    const inRange = segments
      .map((seg, i) => ({ seg, i }))
      .filter(({ seg }) => seg.start * 1000 < e && seg.end * 1000 > s)
    if (inRange.length) {
      const hitAt = inRange.find(({ seg }) => terms.some((term) => words(seg.text).includes(term)))
      const first = hitAt ?? inRange[0]
      const segs = [first.seg]
      const next = segments[first.i + 1]
      if (next && words(first.seg.text).length < SHORT_WORDS) segs.push(next)
      return {
        key: `seg:${first.seg.id}`,
        text: segs.map((x) => x.text.trim()).join(' '),
        start: first.seg.start,
        segments: segs,
        match,
      }
    }
  }
  return {
    key: `hit:${hit.doc_id}`,
    text: snippetAround(hit.text ?? '', terms),
    start: s != null ? s / 1000 : null,
    segments: [],
    match,
  }
}

const TRANSCRIPT = new Set(['transcript', 'quote'])

/** The Brief's results as its two groups: transcript pieces (verbatim first, one per moment), insights. */
export function groupBriefResults(
  hits: SearchHit[],
  segments: Segment[],
  query: string,
): { transcript: BriefPiece[]; insights: SearchHit[] } {
  const terms = searchTerms(query)
  const seen = new Set<string>()
  const transcript: BriefPiece[] = []
  const insights: SearchHit[] = []
  for (const hit of hits) {
    const type = (hit.metadata as Record<string, unknown> | undefined)?.doc_type
    if (type === 'insight') {
      insights.push(hit)
    } else if (typeof type === 'string' && TRANSCRIPT.has(type)) {
      const p = piece(hit, segments, terms)
      if (!seen.has(p.key)) {
        seen.add(p.key)
        transcript.push(p)
      }
    }
  }
  // Verbatim first (the server already orders them so; a stable sort keeps that order).
  transcript.sort((a, b) => Number(b.match === 'phrase') - Number(a.match === 'phrase'))
  return { transcript, insights }
}

/** ``text`` split into runs, the remembered words marked (whole words, any case). */
export function markTerms(text: string, terms: string[]): { text: string; mark: boolean }[] {
  if (!terms.length) return [{ text, mark: false }]
  const out: { text: string; mark: boolean }[] = []
  let last = 0
  for (const m of text.matchAll(WORD)) {
    if (!terms.includes(m[0].toLowerCase())) continue
    const at = m.index ?? 0
    if (at > last) out.push({ text: text.slice(last, at), mark: false })
    out.push({ text: m[0], mark: true })
    last = at + m[0].length
  }
  if (last < text.length) out.push({ text: text.slice(last), mark: false })
  return out
}
