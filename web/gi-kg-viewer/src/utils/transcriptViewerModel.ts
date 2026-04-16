/**
 * Transcript in-app viewer: size guard, highlight range, optional Whisper segments sidecar.
 *
 * Gate 0 (2026-04): sampled transcript `.txt` files under `.test_outputs` are ~10–50 KiB;
 * default cap leaves headroom for long podcasts without Range requests yet.
 */
export const DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES = 5 * 1024 * 1024

export type TranscriptSegment = {
  startSec: number
  endSec: number
  text: string
}

function toFiniteNumber(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) {
    return v
  }
  if (typeof v === 'string' && v.trim()) {
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }
  return null
}

/**
 * Path to `{base}.segments.json` next to the transcript file (pipeline convention).
 * Handles `.cleaned.txt` → `.cleaned.segments.json`.
 */
export function segmentsSidecarRelpathFromTranscriptRelpath(transcriptRelpath: string): string {
  const t = transcriptRelpath
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\/+/, '')
    .replace(/\/+/g, '/')
  if (!t) {
    return '.segments.json'
  }
  const lastDot = t.lastIndexOf('.')
  if (lastDot <= 0) {
    return `${t}.segments.json`
  }
  const base = t.slice(0, lastDot)
  return `${base}.segments.json`
}

/**
 * Parse Whisper-style segment list (seconds). Returns null if not a non-empty array.
 */
export function parseTranscriptSegmentsJson(raw: unknown): TranscriptSegment[] | null {
  if (!Array.isArray(raw) || raw.length === 0) {
    return null
  }
  const out: TranscriptSegment[] = []
  for (const row of raw) {
    if (row == null || typeof row !== 'object') {
      continue
    }
    const o = row as Record<string, unknown>
    const start = toFiniteNumber(o.start ?? o.start_time)
    const end = toFiniteNumber(o.end ?? o.end_time)
    const text = typeof o.text === 'string' ? o.text : ''
    if (start == null && end == null) {
      continue
    }
    out.push({
      startSec: start ?? end ?? 0,
      endSec: end ?? start ?? 0,
      text,
    })
  }
  return out.length > 0 ? out : null
}

export function transcriptExceedsMaxBytes(
  contentLengthHeader: string | null,
  bodyByteLength: number,
  maxBytes: number,
): boolean {
  if (bodyByteLength > maxBytes) {
    return true
  }
  if (contentLengthHeader == null || !contentLengthHeader.trim()) {
    return false
  }
  const n = Number(contentLengthHeader.trim())
  return Number.isFinite(n) && n > maxBytes
}

export type TranscriptHighlightParts = {
  before: string
  highlight: string
  after: string
}

/** Interleaved plain / highlighted runs for multi-span GI quotes (e.g. one insight, many SUPPORTED_BY). */
export type TranscriptHighlightSegment =
  | { type: 'text'; text: string }
  | { type: 'mark'; text: string }

function clampGiCharSliceBounds(
  textLength: number,
  charStart: unknown,
  charEnd: unknown,
): { start: number; end: number } | null {
  if (textLength === 0) {
    return null
  }
  let start = toFiniteNumber(charStart)
  let end = toFiniteNumber(charEnd)
  if (start == null && end == null) {
    return null
  }
  if (start == null) {
    start = 0
  }
  if (end == null) {
    end = textLength
  }
  start = Math.max(0, Math.min(textLength, Math.floor(start)))
  end = Math.max(0, Math.min(textLength, Math.floor(end)))
  if (end < start) {
    const t = start
    start = end
    end = t
  }
  if (start === 0 && end === 0 && textLength > 0) {
    return null
  }
  return { start, end }
}

/**
 * Split full transcript for display with a highlighted GI quote span (character offsets).
 * Returns null if no usable range. Clamps to `text.length`; uses half-open [start, end).
 */
export function splitTranscriptAroundHighlight(
  text: string,
  charStart: unknown,
  charEnd: unknown,
): TranscriptHighlightParts | null {
  const len = text.length
  const b = clampGiCharSliceBounds(len, charStart, charEnd)
  if (!b) {
    return null
  }
  const { start, end } = b
  return {
    before: text.slice(0, start),
    highlight: text.slice(start, end),
    after: text.slice(end),
  }
}

/**
 * Build body segments with merged, non-overlapping ``<mark>`` runs (sorted by start).
 * Skips empty spans. Returns null if nothing to highlight.
 */
export function buildTranscriptHighlightSegments(
  text: string,
  ranges: ReadonlyArray<{ charStart: unknown; charEnd: unknown }>,
): TranscriptHighlightSegment[] | null {
  const len = text.length
  if (len === 0 || ranges.length === 0) {
    return null
  }
  const intervals: { start: number; end: number }[] = []
  for (const r of ranges) {
    const b = clampGiCharSliceBounds(len, r.charStart, r.charEnd)
    if (b != null && b.start < b.end) {
      intervals.push(b)
    }
  }
  if (intervals.length === 0) {
    return null
  }
  intervals.sort((a, b) => a.start - b.start || a.end - b.end)
  const merged: { start: number; end: number }[] = []
  for (const iv of intervals) {
    const prev = merged[merged.length - 1]
    if (!prev || iv.start > prev.end) {
      merged.push({ start: iv.start, end: iv.end })
    } else {
      prev.end = Math.max(prev.end, iv.end)
    }
  }
  const segments: TranscriptHighlightSegment[] = []
  let c = 0
  for (const { start, end } of merged) {
    if (start > c) {
      segments.push({ type: 'text', text: text.slice(c, start) })
    }
    const hl = text.slice(start, end)
    if (hl.length > 0) {
      segments.push({ type: 'mark', text: hl })
    }
    c = end
  }
  if (c < len) {
    segments.push({ type: 'text', text: text.slice(c) })
  }
  return segments.some((s) => s.type === 'mark') ? segments : null
}

export function formatSegmentTimeRange(s: TranscriptSegment): string {
  const a = Number.isFinite(s.startSec) ? s.startSec.toFixed(1) : '?'
  const b = Number.isFinite(s.endSec) ? s.endSec.toFixed(1) : '?'
  return `${a}s – ${b}s`
}
