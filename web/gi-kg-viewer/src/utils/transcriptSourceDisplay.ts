import type { ParsedArtifact } from '../types/artifact'

/**
 * Shown when a GI quote has no resolved speaker (graph **Quote** node detail, Search lifted +
 * supporting quotes, Explore). GI fills `speaker_id` only when timed segments carry diarization
 * labels aligned with the transcript (issue #541) — UI keeps this line short.
 */
export const GI_QUOTE_SPEAKER_UNAVAILABLE_HINT = 'No speaker detected'

/** Playwright: search + Explore supporting-quote rows when attribution is absent. */
export const SUPPORTING_QUOTE_SPEAKER_UNAVAILABLE_TESTID = 'supporting-quote-speaker-unavailable'

/** Playwright: Search **Lifted GI insight** when **`lifted.quote`** has finite **`timestamp_*_ms`** but speaker display does not. */
export const SEARCH_LIFTED_QUOTE_SPEAKER_UNAVAILABLE_TESTID =
  'search-lifted-quote-speaker-unavailable'

/**
 * True when optional **search** ``lifted.quote`` has at least one finite GI timestamp (ms).
 * Gates the muted #541 line when quote timing exists but **`lifted.speaker`** has no display label.
 */
export function liftedQuotePayloadHasUsableTiming(quote: unknown): boolean {
  if (quote == null || typeof quote !== 'object') return false
  const o = quote as Record<string, unknown>
  const a = Number(o.timestamp_start_ms)
  const b = Number(o.timestamp_end_ms)
  return Number.isFinite(a) || Number.isFinite(b)
}

/** Build `/api/corpus/text-file` URL for opening a transcript in a new browser tab. */
export function corpusTextFileViewUrl(corpusRoot: string, relpath: string): string {
  const path = encodeURIComponent(corpusRoot.trim())
  const rel = encodeURIComponent(relpath.trim())
  return `/api/corpus/text-file?path=${path}&relpath=${rel}`
}

/**
 * Pipeline GI files often live under ``.../run_hash/metadata/*.gi.json`` while
 * ``transcript_ref`` is relative to ``run_hash`` (e.g. ``transcripts/ep.txt``), not the
 * session corpus root and not the ``metadata`` folder. When ``giArtifactCorpusRelPath``
 * contains ``/metadata/``, that prefix is the feed run directory used to qualify refs.
 */
function feedRunRootFromGiCorpusRelPath(gi: string): string | null {
  const g = gi.replace(/\\/g, '/').replace(/^\/+/, '')
  const marker = '/metadata/'
  const idx = g.indexOf(marker)
  if (idx >= 0) {
    return g.slice(0, idx)
  }
  return null
}

/**
 * When GI stores a bare filename in ``transcript_ref`` (e.g. ``transcript.txt``), it is
 * relative to the feed run directory (parent of ``metadata`` when present), else the
 * directory of the ``.gi.json``. Multi-segment refs starting with ``transcripts/`` are
 * treated like bare paths under the feed run when ``/metadata/`` is in the GI path.
 * Refs starting with ``feeds/`` are left as corpus-root-relative (already fully qualified).
 */
export function resolveGiPathForTranscript(
  viewArtifact: ParsedArtifact | null | undefined,
  quoteEpisodeId: string | null | undefined,
): string | null | undefined {
  if (!viewArtifact) {
    return null
  }
  const ep =
    quoteEpisodeId != null && String(quoteEpisodeId).trim()
      ? String(quoteEpisodeId).trim()
      : ''
  if (ep && viewArtifact.sourceCorpusRelPathByEpisodeId?.[ep]) {
    return viewArtifact.sourceCorpusRelPathByEpisodeId[ep]
  }
  return viewArtifact.sourceCorpusRelPath ?? null
}

export function resolveTranscriptCorpusRelpath(
  transcriptRef: string,
  giArtifactCorpusRelPath: string | null | undefined,
): string {
  const raw = transcriptRef.trim().replace(/\\/g, '/').replace(/^\.\/+/, '')
  if (!raw) return raw

  const gi = (giArtifactCorpusRelPath || '').trim().replace(/\\/g, '/').replace(/^\/+/, '')
  const feedRoot = gi ? feedRunRootFromGiCorpusRelPath(gi) : null

  if (raw.includes('/')) {
    const stripped = raw.replace(/^\/+/, '')
    if (stripped.startsWith('feeds/')) {
      return stripped
    }
    if (feedRoot && stripped.startsWith('transcripts/')) {
      return `${feedRoot}/${stripped}`
    }
    return stripped
  }

  if (!gi) return raw
  const baseDir =
    feedRoot ??
    (() => {
      const slash = gi.lastIndexOf('/')
      return slash < 0 ? '' : gi.slice(0, slash)
    })()
  if (!baseDir) return raw
  return `${baseDir}/${raw}`
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

/** Human-readable character span in the transcript file (GI Quote fields). */
export function formatTranscriptCharRange(charStart?: unknown, charEnd?: unknown): string | null {
  const cs = toFiniteNumber(charStart)
  const ce = toFiniteNumber(charEnd)
  if (cs == null && ce == null) {
    return null
  }
  if (cs != null && ce != null) {
    return `Characters ${Math.round(cs)}–${Math.round(ce)}`
  }
  if (cs != null) {
    return `From character ${Math.round(cs)}`
  }
  return `Through character ${Math.round(ce!)}`
}

function formatSecondsPlain(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) {
    return '—'
  }
  return `${sec.toFixed(1)}s`
}

/**
 * Human-readable playback window (ms → seconds), or null if no usable timing.
 * When both are 0, returns a short “not specified” line instead of “0–0”.
 */
export function formatAudioTimingRange(
  startMs?: unknown,
  endMs?: unknown,
): string | null {
  const a = toFiniteNumber(startMs)
  const b = toFiniteNumber(endMs)
  if (a == null && b == null) {
    return null
  }
  if (a === 0 && b === 0) {
    return (
      'Audio timing not specified (often no timed transcript segments — e.g. some APIs return text ' +
      'only; see Development Guide, Transcript hash cache / issue 543)'
    )
  }
  const sa = a != null ? a / 1000 : null
  const sb = b != null ? b / 1000 : null
  if (sa != null && sb != null) {
    return `${formatSecondsPlain(sa)} – ${formatSecondsPlain(sb)} in this episode`
  }
  if (sa != null) {
    return `From ${formatSecondsPlain(sa)} in this episode`
  }
  return `Through ${formatSecondsPlain(sb!)} in this episode`
}
