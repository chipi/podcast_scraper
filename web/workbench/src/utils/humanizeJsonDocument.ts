/**
 * Turn corpus / log JSON documents into label–value rows and small tables
 * (no nested JSON blobs in the UI).
 */

export interface LabelValueRow {
  label: string
  value: string
}

/** Paths / prose on ``batch_incidents`` — render full-width below the compact grid. */
export function isBatchIncidentLongValueLabel(label: string): boolean {
  const t = label.trim()
  return (
    t === 'log path' ||
    t === 'semantics note' ||
    t.endsWith(' · log path') ||
    t.endsWith(' · semantics note')
  )
}

export function partitionBatchIncidentRows(rows: LabelValueRow[]): {
  compactRows: LabelValueRow[]
  longRows: LabelValueRow[]
} {
  const compactRows: LabelValueRow[] = []
  const longRows: LabelValueRow[] = []
  for (const r of rows) {
    if (isBatchIncidentLongValueLabel(r.label)) {
      longRows.push(r)
    } else {
      compactRows.push(r)
    }
  }
  return { compactRows, longRows }
}

export interface StringTable {
  headers: string[]
  rows: string[][]
}

function humanLabel(key: string): string {
  return key.replace(/_/g, ' ')
}

/** Section title is already "Corpus batch" — drop a leading ``corpus `` word from labels. */
export function stripCorpusBatchSectionLabel(label: string): string {
  const t = label.trim()
  const stripped = t.replace(/^corpus\s+/i, '').trim()
  return stripped.length > 0 ? stripped : t
}

function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null
}

function cellStr(v: unknown): string {
  if (v == null) {
    return '—'
  }
  if (typeof v === 'boolean') {
    return v ? 'Yes' : 'No'
  }
  if (typeof v === 'number' && Number.isFinite(v)) {
    return String(v)
  }
  if (typeof v === 'string') {
    return v.trim() === '' ? '—' : v
  }
  return '—'
}

function truncateUrl(s: string, max = 44): string {
  if (s.length <= max) {
    return s
  }
  return `${s.slice(0, max - 1)}…`
}

/** Flatten nested plain objects into dotted labels (skip arrays and null leaves). */
export function flattenObjectLeaves(
  obj: unknown,
  prefix = '',
  depth = 0,
  maxDepth = 7,
): LabelValueRow[] {
  if (depth > maxDepth) {
    return []
  }
  const rec = asRecord(obj)
  if (!rec) {
    return []
  }
  const out: LabelValueRow[] = []
  for (const [k, v] of Object.entries(rec)) {
    const label = prefix ? `${prefix} · ${humanLabel(k)}` : humanLabel(k)
    if (v == null || typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
      out.push({ label, value: cellStr(v) })
    } else if (Array.isArray(v)) {
      if (v.length === 0) {
        out.push({ label, value: '—' })
      } else if (v.every((x) => x !== null && typeof x === 'object' && !Array.isArray(x))) {
        /* caller renders tables for known ``feeds`` keys */
      } else {
        out.push({
          label,
          value: v.map((x) => (typeof x === 'object' ? '[object]' : String(x))).join(', '),
        })
      }
    } else if (asRecord(v)) {
      out.push(...flattenObjectLeaves(v, label, depth + 1, maxDepth))
    }
  }
  return out
}

export interface FeedsTableWithUrls {
  table: StringTable
  /** Full ``feed_url`` per ``table.rows`` entry (same order). */
  rowFeedUrls: string[]
}

/** ``feeds`` arrays from ``corpus_run_summary`` / ``multi_feed_batch`` style payloads. */
export function feedsArrayToTableWithUrls(feeds: unknown): FeedsTableWithUrls | null {
  if (!Array.isArray(feeds) || feeds.length === 0) {
    return null
  }
  const rows: string[][] = []
  const rowFeedUrls: string[] = []
  for (const raw of feeds) {
    const f = asRecord(raw)
    if (!f) {
      continue
    }
    const inc = asRecord(f.episode_incidents_unique)
    const urlRaw = f.feed_url
    rowFeedUrls.push(typeof urlRaw === 'string' ? urlRaw.trim() : '')
    rows.push([
      truncateUrl(cellStr(f.feed_url)),
      cellStr(f.ok),
      cellStr(f.episodes_processed),
      cellStr(f.error),
      cellStr(f.finished_at),
      cellStr(f.failure_kind),
      inc ? cellStr(inc.policy) : '—',
      inc ? cellStr(inc.soft) : '—',
      inc ? cellStr(inc.hard) : '—',
    ])
  }
  if (rows.length === 0) {
    return null
  }
  return {
    table: {
      headers: [
        'Feed',
        'OK',
        'Episodes',
        'Error',
        'Finished',
        'Failure kind',
        'Inc. policy',
        'Inc. soft',
        'Inc. hard',
      ],
      rows,
    },
    rowFeedUrls,
  }
}

/** Table only (no per-row URLs). */
export function feedsArrayToTable(feeds: unknown): StringTable | null {
  return feedsArrayToTableWithUrls(feeds)?.table ?? null
}

export interface CorpusLikeDocumentView {
  /** Long corpus root path — render full-width above the compact meta grid. */
  corpusParentRow: LabelValueRow | null
  /** Top-level scalars except ``feeds`` / ``batch_incidents`` / ``corpus_parent``. */
  metaRows: LabelValueRow[]
  incidentRows: LabelValueRow[]
  feedsTable: StringTable | null
  /** Parallel to ``feedsTable.rows`` when present; empty otherwise. */
  feedsRowFeedUrls: string[]
}

const SKIP_META = new Set(['feeds', 'batch_incidents'])

/** Build rows + feed table for ``corpus_run_summary``-shaped objects. */
export function buildCorpusLikeDocumentView(doc: Record<string, unknown> | null): CorpusLikeDocumentView {
  if (!doc) {
    return {
      corpusParentRow: null,
      metaRows: [],
      incidentRows: [],
      feedsTable: null,
      feedsRowFeedUrls: [],
    }
  }
  let corpusParentRow: LabelValueRow | null = null
  const metaRows: LabelValueRow[] = []
  for (const [k, v] of Object.entries(doc)) {
    if (SKIP_META.has(k)) {
      continue
    }
    if (k === 'corpus_parent') {
      if (v == null || typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
        corpusParentRow = {
          label: stripCorpusBatchSectionLabel(humanLabel(k)),
          value: cellStr(v),
        }
      }
      continue
    }
    if (v == null || typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
      metaRows.push({
        label: stripCorpusBatchSectionLabel(humanLabel(k)),
        value: cellStr(v),
      })
    }
  }
  const incidentRows = flattenObjectLeaves(doc.batch_incidents, '', 0, 6)
  const feedsBlock = feedsArrayToTableWithUrls(doc.feeds)
  return {
    corpusParentRow,
    metaRows,
    incidentRows,
    feedsTable: feedsBlock?.table ?? null,
    feedsRowFeedUrls: feedsBlock?.rowFeedUrls ?? [],
  }
}

/** ``multi_feed_batch`` log payload is usually ``{ feeds: [...] }``. */
export function buildMultiFeedBatchView(batch: unknown): {
  metaRows: LabelValueRow[]
  feedsTable: StringTable | null
} {
  const rec = asRecord(batch)
  if (!rec) {
    return { metaRows: [], feedsTable: null }
  }
  const metaRows: LabelValueRow[] = []
  for (const [k, v] of Object.entries(rec)) {
    if (k === 'feeds') {
      continue
    }
    if (v == null || typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
      metaRows.push({ label: humanLabel(k), value: cellStr(v) })
    } else if (asRecord(v)) {
      metaRows.push(...flattenObjectLeaves(v, humanLabel(k), 0, 4))
    }
  }
  return { metaRows, feedsTable: feedsArrayToTable(rec.feeds) }
}
