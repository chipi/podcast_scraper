import { describe, expect, it } from 'vitest'
import {
  buildCorpusLikeDocumentView,
  buildMultiFeedBatchView,
  feedsArrayToTable,
  feedsArrayToTableWithUrls,
  flattenObjectLeaves,
  isBatchIncidentLongValueLabel,
  partitionBatchIncidentRows,
  stripCorpusBatchSectionLabel,
} from './humanizeJsonDocument'

describe('stripCorpusBatchSectionLabel', () => {
  it('removes leading corpus word only', () => {
    expect(stripCorpusBatchSectionLabel('corpus parent')).toBe('parent')
    expect(stripCorpusBatchSectionLabel('Corpus batch id')).toBe('batch id')
    expect(stripCorpusBatchSectionLabel('overall ok')).toBe('overall ok')
    expect(stripCorpusBatchSectionLabel('corpus')).toBe('corpus')
  })
})

describe('flattenObjectLeaves', () => {
  it('flattens nested scalars with readable labels', () => {
    const rows = flattenObjectLeaves({
      a: 1,
      b: { c: 'x', d: { e: false } },
    })
    expect(rows).toContainEqual({ label: 'a', value: '1' })
    expect(rows).toContainEqual({ label: 'b · c', value: 'x' })
    expect(rows).toContainEqual({ label: 'b · d · e', value: 'No' })
  })
})

describe('buildCorpusLikeDocumentView', () => {
  it('splits corpus parent, other meta, incidents, feeds table, and row feed URLs', () => {
    const v = buildCorpusLikeDocumentView({
      corpus_parent: '/data/corpus',
      overall_ok: true,
      corpus_version: '1.0',
      finished_at: '2026-01-01',
      batch_incidents: { lines_in_window: 2 },
      feeds: [{ feed_url: 'https://x', ok: true, episodes_processed: 3, error: null }],
    })
    expect(v.corpusParentRow?.label).toBe('parent')
    expect(v.corpusParentRow?.value).toBe('/data/corpus')
    expect(v.metaRows.some((r) => r.label === 'overall ok' && r.value === 'Yes')).toBe(true)
    expect(v.metaRows.some((r) => r.label === 'version' && r.value === '1.0')).toBe(true)
    expect(v.metaRows.some((r) => r.label === 'corpus parent')).toBe(false)
    expect(v.metaRows.some((r) => r.label === 'corpus version')).toBe(false)
    expect(v.incidentRows.some((r) => r.value === '2')).toBe(true)
    expect(v.feedsTable?.rows.length).toBe(1)
    expect(v.feedsTable?.headers[0]).toBe('Feed')
    expect(v.feedsRowFeedUrls).toEqual(['https://x'])
  })
})

describe('partitionBatchIncidentRows', () => {
  it('splits log path and semantics note for full-width display', () => {
    const { compactRows, longRows } = partitionBatchIncidentRows([
      { label: 'lines in window', value: '3' },
      { label: 'log path', value: '/tmp/a.jsonl' },
      { label: 'semantics note', value: 'long prose…' },
    ])
    expect(compactRows.map((r) => r.label)).toEqual(['lines in window'])
    expect(longRows.map((r) => r.label)).toEqual(['log path', 'semantics note'])
  })
})

describe('buildMultiFeedBatchView', () => {
  it('extracts feeds table from batch payload', () => {
    const v = buildMultiFeedBatchView({
      feeds: [{ feed_url: 'https://a', ok: false, episodes_processed: 0, error: 'x' }],
    })
    expect(v.feedsTable?.rows[0]?.[3]).toBe('x')
  })

  it('returns empty view for non-record input', () => {
    expect(buildMultiFeedBatchView(null)).toEqual({ metaRows: [], feedsTable: null })
    expect(buildMultiFeedBatchView('string')).toEqual({ metaRows: [], feedsTable: null })
    expect(buildMultiFeedBatchView([1, 2])).toEqual({ metaRows: [], feedsTable: null })
  })

  it('collects scalar meta rows and skips the feeds key', () => {
    const v = buildMultiFeedBatchView({
      schema_version: '1',
      overall_ok: true,
      feeds: [{ feed_url: 'https://a' }],
    })
    expect(v.metaRows).toContainEqual({ label: 'schema version', value: '1' })
    expect(v.metaRows).toContainEqual({ label: 'overall ok', value: 'Yes' })
    expect(v.metaRows.some((r) => r.label === 'feeds')).toBe(false)
    expect(v.feedsTable?.rows.length).toBe(1)
  })

  it('flattens nested object meta with a prefixed label', () => {
    const v = buildMultiFeedBatchView({
      totals: { soft: 2, hard: 0 },
    })
    expect(v.metaRows).toContainEqual({ label: 'totals · soft', value: '2' })
    expect(v.metaRows).toContainEqual({ label: 'totals · hard', value: '0' })
    expect(v.feedsTable).toBeNull()
  })
})

describe('isBatchIncidentLongValueLabel', () => {
  it('matches exact and suffixed long-value labels', () => {
    expect(isBatchIncidentLongValueLabel('log path')).toBe(true)
    expect(isBatchIncidentLongValueLabel('semantics note')).toBe(true)
    expect(isBatchIncidentLongValueLabel('feed a · log path')).toBe(true)
    expect(isBatchIncidentLongValueLabel('feed a · semantics note')).toBe(true)
    expect(isBatchIncidentLongValueLabel('  log path  ')).toBe(true)
  })

  it('does not match unrelated or partial labels', () => {
    expect(isBatchIncidentLongValueLabel('lines in window')).toBe(false)
    expect(isBatchIncidentLongValueLabel('log path extra')).toBe(false)
    expect(isBatchIncidentLongValueLabel('note')).toBe(false)
  })
})

describe('flattenObjectLeaves (edge cases)', () => {
  it('returns empty array for non-record input', () => {
    expect(flattenObjectLeaves(null)).toEqual([])
    expect(flattenObjectLeaves('str')).toEqual([])
    expect(flattenObjectLeaves(42)).toEqual([])
    expect(flattenObjectLeaves([1, 2, 3])).toEqual([])
  })

  it('renders null/empty/whitespace-only string leaves as an em dash', () => {
    const rows = flattenObjectLeaves({ a: null, b: '', c: '   ' })
    expect(rows).toContainEqual({ label: 'a', value: '—' })
    expect(rows).toContainEqual({ label: 'b', value: '—' })
    // cellStr trims before the empty check, so whitespace-only collapses to a dash too
    expect(rows).toContainEqual({ label: 'c', value: '—' })
  })

  it('renders empty arrays as an em dash', () => {
    expect(flattenObjectLeaves({ tags: [] })).toContainEqual({ label: 'tags', value: '—' })
  })

  it('joins scalar arrays into a comma-separated value', () => {
    expect(flattenObjectLeaves({ tags: ['a', 'b', 3] })).toContainEqual({
      label: 'tags',
      value: 'a, b, 3',
    })
  })

  it('represents nested objects inside a mixed array as [object]', () => {
    expect(flattenObjectLeaves({ items: ['a', { x: 1 }] })).toContainEqual({
      label: 'items',
      value: 'a, [object]',
    })
  })

  it('skips arrays whose entries are all plain objects (caller renders tables)', () => {
    const rows = flattenObjectLeaves({ feeds: [{ a: 1 }, { b: 2 }], other: 'x' })
    expect(rows.some((r) => r.label === 'feeds')).toBe(false)
    expect(rows).toContainEqual({ label: 'other', value: 'x' })
  })

  it('stops recursing past the max depth', () => {
    const deep = { l0: { l1: { l2: { l3: 'too deep' } } } }
    const rows = flattenObjectLeaves(deep, '', 0, 1)
    expect(rows).toEqual([])
  })
})

describe('feedsArrayToTableWithUrls', () => {
  it('returns null for non-arrays and empty arrays', () => {
    expect(feedsArrayToTableWithUrls(null)).toBeNull()
    expect(feedsArrayToTableWithUrls('feeds')).toBeNull()
    expect(feedsArrayToTableWithUrls([])).toBeNull()
  })

  it('returns null when no array entry is a record', () => {
    expect(feedsArrayToTableWithUrls(['a', 1, null, [2]])).toBeNull()
  })

  it('skips non-record entries but keeps valid feed rows', () => {
    const out = feedsArrayToTableWithUrls([
      'skip me',
      { feed_url: 'https://keep', ok: true, episodes_processed: 5 },
    ])
    expect(out?.table.rows.length).toBe(1)
    expect(out?.rowFeedUrls).toEqual(['https://keep'])
  })

  it('renders incident sub-fields and falls back to dashes when missing', () => {
    const out = feedsArrayToTableWithUrls([
      {
        feed_url: 'https://a',
        ok: false,
        episodes_processed: 0,
        error: 'boom',
        finished_at: '2026-01-01',
        failure_kind: 'timeout',
        episode_incidents_unique: { policy: 'strict', soft: 2, hard: 1 },
      },
      { feed_url: 'https://b' },
    ])
    const r0 = out!.table.rows[0]!
    expect(r0).toEqual([
      'https://a',
      'No',
      '0',
      'boom',
      '2026-01-01',
      'timeout',
      'strict',
      '2',
      '1',
    ])
    const r1 = out!.table.rows[1]!
    // missing incident object -> dash for the three incident columns
    expect(r1.slice(6)).toEqual(['—', '—', '—'])
    // missing scalars -> dash
    expect(r1[1]).toBe('—')
  })

  it('truncates long feed URLs with an ellipsis', () => {
    const longUrl = `https://example.com/${'x'.repeat(60)}`
    const out = feedsArrayToTableWithUrls([{ feed_url: longUrl }])
    const cell = out!.table.rows[0]![0]!
    expect(cell.length).toBe(44)
    expect(cell.endsWith('…')).toBe(true)
    // full untruncated URL retained in rowFeedUrls
    expect(out!.rowFeedUrls[0]).toBe(longUrl)
  })

  it('records an empty rowFeedUrl when feed_url is not a string', () => {
    const out = feedsArrayToTableWithUrls([{ feed_url: 123, ok: true }])
    // not a string -> no full URL captured
    expect(out?.rowFeedUrls).toEqual([''])
    // but cellStr still renders a finite number for the visible cell
    expect(out?.table.rows[0]![0]).toBe('123')
  })
})

describe('feedsArrayToTable', () => {
  it('returns just the table portion', () => {
    const t = feedsArrayToTable([{ feed_url: 'https://a', ok: true }])
    expect(t?.headers[0]).toBe('Feed')
    expect(t?.rows.length).toBe(1)
  })

  it('returns null when no table can be built', () => {
    expect(feedsArrayToTable([])).toBeNull()
    expect(feedsArrayToTable(null)).toBeNull()
  })
})

describe('buildCorpusLikeDocumentView (edge cases)', () => {
  it('returns an empty view for a null document', () => {
    expect(buildCorpusLikeDocumentView(null)).toEqual({
      corpusParentRow: null,
      metaRows: [],
      incidentRows: [],
      feedsTable: null,
      feedsRowFeedUrls: [],
    })
  })

  it('ignores a non-scalar corpus_parent and produces no parent row', () => {
    const v = buildCorpusLikeDocumentView({
      corpus_parent: { nested: 'object' },
      overall_ok: true,
    })
    expect(v.corpusParentRow).toBeNull()
    expect(v.metaRows.some((r) => r.label === 'overall ok')).toBe(true)
  })

  it('omits non-scalar, non-skipped top-level values from meta rows', () => {
    const v = buildCorpusLikeDocumentView({
      overall_ok: true,
      nested_block: { a: 1 },
    })
    expect(v.metaRows.some((r) => r.label.startsWith('nested'))).toBe(false)
    expect(v.metaRows.some((r) => r.label === 'overall ok')).toBe(true)
  })

  it('handles a missing feeds key and missing batch_incidents', () => {
    const v = buildCorpusLikeDocumentView({ overall_ok: true })
    expect(v.feedsTable).toBeNull()
    expect(v.feedsRowFeedUrls).toEqual([])
    expect(v.incidentRows).toEqual([])
  })
})
