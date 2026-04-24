import { describe, expect, it } from 'vitest'
import {
  buildCorpusLikeDocumentView,
  buildMultiFeedBatchView,
  flattenObjectLeaves,
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
})
