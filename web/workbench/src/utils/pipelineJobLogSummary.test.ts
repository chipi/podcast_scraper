import { describe, expect, it } from 'vitest'
import { extractStructuredSummariesFromLogTail } from './pipelineJobLogSummary'

describe('extractStructuredSummariesFromLogTail', () => {
  it('parses multi_feed_batch JSON from tail', () => {
    const tail = `noise\nINFO x multi_feed_batch: {"feeds":[{"feed_url":"https://a","ok":true,"episodes_processed":3}]}\n`
    const out = extractStructuredSummariesFromLogTail(tail)
    expect(out.multiFeedBatch).toEqual({
      feeds: [{ feed_url: 'https://a', ok: true, episodes_processed: 3 }],
    })
    expect(out.corpusMultiFeedSummary).toBeNull()
    expect(out.topicClustersSummary).toBeNull()
  })

  it('prefers last occurrence of marker', () => {
    const tail =
      'multi_feed_batch: {"feeds":[]}\nlater multi_feed_batch: {"feeds":[{"ok":false}]}\n'
    const out = extractStructuredSummariesFromLogTail(tail)
    expect(out.multiFeedBatch).toEqual({ feeds: [{ ok: false }] })
  })

  it('parses corpus_multi_feed_summary line', () => {
    const tail = 'corpus_multi_feed_summary: {"schema_version":"1","overall_ok":true}\n'
    const out = extractStructuredSummariesFromLogTail(tail)
    expect(out.corpusMultiFeedSummary).toEqual({ schema_version: '1', overall_ok: true })
  })

  it('parses topic_clusters_summary line', () => {
    const tail =
      'topic_clusters_summary: {"built":true,"cluster_count":83,"topic_count":205,"singletons":22,"seconds":1.234}\n'
    const out = extractStructuredSummariesFromLogTail(tail)
    expect(out.topicClustersSummary).toEqual({
      built: true,
      cluster_count: 83,
      topic_count: 205,
      singletons: 22,
      seconds: 1.234,
    })
  })

  it('returns nulls for all markers when tail has no JSON', () => {
    const out = extractStructuredSummariesFromLogTail('just some plain log lines\nno markers here\n')
    expect(out.multiFeedBatch).toBeNull()
    expect(out.corpusMultiFeedSummary).toBeNull()
    expect(out.topicClustersSummary).toBeNull()
  })

  it('returns null when marker is present but no JSON bracket follows', () => {
    const out = extractStructuredSummariesFromLogTail('multi_feed_batch: not json at all\n')
    expect(out.multiFeedBatch).toBeNull()
  })

  it('parses a top-level JSON array after a marker', () => {
    const out = extractStructuredSummariesFromLogTail(
      'multi_feed_batch: [{"feed_url":"https://a"},{"feed_url":"https://b"}]\n',
    )
    expect(out.multiFeedBatch).toEqual([{ feed_url: 'https://a' }, { feed_url: 'https://b' }])
  })

  it('parses nested objects and arrays correctly', () => {
    const out = extractStructuredSummariesFromLogTail(
      'corpus_multi_feed_summary: {"feeds":[{"ok":true,"inc":{"soft":1,"hard":0}}],"n":[1,2,3]}\n',
    )
    expect(out.corpusMultiFeedSummary).toEqual({
      feeds: [{ ok: true, inc: { soft: 1, hard: 0 } }],
      n: [1, 2, 3],
    })
  })

  it('ignores brackets that appear inside string values', () => {
    const out = extractStructuredSummariesFromLogTail(
      'multi_feed_batch: {"note":"a } b ] { ["}\n',
    )
    expect(out.multiFeedBatch).toEqual({ note: 'a } b ] { [' })
  })

  it('handles escaped quotes inside string values', () => {
    const out = extractStructuredSummariesFromLogTail(
      'multi_feed_batch: {"note":"he said \\"hi\\" }"}\n',
    )
    expect(out.multiFeedBatch).toEqual({ note: 'he said "hi" }' })
  })

  it('returns null when JSON braces are unterminated', () => {
    const out = extractStructuredSummariesFromLogTail(
      'multi_feed_batch: {"feeds":[{"ok":true}]\n',
    )
    expect(out.multiFeedBatch).toBeNull()
  })

  it('returns null when a closing brace mismatches the opening bracket', () => {
    const out = extractStructuredSummariesFromLogTail('multi_feed_batch: [}\n')
    expect(out.multiFeedBatch).toBeNull()
  })

  it('returns null when a closing bracket mismatches the opening brace', () => {
    const out = extractStructuredSummariesFromLogTail('multi_feed_batch: {]\n')
    expect(out.multiFeedBatch).toBeNull()
  })

  it('returns null when balanced text is not valid JSON (object)', () => {
    const out = extractStructuredSummariesFromLogTail('multi_feed_batch: {not valid}\n')
    expect(out.multiFeedBatch).toBeNull()
  })

  it('returns null when balanced text is not valid JSON (array)', () => {
    const out = extractStructuredSummariesFromLogTail('multi_feed_batch: [1, 2,]\n')
    expect(out.multiFeedBatch).toBeNull()
  })

  it('stops at the first balanced object and ignores trailing log text', () => {
    const out = extractStructuredSummariesFromLogTail(
      'multi_feed_batch: {"ok":true} trailing noise and more text\n',
    )
    expect(out.multiFeedBatch).toEqual({ ok: true })
  })
})
