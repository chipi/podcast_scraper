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
})
