import { describe, expect, it } from 'vitest'
import {
  buildTranscriptHighlightSegments,
  DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES,
  parseTranscriptSegmentsJson,
  segmentsSidecarRelpathFromTranscriptRelpath,
  splitTranscriptAroundHighlight,
  transcriptExceedsMaxBytes,
} from './transcriptViewerModel'

describe('segmentsSidecarRelpathFromTranscriptRelpath', () => {
  it('maps .txt to .segments.json', () => {
    expect(segmentsSidecarRelpathFromTranscriptRelpath('feeds/x/transcripts/ep.txt')).toBe(
      'feeds/x/transcripts/ep.segments.json',
    )
  })

  it('maps .cleaned.txt to .cleaned.segments.json', () => {
    expect(segmentsSidecarRelpathFromTranscriptRelpath('transcripts/foo.cleaned.txt')).toBe(
      'transcripts/foo.cleaned.segments.json',
    )
  })

  it('trims and normalizes slashes', () => {
    expect(segmentsSidecarRelpathFromTranscriptRelpath('  a\\\\b.txt  ')).toBe('a/b.segments.json')
  })
})

describe('parseTranscriptSegmentsJson', () => {
  it('parses Whisper-style list', () => {
    const raw = [
      { start: 0, end: 1.5, text: 'Hi' },
      { start: 1.5, end: 3, text: ' there' },
    ]
    const s = parseTranscriptSegmentsJson(raw)
    expect(s).toHaveLength(2)
    expect(s![0]).toMatchObject({ startSec: 0, endSec: 1.5, text: 'Hi' })
  })

  it('accepts start_time / end_time aliases', () => {
    const s = parseTranscriptSegmentsJson([{ start_time: 2, end_time: 4, text: 'x' }])
    expect(s).toHaveLength(1)
    expect(s![0].startSec).toBe(2)
    expect(s![0].endSec).toBe(4)
  })

  it('returns null for empty or invalid', () => {
    expect(parseTranscriptSegmentsJson([])).toBeNull()
    expect(parseTranscriptSegmentsJson(null)).toBeNull()
    expect(parseTranscriptSegmentsJson([{ foo: 1 }])).toBeNull()
  })
})

describe('transcriptExceedsMaxBytes', () => {
  it('uses body length', () => {
    expect(transcriptExceedsMaxBytes(null, DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES + 1, DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES)).toBe(
      true,
    )
    expect(transcriptExceedsMaxBytes(null, 100, DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES)).toBe(false)
  })

  it('uses Content-Length when larger than body (stream not fully read)', () => {
    expect(
      transcriptExceedsMaxBytes(String(DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES + 10), 0, DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES),
    ).toBe(true)
  })
})

describe('splitTranscriptAroundHighlight', () => {
  it('splits on inclusive-exclusive range', () => {
    const t = 'abcdefghij'
    const p = splitTranscriptAroundHighlight(t, 2, 5)
    expect(p).toEqual({ before: 'ab', highlight: 'cde', after: 'fghij' })
  })

  it('clamps out of range', () => {
    const t = 'ab'
    expect(splitTranscriptAroundHighlight(t, -5, 99)).toEqual({
      before: '',
      highlight: 'ab',
      after: '',
    })
  })

  it('returns null when no offsets', () => {
    expect(splitTranscriptAroundHighlight('hi', null, null)).toBeNull()
  })

  it('returns null for 0–0 window', () => {
    expect(splitTranscriptAroundHighlight('hello', 0, 0)).toBeNull()
  })

  it('swaps inverted range', () => {
    expect(splitTranscriptAroundHighlight('abcde', 4, 1)).toEqual({
      before: 'a',
      highlight: 'bcd',
      after: 'e',
    })
  })
})

describe('buildTranscriptHighlightSegments', () => {
  it('interleaves disjoint highlight spans', () => {
    const t = 'aaaZZZbbbYYYccc'
    const s = buildTranscriptHighlightSegments(t, [
      { charStart: 3, charEnd: 6 },
      { charStart: 9, charEnd: 12 },
    ])
    expect(s).toEqual([
      { type: 'text', text: 'aaa' },
      { type: 'mark', text: 'ZZZ' },
      { type: 'text', text: 'bbb' },
      { type: 'mark', text: 'YYY' },
      { type: 'text', text: 'ccc' },
    ])
  })

  it('merges overlapping ranges into one mark', () => {
    const t = 'abcdefghij'
    const s = buildTranscriptHighlightSegments(t, [
      { charStart: 2, charEnd: 6 },
      { charStart: 4, charEnd: 8 },
    ])
    expect(s).toEqual([
      { type: 'text', text: 'ab' },
      { type: 'mark', text: 'cdefgh' },
      { type: 'text', text: 'ij' },
    ])
  })

  it('returns null when ranges are empty or unusable', () => {
    expect(buildTranscriptHighlightSegments('hi', [])).toBeNull()
    expect(buildTranscriptHighlightSegments('hi', [{ charStart: null, charEnd: null }])).toBeNull()
  })
})
