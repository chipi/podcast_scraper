import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import {
  corpusTextFileViewUrl,
  formatAudioTimingRange,
  formatTranscriptCharRange,
  GI_QUOTE_SPEAKER_UNAVAILABLE_HINT,
  liftedQuotePayloadHasUsableTiming,
  SEARCH_LIFTED_QUOTE_SPEAKER_UNAVAILABLE_TESTID,
  SUPPORTING_QUOTE_SPEAKER_UNAVAILABLE_TESTID,
  resolveGiPathForTranscript,
  resolveTranscriptCorpusRelpath,
} from './transcriptSourceDisplay'

describe('corpusTextFileViewUrl', () => {
  it('builds encoded query string (percent-encoding; avoids +/space ambiguity in query)', () => {
    const u = corpusTextFileViewUrl('/my root', 'feeds/x/transcript.txt')
    expect(u.startsWith('/api/corpus/text-file?')).toBe(true)
    expect(u).toContain('relpath=feeds%2Fx%2Ftranscript.txt')
    expect(u).toContain('path=%2Fmy%20root')
    expect(new URL(u, 'http://x.test').searchParams.get('path')).toBe('/my root')
    expect(new URL(u, 'http://x.test').searchParams.get('relpath')).toBe('feeds/x/transcript.txt')
  })
})

describe('resolveTranscriptCorpusRelpath', () => {
  it('joins bare filename to GI artifact directory', () => {
    expect(
      resolveTranscriptCorpusRelpath('transcript.txt', 'feeds/show/ep/foo.gi.json'),
    ).toBe('feeds/show/ep/transcript.txt')
  })
  it('prefixes transcripts/ with feed run root when GI is under metadata/', () => {
    const gi =
      'feeds/rss_x/run_abc/metadata/0002 - Ep_20260409.gi.json'
    expect(resolveTranscriptCorpusRelpath('transcripts/0002 - Ep_20260409.txt', gi)).toBe(
      'feeds/rss_x/run_abc/transcripts/0002 - Ep_20260409.txt',
    )
  })
  it('joins bare filename to feed run root when GI is under metadata/', () => {
    const gi = 'feeds/show/run_hash/metadata/ep.gi.json'
    expect(resolveTranscriptCorpusRelpath('transcript.txt', gi)).toBe(
      'feeds/show/run_hash/transcript.txt',
    )
  })
  it('leaves multi-segment ref as corpus-relative', () => {
    expect(
      resolveTranscriptCorpusRelpath('feeds/show/ep/transcript.txt', 'other/foo.gi.json'),
    ).toBe('feeds/show/ep/transcript.txt')
  })
  it('no GI path keeps transcripts/ ref unchanged', () => {
    expect(resolveTranscriptCorpusRelpath('transcripts/a.txt', null)).toBe('transcripts/a.txt')
  })
  it('no GI path keeps bare ref', () => {
    expect(resolveTranscriptCorpusRelpath('transcript.txt', null)).toBe('transcript.txt')
  })
})

describe('resolveGiPathForTranscript', () => {
  it('uses per-episode map when merged graph has no single sourceCorpusRelPath', () => {
    const art = {
      sourceCorpusRelPath: null,
      sourceCorpusRelPathByEpisodeId: {
        ep1: 'feeds/x/run/metadata/a.gi.json',
      },
    } as unknown as ParsedArtifact
    expect(resolveGiPathForTranscript(art, 'ep1')).toBe('feeds/x/run/metadata/a.gi.json')
  })
  it('falls back to sourceCorpusRelPath when no map entry', () => {
    const art = {
      sourceCorpusRelPath: 'feeds/single/metadata/x.gi.json',
      sourceCorpusRelPathByEpisodeId: null,
    } as unknown as ParsedArtifact
    expect(resolveGiPathForTranscript(art, 'ep9')).toBe('feeds/single/metadata/x.gi.json')
  })
  it('map wins over global path for matching episode', () => {
    const art = {
      sourceCorpusRelPath: 'feeds/global/metadata/x.gi.json',
      sourceCorpusRelPathByEpisodeId: { ep2: 'feeds/other/metadata/y.gi.json' },
    } as unknown as ParsedArtifact
    expect(resolveGiPathForTranscript(art, 'ep2')).toBe('feeds/other/metadata/y.gi.json')
  })
})

describe('formatTranscriptCharRange', () => {
  it('returns null when both missing', () => {
    expect(formatTranscriptCharRange(undefined, undefined)).toBeNull()
  })
  it('formats span', () => {
    expect(formatTranscriptCharRange(0, 60)).toBe('Characters 0–60')
  })
})

describe('formatAudioTimingRange', () => {
  it('returns null when both missing', () => {
    expect(formatAudioTimingRange(undefined, undefined)).toBeNull()
  })
  it('treats 0,0 as unspecified', () => {
    expect(formatAudioTimingRange(0, 0)).toBe(
      'Audio timing not specified (often no timed transcript segments — e.g. some APIs return text ' +
        'only; see Development Guide, Transcript hash cache / issue 543)',
    )
  })
  it('formats span in seconds', () => {
    expect(formatAudioTimingRange(1000, 2500)).toBe('1.0s – 2.5s in this episode')
  })
})

describe('liftedQuotePayloadHasUsableTiming', () => {
  it('is false for non-objects and missing finite ms', () => {
    expect(liftedQuotePayloadHasUsableTiming(null)).toBe(false)
    expect(liftedQuotePayloadHasUsableTiming(undefined)).toBe(false)
    expect(liftedQuotePayloadHasUsableTiming('x')).toBe(false)
    expect(liftedQuotePayloadHasUsableTiming({})).toBe(false)
    expect(
      liftedQuotePayloadHasUsableTiming({
        timestamp_start_ms: NaN,
        timestamp_end_ms: 'x',
      }),
    ).toBe(false)
  })
  it('is true when either ms field is finite', () => {
    expect(liftedQuotePayloadHasUsableTiming({ timestamp_start_ms: 1000 })).toBe(true)
    expect(liftedQuotePayloadHasUsableTiming({ timestamp_end_ms: 2000 })).toBe(true)
    expect(
      liftedQuotePayloadHasUsableTiming({
        timestamp_start_ms: 0,
        timestamp_end_ms: 500,
      }),
    ).toBe(true)
  })
})

describe('GI quote speaker hint copy', () => {
  it('exports stable hint and test id for search/explore/graph', () => {
    expect(GI_QUOTE_SPEAKER_UNAVAILABLE_HINT).toBe('No speaker detected')
    expect(SUPPORTING_QUOTE_SPEAKER_UNAVAILABLE_TESTID).toBe('supporting-quote-speaker-unavailable')
    expect(SEARCH_LIFTED_QUOTE_SPEAKER_UNAVAILABLE_TESTID).toBe(
      'search-lifted-quote-speaker-unavailable',
    )
  })
})
