import { describe, expect, it } from 'vitest'
import {
  corpusTextFileViewUrl,
  corpusMediaFileViewUrl,
  audioRelpathFromTranscriptRelpath,
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

describe('audioRelpathFromTranscriptRelpath', () => {
  it('flattens nested transcript paths to media/<stem>.mp3 (matches backend)', () => {
    // Backend persists media/<stem><ext> (subdirs dropped); preserving subdirs here
    // produced a path the backend never writes (404). Flatten to the stem.
    expect(audioRelpathFromTranscriptRelpath('feeds/show/ep/episode_01.txt')).toBe(
      'media/episode_01.mp3',
    )
  })

  it('maps bare transcript.txt under media/', () => {
    expect(audioRelpathFromTranscriptRelpath('transcript.txt')).toBe('media/transcript.mp3')
  })
})

describe('corpusMediaFileViewUrl', () => {
  it('builds encoded query string for media relpath', () => {
    const u = corpusMediaFileViewUrl('/my root', 'media/feeds/x/transcript.mp3')
    expect(u.startsWith('/api/corpus/media?')).toBe(true)
    expect(u).toContain('relpath=media%2Ffeeds%2Fx%2Ftranscript.mp3')
    expect(u).toContain('path=%2Fmy%20root')
  })
})
