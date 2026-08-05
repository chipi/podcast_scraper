import { describe, expect, it } from 'vitest'
import { shareCardText } from './useShareCard'
import type { Highlight } from '../services/types'

function hl(over: Partial<Highlight> = {}): Highlight {
  return {
    id: 'h1', episode_slug: 'ep', kind: 'span', start_ms: 65_000, end_ms: null,
    char_start: null, char_end: null, segment_ids: [], quote_text: 'a memorable line',
    speaker: null, source_insight_id: null, color: null, created_at: 1, anchor_status: null,
    graph_refs: [{ id: 'person:jane', kind: 'person', label: 'Jane Doe' }],
    ...over,
  }
}

describe('shareCardText', () => {
  it('includes the quote, episode + timestamp, refs, and the wordmark', () => {
    const text = shareCardText(hl(), 'NVIDIA')
    expect(text).toContain('“a memorable line”')
    expect(text).toContain('— NVIDIA · 1:05')
    expect(text).toContain('Jane Doe')
    expect(text).toContain('closelistening.app')
  })

  it('carries no audio reference (bridge-only)', () => {
    const text = shareCardText(hl(), 'NVIDIA').toLowerCase()
    expect(text).not.toContain('.mp3')
    expect(text).not.toContain('http')
    expect(text).not.toMatch(/audio|enclosure|media_url/)
  })

  it('omits the timestamp line detail when there is no anchor', () => {
    const text = shareCardText(hl({ start_ms: null }), 'NVIDIA')
    expect(text).toContain('— NVIDIA')
    expect(text).not.toContain('·  ') // no dangling timestamp separator
  })
})
