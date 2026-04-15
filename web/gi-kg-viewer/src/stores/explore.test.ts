import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useExploreStore } from './explore'
import type { ExploreApiBody } from '../api/exploreApi'

beforeEach(() => {
  setActivePinia(createPinia())
})

function exploreBody(data: Record<string, unknown>): ExploreApiBody {
  return { kind: 'explore', data }
}

function nlBody(answer: Record<string, unknown>): ExploreApiBody {
  return { kind: 'natural_language', answer }
}

// ── insightRows ──

describe('insightRows', () => {
  it('returns empty when last is null', () => {
    const ex = useExploreStore()
    expect(ex.insightRows).toEqual([])
  })

  it('returns empty when last has error', () => {
    const ex = useExploreStore()
    ex.last = { kind: 'explore', error: 'fail' }
    expect(ex.insightRows).toEqual([])
  })

  it('parses insights from explore data', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [
        {
          insight_id: 'i1',
          text: 'Some insight',
          grounded: true,
          confidence: 0.85,
          episode: { episode_id: 'ep1', title: 'Ep Title' },
        },
      ],
    })
    expect(ex.insightRows).toHaveLength(1)
    expect(ex.insightRows[0]).toMatchObject({
      insight_id: 'i1',
      text: 'Some insight',
      grounded: true,
      confidence: 0.85,
      episode: { episode_id: 'ep1', title: 'Ep Title' },
    })
  })

  it('parses insights from natural_language answer', () => {
    const ex = useExploreStore()
    ex.last = nlBody({
      insights: [{ insight_id: 'i2', text: 'NL insight' }],
    })
    expect(ex.insightRows).toHaveLength(1)
    expect(ex.insightRows[0].insight_id).toBe('i2')
  })

  it('skips entries without insight_id', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [
        { text: 'no id' },
        { insight_id: '', text: 'empty id' },
        { insight_id: 'valid', text: 'ok' },
      ],
    })
    expect(ex.insightRows).toHaveLength(1)
    expect(ex.insightRows[0].insight_id).toBe('valid')
  })

  it('parses supporting quotes with speaker', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [
        {
          insight_id: 'i3',
          text: 'quoted',
          supporting_quotes: [
            {
              text: 'a quote',
              speaker: { name: 'Alice', speaker_id: 's1' },
              timestamp_start_ms: 1000,
              timestamp_end_ms: 2000,
            },
          ],
        },
      ],
    })
    const quotes = ex.insightRows[0].supporting_quotes!
    expect(quotes).toHaveLength(1)
    expect(quotes[0]).toMatchObject({
      text: 'a quote',
      speaker_name: 'Alice',
      speaker_id: 's1',
      start_ms: 1000,
      end_ms: 2000,
    })
  })

  it('falls back to speaker_id when speaker.name is missing', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [
        {
          insight_id: 'i4',
          text: 'x',
          supporting_quotes: [
            { text: 'q', speaker: { speaker_id: 'spk42' } },
          ],
        },
      ],
    })
    const row = ex.insightRows[0].supporting_quotes![0]
    expect(row.speaker_name).toBeUndefined()
    expect(row.speaker_id).toBe('spk42')
  })

  it('uses flat speaker_id when nested speaker object is absent', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [
        {
          insight_id: 'i5',
          text: 'y',
          supporting_quotes: [{ text: 'flat', speaker_id: 'person:bob' }],
        },
      ],
    })
    const row = ex.insightRows[0].supporting_quotes![0]
    expect(row.speaker_name).toBeUndefined()
    expect(row.speaker_id).toBe('person:bob')
  })
})

// ── leaderboardRows ──

describe('leaderboardRows', () => {
  it('returns empty when last is null', () => {
    const ex = useExploreStore()
    expect(ex.leaderboardRows).toEqual([])
  })

  it('parses topics from explore data', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      topics: [
        { topic_id: 't1', label: 'Economy', insight_count: 5 },
        { topic_id: 't2', label: 'Health', insight_count: 3 },
      ],
    })
    expect(ex.leaderboardRows).toHaveLength(2)
    expect(ex.leaderboardRows[0]).toMatchObject({
      topic_id: 't1',
      label: 'Economy',
      insight_count: 5,
    })
  })

  it('parses topics from natural_language answer', () => {
    const ex = useExploreStore()
    ex.last = nlBody({
      topics: [{ topic_id: 'nt1', label: 'NL Topic', insight_count: 1 }],
    })
    expect(ex.leaderboardRows).toHaveLength(1)
  })

  it('defaults missing insight_count to 0', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      topics: [{ topic_id: 't3', label: 'No count' }],
    })
    expect(ex.leaderboardRows[0].insight_count).toBe(0)
  })

  it('filters out entries without label or topic_id', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      topics: [{ insight_count: 1 }, { topic_id: 'ok', label: 'OK' }],
    })
    expect(ex.leaderboardRows).toHaveLength(1)
  })
})

// ── summaryBlock ──

describe('summaryBlock', () => {
  it('returns null when last is null', () => {
    const ex = useExploreStore()
    expect(ex.summaryBlock).toBeNull()
  })

  it('returns null when summary has all zeros', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      summary: { insight_count: 0, grounded_insight_count: 0 },
    })
    expect(ex.summaryBlock).toBeNull()
  })

  it('parses summary from explore data', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      summary: {
        insight_count: 10,
        grounded_insight_count: 7,
        quote_count: 15,
        episode_count: 2,
        speaker_count: 3,
        topic_count: 4,
      },
      episodes_searched: 5,
    })
    const s = ex.summaryBlock!
    expect(s.insight_count).toBe(10)
    expect(s.grounded_insight_count).toBe(7)
    expect(s.quote_count).toBe(15)
    expect(s.episodes_searched).toBe(5)
  })

  it('returns null when no summary object', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ insights: [] })
    expect(ex.summaryBlock).toBeNull()
  })
})

// ── topSpeakers ──

describe('topSpeakers', () => {
  it('returns empty when last is null', () => {
    const ex = useExploreStore()
    expect(ex.topSpeakers).toEqual([])
  })

  it('parses speakers from explore data', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      top_speakers: [
        { speaker_id: 's1', name: 'Alice', quote_count: 5, insight_count: 2 },
        { speaker_id: 's2', name: null, quote_count: 3, insight_count: 1 },
      ],
    })
    expect(ex.topSpeakers).toHaveLength(2)
    expect(ex.topSpeakers[0]).toMatchObject({
      speaker_id: 's1',
      name: 'Alice',
      quote_count: 5,
    })
    expect(ex.topSpeakers[1].name).toBeNull()
  })

  it('filters entries without speaker_id', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      top_speakers: [
        { name: 'No ID' },
        { speaker_id: 'ok', name: 'Valid' },
      ],
    })
    expect(ex.topSpeakers).toHaveLength(1)
  })

  it('defaults missing counts to 0', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      top_speakers: [{ speaker_id: 's3' }],
    })
    expect(ex.topSpeakers[0].quote_count).toBe(0)
    expect(ex.topSpeakers[0].insight_count).toBe(0)
  })
})

// ── clearOutput ──

describe('clearOutput', () => {
  it('resets last and error', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ insights: [] })
    ex.error = 'some error'
    ex.clearOutput()
    expect(ex.last).toBeNull()
    expect(ex.error).toBeNull()
  })
})
