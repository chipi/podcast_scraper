// @vitest-environment happy-dom
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useExploreStore } from './explore'
import type { ExploreApiBody } from '../api/exploreApi'
import {
  fetchExploreFiltered,
  fetchExploreNaturalLanguage,
} from '../api/exploreApi'
import posthog from 'posthog-js'

vi.mock('../api/exploreApi', () => ({
  fetchExploreFiltered: vi.fn(),
  fetchExploreNaturalLanguage: vi.fn(),
}))

vi.mock('posthog-js', () => ({
  default: { capture: vi.fn() },
}))

const mockFiltered = vi.mocked(fetchExploreFiltered)
const mockNL = vi.mocked(fetchExploreNaturalLanguage)
const mockCapture = vi.mocked(posthog.capture)

beforeEach(() => {
  setActivePinia(createPinia())
  vi.clearAllMocks()
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

// ── insightRows: additional branches ──

describe('insightRows (extra branches)', () => {
  it('returns empty when explore data is missing', () => {
    const ex = useExploreStore()
    ex.last = { kind: 'explore' }
    expect(ex.insightRows).toEqual([])
  })

  it('returns empty when insights is not an array', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ insights: 'nope' })
    expect(ex.insightRows).toEqual([])
  })

  it('skips null/non-object entries in the insights array', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [null, 42, 'str', { insight_id: 'keep', text: 'ok' }],
    })
    expect(ex.insightRows).toHaveLength(1)
    expect(ex.insightRows[0].insight_id).toBe('keep')
  })

  it('skips entries whose insight_id is not a string', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ insights: [{ insight_id: 123, text: 'x' }] })
    expect(ex.insightRows).toEqual([])
  })

  it('leaves grounded/confidence undefined when wrong types', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [
        { insight_id: 'i', text: 1, grounded: 'yes', confidence: 'high' },
      ],
    })
    const row = ex.insightRows[0]
    expect(row.text).toBe('')
    expect(row.grounded).toBeUndefined()
    expect(row.confidence).toBeUndefined()
  })

  it('drops the episode when it is not an object', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [{ insight_id: 'i', text: 'x', episode: 'ep1' }],
    })
    expect(ex.insightRows[0].episode).toBeUndefined()
  })

  it('leaves supporting_quotes undefined when array is empty', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [{ insight_id: 'i', text: 'x', supporting_quotes: [] }],
    })
    expect(ex.insightRows[0].supporting_quotes).toBeUndefined()
  })

  it('filters out null/non-object quotes', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [
        {
          insight_id: 'i',
          text: 'x',
          supporting_quotes: [null, 7, { text: 'real' }],
        },
      ],
    })
    const quotes = ex.insightRows[0].supporting_quotes!
    expect(quotes).toHaveLength(1)
    expect(quotes[0].text).toBe('real')
  })

  it('defaults quote text to empty and timestamps to undefined when types are wrong', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [
        {
          insight_id: 'i',
          text: 'x',
          supporting_quotes: [
            { text: 99, timestamp_start_ms: 'a', timestamp_end_ms: 'b' },
          ],
        },
      ],
    })
    const q = ex.insightRows[0].supporting_quotes![0]
    expect(q.text).toBe('')
    expect(q.start_ms).toBeUndefined()
    expect(q.end_ms).toBeUndefined()
  })

  it('ignores blank/whitespace nested speaker name and id', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [
        {
          insight_id: 'i',
          text: 'x',
          supporting_quotes: [
            { text: 'q', speaker: { name: '   ', speaker_id: '  ' } },
          ],
        },
      ],
    })
    const q = ex.insightRows[0].supporting_quotes![0]
    expect(q.speaker_name).toBeUndefined()
    expect(q.speaker_id).toBeUndefined()
  })

  it('prefers nested speaker_id over flat speaker_id', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      insights: [
        {
          insight_id: 'i',
          text: 'x',
          supporting_quotes: [
            { text: 'q', speaker: { speaker_id: 'nested' }, speaker_id: 'flat' },
          ],
        },
      ],
    })
    expect(ex.insightRows[0].supporting_quotes![0].speaker_id).toBe('nested')
  })
})

// ── leaderboardRows: additional branches ──

describe('leaderboardRows (extra branches)', () => {
  it('returns empty when last has error', () => {
    const ex = useExploreStore()
    ex.last = { kind: 'explore', error: 'boom' }
    expect(ex.leaderboardRows).toEqual([])
  })

  it('returns empty when source object is missing', () => {
    const ex = useExploreStore()
    ex.last = { kind: 'explore' }
    expect(ex.leaderboardRows).toEqual([])
  })

  it('returns empty when topics is not an array', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ topics: { not: 'array' } })
    expect(ex.leaderboardRows).toEqual([])
  })

  it('returns empty when topics is an empty array', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ topics: [] })
    expect(ex.leaderboardRows).toEqual([])
  })

  it('keeps entries with only a topic_id (no label) and defaults label to empty', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ topics: [{ topic_id: 'only-id' }] })
    expect(ex.leaderboardRows).toHaveLength(1)
    expect(ex.leaderboardRows[0]).toMatchObject({ topic_id: 'only-id', label: '' })
  })

  it('keeps entries with only a label and defaults topic_id to empty', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ topics: [{ label: 'only-label' }] })
    expect(ex.leaderboardRows[0]).toMatchObject({ topic_id: '', label: 'only-label' })
  })

  it('filters out null entries', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ topics: [null, { topic_id: 'ok' }] })
    expect(ex.leaderboardRows).toHaveLength(1)
  })
})

// ── summaryBlock: additional branches ──

describe('summaryBlock (extra branches)', () => {
  it('returns null when last has error', () => {
    const ex = useExploreStore()
    ex.last = { kind: 'explore', error: 'x' }
    expect(ex.summaryBlock).toBeNull()
  })

  it('returns null when source object is missing', () => {
    const ex = useExploreStore()
    ex.last = { kind: 'explore' }
    expect(ex.summaryBlock).toBeNull()
  })

  it('returns null when summary is not an object', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ summary: 'nope' })
    expect(ex.summaryBlock).toBeNull()
  })

  it('returns a block when only episodes_searched is non-zero', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      summary: { insight_count: 0, grounded_insight_count: 0 },
      episodes_searched: 3,
    })
    expect(ex.summaryBlock).toMatchObject({ episodes_searched: 3 })
  })

  it('returns a block when only grounded count is non-zero', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ summary: { grounded_insight_count: 2 } })
    expect(ex.summaryBlock).toMatchObject({ grounded_insight_count: 2 })
  })

  it('defaults non-numeric summary fields to 0', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({
      summary: {
        insight_count: 5,
        quote_count: 'x',
        episode_count: null,
        speaker_count: undefined,
        topic_count: 'nope',
      },
    })
    const s = ex.summaryBlock!
    expect(s.quote_count).toBe(0)
    expect(s.episode_count).toBe(0)
    expect(s.speaker_count).toBe(0)
    expect(s.topic_count).toBe(0)
  })

  it('reads summary from the natural_language answer', () => {
    const ex = useExploreStore()
    ex.last = nlBody({ summary: { insight_count: 4 }, episodes_searched: 1 })
    expect(ex.summaryBlock).toMatchObject({ insight_count: 4, episodes_searched: 1 })
  })
})

// ── topSpeakers: additional branches ──

describe('topSpeakers (extra branches)', () => {
  it('returns empty when last has error', () => {
    const ex = useExploreStore()
    ex.last = { kind: 'explore', error: 'x' }
    expect(ex.topSpeakers).toEqual([])
  })

  it('returns empty when source object is missing', () => {
    const ex = useExploreStore()
    ex.last = { kind: 'explore' }
    expect(ex.topSpeakers).toEqual([])
  })

  it('returns empty when top_speakers is not an array', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ top_speakers: 'nope' })
    expect(ex.topSpeakers).toEqual([])
  })

  it('returns empty when top_speakers is empty', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ top_speakers: [] })
    expect(ex.topSpeakers).toEqual([])
  })

  it('filters out null entries', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ top_speakers: [null, { speaker_id: 's1' }] })
    expect(ex.topSpeakers).toHaveLength(1)
  })

  it('coerces a non-string name to null', () => {
    const ex = useExploreStore()
    ex.last = exploreBody({ top_speakers: [{ speaker_id: 's1', name: 42 }] })
    expect(ex.topSpeakers[0].name).toBeNull()
  })

  it('reads speakers from the natural_language answer', () => {
    const ex = useExploreStore()
    ex.last = nlBody({ top_speakers: [{ speaker_id: 'nl1', name: 'X' }] })
    expect(ex.topSpeakers).toHaveLength(1)
    expect(ex.topSpeakers[0].speaker_id).toBe('nl1')
  })
})

// ── runFilteredExplore ──

describe('runFilteredExplore', () => {
  it('sets an error and bails when the corpus path is blank', async () => {
    const ex = useExploreStore()
    await ex.runFilteredExplore('   ')
    expect(ex.error).toBe('Set corpus root first.')
    expect(ex.last).toBeNull()
    expect(ex.loading).toBe(false)
    expect(mockFiltered).not.toHaveBeenCalled()
  })

  it('stores the response and toggles loading off on success', async () => {
    const ex = useExploreStore()
    const body = exploreBody({ insights: [{ insight_id: 'i', text: 'ok' }] })
    mockFiltered.mockResolvedValue(body)
    await ex.runFilteredExplore('/corpus')
    expect(ex.last).toEqual(body)
    expect(ex.error).toBeNull()
    expect(ex.loading).toBe(false)
  })

  it('trims the corpus path before calling the api', async () => {
    const ex = useExploreStore()
    mockFiltered.mockResolvedValue(exploreBody({}))
    await ex.runFilteredExplore('  /corpus  ')
    expect(mockFiltered).toHaveBeenCalledWith('/corpus', expect.any(Object))
  })

  it('forwards filter values, parsing a finite minConfidence', async () => {
    const ex = useExploreStore()
    ex.filters.topic = 'econ'
    ex.filters.speaker = 'alice'
    ex.filters.groundedOnly = true
    ex.filters.minConfidence = '0.75'
    ex.filters.sortBy = 'time'
    ex.filters.limit = 25
    ex.filters.strict = true
    mockFiltered.mockResolvedValue(exploreBody({}))
    await ex.runFilteredExplore('/corpus')
    expect(mockFiltered).toHaveBeenCalledWith('/corpus', {
      topic: 'econ',
      speaker: 'alice',
      groundedOnly: true,
      minConfidence: 0.75,
      sortBy: 'time',
      limit: 25,
      strict: true,
    })
  })

  it('passes minConfidence as null when blank', async () => {
    const ex = useExploreStore()
    ex.filters.minConfidence = '   '
    mockFiltered.mockResolvedValue(exploreBody({}))
    await ex.runFilteredExplore('/corpus')
    expect(mockFiltered.mock.calls[0][1].minConfidence).toBeNull()
  })

  it('passes minConfidence as null when not a finite number', async () => {
    const ex = useExploreStore()
    ex.filters.minConfidence = 'abc'
    mockFiltered.mockResolvedValue(exploreBody({}))
    await ex.runFilteredExplore('/corpus')
    expect(mockFiltered.mock.calls[0][1].minConfidence).toBeNull()
  })

  it('emits a posthog event with summary metrics on success', async () => {
    const ex = useExploreStore()
    ex.filters.topic = 't'
    ex.filters.minConfidence = '0.5'
    mockFiltered.mockResolvedValue(
      exploreBody({ summary: { insight_count: 9 }, episodes_searched: 4 }),
    )
    await ex.runFilteredExplore('/corpus')
    expect(mockCapture).toHaveBeenCalledWith(
      'explore_filtered_run',
      expect.objectContaining({
        has_topic_filter: true,
        has_speaker_filter: false,
        has_min_confidence: true,
        insight_count: 9,
        episodes_searched: 4,
      }),
    )
  })

  it('reports null metrics when the response is not an explore body', async () => {
    const ex = useExploreStore()
    mockFiltered.mockResolvedValue(nlBody({ summary: { insight_count: 9 } }))
    await ex.runFilteredExplore('/corpus')
    expect(mockCapture).toHaveBeenCalledWith(
      'explore_filtered_run',
      expect.objectContaining({ insight_count: null, episodes_searched: null }),
    )
  })

  it('captures the error message when the api rejects with an Error', async () => {
    const ex = useExploreStore()
    mockFiltered.mockRejectedValue(new Error('network down'))
    await ex.runFilteredExplore('/corpus')
    expect(ex.error).toBe('network down')
    expect(ex.loading).toBe(false)
    expect(mockCapture).not.toHaveBeenCalled()
  })

  it('stringifies a non-Error rejection', async () => {
    const ex = useExploreStore()
    mockFiltered.mockRejectedValue('plain string failure')
    await ex.runFilteredExplore('/corpus')
    expect(ex.error).toBe('plain string failure')
  })

  it('clears a prior error and result at the start of a run', async () => {
    const ex = useExploreStore()
    ex.error = 'old'
    ex.last = exploreBody({ insights: [] })
    mockFiltered.mockResolvedValue(exploreBody({ summary: { insight_count: 1 } }))
    await ex.runFilteredExplore('/corpus')
    expect(ex.error).toBeNull()
  })

  it('drops a stale success when a newer run superseded it', async () => {
    const ex = useExploreStore()
    let resolveFirst!: (b: ExploreApiBody) => void
    mockFiltered.mockImplementationOnce(
      () => new Promise<ExploreApiBody>((r) => (resolveFirst = r)),
    )
    mockFiltered.mockResolvedValueOnce(exploreBody({ summary: { insight_count: 2 } }))
    const p1 = ex.runFilteredExplore('/corpus')
    const p2 = ex.runFilteredExplore('/corpus')
    await p2
    resolveFirst(exploreBody({ summary: { insight_count: 1 } }))
    await p1
    // The newer run's result wins; the stale one neither overwrites it nor captures.
    expect(ex.summaryBlock!.insight_count).toBe(2)
    expect(mockCapture).toHaveBeenCalledTimes(1)
  })

  it('drops a stale error when a newer run superseded it', async () => {
    const ex = useExploreStore()
    let rejectFirst!: (e: unknown) => void
    mockFiltered.mockImplementationOnce(
      () => new Promise<ExploreApiBody>((_, rej) => (rejectFirst = rej)),
    )
    mockFiltered.mockResolvedValueOnce(exploreBody({ summary: { insight_count: 5 } }))
    const p1 = ex.runFilteredExplore('/corpus')
    const p2 = ex.runFilteredExplore('/corpus')
    await p2
    rejectFirst(new Error('stale boom'))
    await p1
    expect(ex.error).toBeNull()
    expect(ex.summaryBlock!.insight_count).toBe(5)
  })
})

// ── runNaturalLanguage ──

describe('runNaturalLanguage', () => {
  it('sets an error and bails when the corpus path is blank', async () => {
    const ex = useExploreStore()
    ex.nlQuestion = 'a question'
    await ex.runNaturalLanguage('  ')
    expect(ex.error).toBe('Set corpus root first.')
    expect(mockNL).not.toHaveBeenCalled()
  })

  it('sets an error and bails when the question is blank', async () => {
    const ex = useExploreStore()
    ex.nlQuestion = '   '
    await ex.runNaturalLanguage('/corpus')
    expect(ex.error).toBe('Enter a question.')
    expect(ex.last).toBeNull()
    expect(mockNL).not.toHaveBeenCalled()
  })

  it('calls the api with the trimmed question and forwards limit/strict', async () => {
    const ex = useExploreStore()
    ex.nlQuestion = '  what happened?  '
    ex.filters.limit = 10
    ex.filters.strict = true
    mockNL.mockResolvedValue(nlBody({ insights: [] }))
    await ex.runNaturalLanguage('/corpus')
    expect(mockNL).toHaveBeenCalledWith('/corpus', 'what happened?', {
      limit: 10,
      strict: true,
    })
  })

  it('stores the response and emits a posthog event on success', async () => {
    const ex = useExploreStore()
    ex.nlQuestion = 'hello?'
    const body = nlBody({ insights: [{ insight_id: 'x', text: 't' }] })
    mockNL.mockResolvedValue(body)
    await ex.runNaturalLanguage('/corpus')
    expect(ex.last).toEqual(body)
    expect(ex.loading).toBe(false)
    expect(mockCapture).toHaveBeenCalledWith(
      'explore_natural_language_run',
      expect.objectContaining({ question_length: 6 }),
    )
  })

  it('captures the error message when the api rejects with an Error', async () => {
    const ex = useExploreStore()
    ex.nlQuestion = 'q'
    mockNL.mockRejectedValue(new Error('nl failed'))
    await ex.runNaturalLanguage('/corpus')
    expect(ex.error).toBe('nl failed')
    expect(ex.loading).toBe(false)
  })

  it('stringifies a non-Error rejection', async () => {
    const ex = useExploreStore()
    ex.nlQuestion = 'q'
    mockNL.mockRejectedValue({ code: 500 })
    await ex.runNaturalLanguage('/corpus')
    expect(ex.error).toBe('[object Object]')
  })

  it('drops a stale success when a newer run superseded it', async () => {
    const ex = useExploreStore()
    ex.nlQuestion = 'q'
    let resolveFirst!: (b: ExploreApiBody) => void
    mockNL.mockImplementationOnce(
      () => new Promise<ExploreApiBody>((r) => (resolveFirst = r)),
    )
    mockNL.mockResolvedValueOnce(nlBody({ summary: { insight_count: 8 } }))
    const p1 = ex.runNaturalLanguage('/corpus')
    const p2 = ex.runNaturalLanguage('/corpus')
    await p2
    resolveFirst(nlBody({ summary: { insight_count: 1 } }))
    await p1
    expect(ex.summaryBlock!.insight_count).toBe(8)
    expect(mockCapture).toHaveBeenCalledTimes(1)
  })

  it('drops a stale error when a newer run superseded it', async () => {
    const ex = useExploreStore()
    ex.nlQuestion = 'q'
    let rejectFirst!: (e: unknown) => void
    mockNL.mockImplementationOnce(
      () => new Promise<ExploreApiBody>((_, rej) => (rejectFirst = rej)),
    )
    mockNL.mockResolvedValueOnce(nlBody({ summary: { insight_count: 6 } }))
    const p1 = ex.runNaturalLanguage('/corpus')
    const p2 = ex.runNaturalLanguage('/corpus')
    await p2
    rejectFirst(new Error('stale'))
    await p1
    expect(ex.error).toBeNull()
    expect(ex.summaryBlock!.insight_count).toBe(6)
  })
})
