import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import type { Highlight } from '../services/types'
import { useCaptureStore } from './capture'

function hl(over: Partial<Highlight> = {}): Highlight {
  return {
    id: 'h1',
    episode_slug: 'show-ep01',
    kind: 'moment',
    start_ms: 1000,
    end_ms: null,
    char_start: null,
    char_end: null,
    segment_ids: [],
    quote_text: null,
    speaker: null,
    source_insight_id: null,
    color: null,
    created_at: 1,
    anchor_status: null,
    ...over,
  }
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(api, 'getNotes').mockResolvedValue([])
})
afterEach(() => vi.restoreAllMocks())

describe('capture store', () => {
  it('load() pulls highlights from the API', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([hl(), hl({ id: 'h2' })])
    const c = useCaptureStore()
    await c.load()
    expect(c.count).toBe(2)
    expect(c.loaded).toBe(true)
  })

  it('captureMoment() appends a moment highlight', async () => {
    const created = hl({ id: 'm1', kind: 'moment', start_ms: 42_000 })
    vi.spyOn(api, 'createHighlight').mockResolvedValue(created)
    const c = useCaptureStore()
    await c.captureMoment('show-ep01', 42, 'person:guest')
    expect(api.createHighlight).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'moment', episode_slug: 'show-ep01', start_ms: 42_000 }),
    )
    expect(c.forEpisode('show-ep01')).toHaveLength(1)
  })

  it('captureSpan() saves a span, then toggles it off when the same span is saved again', async () => {
    const span = {
      start_ms: 10_000, end_ms: 14_000, segment_ids: ['s5'], char_start: 0, char_end: 7,
      quote_text: 'a quote', speaker: 'person:g',
    }
    const saved = hl({ id: 'sp1', kind: 'span', segment_ids: ['s5'], quote_text: 'a quote' })
    vi.spyOn(api, 'createHighlight').mockResolvedValue(saved)
    vi.spyOn(api, 'deleteHighlight').mockResolvedValue([])
    const c = useCaptureStore()

    await c.captureSpan('show-ep01', span)
    expect(api.createHighlight).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'span', segment_ids: ['s5'], quote_text: 'a quote' }),
    )
    expect(c.savedSegmentIds.has('s5')).toBe(true)

    // saving the identical span (same quote + segments) removes it
    await c.captureSpan('show-ep01', span)
    expect(api.deleteHighlight).toHaveBeenCalledWith('sp1')
    expect(c.savedSegmentIds.has('s5')).toBe(false)
  })

  it('captureSpan() adds distinct spans (a phrase ≠ the whole paragraph) without toggling', async () => {
    let n = 0
    vi.spyOn(api, 'createHighlight').mockImplementation(
      async (b) => hl({ id: `sp${++n}`, kind: 'span', segment_ids: b.segment_ids, quote_text: b.quote_text }),
    )
    const del = vi.spyOn(api, 'deleteHighlight')
    const c = useCaptureStore()
    const whole = {
      start_ms: 10_000, end_ms: 14_000, segment_ids: ['s5', 's6'], char_start: 0, char_end: 30,
      quote_text: 'deep sleep consolidates memory', speaker: 'person:g',
    }
    const phrase = { ...whole, segment_ids: ['s5'], char_start: 5, char_end: 10, quote_text: 'sleep' }
    await c.captureSpan('show-ep01', whole)
    await c.captureSpan('show-ep01', phrase) // different quote/segments → a second, independent span
    expect(del).not.toHaveBeenCalled()
    expect(c.forEpisode('show-ep01')).toHaveLength(2)
  })

  it('captureInsight() saves by source_insight_id, then toggles off', async () => {
    const ins = hl({ id: 'i1', kind: 'insight', source_insight_id: 'gi-3', quote_text: 'claim' })
    vi.spyOn(api, 'createHighlight').mockResolvedValue(ins)
    vi.spyOn(api, 'deleteHighlight').mockResolvedValue([])
    const c = useCaptureStore()

    await c.captureInsight('show-ep01', { id: 'gi-3', text: 'claim', start_ms: 5000 })
    expect(c.savedInsightIds.has('gi-3')).toBe(true)

    await c.captureInsight('show-ep01', { id: 'gi-3', text: 'claim' })
    expect(api.deleteHighlight).toHaveBeenCalledWith('i1')
    expect(c.savedInsightIds.has('gi-3')).toBe(false)
  })

  it('swallows API errors (signed out) without throwing', async () => {
    vi.spyOn(api, 'createHighlight').mockRejectedValue(new Error('401'))
    const c = useCaptureStore()
    await expect(c.captureMoment('show-ep01', 1)).resolves.toBeUndefined()
    expect(c.count).toBe(0)
  })

  it('load() pulls highlights and notes together', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([hl()])
    vi.spyOn(api, 'getNotes').mockResolvedValue([
      { id: 'n1', target: 'highlight', target_id: 'h1', text: 'note', created_at: 1, updated_at: 1 },
    ])
    const c = useCaptureStore()
    await c.load()
    expect(c.notesFor('highlight', 'h1')).toHaveLength(1)
  })

  it('addNote / editNote / removeNote keep local notes in sync', async () => {
    const c = useCaptureStore()
    vi.spyOn(api, 'createNote').mockResolvedValue({
      id: 'n1', target: 'highlight', target_id: 'h1', text: 'first', created_at: 1, updated_at: 1,
    })
    await c.addNote('highlight', 'h1', 'first')
    expect(c.notesFor('highlight', 'h1')[0].text).toBe('first')

    vi.spyOn(api, 'patchNote').mockResolvedValue({
      id: 'n1', target: 'highlight', target_id: 'h1', text: 'edited', created_at: 1, updated_at: 2,
    })
    await c.editNote('n1', 'edited')
    expect(c.notesFor('highlight', 'h1')[0].text).toBe('edited')

    vi.spyOn(api, 'deleteNote').mockResolvedValue([])
    await c.removeNote('n1')
    expect(c.notes).toHaveLength(0)
  })

  it('setColor patches the colour and updates local state', async () => {
    const c = useCaptureStore()
    c.highlights = [hl({ id: 'h1', color: null })]
    vi.spyOn(api, 'patchHighlight').mockResolvedValue(hl({ id: 'h1', color: 'amber' }))
    await c.setColor('h1', 'amber')
    expect(api.patchHighlight).toHaveBeenCalledWith('h1', { color: 'amber' })
    expect(c.highlights[0].color).toBe('amber')
  })

  it('removing a highlight also drops its local notes', async () => {
    const c = useCaptureStore()
    c.highlights = [hl({ id: 'h1' })]
    c.notes = [
      { id: 'n1', target: 'highlight', target_id: 'h1', text: 'x', created_at: 1, updated_at: 1 },
    ]
    vi.spyOn(api, 'deleteHighlight').mockResolvedValue([])
    await c.remove('h1')
    expect(c.notes).toHaveLength(0)
  })
})
