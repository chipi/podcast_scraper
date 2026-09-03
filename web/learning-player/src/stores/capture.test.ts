import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { ApiError } from '../services/api'
import * as outbox from '../services/outbox'
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

  it('never throws on an API error, but REPORTS the failure (S8)', async () => {
    // Swallowing the throw is deliberate — callers use `void capture.x()` and an unhandled
    // rejection would be worse. Swallowing the OUTCOME was the bug: callers announced "Saved" to
    // screen readers regardless, so a failed POST told the user their highlight was stored.
    // An ApiError, because that is what a REFUSAL is: the server answered, and the answer was no.
    // A request that never got an answer is a different case entirely — see the offline block.
    // A REFUSAL is 4xx-that-is-not-401/403: the server answered no. A 401 means the session
    // died and the capture is queued instead (advisor 1.1), which is a truthful "Marked".
    vi.spyOn(api, 'createHighlight').mockRejectedValue(new ApiError(404, 'gone'))
    const c = useCaptureStore()
    await expect(c.captureMoment('show-ep01', 1)).resolves.toBe(false)
    expect(c.count).toBe(0)
  })

  it('reports success so the caller can confirm truthfully', async () => {
    vi.spyOn(api, 'createHighlight').mockResolvedValue(hl())
    const c = useCaptureStore()
    await expect(c.captureMoment('show-ep01', 1)).resolves.toBe(true)
    expect(c.count).toBe(1)
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

/**
 * Capture offline (#1925). This was the last thing that silently lost user work: a highlight made
 * with no network vanished, and the caller announced "Marked" anyway. Client-minted ids let the
 * create sit in the outbox — a replay whose first response was lost cannot duplicate the row.
 */
describe('capture offline', () => {
  it('keeps the highlight on screen and queues the write', async () => {
    vi.spyOn(api, 'createHighlight').mockRejectedValue(new TypeError('Failed to fetch'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const c = useCaptureStore()

    await expect(c.captureMoment('show-ep01', 12)).resolves.toBe(true)
    expect(c.count).toBe(1)
    const queued = enqueue.mock.calls[0][0]
    expect(queued.op).toBe('highlight.create')
    // The row on screen and the row the server will store share ONE id — that is the mechanism.
    expect(queued.op === 'highlight.create' && queued.body.client_id).toBe(c.highlights[0].id)
  })

  it('does not lose a capture to a 502 — a bad gateway is not a refusal', async () => {
    vi.spyOn(api, 'createHighlight').mockRejectedValue(new ApiError(502, 'bad gateway'))
    vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const c = useCaptureStore()
    await expect(c.captureMoment('show-ep01', 12)).resolves.toBe(true)
    expect(c.count).toBe(1)
  })

  it('adopts the server row once the write lands, without duplicating', async () => {
    vi.spyOn(api, 'createHighlight').mockImplementation(async (body) =>
      hl({ id: body.client_id, quote_text: 'from the server' }),
    )
    const c = useCaptureStore()
    await c.captureMoment('show-ep01', 12)
    expect(c.count).toBe(1)
    expect(c.highlights[0].quote_text).toBe('from the server')
  })

  it('queues an offline note too, and keeps it visible', async () => {
    vi.spyOn(api, 'createNote').mockRejectedValue(new TypeError('Failed to fetch'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const c = useCaptureStore()

    await c.addNote('highlight', 'h1', 'a thought')
    expect(c.notes).toHaveLength(1)
    expect(c.notes[0].text).toBe('a thought')
    expect(enqueue.mock.calls[0][0].op).toBe('note.create')
  })

  it('drops a note the server REFUSES', async () => {
    vi.spyOn(api, 'createNote').mockRejectedValue(new ApiError(404, 'gone'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const c = useCaptureStore()
    await c.addNote('highlight', 'h1', 'a thought')
    expect(c.notes).toHaveLength(0)
    expect(enqueue).not.toHaveBeenCalled()
  })
})

/**
 * A capture created offline and undone before it flushed (advisor 3.1).
 *
 * The row exists on screen under a client-minted id and has never reached the server, so deleting
 * it there 404s — and a 404 is a refusal, which RESTORED the row. It was then undeletable until
 * the outbox happened to create it first.
 */
describe('undoing an offline capture', () => {
  beforeEach(() => outbox.__resetOutbox())

  it('withdraws the queued create instead of deleting a row the server never had', async () => {
    vi.spyOn(api, 'createHighlight').mockRejectedValue(new TypeError('Failed to fetch'))
    const del = vi.spyOn(api, 'deleteHighlight')
    const c = useCaptureStore()

    await c.captureMoment('show-ep01', 12)
    expect(c.count).toBe(1)
    const id = c.highlights[0].id

    await c.remove(id)
    expect(c.count).toBe(0)
    // No delete is attempted at all: there is nothing on the server to delete, and the 404 it
    // would return is what used to make the row come back.
    expect(del).not.toHaveBeenCalled()
  })

  it('withdraws the create but STILL queues the delete', async () => {
    // The create may have reached the server with only its RESPONSE lost — the ordinary offline
    // failure — in which case the row exists and withdrawing alone would orphan it forever
    // (advisor-2 #2). The flush drops a delete on 404, which is exactly the never-existed case,
    // so queuing it is safe in both worlds.
    vi.spyOn(api, 'createHighlight').mockRejectedValue(new TypeError('Failed to fetch'))
    const c = useCaptureStore()
    await c.captureMoment('show-ep01', 12)
    const id = c.highlights[0].id

    await c.remove(id)
    const queued = outbox.pendingWrites().filter((e) => e.action.op.startsWith('highlight'))
    expect(queued.map((e) => e.action.op)).toEqual(['highlight.remove'])
  })

  it('does the same for a note', async () => {
    vi.spyOn(api, 'createNote').mockRejectedValue(new TypeError('Failed to fetch'))
    const del = vi.spyOn(api, 'deleteNote')
    const c = useCaptureStore()

    await c.addNote('highlight', 'h1', 'a thought')
    expect(c.notes).toHaveLength(1)
    await c.removeNote(c.notes[0].id)

    expect(c.notes).toHaveLength(0)
    expect(del).not.toHaveBeenCalled()
  })

  it('still deletes normally when the capture DID reach the server', async () => {
    vi.spyOn(api, 'createHighlight').mockImplementation(async (body) => hl({ id: body.client_id }))
    const del = vi.spyOn(api, 'deleteHighlight').mockResolvedValue([])
    const c = useCaptureStore()

    await c.captureMoment('show-ep01', 12)
    await c.remove(c.highlights[0].id)
    expect(del).toHaveBeenCalled()
  })
})
