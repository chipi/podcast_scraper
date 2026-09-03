/**
 * Capture store (Pinia ↔ /api/app/highlights) — the P2 "mark this moment" surface (PRD-040).
 * Mirrors the favorites store: auth-gated (empty + no-op signed out), every mutation persists
 * and reconciles from the server response (no optimistic drift). Holds the signed-in user's
 * highlights; the Library Highlights view (#1117) reads the same store.
 *
 * Offline (#1925): a capture is shown immediately under a CLIENT-minted id and queued in the
 * outbox. That id is the whole mechanism — the optimistic row and the row the server eventually
 * stores are the same row, so a replay cannot duplicate it. Before this, a capture made with no
 * network was simply lost, silently, which is the worst thing this store can do.
 */
import { defineStore } from 'pinia'
import {
  createHighlight,
  createNote,
  deleteHighlight,
  deleteNote,
  getHighlights,
  getNotes,
  patchHighlight,
  patchNote,
} from '../services/api'
import { newCaptureId } from '../services/captureIds'
import { identityChangedSince, identityEpoch } from '../services/identity'
import { enqueue, isPermanent, withdrawPendingCreate } from '../services/outbox'
import type { Highlight, HighlightCreate, Note, NoteCreate } from '../services/types'
import type { ParagraphSpan } from '../player/transcriptCapture'

interface CaptureState {
  highlights: Highlight[]
  notes: Note[]
  loaded: boolean
}

/** Seconds → integer milliseconds (the highlight anchor unit). */
function ms(seconds: number): number {
  return Math.max(0, Math.round(seconds * 1000))
}

export const useCaptureStore = defineStore('capture', {
  state: (): CaptureState => ({ highlights: [], notes: [], loaded: false }),
  getters: {
    /** Highlights for one episode (newest-last as stored). */
    forEpisode:
      (s) =>
      (slug: string): Highlight[] =>
        s.highlights.filter((h) => h.episode_slug === slug),
    /** Notes attached to a given target (highlight / insight / episode). */
    notesFor:
      (s) =>
      (target: string, targetId: string): Note[] =>
        s.notes.filter((n) => n.target === target && n.target_id === targetId),
    /** Source-insight ids already saved as insight highlights (drives the insight save toggle). */
    savedInsightIds: (s): Set<string> =>
      new Set(s.highlights.filter((h) => h.source_insight_id).map((h) => h.source_insight_id!)),
    /** Segment ids already captured as a span (drives the transcript-line save toggle). */
    savedSegmentIds: (s): Set<string> => {
      const out = new Set<string>()
      for (const h of s.highlights) for (const sid of h.segment_ids) out.add(sid)
      return out
    },
    count: (s): number => s.highlights.length,
  },
  actions: {
    async load(): Promise<void> {
      const [highlights, notes] = await Promise.all([getHighlights(), getNotes()])
      this.highlights = highlights
      this.notes = notes
      this.loaded = true
    },
    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.load()
    },
    /** Replace local state from a server list (after a mutation). */
    _sync(items: Highlight[]): void {
      this.highlights = items
      this.loaded = true
    },
    /**
     * Capture one highlight, surviving an offline moment (#1925).
     *
     * The id is minted HERE, before the request, which is the whole mechanism: the row we show
     * optimistically and the row the server eventually stores are the same row, so a replay cannot
     * duplicate it and a later load reconciles without a flicker.
     *
     * A server REFUSAL (4xx) is an answer — the capture is dropped and `false` reported, because
     * callers announce "Saved" to screen readers and must not say it when nothing was saved. A
     * transport failure is not an answer: the highlight stays on screen and the write is queued.
     */
    async _capture(body: HighlightCreate): Promise<boolean> {
      const generation = identityEpoch()
      const client_id = newCaptureId('h')
      const withId: HighlightCreate = { ...body, client_id }
      // Shown immediately, and shaped like the server's row so nothing downstream has to know
      // which of the two it is holding.
      const optimistic: Highlight = {
        segment_ids: [],
        ...withId,
        id: client_id,
        created_at: Math.floor(Date.now() / 1000),
        graph_refs: [],
      } as unknown as Highlight
      this.highlights = [...this.highlights, optimistic]
      this.loaded = true
      try {
        const saved = await createHighlight(withId)
        // A response landing after an account switch belongs to nobody now (advisor 1.4).
        if (identityChangedSince(generation)) return false
        this.highlights = this.highlights.map((h) => (h.id === client_id ? saved : h))
        return true
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return false
        // Only a REFUSAL discards the capture. A 502 or a dead socket is not an answer, and
        // dropping the user's highlight on one would lose it for good.
        if (isPermanent(err)) {
          this.highlights = this.highlights.filter((h) => h.id !== client_id)
          return false
        }
        enqueue({ op: 'highlight.create', body: withId })
        return true
      }
    },
    /** Delete a highlight, queuing the delete when the request never lands. */
    async _uncapture(id: string): Promise<boolean> {
      const generation = identityEpoch()
      const prev = this.highlights
      // A capture made offline has never reached the server, so deleting it there would 404 —
      // and a 404 is "permanent", which would RESTORE the row and leave it undeletable until the
      // outbox happened to create it (advisor 3.1). The queued create is simply withdrawn
      // instead: create-then-delete offline collapses to nothing, which is what the user did.
      const wasPendingCreate = withdrawPendingCreate(id)
      this.highlights = this.highlights.filter((h) => h.id !== id)
      if (wasPendingCreate) return true
      try {
        const items = await deleteHighlight(id)
        if (identityChangedSince(generation)) return false
        this._sync(items)
        return true
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return false
        if (isPermanent(err)) {
          this.highlights = prev
          return false
        }
        enqueue({ op: 'highlight.remove', id })
        return true
      }
    },
    /** One-tap "mark this moment" at a content-time position (seconds). */
    async captureMoment(
      slug: string,
      contentSeconds: number,
      speaker?: string | null,
    ): Promise<boolean> {
      // Swallowed so `void capture.x()` can never raise, but the OUTCOME is reported: callers
      // were announcing "Saved" to screen readers unconditionally, so a failed POST told a blind
      // user their highlight was stored when nothing was (#1590 review, S8). Offline is now a
      // SUCCESS by that measure — the capture is kept and replayed (#1925).
      return this._capture({
        episode_slug: slug,
        kind: 'moment',
        start_ms: ms(contentSeconds),
        speaker: speaker ?? null,
      })
    },
    /**
     * Save a transcript span — a selected phrase or a whole paragraph (PRD-040 FR1.2). The span is
     * pre-computed (`spanFromParagraph`). An identical span (same verbatim text over the same
     * segments) *toggles* — a second save removes it; otherwise it *adds*.
     */
    async captureSpan(slug: string, span: ParagraphSpan): Promise<boolean> {
      const key = span.segment_ids.join(',')
      const existing = this.highlights.find(
        (h) =>
          h.kind === 'span' && h.quote_text === span.quote_text && h.segment_ids.join(',') === key,
      )
      if (existing) return this._uncapture(existing.id)
      return this._capture({ episode_slug: slug, kind: 'span', ...span })
    },
    /** Save a grounded insight as an insight highlight (toggles off if already saved). */
    async captureInsight(
      slug: string,
      insight: { id: string; text: string; start_ms?: number | null },
    ): Promise<boolean> {
      const existing = this.highlights.find((h) => h.source_insight_id === insight.id)
      if (existing) return this._uncapture(existing.id)
      return this._capture({
        episode_slug: slug,
        kind: 'insight',
        source_insight_id: insight.id,
        quote_text: insight.text,
        start_ms: insight.start_ms ?? null,
      })
    },
    /** Set (or clear, with null) a highlight's colour token. */
    async setColor(id: string, color: string | null): Promise<void> {
      try {
        const updated = await patchHighlight(id, { color })
        this.highlights = this.highlights.map((h) => (h.id === id ? updated : h))
      } catch {
        /* signed out / transient */
      }
    },
    /** Remove a highlight by id (and any notes that targeted it, locally). */
    async remove(id: string): Promise<void> {
      await this._uncapture(id)
      this.notes = this.notes.filter((n) => !(n.target === 'highlight' && n.target_id === id))
    },
    /** Attach a note to a target (highlight / insight / episode). Survives offline (#1925). */
    async addNote(target: Note['target'], targetId: string, text: string): Promise<void> {
      const generation = identityEpoch()
      const client_id = newCaptureId('n')
      const body: NoteCreate = { target, target_id: targetId, text, client_id }
      const now = Math.floor(Date.now() / 1000)
      this.notes = [...this.notes, { ...body, id: client_id, created_at: now, updated_at: now }]
      try {
        const saved = await createNote(body)
        if (identityChangedSince(generation)) return
        this.notes = this.notes.map((n) => (n.id === client_id ? saved : n))
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return
        // A refusal drops it; anything else keeps it and queues the write.
        if (isPermanent(err)) this.notes = this.notes.filter((n) => n.id !== client_id)
        else enqueue({ op: 'note.create', body })
      }
    },
    /** Edit a note's text. */
    async editNote(id: string, text: string): Promise<void> {
      try {
        const updated = await patchNote(id, text)
        this.notes = this.notes.map((n) => (n.id === id ? updated : n))
      } catch {
        /* signed out / transient */
      }
    },
    /** Remove a note by id. */
    async removeNote(id: string): Promise<void> {
      const generation = identityEpoch()
      const prev = this.notes
      // Same withdrawal as _uncapture: a note created offline and removed before it flushed has
      // no server row to delete, and the 404 would restore it permanently (advisor 3.1).
      const wasPendingCreate = withdrawPendingCreate(id)
      this.notes = this.notes.filter((n) => n.id !== id)
      if (wasPendingCreate) return
      try {
        const items = await deleteNote(id)
        if (identityChangedSince(generation)) return
        this.notes = items
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return
        if (isPermanent(err)) this.notes = prev
        else enqueue({ op: 'note.remove', id })
      }
    },
  },
})
