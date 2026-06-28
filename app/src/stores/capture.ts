/**
 * Capture store (Pinia ↔ /api/app/highlights) — the P2 "mark this moment" surface (PRD-040).
 * Mirrors the favorites store: auth-gated (empty + no-op signed out), every mutation persists
 * and reconciles from the server response (no optimistic drift). Holds the signed-in user's
 * highlights; the Library Highlights view (#1117) reads the same store.
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
import type { Highlight, Note, Segment } from '../services/types'
import type { SubRange } from '../player/transcriptCapture'

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
    /** One-tap "mark this moment" at a content-time position (seconds). */
    async captureMoment(slug: string, contentSeconds: number, speaker?: string | null): Promise<void> {
      try {
        const h = await createHighlight({
          episode_slug: slug,
          kind: 'moment',
          start_ms: ms(contentSeconds),
          speaker: speaker ?? null,
        })
        this.highlights = [...this.highlights, h]
        this.loaded = true
      } catch {
        /* signed out / transient — next load reconciles */
      }
    },
    /**
     * Save a transcript line as a span highlight. With a `sub` selection (PRD-040 FR1.2) it captures
     * that exact phrase and always *adds* (a line can hold several phrase highlights). Without one it
     * captures the whole line and *toggles* — a second tap on an already-saved line removes it.
     */
    async captureSegment(slug: string, seg: Segment, sub?: SubRange | null): Promise<void> {
      try {
        if (sub) {
          const h = await createHighlight({
            episode_slug: slug,
            kind: 'span',
            start_ms: ms(seg.start),
            end_ms: ms(seg.end),
            segment_ids: [seg.id],
            char_start: sub.char_start,
            char_end: sub.char_end,
            quote_text: sub.quote_text,
            speaker: seg.speaker,
          })
          this.highlights = [...this.highlights, h]
          this.loaded = true
          return
        }
        // whole line: toggle (match the existing whole-line span by its verbatim text)
        const existing = this.highlights.find(
          (h) =>
            h.kind === 'span' && h.segment_ids.includes(seg.id) && h.quote_text === seg.text,
        )
        if (existing) {
          this._sync(await deleteHighlight(existing.id))
          return
        }
        const h = await createHighlight({
          episode_slug: slug,
          kind: 'span',
          start_ms: ms(seg.start),
          end_ms: ms(seg.end),
          segment_ids: [seg.id],
          char_start: 0,
          char_end: seg.text.length,
          quote_text: seg.text,
          speaker: seg.speaker,
        })
        this.highlights = [...this.highlights, h]
        this.loaded = true
      } catch {
        /* signed out / transient */
      }
    },
    /** Save a grounded insight as an insight highlight (toggles off if already saved). */
    async captureInsight(
      slug: string,
      insight: { id: string; text: string; start_ms?: number | null },
    ): Promise<void> {
      const existing = this.highlights.find((h) => h.source_insight_id === insight.id)
      try {
        if (existing) {
          this._sync(await deleteHighlight(existing.id))
          return
        }
        const h = await createHighlight({
          episode_slug: slug,
          kind: 'insight',
          source_insight_id: insight.id,
          quote_text: insight.text,
          start_ms: insight.start_ms ?? null,
        })
        this.highlights = [...this.highlights, h]
        this.loaded = true
      } catch {
        /* signed out / transient */
      }
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
      try {
        this._sync(await deleteHighlight(id))
        this.notes = this.notes.filter((n) => !(n.target === 'highlight' && n.target_id === id))
      } catch {
        /* signed out / transient */
      }
    },
    /** Attach a note to a target (highlight / insight / episode). */
    async addNote(target: Note['target'], targetId: string, text: string): Promise<void> {
      try {
        const n = await createNote({ target, target_id: targetId, text })
        this.notes = [...this.notes, n]
      } catch {
        /* signed out / transient */
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
      try {
        this.notes = await deleteNote(id)
      } catch {
        /* signed out / transient */
      }
    },
  },
})
