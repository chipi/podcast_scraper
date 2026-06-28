/**
 * Capture store (Pinia ↔ /api/app/highlights) — the P2 "mark this moment" surface (PRD-040).
 * Mirrors the favorites store: auth-gated (empty + no-op signed out), every mutation persists
 * and reconciles from the server response (no optimistic drift). Holds the signed-in user's
 * highlights; the Library Highlights view (#1117) reads the same store.
 */
import { defineStore } from 'pinia'
import {
  createHighlight,
  deleteHighlight,
  getHighlights,
} from '../services/api'
import type { Highlight, Segment } from '../services/types'

interface CaptureState {
  highlights: Highlight[]
  loaded: boolean
}

/** Seconds → integer milliseconds (the highlight anchor unit). */
function ms(seconds: number): number {
  return Math.max(0, Math.round(seconds * 1000))
}

export const useCaptureStore = defineStore('capture', {
  state: (): CaptureState => ({ highlights: [], loaded: false }),
  getters: {
    /** Highlights for one episode (newest-last as stored). */
    forEpisode:
      (s) =>
      (slug: string): Highlight[] =>
        s.highlights.filter((h) => h.episode_slug === slug),
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
      this.highlights = await getHighlights()
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
    /** Save a transcript segment as a span highlight (toggles off if already saved). */
    async captureSegment(slug: string, seg: Segment): Promise<void> {
      const existing = this.highlights.find(
        (h) => h.kind === 'span' && h.segment_ids.includes(seg.id),
      )
      try {
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
    /** Remove a highlight by id. */
    async remove(id: string): Promise<void> {
      try {
        this._sync(await deleteHighlight(id))
      } catch {
        /* signed out / transient */
      }
    },
  },
})
