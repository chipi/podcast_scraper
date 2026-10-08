/**
 * Capture store (Pinia ↔ /api/app/highlights) — the P2 "mark this moment" surface (PRD-040).
 * Mirrors the favorites store: auth-gated (empty + no-op signed out), every mutation persists
 * and reconciles from the server response (no optimistic drift). Holds the signed-in user's
 * highlights; the Library Highlights view (#1117) reads the same store.
 *
 * Since 2026-10-08 it is a CACHE of what has been loaded, not the whole library: the player asks for
 * its episode (`ensureEpisode`), a note composer for its target (`ensureNotesFor`), and the Saved
 * and Boards lists read server PAGES (`getHighlightsPage` / `getNotesPage`) and `merge` them in.
 * `count` is the server's total. Every change moves `version`, so the paged lists refetch.
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
  getHighlightsPage,
  getNotes,
  patchHighlight,
  patchNote,
  unretireHighlight,
} from '../services/api'
import { newCaptureId } from '../services/captureIds'
import { serialWrites } from '../services/serialWrites'
import { hasArrayFields, readCached, writeCached } from '../services/contentCache'
import { identityChangedSince, identityEpoch } from '../services/identity'
import {
  enqueue,
  hasPendingHighlightCreate,
  hasPendingNoteCreate,
  isPermanent,
  updatePendingHighlightColor,
  updatePendingNoteText,
  withdrawPendingCreate,
} from '../services/outbox'
import type { Highlight, HighlightCreate, Note, NoteCreate } from '../services/types'
import type { ParagraphSpan } from '../player/transcriptCapture'

interface CaptureState {
  /** Every highlight loaded so far — the open episodes', and the rows the lists merged in. */
  highlights: Highlight[]
  /** Every note loaded so far. */
  notes: Note[]
  /** The server's count of ALL the user's highlights (null until asked). */
  total: number | null
  /** Moves on every change — the paged lists refetch. */
  version: number
  /** Episodes and note targets loaded, so `load()` can revalidate exactly those. */
  episodes: string[]
  noteTargets: string[]
  loaded: boolean
  /** Showing a cached copy the server has not confirmed this session. */
  stale: boolean
  /** Nothing fetched AND nothing cached — an unknown library, not an empty one. */
  unavailable: boolean
}

/** `a` wins over `b` on the same id; order: `b`'s rows that `a` does not replace, then `a`. */
function mergeById<T extends { id: string }>(a: T[], b: T[]): T[] {
  const ids = new Set(a.map((x) => x.id))
  return [...b.filter((x) => !ids.has(x.id)), ...a]
}

/** A row minted here (`newCaptureId`: hc_ / nc_) that the server has not answered for yet. */
function isLocalOnly(id: string): boolean {
  return /^[hn]c_/.test(id) && !serverIdFor.has(id)
}

function splitTarget(key: string): [string, string] {
  const i = key.indexOf('\u0000')
  return [key.slice(0, i), key.slice(i + 1)]
}

/** Seconds → integer milliseconds (the highlight anchor unit). */
function ms(seconds: number): number {
  return Math.max(0, Math.round(seconds * 1000))
}

/**
 * In-flight `load()` dedup (singleton store).
 *
 * NoteComposer self-hydrates via `onMounted → ensureLoaded()`, so its initial GET is in flight
 * exactly when a fast user adds a note. Without dedup, a second concurrent `ensureLoaded()` fired a
 * SECOND fetch whose late response overwrote the optimistic note with the server's (still noteless)
 * list — the note appeared, then silently vanished until the next reload. One shared promise means
 * `addNote`'s `await ensureLoaded()` waits for the SAME load, so its optimistic write lands after
 * the server list, never before it. Not in reactive state — a Promise does not belong there.
 */
let loadPromise: Promise<void> | null = null

/**
 * Save / unsave writes go through the shared serializer (see `services/serialWrites`), so a quick
 * save-then-unsave reaches the server in that order and the unsave's answer is the one that counts.
 *
 * Two facts the delete needs about the create it may be chasing: the id the SERVER gave the row
 * (the tap only knew the client id), and whether the server refused the create — then there is
 * nothing to delete. Before this, a tap-tap sent DELETE /highlights/<client id> while the create was
 * in flight; the server answered 404, the store read that as a refusal and put the row back, and the
 * create then landed — the insight the user unsaved stayed saved.
 */
const writes = serialWrites()
const serverIdFor = new Map<string, string>()
const refusedCreates = new Set<string>()

export const useCaptureStore = defineStore('capture', {
  state: (): CaptureState => ({
    highlights: [],
    notes: [],
    total: null,
    version: 0,
    episodes: [],
    noteTargets: [],
    loaded: false,
    stale: false,
    unavailable: false,
  }),
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
    /** How many highlights the user has — the server's total once asked, else what is loaded. */
    count: (s): number => s.total ?? s.highlights.length,
  },
  actions: {
    /**
     * How many highlights the user has (one row from the server), and a revalidation of whatever
     * episodes and note targets are already loaded. Cached per account (#1909): offline, the
     * cached copy stands and is marked stale; with nothing cached the library is UNKNOWN, not empty.
     *
     * Does not reject once it has recovered from cache: every caller treats this as fire-and-forget.
     */
    async load(): Promise<void> {
      // A load in flight across an account switch must not write A's captures into B's store, nor
      // cache them under B's namespace. A response that lands after the switch belongs to nobody.
      const generation = identityEpoch()
      try {
        const page = await writes.fresh(() => getHighlightsPage({ limit: 1, perEpisode: 1 }))
        if (identityChangedSince(generation)) return
        this.total = page.total
        this.loaded = true
        this.stale = false
        this.unavailable = false
        await Promise.all([
          ...this.episodes.map((slug) => this._fetchEpisode(slug)),
          ...this.noteTargets.map((key) => {
            const [target, id] = splitTarget(key)
            return this._fetchNotes(target, id)
          }),
        ])
        this._bump()
      } catch {
        if (identityChangedSince(generation)) return
        const cached = await readCached<{ highlights: Highlight[]; notes: Note[] }>(
          'captures',
          hasArrayFields('highlights', 'notes'),
        )
        if (identityChangedSince(generation)) return
        if (cached) {
          this.highlights = mergeById(cached.highlights, this.highlights)
          this.notes = mergeById(cached.notes, this.notes)
          this.loaded = true
          this.stale = true
          this.unavailable = false
          return
        }
        // Nothing fetched and nothing stored. NOT an empty account — the tab must say so rather
        // than render the "you have kept nothing yet" state over a library we simply cannot see.
        this.unavailable = true
      }
    },
    async ensureLoaded(): Promise<void> {
      if (this.loaded) return
      // Share ONE in-flight load (see `loadPromise`).
      if (!loadPromise) loadPromise = this.load().finally(() => (loadPromise = null))
      await loadPromise
    },
    /**
     * One episode's highlights — the player and its notes panel. Replaces that episode's loaded
     * rows with the server's, keeping a capture still on its way (outbox, or not yet answered).
     * Offline, the cached rows stand.
     */
    async ensureEpisode(slug: string): Promise<void> {
      if (!slug) return
      if (!this.episodes.includes(slug)) this.episodes = [...this.episodes, slug]
      await this._fetchEpisode(slug)
    },
    async _fetchEpisode(slug: string): Promise<void> {
      const generation = identityEpoch()
      try {
        const rows = await writes.fresh(() => getHighlights(slug))
        if (identityChangedSince(generation)) return
        const keep = this.highlights.filter(
          (h) => h.episode_slug !== slug || hasPendingHighlightCreate(h.id) || isLocalOnly(h.id),
        )
        this.highlights = mergeById(rows, keep)
        this.loaded = true
        this._cache()
      } catch {
        if (identityChangedSince(generation)) return
        const cached = await readCached<{ highlights: Highlight[]; notes: Note[] }>(
          'captures',
          hasArrayFields('highlights', 'notes'),
        )
        if (cached) {
          this.highlights = mergeById(
            cached.highlights.filter((h) => h.episode_slug === slug),
            this.highlights,
          )
          this.stale = true
        }
      }
    },
    /** Notes on one target — a note composer. Same rules as `ensureEpisode`. */
    async ensureNotesFor(target: string, targetId: string): Promise<void> {
      if (!targetId) return
      const key = `${target}\u0000${targetId}`
      if (!this.noteTargets.includes(key)) this.noteTargets = [...this.noteTargets, key]
      await this._fetchNotes(target, targetId)
    },
    async _fetchNotes(target: string, targetId: string): Promise<void> {
      const generation = identityEpoch()
      try {
        const rows = await writes.fresh(() => getNotes(target, targetId))
        if (identityChangedSince(generation)) return
        const keep = this.notes.filter(
          (n) =>
            !(n.target === target && n.target_id === targetId) ||
            hasPendingNoteCreate(n.id) ||
            isLocalOnly(n.id),
        )
        this.notes = mergeById(rows, keep)
        this._cache()
      } catch {
        if (identityChangedSince(generation)) return
        const cached = await readCached<{ highlights: Highlight[]; notes: Note[] }>(
          'captures',
          hasArrayFields('highlights', 'notes'),
        )
        if (cached) {
          this.notes = mergeById(
            cached.notes.filter((n) => n.target === target && n.target_id === targetId),
            this.notes,
          )
          this.stale = true
        }
      }
    },
    /** Rows a paged list fetched — kept so their edits, colours and note links work like any. */
    merge(highlights: Highlight[], notes: Note[] = []): void {
      this.highlights = mergeById(highlights, this.highlights)
      this.notes = mergeById(notes, this.notes)
    },
    _bump(): void {
      this.version++
    },
    _cache(): void {
      void writeCached('captures', { highlights: this.highlights, notes: this.notes })
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
      return writes.run(async () => {
        try {
          const saved = await createHighlight(withId)
          // A response landing after an account switch belongs to nobody now (advisor 1.4).
          if (identityChangedSince(generation)) return false
          serverIdFor.set(client_id, saved.id)
          // Row-level: replaces only this capture's own row (a no-op if the user already unsaved
          // it — the delete queued behind this write will remove the server's row).
          this.highlights = this.highlights.map((h) => (h.id === client_id ? saved : h))
          if (this.total !== null) this.total++
          this._bump()
          return true
        } catch (err: unknown) {
          if (identityChangedSince(generation)) return false
          // Only a REFUSAL discards the capture. A 502 or a dead socket is not an answer, and
          // dropping the user's highlight on one would lose it for good.
          if (isPermanent(err)) {
            refusedCreates.add(client_id)
            this.highlights = this.highlights.filter((h) => h.id !== client_id)
            return false
          }
          enqueue({ op: 'highlight.create', body: withId })
          return true
        }
      })
    },
    /** Delete a highlight, queuing the delete when the request never lands. */
    async _uncapture(id: string): Promise<boolean> {
      const generation = identityEpoch()
      const prev = this.highlights
      // A capture made offline has never reached the server, so deleting it there would 404 —
      // and a 404 is "permanent", which would RESTORE the row and leave it undeletable until the
      // outbox happened to create it (advisor 3.1). The queued create is withdrawn instead.
      //
      // Withdrawing does NOT end it: the POST may have reached the server and only its RESPONSE
      // been lost, in which case the row exists and nothing would ever delete it (advisor-2 #2).
      // So the delete is still queued — the flush drops it on a 404, which is exactly the
      // never-existed case.
      const wasPendingCreate = withdrawPendingCreate(id)
      this.highlights = this.highlights.filter((h) => h.id !== id)
      if (wasPendingCreate) {
        enqueue({ op: 'highlight.remove', id })
        return true
      }
      return writes.run(async (isLatest) => {
        // The create this delete was chasing has finished by now (writes are serialised). If the
        // server refused it there is nothing to delete; if it failed in transit it was QUEUED
        // meanwhile, so withdraw it exactly as the offline case above does.
        if (refusedCreates.has(id)) return true
        if (withdrawPendingCreate(id)) {
          enqueue({ op: 'highlight.remove', id })
          return true
        }
        const target = serverIdFor.get(id) ?? id
        try {
          await deleteHighlight(target)
          if (identityChangedSince(generation)) return false
          // The answer is the WHOLE remaining list (the contract older clients read); this store
          // holds only what it loaded, so the row it already removed is the whole change.
          if (this.total !== null) this.total = Math.max(0, this.total - 1)
          if (isLatest()) this._bump()
          return true
        } catch (err: unknown) {
          if (identityChangedSince(generation)) return false
          if (isPermanent(err)) {
            if (isLatest()) this.highlights = prev
            return false
          }
          enqueue({ op: 'highlight.remove', id: target })
          return true
        }
      })
    },
    /** One-tap "mark this moment" at a content-time position (seconds). */
    async captureMoment(
      slug: string,
      contentSeconds: number,
      speaker?: string | null,
      quoteText?: string | null,
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
        // WHAT was marked, not just when. A moment stored only a timestamp and a speaker, so every
        // card in the Library read "Marked moment" with no way to tell one from another — the user
        // could see that they had marked something and not what (operator 2026-09-17). The player
        // already has the active segment in hand; its text is the thing being marked.
        quote_text: quoteText ?? null,
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
    /**
     * Set (or clear, with null) a highlight's colour token. Optimistic like the other capture
     * writes: paint now, and on a transient failure keep the paint AND queue it to replay, so an
     * offline recolour is not a silent no-op (#2004 #12). A refusal reverts. If the highlight is
     * itself still a queued create (offline create-then-recolour), fold the colour into that create
     * rather than queue a `highlight.edit` that would evict it — same rule as `editNote`.
     */
    async setColor(id: string, color: string | null): Promise<void> {
      await this.ensureLoaded()
      const generation = identityEpoch()
      const prev = this.highlights
      this.highlights = this.highlights.map((h) => (h.id === id ? { ...h, color } : h))
      try {
        const updated = await patchHighlight(id, { color })
        if (identityChangedSince(generation)) return
        this.highlights = this.highlights.map((h) => (h.id === id ? updated : h))
        this._bump()
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return
        if (isPermanent(err)) this.highlights = prev
        else if (!updatePendingHighlightColor(id, color) && !hasPendingHighlightCreate(id)) {
          enqueue({ op: 'highlight.edit', id, color })
        }
      }
    },
    /**
     * Resume resurfacing a capture the user retired — the undo for the Revisit bell.
     *
     * Only ever un-retires. Retiring happens on Revisit, where the card is already leaving the
     * list; here the item STAYS in Saved and only its badge changes, so the optimistic update is
     * a field flip rather than a removal. On failure the flag goes back, because a badge that
     * silently disagrees with the server is worse than one that visibly did not change.
     */
    async unretire(id: string): Promise<void> {
      const prev = this.highlights
      this.highlights = this.highlights.map((h) => (h.id === id ? { ...h, retired: false } : h))
      try {
        await unretireHighlight(id)
        this._bump()
      } catch {
        this.highlights = prev
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
      // Let any in-flight initial load settle FIRST, so its server list cannot overwrite the
      // optimistic note appended just below (the NoteComposer self-hydrate race).
      await this.ensureLoaded()
      if (identityChangedSince(generation)) return
      const client_id = newCaptureId('n')
      const body: NoteCreate = { target, target_id: targetId, text, client_id }
      const now = Math.floor(Date.now() / 1000)
      this.notes = [...this.notes, { ...body, id: client_id, created_at: now, updated_at: now }]
      try {
        const saved = await createNote(body)
        if (identityChangedSince(generation)) return
        this.notes = this.notes.map((n) => (n.id === client_id ? saved : n))
        this._bump()
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return
        // A refusal drops it; anything else keeps it and queues the write.
        if (isPermanent(err)) this.notes = this.notes.filter((n) => n.id !== client_id)
        else enqueue({ op: 'note.create', body })
      }
    },
    /** Edit a note's text. */
    async editNote(id: string, text: string): Promise<void> {
      const generation = identityEpoch()
      // Let any in-flight initial load settle first, so its server list can't clobber the edit
      // (the same race addNote guards against).
      await this.ensureLoaded()
      if (identityChangedSince(generation)) return
      const prev = this.notes
      // Optimistic: show the edit immediately. Before this, a failed patch left the note UNCHANGED
      // with no feedback — the user's edit silently did nothing offline.
      this.notes = this.notes.map((n) => (n.id === id ? { ...n, text } : n))
      try {
        const updated = await patchNote(id, text)
        if (identityChangedSince(generation)) return
        this.notes = this.notes.map((n) => (n.id === id ? updated : n))
        this._bump()
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return
        // A refusal reverts; a transient error keeps the optimistic edit AND queues it to replay on
        // reconnect. If a CREATE for this note is still queued (offline create-then-edit), fold the
        // text into that create — NEVER enqueue a note.edit, which shares the queue slot and would
        // evict the create (losing the note if the create later fails). `updatePendingNoteText` is a
        // no-op mid-flush; the `hasPendingNoteCreate` guard then still refuses the evicting enqueue,
        // and the create replays as-is (the edit persists on screen, best-effort).
        if (isPermanent(err)) this.notes = prev
        else if (!updatePendingNoteText(id, text) && !hasPendingNoteCreate(id)) {
          enqueue({ op: 'note.edit', id, text })
        }
      }
    },
    /** Remove a note by id. */
    async removeNote(id: string): Promise<void> {
      const generation = identityEpoch()
      // Settle any in-flight initial load first (same race guard as addNote/editNote).
      await this.ensureLoaded()
      if (identityChangedSince(generation)) return
      const prev = this.notes
      // Same withdrawal as _uncapture, and the same reason it is not sufficient on its own: the
      // POST may have landed with only its response lost (advisor-2 #2).
      const wasPendingCreate = withdrawPendingCreate(id)
      this.notes = this.notes.filter((n) => n.id !== id)
      if (wasPendingCreate) {
        enqueue({ op: 'note.remove', id })
        return
      }
      try {
        await deleteNote(id)
        if (identityChangedSince(generation)) return
        // The answer is every remaining note; the one removed above is the whole change here.
        this._bump()
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return
        if (isPermanent(err)) this.notes = prev
        else enqueue({ op: 'note.remove', id })
      }
    },
  },
})
