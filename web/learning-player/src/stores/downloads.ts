/**
 * Offline-download registry (#1905) — which episodes are on THIS device's disk.
 *
 * Unlike every other store here, this one wraps no `/api/app/*` endpoint. A download is
 * device-local by decision: one phone may be on a metered plan while a tablet is not, so the
 * registry lives in Capacitor Preferences via `services/deviceStore`. That is also why its
 * actions are absent from `__checks__/auth-gate.test.ts`'s `GATED_WRITES` — they write no
 * per-user *server* state and so can never 401. Adding a gate here would be cargo-culted.
 *
 * This store owns STATE ONLY. The transfer itself (`audio-source` → `Filesystem.downloadFile`)
 * is slice 2 and drives this store through the setters below; the network policy that decides
 * *when* a `queued` entry may start is slice 3.
 *
 * The store is platform-agnostic on purpose — the *feature surface* is gated on `isNative()`
 * at the UI, not here, so this logic stays unit-testable under happy-dom.
 */

import { defineStore } from 'pinia'
import { getDeviceJson, setDeviceJson } from '../services/deviceStore'

export type DownloadState = 'queued' | 'downloading' | 'downloaded' | 'failed'

export interface DownloadEntry {
  slug: string
  state: DownloadState
  /** Native file URI. Set only once the transfer completes. */
  uri?: string
  /**
   * Directory-relative path the file was written to. Recorded when the transfer STARTS, because
   * deleting later must not depend on re-resolving the audio source over the network.
   */
  path?: string
  /** Size on disk, for the Settings "storage used" readout. */
  bytes?: number
  /** Last error, when `state === 'failed'`. */
  error?: string
  /**
   * Whether retrying could ever succeed. Without this the drain retries a permanently gone
   * episode (corpus removal → 404) on every network change, forever.
   */
  errorKind?: 'retryable' | 'permanent'
  /**
   * Display metadata, captured at download time. The API is unreachable offline, so anything
   * the Downloaded list or the lock screen needs must live here — `stores/README.md` requires
   * every path into `player.load()` to carry title and artwork, or the lock screen shows the
   * previous episode.
   */
  title?: string
  showTitle?: string
  durationSeconds?: number
  /** Directory-relative path of the downloaded artwork, when it was fetched successfully. */
  artworkPath?: string
  /** Resolved native URI for the artwork, refreshed at boot alongside the audio URI. */
  artworkUri?: string
  /** Directory-relative path of the cached transcript JSON, when it was fetched successfully. */
  transcriptPath?: string
  updatedAt: number
}

/** Offline display metadata, captured from the episode detail at download time. */
export interface DownloadMeta {
  title?: string
  showTitle?: string
  durationSeconds?: number
}

interface DownloadsState {
  entries: Record<string, DownloadEntry>
  /** In-memory ONLY: progress is meaningless across a restart, because there is no resume. */
  progress: Record<string, number>
  loaded: boolean
}

export const REGISTRY_KEY = 'downloads.registry'

// Module-scoped (the store is an app singleton): coalesces concurrent loads onto one read, so
// N components mounting at once cannot clobber each other. Same shape as `queue.ts`.
let inflightLoad: Promise<void> | null = null

export const useDownloadsStore = defineStore('downloads', {
  state: (): DownloadsState => ({ entries: {}, progress: {}, loaded: false }),

  getters: {
    entry:
      (s) =>
      (slug: string): DownloadEntry | null =>
        s.entries[slug] ?? null,
    stateOf:
      (s) =>
      (slug: string): DownloadState | null =>
        s.entries[slug]?.state ?? null,
    isDownloaded:
      (s) =>
      (slug: string): boolean =>
        s.entries[slug]?.state === 'downloaded',
    progressOf:
      (s) =>
      (slug: string): number =>
        s.progress[slug] ?? 0,
    /** Slugs waiting on the network policy, oldest first — slice 3 drains in this order. */
    queued: (s): string[] =>
      Object.values(s.entries)
        .filter((e) => e.state === 'queued')
        .sort((a, b) => a.updatedAt - b.updatedAt)
        .map((e) => e.slug),
    /** Bytes across COMPLETED downloads only; a partial file is not on disk to account for. */
    bytesOnDisk: (s): number =>
      Object.values(s.entries).reduce(
        (n, e) => n + (e.state === 'downloaded' ? (e.bytes ?? 0) : 0),
        0,
      ),
    downloadedCount: (s): number =>
      Object.values(s.entries).filter((e) => e.state === 'downloaded').length,
  },

  actions: {
    async load(): Promise<void> {
      if (inflightLoad) return inflightLoad
      inflightLoad = (async (): Promise<void> => {
        const stored = (await getDeviceJson<Record<string, DownloadEntry>>(REGISTRY_KEY)) ?? {}
        // A `downloading` entry cannot survive a restart: the transfer died with the process,
        // and `Filesystem.downloadFile` has no resume (#1905). Demote it to `queued` so the
        // drain restarts it from zero, rather than leaving a spinner that will never move.
        for (const e of Object.values(stored)) {
          if (e.state === 'downloading') e.state = 'queued'
        }
        // In-memory entries WIN over stored ones: anything already in `entries` was written by
        // a mutation that ran before this first load resolved, so it is strictly newer than the
        // disk copy. Assigning `stored` outright would silently drop it — the same "dropped add"
        // race `queue.ts` guards with ensureLoaded(), except the setters here are sync and
        // cannot await.
        const pending = Object.keys(this.entries).length > 0
        this.entries = { ...stored, ...this.entries }
        this.loaded = true
        // Those in-memory entries were never written: `_persist` refuses to run before the
        // first load (it would overwrite the stored registry with a near-empty map). Now that
        // the merge is done, flush the union so they survive the next launch.
        if (pending) await this._persist()
      })().finally(() => {
        inflightLoad = null
      })
      return inflightLoad
    },

    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.load()
    },

    async _persist(): Promise<void> {
      // Refuse to write before the first load: `this.entries` is not yet the union of what is
      // on disk, so writing it would clobber every previously downloaded episode's record and
      // orphan its file. `load()` flushes the merge instead.
      if (!this.loaded) return
      // Swallowed by convention (stores/README §2): a failed write costs the flag on next
      // launch, which the UI renders as not-downloaded. Never throw — callers fire these
      // straight from template handlers.
      try {
        await setDeviceJson(REGISTRY_KEY, this.entries)
      } catch {
        // Device storage full or unavailable; nothing useful to do here.
      }
    },

    _put(slug: string, patch: Partial<DownloadEntry> & { state: DownloadState }): void {
      this.entries[slug] = {
        ...(this.entries[slug] ?? { slug }),
        ...patch,
        slug,
        updatedAt: Date.now(),
      }
    },

    /**
     * Flag an episode for offline listening. Returns whether anything changed, so the caller
     * can announce the outcome (stores/README §2 — a write the user perceives returns it).
     */
    async mark(slug: string): Promise<boolean> {
      await this.ensureLoaded()
      const current = this.entries[slug]?.state
      // Already on disk, or already moving — re-flagging must not restart a live transfer.
      if (current === 'downloaded' || current === 'downloading' || current === 'queued') {
        return false
      }
      this._put(slug, { state: 'queued', error: undefined })
      await this._persist()
      return true
    },

    /** Record what the UI needs offline. Safe before the file exists. */
    setMetadata(slug: string, meta: DownloadMeta): void {
      const existing = this.entries[slug]
      if (!existing) return
      this.entries[slug] = { ...existing, ...meta }
      void this._persist()
    },

    setArtworkPath(slug: string, artworkPath: string, artworkUri?: string): void {
      const existing = this.entries[slug]
      if (!existing) return
      this.entries[slug] = { ...existing, artworkPath, ...(artworkUri ? { artworkUri } : {}) }
      void this._persist()
    },

    setTranscriptPath(slug: string, transcriptPath: string): void {
      const existing = this.entries[slug]
      if (!existing) return
      this.entries[slug] = { ...existing, transcriptPath }
      void this._persist()
    },

    setDownloading(slug: string, path?: string): void {
      this._put(slug, { state: 'downloading', error: undefined, ...(path ? { path } : {}) })
      void this._persist()
    },

    /** `fraction` is 0–1. Transient by design: not persisted, cleared on completion. */
    setProgress(slug: string, fraction: number): void {
      this.progress[slug] = Math.max(0, Math.min(1, fraction))
    },

    async setDownloaded(slug: string, uri: string, bytes: number): Promise<void> {
      this._put(slug, { state: 'downloaded', uri, bytes, error: undefined })
      delete this.progress[slug]
      await this._persist()
    },

    async setFailed(
      slug: string,
      error: string,
      errorKind: 'retryable' | 'permanent' = 'retryable',
    ): Promise<void> {
      this._put(slug, { state: 'failed', error, errorKind })
      delete this.progress[slug]
      await this._persist()
    },

    /**
     * Drop the RECORD only — the file is left on disk. Underscore-prefixed deliberately: this is
     * internal, and a component that wires it instead of `services/downloads.deleteEpisode()`
     * leaks the bytes forever. (`__checks__/auth-gate.test.ts:182` treats `_` as
     * component-unreachable, which is the convention being borrowed here.)
     */
    async _forget(slug: string): Promise<boolean> {
      await this.ensureLoaded()
      if (!this.entries[slug]) return false
      delete this.entries[slug]
      delete this.progress[slug]
      await this._persist()
      return true
    },
  },
})
