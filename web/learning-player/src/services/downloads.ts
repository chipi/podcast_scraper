/**
 * Offline transfer (#1905) — origin host → this device's disk.
 *
 * PRD-035 Principle 4 ("bridge, never rehost") is preserved: the bytes travel from the ORIGIN
 * host straight to the device. Our API only ever hands over the URL
 * (`GET /api/app/episodes/{slug}/audio-source`); no audio is stored on, proxied by, or served
 * from our infrastructure. Nothing here touches the service worker either, so the
 * `e2e/offline.spec.ts` "audio is never cached" invariant is untouched.
 *
 * Three hard constraints, all visible in the behaviour below:
 *
 * 1. **No cancel.** `@capacitor/filesystem` exposes no abort for `downloadFile`. "Cancel" is
 *    *unmark-then-reconcile*: the record is dropped at once so the UI responds, and the file
 *    that later lands is deleted as an orphan.
 * 2. **No resume.** An interrupted transfer restarts from zero; the store demotes an
 *    interrupted `downloading` entry back to `queued` on the next launch.
 * 3. **No concurrency from the plugin.** Two transfers writing one path truncate each other, so
 *    same-slug calls are coalesced onto one in-flight promise here.
 *
 * NOTE (#1063/#1066): the consumer episode routes are documented to become auth-gated later
 * (`server/routes/app_episodes.py:5`). When that lands, `getAudioSource` will 401 for a
 * signed-out user and a mark→download would produce a silent `failed` row. The mark control
 * must be gated then — and `__checks__/auth-gate.test.ts` will NOT catch it, because it scans
 * store actions, not service functions.
 */

import { Capacitor, type PluginListenerHandle } from '@capacitor/core'
import { Directory, Encoding, Filesystem } from '@capacitor/filesystem'
import { useDownloadsStore } from '../stores/downloads'
import { episodeArtwork } from '../utils/episode'
import { ApiError, getAudioSource, getEpisode, getSegments } from './api'
import type { SegmentsResponse } from './types'
import { isNative } from './native'
import { localPosition } from './playbackPositions'

/**
 * `LibraryNoCloud`, not `Data`/`Documents`: episode audio is re-downloadable and often hundreds
 * of MB, so it must be excluded from iCloud backup. `Cache` is wrong in the other direction —
 * the OS may reap it, and a download the user explicitly asked for should not evaporate.
 */
export const DOWNLOAD_DIR = Directory.LibraryNoCloud
export const DOWNLOAD_FOLDER = 'offline-audio'
export const ARTWORK_FOLDER = 'offline-artwork'
export const TRANSCRIPT_FOLDER = 'offline-transcripts'

/**
 * How much audio one account may keep on this device.
 *
 * A starting figure, not a tuned one — it should become a device setting once there is any
 * evidence about what people actually keep.
 */
export const DOWNLOAD_CAP_BYTES = 4 * 1024 * 1024 * 1024

/**
 * Coalesces same-slug calls. The slice-3 drain fires on flag, on network change, and on app
 * resume, so a user tap landing beside a drain tick is routine rather than exotic — and the
 * plugin opens the target truncating, so two transfers would interleave into one corrupt file.
 */
const inflight = new Map<string, Promise<boolean>>()

/**
 * Keys are namespace-scoped (#1905): keyed by slug alone, an account switch mid-transfer let B's
 * `downloadEpisode` join A's in-flight promise and stamp B's registry with a URI pointing into
 * A's folder — B playing A's file, and B's delete removing A's bytes.
 */
const nsKey = (slug: string): string => `${useDownloadsStore().namespace}\u0000${slug}`

/**
 * Bumped whenever a record is deliberately dropped. A transfer captures the epoch at its start
 * and refuses to touch the registry if it changed, so a cancelled transfer's epilogue cannot
 * stamp a stale error onto an entry the user has since re-created.
 */
const epochs = new Map<string, number>()
const epochOf = (slug: string): number => epochs.get(nsKey(slug)) ?? 0

/** Filenames are derived from the slug, so they cannot collide or escape the folder. */
function nameFor(slug: string, url: string, fallbackExt: string): string {
  const safe = slug.replace(/[^a-zA-Z0-9._-]/g, '_')
  const ext = /\.([a-zA-Z0-9]{1,5})(?:$|[?#])/.exec(url)?.[1]?.toLowerCase() ?? fallbackExt
  return `${safe}.${ext}`
}

/**
 * Paths are per-ACCOUNT (#1905). Two accounts on one device that download the same episode each
 * get their own copy rather than sharing — the accepted cost of not letting one see or delete the
 * other's downloads.
 */
export function pathFor(slug: string, url: string): string {
  return `${useDownloadsStore().folderFor(DOWNLOAD_FOLDER)}/${nameFor(slug, url, 'mp3')}`
}

export function artworkPathFor(slug: string, url: string): string {
  return `${useDownloadsStore().folderFor(ARTWORK_FOLDER)}/${nameFor(slug, url, 'jpg')}`
}

export function transcriptPathFor(slug: string): string {
  const safe = slug.replace(/[^a-zA-Z0-9._-]/g, '_')
  return `${useDownloadsStore().folderFor(TRANSCRIPT_FOLDER)}/${safe}.json`
}

/**
 * The bridge returns an absolute origin enclosure URL in production. A RELATIVE url occurs only
 * against the local validation corpus, where resolving against the document origin is correct.
 * Anything that is not http(s) after resolution is refused rather than handed to the plugin.
 */
export function absolutize(url: string): string {
  const resolved = new URL(url, window.location.origin).toString()
  if (!/^https?:/i.test(resolved)) {
    throw new Error(`audio source is not an http(s) URL: ${resolved}`)
  }
  return resolved
}

/** A gone episode must not be retried by the drain on every network change, forever. */
function classify(err: unknown): 'retryable' | 'permanent' {
  return err instanceof ApiError && (err.status === 404 || err.status === 410)
    ? 'permanent'
    : 'retryable'
}

/**
 * Fetch one episode to disk and record it. Returns whether the file ended up on disk.
 *
 * Never throws: callers fire this from a template handler, and the outcome is reported through
 * the store (and this return value) rather than as a rejection.
 */
export function downloadEpisode(slug: string): Promise<boolean> {
  const key = nsKey(slug)
  const existing = inflight.get(key)
  if (existing) return existing
  const run = runDownload(slug).finally(() => inflight.delete(key))
  inflight.set(key, run)
  return run
}

async function runDownload(slug: string): Promise<boolean> {
  // Defence in depth: Capacitor's WEB Filesystem writes into IndexedDB, so a missing UI gate
  // would quietly store hundreds of MB of third-party audio in browser storage — the spirit of
  // the bridge-never-rehost invariant, and `e2e/offline.spec.ts` would not catch it (it asserts
  // service-worker cache behaviour only).
  if (!isNative()) return false

  const store = useDownloadsStore()
  await store.ensureLoaded()
  if (store.isDownloaded(slug)) return true

  // Ensure a record EXISTS before any network call. Without this, a failure in `audio-source`
  // or `episode` (the calls that happen before `setDownloading`) had no entry to attach itself
  // to, so a direct `downloadEpisode()` for an un-marked slug failed completely silently.
  // A no-op when the drain already queued it.
  await store.mark(slug)

  // At the cap: reclaim finished episodes, and if that is not enough, refuse rather than delete
  // something the user has not listened to yet.
  if (store.bytesOnDisk >= DOWNLOAD_CAP_BYTES) {
    await reclaimFinished()
    if (store.bytesOnDisk >= DOWNLOAD_CAP_BYTES) {
      await store.setFailed(slug, 'not enough space on this device', 'needs-space')
      return false
    }
  }

  let handle: PluginListenerHandle | null = null
  let path: string | null = null
  const epoch = epochOf(slug)
  // Captured at the start: every registry write below must belong to the account that asked.
  const startedIn = store.namespace
  try {
    const [source, detail] = await Promise.all([
      getAudioSource(slug),
      // Offline display metadata — the API is unreachable when the user actually needs it.
      getEpisode(slug).catch(() => null),
    ])
    const url = absolutize(source.url)
    path = pathFor(slug, url)
    store.setDownloading(slug, path)
    if (detail) {
      store.setMetadata(slug, {
        title: detail.title,
        showTitle: detail.podcast_title ?? undefined,
        feedId: detail.feed_id || undefined,
        durationSeconds: detail.duration_seconds ?? undefined,
      })
    }

    handle = await Filesystem.addListener('progress', (p) => {
      // The listener is global; ignore chunks belonging to a sibling transfer.
      if (p.url !== url || !p.contentLength) return
      store.setProgress(slug, p.bytes / p.contentLength)
    })

    await Filesystem.downloadFile({
      url,
      path,
      directory: DOWNLOAD_DIR,
      progress: true,
      // The folder does not exist on a fresh install.
      recursive: true,
    })

    // Dropped while the bytes were still arriving (see "no cancel" above), or dropped and
    // re-created: either way the file that just landed is untracked disk.
    if (store.namespace !== startedIn || epochOf(slug) !== epoch || store.stateOf(slug) === null) {
      await removeFile(path)
      return false
    }

    const [{ uri }, stat] = await Promise.all([
      Filesystem.getUri({ directory: DOWNLOAD_DIR, path }),
      Filesystem.stat({ directory: DOWNLOAD_DIR, path }),
    ])
    await store.setDownloaded(slug, uri, stat.size)
    // Best-effort: artwork is needed for the offline list and the lock screen, but a missing
    // image must not turn a perfectly good audio download into a failure.
    if (detail) void cacheArtwork(slug, detail, epoch)
    void cacheTranscript(slug, epoch)
    return true
  } catch (err: unknown) {
    // Do NOT resurrect a record the user cancelled, and do not stamp a stale error onto an
    // entry they have since re-created — a "failed" row on a screen just cleared is a bug.
    if (store.namespace === startedIn && epochOf(slug) === epoch && store.entry(slug)) {
      await store.setFailed(slug, err instanceof Error ? err.message : String(err), classify(err))
    } else if (path) {
      await removeFile(path)
    }
    return false
  } finally {
    await handle?.remove()
  }
}

async function cacheArtwork(
  slug: string,
  detail: Parameters<typeof episodeArtwork>[0],
  epoch: number,
): Promise<void> {
  try {
    const raw = episodeArtwork(detail)
    if (!raw) return
    const url = absolutize(raw)
    const path = artworkPathFor(slug, url)
    await Filesystem.downloadFile({ url, path, directory: DOWNLOAD_DIR, recursive: true })
    if (epochOf(slug) !== epoch) {
      await removeFile(path)
      return
    }
    const { uri } = await Filesystem.getUri({ directory: DOWNLOAD_DIR, path })
    useDownloadsStore().setArtworkPath(slug, path, uri)
  } catch {
    // The episode is still fully playable offline without its art.
  }
}

/**
 * Store the transcript beside the audio. A downloaded episode must be DETERMINISTICALLY complete
 * offline — "complete if you happened to open it recently" is not a feature. Transcripts are our
 * own artifact, not origin media, so Principle 4 does not apply to caching them.
 */
async function cacheTranscript(slug: string, epoch: number): Promise<void> {
  try {
    const segments = await getSegments(slug)
    const path = transcriptPathFor(slug)
    await Filesystem.writeFile({
      path,
      directory: DOWNLOAD_DIR,
      data: JSON.stringify(segments),
      encoding: Encoding.UTF8,
      recursive: true,
    })
    if (epochOf(slug) !== epoch) {
      await removeFile(path)
      return
    }
    useDownloadsStore().setTranscriptPath(slug, path)
  } catch {
    // The episode still plays offline; only the transcript is missing.
  }
}

/** Playable artwork src for a downloaded episode, or null to fall back to the network. */
export function localArtworkFor(slug: string): string | null {
  if (!isNative()) return null
  const uri = useDownloadsStore().entry(slug)?.artworkUri
  return uri ? Capacitor.convertFileSrc(uri) : null
}

/** The cached transcript for a downloaded episode, or null to fetch it from the API. */
export async function localTranscriptFor(slug: string): Promise<SegmentsResponse | null> {
  if (!isNative()) return null
  const path = useDownloadsStore().entry(slug)?.transcriptPath
  if (!path) return null
  try {
    const { data } = await Filesystem.readFile({
      path,
      directory: DOWNLOAD_DIR,
      encoding: Encoding.UTF8,
    })
    return JSON.parse(typeof data === 'string' ? data : '') as SegmentsResponse
  } catch {
    return null
  }
}

/**
 * Reclaim room by deleting episodes the user has FINISHED, oldest first.
 *
 * This is the whole eviction policy (#1905): a finished episode is done, so removing it takes
 * nothing the user still wants — which is what keeps the `LibraryNoCloud` rationale honest ("a
 * download the user explicitly asked for should not evaporate"). Nothing unplayed is ever
 * removed automatically; when reclaiming is not enough, the next download is REFUSED instead.
 */
export async function reclaimFinished(): Promise<number> {
  const store = useDownloadsStore()
  const finished = Object.values(store.entries)
    .filter((e) => e.state === 'downloaded' && localPosition(e.slug)?.finished)
    .sort((a, b) => a.updatedAt - b.updatedAt)

  let reclaimed = 0
  for (const e of finished) {
    if (store.bytesOnDisk < DOWNLOAD_CAP_BYTES) break
    await deleteEpisode(e.slug)
    reclaimed += 1
  }
  return reclaimed
}

/** Drop the record and the bytes. Safe to call for an episode that was never downloaded. */
export async function deleteEpisode(slug: string): Promise<void> {
  const store = useDownloadsStore()
  await store.ensureLoaded()
  const entry = store.entry(slug)
  // Invalidate any transfer still running for this slug before dropping the record.
  epochs.set(nsKey(slug), epochOf(slug) + 1)
  await store._forget(slug)
  if (entry?.path) await removeFile(entry.path)
  if (entry?.artworkPath) await removeFile(entry.artworkPath)
  if (entry?.transcriptPath) await removeFile(entry.transcriptPath)
}

/** Best-effort unlink — a missing file is already the desired end state. */
async function removeFile(path: string): Promise<void> {
  try {
    await Filesystem.deleteFile({ directory: DOWNLOAD_DIR, path })
  } catch {
    // Already gone, or never written.
  }
}

/**
 * Re-resolve every downloaded file's URI, and drop records whose file is gone.
 *
 * iOS regenerates the app container UUID on update, so an absolute `file:///…/<UUID>/…` persisted
 * yesterday is dead today. `path` + `DOWNLOAD_DIR` is the durable identity, so the URI is derived
 * fresh at each launch rather than trusted from disk. Also catches files removed by the user or
 * reaped by the OS: better to show "not downloaded" than to hand the player a dead src.
 *
 * Call once at boot, after the registry has loaded.
 */
export async function refreshLocalUris(): Promise<void> {
  if (!isNative()) return
  const store = useDownloadsStore()
  await store.ensureLoaded()
  for (const entry of Object.values(store.entries)) {
    if (entry.state !== 'downloaded' || !entry.path) continue
    try {
      const [{ uri }] = await Promise.all([
        Filesystem.getUri({ directory: DOWNLOAD_DIR, path: entry.path }),
        // getUri is pure string maths and succeeds for a missing file; stat is what proves it.
        Filesystem.stat({ directory: DOWNLOAD_DIR, path: entry.path }),
      ])
      if (uri !== entry.uri) await store.setDownloaded(entry.slug, uri, entry.bytes ?? 0)
      // Artwork shares the container, so its URI goes stale on the same app update.
      if (entry.artworkPath) {
        try {
          const art = await Filesystem.getUri({ directory: DOWNLOAD_DIR, path: entry.artworkPath })
          if (art.uri !== entry.artworkUri) {
            store.setArtworkPath(entry.slug, entry.artworkPath, art.uri)
          }
        } catch {
          // Art is optional; the episode still plays.
        }
      }
    } catch {
      await store._forget(entry.slug)
    }
  }
}

/**
 * The player's source resolver: a playable src for a downloaded episode, else null to stream.
 * Sync, because `player.load()` is sync — it reads the already-hydrated registry, and the URIs
 * were refreshed at boot by `refreshLocalUris()`.
 */
export function localSourceFor(slug: string): string | null {
  if (!isNative()) return null
  const entry = useDownloadsStore().entry(slug)
  if (!entry || entry.state !== 'downloaded' || !entry.uri) return null
  return Capacitor.convertFileSrc(entry.uri)
}
