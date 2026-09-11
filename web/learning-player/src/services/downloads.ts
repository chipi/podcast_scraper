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
import { ApiError, getAudioSource, getEntities, getEpisode, getInsights, getSegments } from './api'
import type { Entity, EpisodeDetail, Insight, SegmentsResponse, Topic } from './types'
import { getDeviceJson, setDeviceJson } from './deviceStore'
import { isNative } from './native'
import { resolveMediaUrl } from './tier'
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
export const KNOWLEDGE_FOLDER = 'offline-knowledge'

/**
 * How much audio one account may keep on this device.
 *
 * A starting figure, not a tuned one — it should become a device setting once there is any
 * evidence about what people actually keep.
 */
export const CAP_KEY = 'downloads.capBytes'
export const CAP_CHOICES = [
  2 * 1024 * 1024 * 1024,
  4 * 1024 * 1024 * 1024,
  8 * 1024 * 1024 * 1024,
] as const
export const DEFAULT_CAP_BYTES: number = CAP_CHOICES[1]

/**
 * How much audio one account may keep on this device — a per-DEVICE setting for the same reason
 * the network policy is: a 64GB phone and a 1TB tablet should not be forced to agree.
 *
 * NOTE this is OUR cap, not the device's free space. `@capacitor/filesystem` exposes no
 * free-space API, so a genuine disk-full preflight needs a native plugin; refusing against the
 * cap is the honest approximation and is what the UI describes.
 */
export async function getDownloadCap(): Promise<number> {
  const stored = await getDeviceJson<number>(CAP_KEY)
  return typeof stored === 'number' && stored > 0 ? stored : DEFAULT_CAP_BYTES
}

export async function setDownloadCap(bytes: number): Promise<void> {
  await setDeviceJson(CAP_KEY, bytes)
}

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
/**
 * Create a download folder if it is not already there. `recursive: true` covers this for the
 * plain Filesystem writes; doing it explicitly costs one call and removes the assumption.
 *
 * An "already exists" rejection is the desired end state, so it is swallowed.
 */
async function ensureFolder(path: string, directory = DOWNLOAD_DIR): Promise<void> {
  const folder = path.slice(0, path.lastIndexOf('/'))
  if (!folder) return
  try {
    await Filesystem.mkdir({ path: folder, directory, recursive: true })
  } catch {
    /* already there */
  }
}

/**
 * Put a just-downloaded file where the rest of this module expects it (#1925, decision 4).
 *
 * `@capacitor/filesystem` 8.1.2 IGNORES `directory` in `downloadFile` on iOS: whatever we ask
 * for, the bytes land under `Directory.Documents` at the same relative path. Every other call
 * here — `stat`, `getUri`, `readdir`, `deleteFile`, and the URI the player feeds to the audio
 * element — uses `LibraryNoCloud`. The consequences were both worse than they look:
 *
 *  1. Every download FAILED. `downloadFile` resolves, the `stat` that follows finds nothing, and
 *     the entry is recorded as retryable — so the user sees "Download failed — tap to retry", and
 *     the retry fails identically, for ever.
 *  2. The bytes were still on the device, in `Documents` — which iOS backs up to iCloud. Choosing
 *     `LibraryNoCloud` is precisely how this app keeps third-party podcast audio OUT of a user's
 *     backup, so the misplacement silently defeated that. The orphan sweep could not reclaim them
 *     either: it reads `LibraryNoCloud`, so they would accumulate for ever.
 *
 * No device test caught it because they all seeded files into `LibraryNoCloud` with `cp` and
 * asserted playback from there. Downloading through the UI is what made it visible.
 *
 * Written to survive the plugin being FIXED: if the file is already in the right place we do
 * nothing, so this becomes a no-op rather than a breakage on a future upgrade.
 */
async function settleDownloadedFile(path: string): Promise<void> {
  try {
    await Filesystem.stat({ directory: DOWNLOAD_DIR, path })
    return // already where it belongs — a fixed plugin, or a non-iOS platform
  } catch {
    /* fall through to the rescue */
  }
  await ensureFolder(path)
  await Filesystem.rename({
    from: path,
    directory: Directory.Documents,
    to: path,
    toDirectory: DOWNLOAD_DIR,
  })
}

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

export function knowledgePathFor(slug: string): string {
  const safe = slug.replace(/[^a-zA-Z0-9._-]/g, '_')
  return `${useDownloadsStore().folderFor(KNOWLEDGE_FOLDER)}/${safe}.json`
}

/**
 * The bridge returns an absolute origin enclosure URL in production. A RELATIVE url occurs only
 * against the local validation corpus, where resolving against the document origin is correct.
 * Anything that is not http(s) after resolution is refused rather than handed to the plugin.
 */
export function absolutize(url: string): string {
  // Against the API base, NOT the document origin: on native the document origin is
  // capacitor://localhost, so a relative media url resolved that way is both wrong and then
  // rejected below as non-http — a confusing failure for an episode that is perfectly fetchable.
  // On native this yields an absolute API-based url; on web it returns the relative string
  // unchanged (origin-relative is already correct there), so fall back to the document origin.
  const viaApi = resolveMediaUrl(url)
  const resolved =
    viaApi && /^https?:/i.test(viaApi) ? viaApi : new URL(url, window.location.origin).toString()
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


  let handle: PluginListenerHandle | null = null
  let path: string | null = null
  const epoch = epochOf(slug)
  // Captured at the start: every registry write below must belong to the account that asked.
  const startedIn = store.namespace
  try {
    const [source, detail] = await Promise.all([
      // validate=true HEADs the origin so `content_length` is populated — that is what makes a
      // real preflight possible instead of only noticing at 95%.
      getAudioSource(slug, true),
      // Offline display metadata — the API is unreachable when the user actually needs it.
      getEpisode(slug).catch(() => null),
    ])
    const url = absolutize(source.url)

    // Preflight: reclaim finished episodes, then refuse if the incoming file still cannot fit.
    // Refusing BEFORE the transfer is the point — the old check only fired once already at the
    // cap, so a 200MB download could start and die at the end.
    const cap = await getDownloadCap()
    const incoming = source.content_length ?? 0
    if (store.bytesOnDisk + incoming >= cap) {
      await reclaimFinished()
      if (store.bytesOnDisk + incoming >= cap) {
        await store.setFailed(slug, 'not enough space for this episode', 'needs-space')
        return false
      }
    }

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

    await ensureFolder(path)
    await Filesystem.downloadFile({
      url,
      path,
      directory: DOWNLOAD_DIR,
      progress: true,
      recursive: true,
    })
    // ...and put it where `directory` asked for, which the plugin does not honour.
    await settleDownloadedFile(path)

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
    // Re-checked AFTER these awaits, not only before them (advisor 2.1): the switch can land
    // while this very entry is being stat'd, and `setDownloaded` would then stamp a path in A's
    // folder into B's registry — after which B's delete removes A's file.
    if (store.namespace !== startedIn || epochOf(slug) !== epoch) {
      await removeFile(path)
      return false
    }
    await store.setDownloaded(slug, uri, stat.size)
    // Best-effort: artwork is needed for the offline list and the lock screen, but a missing
    // image must not turn a perfectly good audio download into a failure.
    if (detail) void cacheArtwork(slug, detail, epoch)
    void cacheTranscript(slug, epoch)
    void cacheKnowledge(slug, detail, epoch)
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

/**
 * Rename a downloaded image to the extension its BYTES say it is, returning the path to record.
 *
 * `nameFor` derives the extension from the URL and falls back to `jpg`. Artwork is fetched from
 * `/api/app/artwork?ref=…`, which carries no extension at all — so EVERY downloaded cover was
 * written as `.jpg` whatever the server actually sent. iOS serves a local file's MIME type from its
 * extension, so an SVG (or PNG, or WebP) stored as `.jpg` reaches the WebView as `image/jpeg`,
 * fails to decode, and the player shows a broken-image box for an episode that is fully downloaded.
 * Observed on the simulator: every `offline-artwork/*.jpg` was 2018 bytes of `<svg xmlns=…`.
 *
 * The URL is not a fact about the bytes; the bytes are. Sniffing the magic number is what a
 * `Content-Type` would have told us, and `Filesystem.downloadFile` does not surface response
 * headers. Any failure here leaves the file exactly where it is — wrong art is survivable, and
 * throwing would lose a cover we successfully fetched.
 */
async function correctImageExtension(path: string): Promise<string> {
  try {
    const { data } = await Filesystem.readFile({ path, directory: DOWNLOAD_DIR })
    // Native returns base64. 24 chars decode to 18 bytes — past every magic number below, and far
    // cheaper than pulling a whole image into a string to look at its first byte.
    const head = typeof data === 'string' ? atob(data.slice(0, 24)) : ''
    const b = (i: number): number => head.charCodeAt(i)
    let ext: string | null = null
    if (b(0) === 0xff && b(1) === 0xd8 && b(2) === 0xff) ext = 'jpg'
    else if (head.startsWith('\x89PNG')) ext = 'png'
    else if (head.startsWith('GIF8')) ext = 'gif'
    else if (head.startsWith('RIFF') && head.slice(8, 12) === 'WEBP') ext = 'webp'
    // SVG is text and has no magic number: an XML declaration, a comment, or the tag itself.
    else if (/^\s*(<\?xml|<!--|<svg)/i.test(head)) ext = 'svg'
    if (!ext) return path
    const correct = path.replace(/\.[^./]*$/, `.${ext}`)
    if (correct === path) return path
    await Filesystem.rename({ from: path, to: correct, directory: DOWNLOAD_DIR, toDirectory: DOWNLOAD_DIR })
    return correct
  } catch {
    // An unreadable or unrenameable file is still the art we downloaded — keep it.
    return path
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
    const downloadedTo = artworkPathFor(slug, url)
    await ensureFolder(downloadedTo)
    await Filesystem.downloadFile({ url, path: downloadedTo, directory: DOWNLOAD_DIR, recursive: true })
    // Artwork lands in Documents too; this one failed SILENTLY (the catch below is deliberate),
    // so the episode played offline with no cover art and nothing said why.
    await settleDownloadedFile(downloadedTo)
    // The name was a guess off the URL; the bytes are the fact. Do this BEFORE the epoch check so a
    // cancelled download removes the file that is actually on disk.
    const path = await correctImageExtension(downloadedTo)
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

/**
 * Everything the episode PAGE needs, stored beside the audio (#1905 follow-up).
 *
 * A downloaded episode carried audio, a transcript and three display fields — title, show,
 * duration. Everything else you open an episode FOR came from the API: the summary, the insights
 * list, the topics and the people. So on a plane the page was a player, a wall of transcript, and
 * nothing else — and `offlineEpisodeDetail` reported `has_summary: false` for an episode whose
 * summary the server had already written.
 *
 * ONE file, not three. These are fetched together, invalidated together and deleted together;
 * three sidecars would be three chances for an episode to end up partly complete, which is the
 * state that produces a page with insights but no summary and nothing to explain the difference.
 *
 * Best-effort, like the transcript: a missing knowledge file must never fail an audio download
 * that succeeded. The page then degrades to exactly what it does today, rather than to worse.
 */
async function cacheKnowledge(
  slug: string,
  detail: EpisodeDetail | null,
  epoch: number,
): Promise<void> {
  try {
    // Caught individually: an episode with no insights yet is normal, and it must not cost us the
    // entities as well.
    const [insights, entities] = await Promise.all([
      getInsights(slug).catch(() => null),
      getEntities(slug).catch(() => null),
    ])
    if (!detail && !insights && !entities) return
    const path = knowledgePathFor(slug)
    await Filesystem.writeFile({
      path,
      directory: DOWNLOAD_DIR,
      data: JSON.stringify({
        detail,
        insights: insights?.insights ?? [],
        topics: entities?.topics ?? [],
        persons: entities?.persons ?? [],
      }),
      encoding: Encoding.UTF8,
      recursive: true,
    })
    if (epochOf(slug) !== epoch) {
      await removeFile(path)
      return
    }
    useDownloadsStore().setKnowledgePath(slug, path)
  } catch {
    // The episode still plays, and still has its transcript.
  }
}

/**
 * Write the display metadata a marked episode needs to be RECOGNISABLE, before any bytes move.
 *
 * The registry entry is created by `mark()` with nothing but a slug, and the title, show and
 * duration were written inside the transfer — so an episode queued behind a Wi-Fi-only policy on a
 * cellular connection never got them. The Downloaded list falls back to `e.title ?? e.slug`, so the
 * row rendered as a raw `sha256…`, next to finished rows with covers and titles. Reported from a
 * real phone: "I've never seen this screen before."
 *
 * Best-effort and fire-and-forget: this is what a row LOOKS like, not whether it can download.
 * Offline it fetches nothing and the row keeps its slug until a connection returns — which is the
 * honest state, since we have never been told what this episode is.
 */
export async function captureDisplayMetadata(slug: string): Promise<void> {
  const store = useDownloadsStore()
  const entry = store.entry(slug)
  if (!entry || entry.title) return
  const startedIn = store.namespace
  try {
    const detail = await getEpisode(slug)
    if (store.namespace !== startedIn || !store.entry(slug)) return
    store.setMetadata(slug, {
      title: detail.title,
      showTitle: detail.podcast_title ?? undefined,
      feedId: detail.feed_id || undefined,
      durationSeconds: detail.duration_seconds ?? undefined,
      artworkUrl: episodeArtwork(detail) ?? undefined,
    })
  } catch {
    // No network, or the episode is gone. The row keeps its slug rather than inventing a title.
  }
}

/**
 * Fetch the knowledge sidecar for episodes downloaded BEFORE it existed.
 *
 * `cacheKnowledge` runs at download time, so every episode already on a device when this shipped
 * has audio, a transcript and nothing else — the summary and insights would stay missing until the
 * user happened to delete and re-download, which nobody will do and nobody should have to.
 *
 * Runs at boot beside the URI repair, sequentially: it is backfill, not a race, and firing N
 * episode fetches at once on a cold launch competes with the screen the user is actually looking
 * at. Every step is best-effort — offline this simply fetches nothing and tries again next launch.
 *
 * The namespace is re-checked each iteration: an account switch mid-backfill must not write one
 * user's episodes into the other's registry.
 */
export async function backfillKnowledge(): Promise<void> {
  if (!isNative()) return
  const store = useDownloadsStore()
  const startedIn = store.namespace
  const pending = Object.values(store.entries)
    .filter((e) => e.state === 'downloaded' && !e.knowledgePath)
    .map((e) => e.slug)
  for (const slug of pending) {
    if (store.namespace !== startedIn) return
    const detail = await getEpisode(slug).catch(() => null)
    await cacheKnowledge(slug, detail, epochOf(slug))
  }
}

/** What `cacheKnowledge` wrote, or null to ask the API. */
export interface LocalKnowledge {
  detail: EpisodeDetail | null
  insights: Insight[]
  topics: Topic[]
  persons: Entity[]
}

export async function localKnowledgeFor(slug: string): Promise<LocalKnowledge | null> {
  if (!isNative()) return null
  const path = useDownloadsStore().entry(slug)?.knowledgePath
  if (!path) return null
  try {
    const { data } = await Filesystem.readFile({
      path,
      directory: DOWNLOAD_DIR,
      encoding: Encoding.UTF8,
    })
    return JSON.parse(typeof data === 'string' ? data : '') as LocalKnowledge
  } catch {
    return null
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
  const cap = await getDownloadCap()
  const finished = Object.values(store.entries)
    .filter((e) => e.state === 'downloaded' && localPosition(e.slug)?.finished)
    .sort((a, b) => a.updatedAt - b.updatedAt)

  let reclaimed = 0
  for (const e of finished) {
    if (store.bytesOnDisk < cap) break
    await deleteEpisode(e.slug)
    reclaimed += 1
  }
  return reclaimed
}

/**
 * Delete audio files this account's registry does not reference (#1911).
 *
 * A failed transfer can leave partial bytes behind, and a retry whose URL extension changed
 * orphans the previous attempt — neither is visible to `bytesOnDisk`, so the cap silently
 * under-counts and the user pays for space they cannot see or free. Runs at boot, after the URI
 * refresh has already dropped records whose file is gone; this is the other direction.
 */
export async function reconcileDownloadFolders(): Promise<number> {
  if (!isNative()) return 0
  // A transfer the drain starts AFTER we snapshot `referenced` writes a file this sweep would not
  // recognise — and delete out from under `Filesystem.downloadFile`. Refuse to run while anything
  // is in flight; boot fires this alongside the first drain, so the overlap is the normal case.
  if (inflight.size > 0) return 0
  const store = useDownloadsStore()
  await store.ensureLoaded()

  const referenced = new Set<string>()
  for (const e of Object.values(store.entries)) {
    if (e.path) referenced.add(e.path)
    if (e.artworkPath) referenced.add(e.artworkPath)
    if (e.transcriptPath) referenced.add(e.transcriptPath)
    if (e.knowledgePath) referenced.add(e.knowledgePath)
  }

  let removed = 0
  for (const kind of [DOWNLOAD_FOLDER, ARTWORK_FOLDER, TRANSCRIPT_FOLDER]) {
    const folder = store.folderFor(kind)
    try {
      const { files } = await Filesystem.readdir({ path: folder, directory: DOWNLOAD_DIR })
      for (const f of files) {
        const name = typeof f === 'string' ? f : f.name
        const full = `${folder}/${name}`
        if (referenced.has(full)) continue
        // A drain may have started while we were listing; anything now in flight is off limits.
        if (inflight.size > 0) return removed
        await removeFile(full)
        removed += 1
      }
    } catch {
      // The folder does not exist yet — nothing downloaded for this account.
    }
  }
  return removed
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
  if (entry?.knowledgePath) await removeFile(entry.knowledgePath)
}

/** Remove EVERY downloaded episode for this account (Config → "Remove downloads"). Returns the
 * count removed. Deletes records + files via deleteEpisode; safe when nothing is downloaded. */
export async function clearAllDownloads(): Promise<number> {
  const store = useDownloadsStore()
  await store.ensureLoaded()
  const slugs = Object.keys(store.entries)
  // Isolate per-item failures: one filesystem error (common on iOS with container-UUID churn) must
  // not abort the sweep and leave the rest untouched. Count what actually went.
  let removed = 0
  for (const slug of slugs) {
    try {
      await deleteEpisode(slug)
      removed += 1
    } catch {
      /* leave this one; keep clearing the others */
    }
  }
  return removed
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
  // The same guard the drain has: an account switch mid-loop would write A's repaired URIs into
  // B's registry, and `_forget` would delete B's records for files that are A's.
  const startedIn = store.namespace
  for (const entry of Object.values(store.entries)) {
    if (store.namespace !== startedIn) return
    if (entry.state !== 'downloaded' || !entry.path) continue
    try {
      const [{ uri }] = await Promise.all([
        Filesystem.getUri({ directory: DOWNLOAD_DIR, path: entry.path }),
        // getUri is pure string maths and succeeds for a missing file; stat is what proves it.
        Filesystem.stat({ directory: DOWNLOAD_DIR, path: entry.path }),
      ])
      // Re-checked AFTER the await, not just at the top of the loop: the switch can land while
      // this very entry is being stat'd, and writing here would put A's URI in B's registry.
      if (store.namespace !== startedIn) return
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
      if (store.namespace !== startedIn) return
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
