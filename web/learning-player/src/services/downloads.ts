/**
 * Offline transfer (#1905, slice 2) — origin host → this device's disk.
 *
 * PRD-035 Principle 4 ("bridge, never rehost") is preserved: the bytes travel from the ORIGIN
 * host straight to the device. Our API only ever hands over the URL
 * (`GET /api/app/episodes/{slug}/audio-source`); no audio is stored on, proxied by, or served
 * from our infrastructure. Nothing here touches the service worker either, so the
 * `e2e/offline.spec.ts` "audio is never cached" invariant is untouched.
 *
 * Two hard constraints of the plugin, both visible in the behaviour below:
 *
 * 1. **No cancel.** `@capacitor/filesystem` exposes no abort for `downloadFile`. "Cancel" is
 *    therefore *unmark-then-reconcile*: the record is dropped immediately so the UI responds,
 *    and when the transfer eventually lands we delete the orphaned file.
 * 2. **No resume.** An interrupted transfer restarts from zero; the store demotes an
 *    interrupted `downloading` entry back to `queued` on the next launch.
 */

import type { PluginListenerHandle } from '@capacitor/core'
import { Directory, Filesystem } from '@capacitor/filesystem'
import { useDownloadsStore } from '../stores/downloads'
import { getAudioSource } from './api'

/**
 * `LibraryNoCloud`, not `Data`/`Documents`: episode audio is re-downloadable and often hundreds
 * of MB, so it must be excluded from iCloud backup. `Cache` is wrong in the other direction —
 * the OS may reap it, and a download the user explicitly asked for should not evaporate.
 */
export const DOWNLOAD_DIR = Directory.LibraryNoCloud
export const DOWNLOAD_FOLDER = 'offline-audio'

/** Filenames are derived from the slug, so they cannot collide or escape the folder. */
export function pathFor(slug: string, url: string): string {
  const safe = slug.replace(/[^a-zA-Z0-9._-]/g, '_')
  const ext = /\.([a-zA-Z0-9]{1,5})(?:$|[?#])/.exec(url)?.[1]?.toLowerCase() ?? 'mp3'
  return `${DOWNLOAD_FOLDER}/${safe}.${ext}`
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

/**
 * Fetch one episode to disk and record it. Returns whether the file ended up on disk.
 *
 * Never throws: callers fire this from a template handler, and the outcome is reported through
 * the store (and this return value) rather than as a rejection.
 */
export async function downloadEpisode(slug: string): Promise<boolean> {
  const store = useDownloadsStore()
  await store.ensureLoaded()
  if (store.isDownloaded(slug)) return true

  let handle: PluginListenerHandle | null = null
  let path: string | null = null
  try {
    const source = await getAudioSource(slug)
    const url = absolutize(source.url)
    path = pathFor(slug, url)
    store.setDownloading(slug, path)

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

    // Unmarked while the bytes were still arriving (see "no cancel" above): the file that just
    // landed is untracked disk. Delete it instead of leaking it.
    if (store.stateOf(slug) === null) {
      await removeFile(path)
      return false
    }

    const [{ uri }, stat] = await Promise.all([
      Filesystem.getUri({ directory: DOWNLOAD_DIR, path }),
      Filesystem.stat({ directory: DOWNLOAD_DIR, path }),
    ])
    await store.setDownloaded(slug, uri, stat.size)
    return true
  } catch (err: unknown) {
    // Do NOT resurrect a record the user cancelled — that would put a "failed" row back on a
    // screen they just cleared.
    if (store.entry(slug)) {
      await store.setFailed(slug, err instanceof Error ? err.message : String(err))
    } else if (path) {
      await removeFile(path)
    }
    return false
  } finally {
    await handle?.remove()
  }
}

/** Drop the record and the bytes. Safe to call for an episode that was never downloaded. */
export async function deleteEpisode(slug: string): Promise<void> {
  const store = useDownloadsStore()
  await store.ensureLoaded()
  const path = store.entry(slug)?.path ?? null
  await store.unmark(slug)
  if (path) await removeFile(path)
}

/** Best-effort unlink — a missing file is already the desired end state. */
async function removeFile(path: string): Promise<void> {
  try {
    await Filesystem.deleteFile({ directory: DOWNLOAD_DIR, path })
  } catch {
    // Already gone, or never written.
  }
}
