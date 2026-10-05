/**
 * Share an entity's card — the SERVER's card (`server/og/card.py`), the one design for every kind
 * (operator 2026-10-05). It replaced a second, plainer card the app drew itself on a canvas, which
 * had drifted from the server's: no artwork, no topics, an empty black middle.
 *
 * Native: the image goes to the share sheet as a FILE (the same path as the Obsidian export) — never
 * a text file, which is what the old card fell back to when the web view could not share files.
 * Web: the browser's file share where it has one (phones), else a PNG download.
 */
import { fetchShareCard, type ShareCardKind } from '../services/api'
import { deliverFile, isNative } from '../services/native'
import { exportFilename } from '../utils/exportFilename'

export async function shareCard(kind: ShareCardKind, id: string, title: string): Promise<void> {
  const blob = await fetchShareCard(kind, id)
  const name = exportFilename(title, 'png', 'closelistening-card')
  if (isNative()) {
    await deliverFile(name, blob)
    return
  }
  const file = new File([blob], name, { type: 'image/png' })
  if (typeof navigator.canShare === 'function' && navigator.canShare({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title })
      return
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return // the reader closed the sheet — done
    }
  }
  await deliverFile(name, blob)
}
