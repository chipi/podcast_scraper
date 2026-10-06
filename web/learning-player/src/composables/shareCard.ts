/**
 * Share a card — the SERVER's card (`server/og/card.py`), the one design for every kind and for a
 * highlight (operator 2026-10-05). It replaced two canvas cards the app drew itself (an entity card
 * and a highlight card), each plainer than the server's and each drifting from it.
 *
 * Native: the image goes to the share sheet as a FILE (the same path as the Obsidian export) — never
 * a text file, which is what both old cards fell back to when the web view could not share files.
 * Web: the browser's file share where it has one (phones), else a PNG download.
 */
import type { Highlight } from '../services/types'
import { fetchHighlightCard, fetchShareCard, type ShareCardKind } from '../services/api'
import { deliverFile, isNative } from '../services/native'
import { exportFilename } from '../utils/exportFilename'

/** Hand a card PNG to the platform. */
async function shareImage(blob: Blob, title: string): Promise<void> {
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

/** An entity's card: episode, show, topic, person, storyline, theme or organization. */
export async function shareCard(kind: ShareCardKind, id: string, title: string): Promise<void> {
  await shareImage(await fetchShareCard(kind, id), title)
}

/** A highlight's quote card, named after its episode. */
export async function shareHighlightCard(h: Highlight, episodeTitle: string): Promise<void> {
  await shareImage(await fetchHighlightCard(h.id), `${episodeTitle} highlight`)
}
