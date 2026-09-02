/**
 * What plays after the current episode (#1905/#1906).
 *
 * Extracted from the shell so it can be tested: auto-advance runs with NO view mounted, it is the
 * one path that must work with no network for a downloaded queue, and it is also where a listen
 * event is recorded — none of which any view-level test can observe.
 */

import { getAudioSource, getEpisode } from './api'
import { localArtworkFor, localSourceFor } from './downloads'
import { useDownloadsStore } from '../stores/downloads'
import { episodeArtwork } from '../utils/episode'
import type { NextUp } from '../stores/player'

export async function resolveNextUpFor(
  currentSlug: string | null,
  nextAfter: (slug: string) => string | null,
): Promise<NextUp | null> {
  if (!currentSlug) return null
  const next = nextAfter(currentSlug)
  if (!next) return null

  const [src, detail] = await Promise.all([
    getAudioSource(next).catch(() => null),
    getEpisode(next).catch(() => null),
  ])

  // Offline both calls fail — but a downloaded next episode can still play. Without this,
  // auto-advance stopped at the end of every episode even with the whole queue on disk, which is
  // exactly the journey offline downloads exist for.
  const entry = useDownloadsStore().entry(next)
  const localSrc = localSourceFor(next)
  if (!src?.url && !localSrc) return null

  return {
    slug: next,
    url: src?.url ?? localSrc ?? '',
    // Metadata comes from the registry when the API cannot answer: auto-advance runs with no view
    // mounted, so without a title here the lock screen keeps showing the PREVIOUS episode.
    title: detail?.title ?? entry?.title ?? null,
    artwork: (detail ? episodeArtwork(detail) : null) ?? localArtworkFor(next),
  }
}
