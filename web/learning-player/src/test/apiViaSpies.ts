import { vi } from 'vitest'
import * as api from '../services/api'
import {
  pageCollectionLocally,
  pageFavoritesLocally,
  pageHighlightsLocally,
  pageNotesLocally,
  pagePodcastsLocally,
  pageResurfacingLocally,
} from './localPagers'
import type { EpisodeDetail } from '../services/types'

/**
 * Route `getEpisodesBatch` through the test's own `getEpisode` spy, one call per slug.
 *
 * The lists hydrate with ONE batch request now, but their specs describe what each screen asks for
 * per episode ("the ten rows it shows", "the next five only on Show more"). Answering the batch from
 * the per-slug spy keeps those assertions meaningful: a slug the screen did not ask for never
 * reaches `getEpisode`. A slug whose `getEpisode` rejects is absent, as the server reports it.
 * The batch request itself is tested in `services/api.test.ts`.
 */
export function batchViaGetEpisode(): void {
  vi.spyOn(api, 'getEpisodesBatch').mockImplementation(async (slugs: string[]) => {
    const out: Record<string, EpisodeDetail> = {}
    for (const s of [...new Set(slugs)]) {
      try {
        out[s] = await api.getEpisode(s)
      } catch {
        // absent, as an unknown slug is
      }
    }
    return out
  })
}

/**
 * Answer the favourites reads from the test's own `getFavorites` spy: identities derived from it,
 * and each Saved page cut from it by the same code an older server's full list goes through
 * (`pageFavoritesLocally`, test/localPagers.ts). The server's own paging is tested in the API's integration tests.
 */
export function favoritesViaGetFavorites(): void {
  vi.spyOn(api, 'getFavoriteRefs').mockImplementation(async () =>
    api.favoriteRefsOf(await api.getFavorites()),
  )
  vi.spyOn(api, 'getFavoritesPage').mockImplementation(async (query) =>
    pageFavoritesLocally(await api.getFavorites(), query),
  )
}

/**
 * Answer the paged capture reads from the test's own `getNotes` / `getHighlights` spies, cut by the
 * same code an older server's full list goes through (`pageNotesLocally` / `pageHighlightsLocally`).
 * The server's own paging is tested in the API's integration tests.
 */
export function capturesViaFullLists(): void {
  vi.spyOn(api, 'getNotesPage').mockImplementation(async (query) =>
    pageNotesLocally(await api.getNotes(), await api.getHighlights().catch(() => []), query),
  )
  vi.spyOn(api, 'getHighlightsPage').mockImplementation(async (query) =>
    pageHighlightsLocally(
      await api.getHighlights(),
      await api.getNotes('highlight').catch(() => []),
      query,
    ),
  )
}

/** Answer `getResurfacingPage` from the test's own `getResurfacing` spy (older-server paging). */
export function resurfacingViaGetResurfacing(): void {
  vi.spyOn(api, 'getResurfacingPage').mockImplementation(async (query) =>
    pageResurfacingLocally(await api.getResurfacing(), query),
  )
}

/** Answer `getCollectionPage` from the test's own `getCollection` spy (older-server paging). */
export function collectionsViaGetCollection(): void {
  vi.spyOn(api, 'getCollectionPage').mockImplementation(async (id, query) =>
    pageCollectionLocally(await api.getCollection(id), query),
  )
}

/**
 * Answer the paged catalogue reads (`getPodcastsPage`, `getPodcastsByIds`) from the test's own
 * `getPodcasts` spy, cut by `pagePodcastsLocally` — the older-server path.
 */
export function podcastsViaGetPodcasts(): void {
  vi.spyOn(api, 'getPodcastsPage').mockImplementation(async (query) =>
    pagePodcastsLocally(await api.getPodcasts(), query),
  )
  vi.spyOn(api, 'getPodcastsByIds').mockImplementation(async (ids) =>
    pagePodcastsLocally(await api.getPodcasts(), { feedIds: ids, limit: ids.length || 1 }).items,
  )
}
