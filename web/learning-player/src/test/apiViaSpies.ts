import { vi } from 'vitest'
import * as api from '../services/api'
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
 * (`pageFavoritesLocally`). The server's own paging is tested in the API's integration tests.
 */
export function favoritesViaGetFavorites(): void {
  vi.spyOn(api, 'getFavoriteRefs').mockImplementation(async () =>
    api.favoriteRefsOf(await api.getFavorites()),
  )
  vi.spyOn(api, 'getFavoritesPage').mockImplementation(async (query) =>
    api.pageFavoritesLocally(await api.getFavorites(), query),
  )
}

/**
 * Answer the paged capture reads from the test's own `getNotes` / `getHighlights` spies, cut by the
 * same code an older server's full list goes through (`pageNotesLocally` / `pageHighlightsLocally`).
 * The server's own paging is tested in the API's integration tests.
 */
export function capturesViaFullLists(): void {
  vi.spyOn(api, 'getNotesPage').mockImplementation(async (query) =>
    api.pageNotesLocally(await api.getNotes(), await api.getHighlights().catch(() => []), query),
  )
  vi.spyOn(api, 'getHighlightsPage').mockImplementation(async (query) =>
    api.pageHighlightsLocally(
      await api.getHighlights(),
      await api.getNotes('highlight').catch(() => []),
      query,
    ),
  )
}
