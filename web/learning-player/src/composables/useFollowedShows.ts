import { computed, ref } from 'vue'
import { getPodcasts } from '../services/api'
import { useAuthStore } from '../stores/auth'
import { useLibraryStore } from '../stores/library'
import type { Podcast } from '../services/types'

/**
 * The shows the user follows, resolved to full `Podcast` records — shared by Home's "Your shows"
 * dispatch rail and the Library "Shows" management tab so the two can't drift (UXS-014: define the
 * pattern once, apply it app-wide).
 *
 * The library API returns subscriptions (feed_id + title + added_at), not catalogue metadata, so
 * artwork/episode counts are joined from the public catalogue; a followed feed that has left the
 * corpus still renders from its stored title rather than vanishing. `shows` is derived, so following
 * or unfollowing anywhere updates every consumer instantly with no reload.
 */
export function useFollowedShows() {
  const auth = useAuthStore()
  const library = useLibraryStore()
  const catalogue = ref<Podcast[]>([])

  /** Load the public catalogue (artwork) + the user's follows. Both must resolve for "you follow
   *  nothing" to be a truthful render. Wrap this in the caller's own section-state. */
  async function load(): Promise<void> {
    const [cat] = await Promise.all([
      getPodcasts(),
      auth.isAuthenticated ? library.ensureLoaded() : Promise.resolve(),
    ])
    catalogue.value = cat
  }

  const shows = computed<Podcast[]>(() => {
    if (!auth.isAuthenticated) return []
    const byId = new Map(catalogue.value.map((p) => [p.feed_id, p]))
    return library.items.map(
      (i) =>
        byId.get(i.feed_id) ?? {
          feed_id: i.feed_id,
          title: i.title,
          artwork_url: null,
          image_url: null,
          description: null,
          episode_count: 0,
        },
    )
  })

  /** Catalogue shows the user does NOT follow — what an empty state offers so following is
   *  completable in place rather than described. */
  const suggested = computed<Podcast[]>(() =>
    catalogue.value.filter((p) => !library.has(p.feed_id)).slice(0, 6),
  )

  return { catalogue, load, shows, suggested }
}
