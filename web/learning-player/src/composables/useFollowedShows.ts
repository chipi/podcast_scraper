import { computed, ref, watch } from 'vue'
import { getPodcastsByIds, getPodcastsPage, getSuggestedShows } from '../services/api'
import { useAuthStore } from '../stores/auth'
import { useLibraryStore } from '../stores/library'
import type { Podcast } from '../services/types'

/**
 * The shows the user follows, resolved to full `Podcast` records — shared by Home's "Your shows"
 * dispatch rail and the Library "Shows" management tab so the two can't drift (UXS-014: define the
 * pattern once, apply it app-wide).
 *
 * The library API returns subscriptions (feed_id + title + added_at), not catalogue metadata, so
 * artwork/episode counts are joined from the catalogue — for the FOLLOWED shows only, by id, since
 * 2026-10-08 (it loaded every show to join a handful); a followed feed that has left the
 * corpus still renders from its stored title rather than vanishing. `shows` is derived, so following
 * or unfollowing anywhere updates every consumer instantly with no reload.
 */
export function useFollowedShows() {
  const auth = useAuthStore()
  const library = useLibraryStore()
  /** Catalogue records for the followed shows (looked up by id). */
  const catalogue = ref<Podcast[]>([])
  /** Shows to offer when the user follows none (the server leaves out the followed ones). */
  const suggestedShows = ref<Podcast[]>([])

  /**
   * Load the public catalogue (artwork) + the user's follows.
   *
   * Both must resolve for "you follow nothing" to be a truthful render — this comment already said
   * so, and the code did not enforce it. `library.ensureLoaded()` never throws (deliberately: boot
   * and reconnect both call it and must not be aborted by an offline library). So a failed
   * `/library` resolved quietly, the section went READY, and Library rendered "You're not following
   * any shows yet" plus six shows to follow — a confident claim about the user's data, made from no
   * data at all.
   *
   * The store leaves `loaded` false in exactly that case — no fresh answer AND no cache — so that
   * is the condition to convert into a throw. The caller's section-state then renders error+retry,
   * which is what its own docblock always promised.
   */
  async function load(): Promise<void> {
    if (auth.isAuthenticated) await library.ensureLoaded()
    if (auth.isAuthenticated && !library.loaded) {
      throw new Error('followed shows unavailable')
    }
    const ids = library.items.map((i) => i.feed_id)
    const [cat, offer] = await Promise.all([
      ids.length ? getPodcastsByIds(ids) : Promise.resolve([] as Podcast[]),
      // Signed out there is nobody to rank for: the catalogue's newest few stand in.
      auth.isAuthenticated
        ? getSuggestedShows(6).catch(() => [] as Podcast[])
        : getPodcastsPage({ limit: 6 }).then((p) => p.items).catch(() => [] as Podcast[]),
    ])
    catalogue.value = cat
    suggestedShows.value = offer
  }

  // A show followed after the load (from a suggestion, say) needs its record too.
  watch(
    () => library.items.map((i) => i.feed_id),
    async (ids) => {
      const have = new Set(catalogue.value.map((p) => p.feed_id))
      const missing = ids.filter((id) => !have.has(id))
      if (!missing.length) return
      const got = await getPodcastsByIds(missing).catch(() => [] as Podcast[])
      catalogue.value = [...catalogue.value, ...got]
    },
  )

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
    suggestedShows.value.filter((p) => !library.has(p.feed_id)).slice(0, 6),
  )

  return { catalogue, load, shows, suggested }
}
