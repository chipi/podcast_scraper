<script setup lang="ts">
/**
 * "Recently played" — playback history, as its own component (operator 2026-09-23).
 *
 * It lived inside `QueuePanel`, which was fine while the panel was the only way to see a queue.
 * The masthead now carries a queue control at every width and that control goes to the `/queue`
 * PAGE — so leaving the history in the panel would have made the queue's two halves live on two
 * different surfaces depending on how you got there, and the page would have been the half that
 * omits things. A destination that shows half the thing is the same complaint as a panel that
 * works offline on top and not on the bottom.
 *
 * The point of this list is to FIND and resume something you heard, not to re-queue it — hence
 * `hide-queue` (the queue toggle moves into the ⋯) and `hide-played` (every row here is history, so
 * a per-row played marker would only repeat the heading).
 *
 * Cached, because the surface most worth having offline is the one listing what you already heard.
 * `GET /playback` has no cache of its own, so opening this on a plane used to show nothing at all.
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import EpisodeCard from './EpisodeCard.vue'
import { getEpisodesBatch, getPlaybackList } from '../services/api'
import { isArrayCache, readCached, writeCached } from '../services/contentCache'
import type { EpisodeDetail } from '../services/types'
import { summaryFromDetail } from '../utils/episode'
import { formatPlayedAt } from '../utils/format'

const { t, locale } = useI18n()

type RecentRow = { detail: EpisodeDetail; playedAt: number | null }

const RECENT_CACHE_KEY = 'queue.recent'

/**
 * The episode AND when it was last played.
 *
 * This used to keep only the `EpisodeDetail` and drop the position record it came from — so the
 * list was ORDERED by a timestamp it then refused to show, and two sittings with the same show
 * were indistinguishable.
 */
const recent = ref<RecentRow[]>([])
const loading = ref(true)

/** The stamp split for stacking — see `formatPlayedAt`. */
const playedAtParts = (at: number | null): { date: string; time: string } | null =>
  formatPlayedAt(at, locale.value)

onMounted(async () => {
  try {
    // Paint the cached copy FIRST, so an offline open shows the history instead of "nothing here".
    const cached = await readCached<RecentRow[]>(RECENT_CACHE_KEY, isArrayCache)
    if (cached?.length) {
      recent.value = cached
      loading.value = false
    }

    // The newest thirty from the server — not every position, to keep thirty (2026-10-08).
    const positions = await getPlaybackList({ limit: 30 }).catch(() => [])
    // No positions AND a cached copy means the request failed, not that the history is empty —
    // overwriting with [] would discard the only copy at the moment it is the only copy.
    if (!positions.length && cached?.length) return
    // One request for all of them (`/episodes/batch`), not one per row.
    const details = await getEpisodesBatch(positions.map((p) => p.slug)).catch(
      () => ({}) as Record<string, EpisodeDetail>,
    )
    const rows: RecentRow[] = positions.flatMap((p) =>
      details[p.slug] ? [{ detail: details[p.slug]!, playedAt: p.updated_at }] : [],
    )
    // Same rule one level down: every hydrate failing is a network fault, not an empty history.
    if (!rows.length && cached?.length) return
    recent.value = rows
    void writeCached(RECENT_CACHE_KEY, rows)
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <section>
    <h3 class="lp-kicker mb-2">{{ t('queue.recentlyPlayed') }}</h3>
    <p v-if="loading" class="text-sm text-muted">{{ t('catalog.loading') }}</p>
    <p v-else-if="!recent.length" class="text-sm text-muted">{{ t('queue.recentEmpty') }}</p>
    <div v-else class="flex flex-col" data-testid="queue-panel-recent">
      <EpisodeCard
        v-for="r in recent"
        compact
        hide-queue
        hide-played
        :key="r.detail.slug"
        :episode="summaryFromDetail(r.detail)"
      >
        <!-- Under the artwork: WHEN this was last played. The list is already ordered by it;
             stating it is what lets you tell two sittings apart.
             Date and time on SEPARATE lines — as one string it was wider than the artwork and
             stretched the whole left column. -->
        <template v-if="playedAtParts(r.playedAt)" #aside>
          <time
            :datetime="new Date((r.playedAt ?? 0) * 1000).toISOString()"
            class="block leading-tight"
          >
            <span class="block">{{ playedAtParts(r.playedAt)!.date }}</span>
            <span class="block">{{ playedAtParts(r.playedAt)!.time }}</span>
          </time>
        </template>
      </EpisodeCard>
    </div>
  </section>
</template>
