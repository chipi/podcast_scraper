<script setup lang="ts">
/**
 * Your Library (UXS-014) — the hub for everything per-user, tabbed: Saved (favourited things, in
 * per-kind sections — episodes, insights, …) · Highlights · Revisit · Queue · Recent. One place,
 * tabbed; the Saved tab grows a new section as new favourite kinds arrive. Auth-gated.
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
defineOptions({ name: 'LibraryView' }) // stable name for <keep-alive :include> (App.vue)
import { RouterLink, useRoute } from 'vue-router'
import { useFavoritesStore } from '../stores/favorites'
import { useSavedQueriesStore } from '../stores/savedQueries'
import { useUserPreferencesStore } from '../stores/userPreferences'
import { useFollowedShows } from '../composables/useFollowedShows'
import { useSectionState } from '../composables/useSectionState'
import { formatTime } from '../player/transcriptSync'
import EpisodeCard from '../components/EpisodeCard.vue'
import SectionStatus from '../components/SectionStatus.vue'
import ShowTile from '../components/ShowTile.vue'
import FollowedInterests from '../components/FollowedInterests.vue'
import HighlightsView from './HighlightsView.vue'
import ResurfacingInbox from './ResurfacingInbox.vue'
import CollectionsView from './CollectionsView.vue'

const { t } = useI18n()
const favorites = useFavoritesStore()
const savedQueries = useSavedQueriesStore()
const userPrefs = useUserPreferencesStore()

// Tabs: Shows (the feeds you follow) · Saved · Revisit · Queue · Recent — five fit a phone row with
// no scroll. Highlights + Collections are now SECTIONS inside Saved (everything you deliberately kept
// under one tab), which is what kept the strip short. Shows is the follow-management home.
// Queue + Recent moved to the player surface (#1838) — reachable from the mini/full player's queue
// button, not Library — which frees the tab strip (and the slot the Collections tab will take).
type Tab = 'shows' | 'saved' | 'revisit'
const tabs: { key: Tab; label: string }[] = [
  { key: 'shows', label: 'library.following' },
  { key: 'saved', label: 'library.saved' },
  { key: 'revisit', label: 'library.revisit' },
]
// Home's "See all N shows →" deep-links here with ?tab=shows so it lands on the follows, not Saved.
const route = useRoute()
const initialTab = String(route.query.tab || '')
const tab = ref<Tab>(tabs.some((tb) => tb.key === initialTab) ? (initialTab as Tab) : 'saved')

// Followed shows — the same derivation Home's "Your shows" rail uses (shared so they can't drift).
// Section-state so a catalogue/library outage renders error+retry, never a fake "you follow nothing".
const { shows: followedShows, suggested: suggestedShows, load: loadFollows } = useFollowedShows()
const showsSection = useSectionState<null>(null)
function loadFollowedShows(): Promise<void> {
  return showsSection.load(async () => {
    await loadFollows()
    return null
  })
}

onMounted(async () => {
  await favorites.ensureLoaded()
  // #1261-8: fire-and-forget the USERPREFS-1 hydrate so the saved-queries store picks up the
  // cross-device list (the preferences endpoint being offline shouldn't gate the tab).
  void userPrefs.hydrate()
  void loadFollowedShows()
})
</script>

<template>
  <section>
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">{{ t('library.title') }}</h1>

    <!-- Tabs — five equal-width columns so all fit ONE phone row (Recent used to wrap to a second
         row at px-3/text-sm). flex-1 + text-xs + tight padding keeps them on one line; whitespace-
         nowrap keeps each label intact rather than truncating. -->
    <div class="mb-6 flex gap-0.5 border-b border-border">
      <button
        v-for="tb in tabs"
        :key="tb.key"
        type="button"
        class="-mb-px flex-1 whitespace-nowrap border-b-2 px-1 py-2 text-center text-xs font-bold transition"
        :class="tab === tb.key ? 'border-accent text-canvas-foreground' : 'border-transparent text-muted hover:text-canvas-foreground'"
        @click="tab = tb.key"
      >{{ t(tb.label) }}</button>
    </div>

    <!-- Following — everything you follow: shows (feeds) plus the topics / people / storylines you
         followed via ＋. The follow-management home: Home's "See all N shows →" deep-links here
         (?tab=shows). Sectioned by kind, like Saved. -->
    <div v-show="tab === 'shows'">
      <section class="mb-6">
        <h3 class="lp-kicker mb-2">{{ t('library.followingShows') }}</h3>
        <SectionStatus :phase="showsSection.phase.value" :rows="2" @retry="loadFollowedShows" />
        <div
          v-if="showsSection.isReady.value && !followedShows.length"
          class="rounded-xl border border-dashed border-border p-4"
        >
          <p class="text-sm text-muted">{{ t('library.showsEmpty') }}</p>
          <ul v-if="suggestedShows.length" class="mt-3 grid grid-cols-3 gap-3 sm:grid-cols-6">
            <li v-for="p in suggestedShows" :key="p.feed_id"><ShowTile :show="p" followable /></li>
          </ul>
          <RouterLink
            :to="{ name: 'catalog' }"
            class="mt-3 inline-block text-xs font-bold text-accent no-underline"
          >{{ t('library.showsBrowse') }}</RouterLink>
        </div>
        <ul
          v-else-if="followedShows.length"
          class="grid grid-cols-3 gap-3 sm:grid-cols-4"
          data-testid="library-shows-grid"
        >
          <li v-for="p in followedShows" :key="p.feed_id"><ShowTile :show="p" followable /></li>
        </ul>
      </section>

      <!-- Topics / People / Storylines you follow (previously invisible — the interests profile). -->
      <FollowedInterests />
    </div>

    <!-- Saved — everything you deliberately kept, one section per kind: searches, episodes, insights,
         plus Highlights and Collections (folded in from their old tabs to keep the strip to five).
         Each section owns its own presence/empty state, so there is no separate "nothing saved" line. -->
    <div v-show="tab === 'saved'">
        <!-- #1261-8: Saved searches — power-listener persistent queries.
             Tap the query to re-run the search; ×  removes it. -->
        <section
          v-if="savedQueries.count"
          class="mb-6"
          data-testid="saved-searches-section"
        >
          <h2 class="lp-section mb-2">{{ t('library.savedSearches') }}</h2>
          <ul class="flex flex-col">
            <li
              v-for="q in savedQueries.list"
              :key="q.scope + '|' + q.q"
              class="flex items-center gap-2 border-b border-border py-2"
            >
              <RouterLink
                :to="{ name: 'search', query: { q: q.q, scope: q.scope } }"
                class="min-w-0 flex-1 text-sm font-semibold leading-snug text-canvas-foreground no-underline"
              >
                {{ q.q }}
                <span class="lp-kicker ml-1 font-normal">
                  {{ q.scope === 'mine' ? t('search.scopeMine') : t('search.scopeAll') }}
                </span>
              </RouterLink>
              <button
                type="button"
                class="shrink-0 rounded-full border border-border px-2 py-1 text-xs font-semibold text-muted transition hover:text-canvas-foreground"
                :aria-label="t('library.savedSearchRemove', { q: q.q })"
                @click="savedQueries.remove(q.q, q.scope)"
              >
                ×
              </button>
            </li>
          </ul>
        </section>
        <!-- Episodes -->
        <section v-if="favorites.episodes.length" class="mb-6">
          <h2 class="lp-section mb-2">{{ t('library.savedEpisodes') }}</h2>
          <div class="flex flex-col">
            <EpisodeCard v-for="e in favorites.episodes" :key="e.slug" :episode="e" />
          </div>
        </section>
        <!-- Insights (snapshot text + jump-to-moment).
             LEGACY, read-only since #1593: insights used to be savable BOTH here (heart) and to
             Highlights (bookmark) — same text, two lists. The heart was removed from the Knowledge
             Panel, so nothing writes here any more. Existing saves stay readable and removable, and
             this section disappears on its own once a user's are gone. Do not add a new write path;
             Highlights is the destination. -->
        <section v-if="favorites.insights.length">
          <h2 class="lp-section mb-2">{{ t('library.savedInsights') }}</h2>
          <ul class="flex flex-col">
            <li v-for="ins in favorites.insights" :key="ins.ref" class="border-b border-border py-3">
              <RouterLink
                v-if="ins.episode_slug"
                :to="{ name: 'player', params: { slug: ins.episode_slug }, query: ins.start_ms != null ? { t: String(Math.floor(ins.start_ms / 1000)) } : {} }"
                class="block no-underline text-canvas-foreground"
              >
                <p class="text-sm font-semibold leading-snug">{{ ins.text }}</p>
                <p class="lp-kicker mt-1">
                  {{ ins.podcast_title
                  }}<template v-if="ins.podcast_title && ins.start_ms != null"> · </template
                  ><span v-if="ins.start_ms != null" class="text-accent">▶ {{ formatTime(ins.start_ms / 1000) }}</span>
                </p>
              </RouterLink>
              <p v-else class="text-sm font-semibold leading-snug text-muted">{{ ins.text }}</p>
            </li>
          </ul>
        </section>
        <!-- Highlights — captured moments / spans / saved insights, grouped by episode, with notes.
             Folded in from its old tab; HighlightsView owns its own empty state. -->
        <section class="mb-6">
          <h2 class="lp-section mb-2">{{ t('library.highlights') }}</h2>
          <HighlightsView />
        </section>
        <!-- Collections — boards of highlights. Folded in from its old tab; owns its empty state. -->
        <section>
          <h2 class="lp-section mb-2">{{ t('library.collections') }}</h2>
          <CollectionsView />
        </section>
    </div>

    <!-- Revisit — spaced resurfacing of past highlights with reflection prompts. -->
    <div v-show="tab === 'revisit'">
      <ResurfacingInbox />
    </div>
  </section>
</template>
