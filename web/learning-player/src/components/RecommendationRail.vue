<script setup lang="ts">
/**
 * One Discover rail of recommendations (ADR-163) — the shape every rail shares: episodes, each
 * card led by WHY it is here (`RecommendationWhy`: the reason line, the quote that grounds it,
 * "▶ Play from"), measured in Umami through `useRailTelemetry(rail)`.
 *
 * Paged on the SERVER (operator 2026-10-11): Discover is a taste, not a scroll. The page hands the
 * rail its first page (`GET /discover/rails`, one card per episode across the page) and how to ask
 * for more; "Show more (N)" fetches the next page (`GET /discover/rails/{rail}`), N being what the
 * server says is left. "Show less" returns to the first page.
 *
 * - The card opens the episode; "Play from" opens it playing at the quote (ADR-163 §8); "▶ Moments"
 *   (the shared `MomentsLink`, only when the episode has moments) samples it.
 * - "Less like this" (ADR-163 §6), on a card whose reason names a subject: hides that topic, person
 *   or storyline from every rail (server-side), drops its cards here at once, and offers Undo —
 *   which asks the page to reload. Not offered on the cold-start fallback (Everyone's).
 * - `page.fallback` (cold start, decided by the server): Everyone's cards standing in for an empty
 *   Mine, with the `fallback-note` slot under the heading, measured as scope `everyone`.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  suggestLess,
  undoSuggestLess,
  type DiscoverRail,
  type TrendingEpisodeCard,
} from '../services/api'
import type { RailId } from '../services/analytics'
import { episodeArtwork } from '../utils/episode'
import { useRailTelemetry } from '../composables/useRailTelemetry'
import EpisodeActions from './EpisodeActions.vue'
import RecommendationWhy from './RecommendationWhy.vue'
import SectionHeading from './SectionHeading.vue'
import ShowAllToggle from './ShowAllToggle.vue'

const props = defineProps<{
  rail: RailId
  title: string
  /** The rail's first page, from the server. */
  page: DiscoverRail
  /** The next page from the server: `offset` counts within this rail, after the rails above. */
  more: (offset: number, limit: number) => Promise<DiscoverRail>
  /** Telemetry scope of what this rail shows. */
  scope: 'mine' | 'everyone'
  /** The rail's own testids, passed as literals so the surface map can name them. */
  cardsTestid: string
  cardTestid: string
  openTestid: string
}>()
const emit = defineEmits<{ (e: 'changed'): void }>()

/** Cards each "Show more" asks for: two rows on desktop. */
const MORE = 4

const { t } = useI18n()
const cards = ref<TrendingEpisodeCard[]>([])
const total = ref(0)
const loadingMore = ref(false)
/** The subject just hidden, for the Undo line. */
const lessOf = ref<{ token: string; label: string } | null>(null)
let telemetry = useRailTelemetry(props.rail)
const usingFallback = computed(() => props.page.fallback)
const remaining = computed(() => Math.max(0, total.value - cards.value.length))
const expanded = computed(() => cards.value.length > props.page.items.length)

// A new first page (a load, a scope switch, a return to Discover) is a new rail impression.
watch(
  () => props.page,
  (page) => {
    cards.value = [...page.items]
    total.value = page.total
    lessOf.value = null
    telemetry = useRailTelemetry(props.rail)
    telemetry.shown(cards.value, page.fallback ? 'everyone' : props.scope)
  },
  { immediate: true },
)

async function showMoreOrLess(): Promise<void> {
  if (remaining.value <= 0) {
    cards.value = [...props.page.items]
    total.value = props.page.total
    return
  }
  if (loadingMore.value) return
  loadingMore.value = true
  try {
    const next = await props.more(cards.value.length, MORE)
    const have = new Set(cards.value.map((c) => c.episode.slug))
    cards.value = [...cards.value, ...next.items.filter((c) => !have.has(c.episode.slug))]
    total.value = next.total
  } catch {
    // The page stays as it was; the control is still there to try again.
  } finally {
    loadingMore.value = false
  }
}

async function less(c: TrendingEpisodeCard, i: number): Promise<void> {
  const subject = c.why.subject
  if (!subject) return
  telemetry.tapped(c, i, 'less')
  try {
    await suggestLess(subject.id, subject.label)
  } catch {
    return // nothing changed server-side: keep the card rather than pretend
  }
  const left = cards.value.filter((x) => x.why.subject?.id !== subject.id)
  total.value -= cards.value.length - left.length
  cards.value = left
  lessOf.value = { token: subject.id, label: subject.label }
}

async function undoLess(): Promise<void> {
  const hidden = lessOf.value
  if (!hidden) return
  await undoSuggestLess(hidden.token).catch(() => undefined)
  lessOf.value = null
  emit('changed')
}
</script>

<template>
  <section v-if="cards.length || lessOf" class="mt-7" :data-testid="cardsTestid">
    <SectionHeading :title="title" />
    <slot v-if="usingFallback" name="fallback-note" />
    <p v-if="lessOf" class="-mt-1 mb-3 flex flex-wrap items-center gap-x-3 text-sm text-muted" role="status" data-testid="rail-less-done">
      <span>{{ t('discover.less.done', { subject: lessOf.label }) }}</span>
      <button type="button" class="lp-tap font-bold text-accent" data-testid="rail-less-undo" @click="undoLess">
        {{ t('discover.less.undo') }}
      </button>
    </p>
    <ul class="grid gap-3 lg:grid-cols-2">
      <li v-for="(c, i) in cards" :key="c.episode.slug">
        <article class="flex h-full flex-col gap-3 rounded-xl border border-border p-4" :data-testid="cardTestid">
          <div class="flex min-w-0 items-center gap-3">
            <RouterLink
              :to="{ name: 'player', params: { slug: c.episode.slug } }"
              class="shrink-0"
              tabindex="-1"
              aria-hidden="true"
              @click="telemetry.tapped(c, i, 'open')"
            >
              <img
                v-if="episodeArtwork(c.episode)"
                :src="episodeArtwork(c.episode) ?? ''"
                alt=""
                class="h-10 w-10 rounded-md object-cover"
              />
            </RouterLink>
            <span class="min-w-0">
              <RouterLink
                :to="{ name: 'player', params: { slug: c.episode.slug } }"
                class="lp-tap line-clamp-2 text-sm font-bold no-underline"
                :data-testid="openTestid"
                @click="telemetry.tapped(c, i, 'open')"
              >{{ c.episode.title }}</RouterLink>
              <!-- The show is named, so it is a way there (operator 2026-10-11) — a link only when the
                   card knows the feed: routing to a podcast with no id throws (EpisodeCard's rule). -->
              <RouterLink
                v-if="c.episode.feed_id && c.episode.podcast_title"
                :to="{ name: 'podcast', params: { feedId: c.episode.feed_id } }"
                class="lp-tap lp-show-name text-xs text-muted no-underline hover:text-canvas-foreground"
                data-testid="rail-show-link"
              >{{ c.episode.podcast_title }}</RouterLink>
              <span v-else class="lp-show-name text-xs text-muted">{{ c.episode.podcast_title }}</span>
            </span>
          </div>
          <RecommendationWhy :why="c.why" @play="telemetry.tapped(c, i, 'play_from')" />
          <!-- The standard episode actions, as on every episode card (UXS-014): save, queue, and ⋯ —
               which leads with "Play moments" when the episode has them. -->
          <div class="mt-auto flex flex-wrap items-center justify-between gap-x-4">
            <EpisodeActions :slug="c.episode.slug" :moments="!!c.episode.has_moments" />
            <button
              v-if="c.why.subject && !usingFallback"
              type="button"
              class="lp-tap relative -my-3 ml-auto min-h-[44px] px-1 text-xs text-muted transition hover:text-canvas-foreground"
              :aria-label="t('discover.less.label', { subject: c.why.subject.label })"
              data-testid="rail-less"
              @click="less(c, i)"
            >{{ t('discover.less.button') }}</button>
          </div>
        </article>
      </li>
    </ul>
    <ShowAllToggle
      v-if="total > page.items.length"
      :expanded="expanded"
      :count="total"
      :remaining="remaining"
      @toggle="showMoreOrLess"
    />
  </section>
</template>
