<script setup lang="ts">
/**
 * One Discover rail of recommendations (ADR-163) — the shape every rail shares: episodes, each
 * card led by WHY it is here (`RecommendationWhy`: the reason line, the quote that grounds it,
 * "▶ Play from"), measured in Umami through `useRailTelemetry(rail)`.
 *
 * The rail supplies its id, its title and how to load its cards; everything else is the same for
 * every rail, so a new rail cannot drift in how it explains itself or how it is measured.
 *
 * - The card opens the episode; "Play from" opens it playing at the quote (ADR-163 §8); "▶ Moments"
 *   (the shared `MomentsLink`, only when the episode has moments) samples it: its strongest moments,
 *   back to back (Discover arc Step 3, "Sample an episode").
 * - A rail with nothing to show hides itself (thin data is not padded out).
 * - `reloadKey` changing (e.g. the page scope) reloads it and counts a fresh impression.
 * - "Less like this" (ADR-163 §6), on a card whose reason names a subject: hides that topic, person
 *   or storyline from every rail (server-side), drops its cards here at once, and offers Undo.
 *   Profile › Interests lists what was hidden. Not offered on the cold-start fallback (Everyone's).
 * - `fallback` (ADR-163 §6, cold start): when `load` finds nothing, Everyone's cards are shown in
 *   its place, measured as scope `everyone`, with the `fallback-note` slot under the heading — the
 *   page says whose they are and how to make them yours. Never presented as the listener's own.
 * - Discover is kept alive, so returning to it reloads too — a highlight made a minute ago must be
 *   able to seed "More from what you captured" without an app restart.
 */
import { onActivated, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { suggestLess, undoSuggestLess, type TrendingEpisodeCard } from '../services/api'
import type { RailId } from '../services/analytics'
import { episodeArtwork } from '../utils/episode'
import { useRailTelemetry } from '../composables/useRailTelemetry'
import MomentsLink from './MomentsLink.vue'
import RecommendationWhy from './RecommendationWhy.vue'
import SectionHeading from './SectionHeading.vue'

const props = defineProps<{
  rail: RailId
  title: string
  load: () => Promise<TrendingEpisodeCard[]>
  /** Telemetry scope of what this rail shows. */
  scope: 'mine' | 'everyone'
  /** The rail's own testids, passed as literals so the surface map can name them. */
  cardsTestid: string
  cardTestid: string
  openTestid: string
  reloadKey?: string
  /** Everyone's cards, shown (and labelled by the `fallback-note` slot) when `load` finds none. */
  fallback?: () => Promise<TrendingEpisodeCard[]>
}>()

const { t } = useI18n()
const cards = ref<TrendingEpisodeCard[]>([])
const usingFallback = ref(false)
/** The subject just hidden, for the Undo line. */
const lessOf = ref<{ token: string; label: string } | null>(null)

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
  // Only Everyone's outside-share cards left: the server shows no rail then (the share only rides
  // along with the listener's own cards), so neither does this.
  cards.value = left.every((x) => x.why.outside_habits) ? [] : left
  lessOf.value = { token: subject.id, label: subject.label }
}

async function undoLess(): Promise<void> {
  const hidden = lessOf.value
  if (!hidden) return
  await undoSuggestLess(hidden.token).catch(() => undefined)
  lessOf.value = null
  reload()
}
let telemetry = useRailTelemetry(props.rail)

let latest = 0
async function fetchCards(): Promise<void> {
  // Only the latest load may write: a scope switch starts a new one while the old is in flight, and
  // the old one finishing last must not put the previous scope's answer (or nothing) on screen.
  const run = ++latest
  const { load, fallback, scope } = props
  let got = await load().catch(() => [] as TrendingEpisodeCard[])
  let fellBack = false
  if (!got.length && fallback) {
    got = await fallback().catch(() => [] as TrendingEpisodeCard[])
    fellBack = got.length > 0
  }
  if (run !== latest) return
  usingFallback.value = fellBack
  cards.value = got
  telemetry.shown(cards.value, fellBack ? 'everyone' : scope)
}
function reload(): void {
  telemetry = useRailTelemetry(props.rail) // a new visit or scope is a new rail impression
  void fetchCards()
}
onMounted(fetchCards)
let mounted = false
onActivated(() => {
  if (mounted) reload() // the first activation is the mount itself
  mounted = true
})
watch(() => props.reloadKey, reload)
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
          <RouterLink
            :to="{ name: 'player', params: { slug: c.episode.slug } }"
            class="lp-tap flex min-w-0 items-center gap-3"
            :data-testid="openTestid"
            @click="telemetry.tapped(c, i, 'open')"
          >
            <img
              v-if="episodeArtwork(c.episode)"
              :src="episodeArtwork(c.episode) ?? ''"
              alt=""
              class="h-10 w-10 shrink-0 rounded-md object-cover"
            />
            <span class="min-w-0">
              <span class="line-clamp-2 text-sm font-bold">{{ c.episode.title }}</span>
              <span class="block break-words text-xs text-muted">{{ c.episode.podcast_title }}</span>
            </span>
          </RouterLink>
          <RecommendationWhy :why="c.why" @play="telemetry.tapped(c, i, 'play_from')" />
          <div class="mt-auto flex flex-wrap items-center justify-between gap-x-4">
            <MomentsLink
              v-if="c.episode.has_moments"
              :slug="c.episode.slug"
              @click="telemetry.tapped(c, i, 'moments')"
            />
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
  </section>
</template>
