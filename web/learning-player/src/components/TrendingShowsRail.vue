<script setup lang="ts">
/**
 * Trending shows (RFC-103 §show) — the standard rail of standard `ShowTile`s, top N by velocity.
 *
 * It used to have a second shape: full-width artwork "slices" with a sparkline horizon, used only on
 * Home. Every rail is one shape now — `CardRail`, one slot width, three reserved title lines — and
 * Home no longer carries trending shows (operator 2026-10-05), so the slices went with it.
 *
 * A trending show's entity_id IS its feed_id; artwork joins from the shows SHOWN, looked up by id
 * (2026-10-08 — the caller used to pass the whole catalogue for five tiles).
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useSectionState } from '../composables/useSectionState'
import SectionStatus from './SectionStatus.vue'
import { getPodcastsByIds, getTrending } from '../services/api'
import type { Podcast, TrendingEntity } from '../services/types'
import CardRail from './CardRail.vue'
import SectionHeading from './SectionHeading.vue'
import ShowTile from './ShowTile.vue'

const props = withDefaults(
  defineProps<{
    title: string
    /** Records to join from; looked up by id when not given. */
    podcasts?: Podcast[]
    scope?: 'corpus' | 'mine'
    top?: number
    // Discover surfaces this rail above the dashboard with a "See all →" into the Shows tab (operator
    // 2026-09-14).
    seeAll?: boolean
  }>(),
  { scope: 'corpus', top: 5, seeAll: false },
)
const emit = defineEmits<{ (e: 'see-all'): void; (e: 'show-everyone'): void }>()
const { t } = useI18n()

// #1591 — a rejection lands in the error phase rather than collapsing into empty, so an outage
// stops rendering identically to "the corpus has no trending shows".
// Keyed per scope: "mine" and "everyone" are different lists, so neither may hydrate from, or stand
// in for, the other (2026-10-09).
const section = useSectionState<TrendingEntity[]>([], {
  cacheKey: () => `home.trendingshows.${props.scope}`,
})
function load(): Promise<void> {
  return section.load(() => getTrending('show', props.scope, 12))
}
void load()
// #2030 — re-fetch when the app-level trending lens (Corpus ⇄ My listening) flips.
watch(() => props.scope, load)
const shown = computed(() => section.data.value.slice(0, props.top))
const hasAny = computed(() => shown.value.length > 0)
// "Mine" with nothing in it says so and offers everyone's, as Trends does — a rail that silently
// vanished while the switch is lit reads as broken (operator 2026-10-09).
const mineEmpty = computed(() => props.scope === 'mine' && section.isReady.value && !hasAny.value)

// The standard ShowTile needs a full Podcast, so resolve each from the catalogue; a show that has
// left the catalogue still renders from its trending label rather than vanishing.
const looked = ref<Podcast[]>([])
watch(
  () => shown.value.map((e) => e.entity_id),
  async (ids) => {
    if (props.podcasts || !ids.length) return
    looked.value = await getPodcastsByIds(ids).catch(() => [] as Podcast[])
  },
  { immediate: true },
)
const shownPodcasts = computed<Podcast[]>(() => {
  const byId = new Map((props.podcasts ?? looked.value).map((p) => [p.feed_id, p]))
  return shown.value.map(
    (e) =>
      byId.get(e.entity_id) ?? {
        feed_id: e.entity_id,
        title: e.label,
        artwork_url: null,
        image_url: null,
        description: null,
        episode_count: 0,
      }
  )
})

/**
 * The header is exactly as wide as the tiles below it, so "all ›" ends where the last tile ends —
 * with two shows it sat at the far edge of an empty half-row (operator 2026-10-05). The counts match
 * `.lp-rail-item`: 3 to a row on a phone, 4 from `sm`; a full row is the full width.
 */
const tileHeader = computed<Record<string, string> | null>(() => {
  const n = shownPodcasts.value.length
  if (n === 0) return null
  return { '--n3': String(Math.min(n, 3)), '--n4': String(Math.min(n, 4)) }
})
</script>

<template>
  <section v-if="hasAny || mineEmpty || !section.isReady.value" class="mt-7" data-testid="trending-shows-rail">
    <div
      :class="tileHeader ? 'w-[calc(var(--n3)*(100%_-_1.5rem)/3_+_(var(--n3)_-_1)*0.75rem)] sm:w-[calc(var(--n4)*(100%_-_2.25rem)/4_+_(var(--n4)_-_1)*0.75rem)]' : ''"
      :style="tileHeader ?? undefined"
      data-testid="trending-shows-header"
    >
      <SectionHeading :title="title">
        <template v-if="seeAll" #action>
          <button
            type="button"
            class="whitespace-nowrap text-sm font-bold text-accent"
            data-testid="trending-shows-seeall"
            @click="emit('see-all')"
          >
            {{ t('home.seeAll') }} ›
          </button>
        </template>
      </SectionHeading>
    </div>
    <SectionStatus :phase="section.phase.value" :rows="2" @retry="load" />
    <div
      v-if="mineEmpty"
      class="rounded-xl border border-border px-4 py-3 text-sm text-muted"
      data-testid="trending-shows-mine-empty"
    >
      <p>{{ t('home.trendingShowsMineEmpty') }}</p>
      <button
        type="button"
        class="mt-2 font-bold text-accent"
        data-testid="trending-shows-show-everyone"
        @click="emit('show-everyone')"
      >{{ t('home.trendingShowsShowEveryone') }}</button>
    </div>

    <!-- `followable`, so the rail carries the identical Follow + save pair as the Shows tab's grid
         rather than being the one show surface you cannot act on (operator 2026-09-17). -->
    <CardRail v-if="hasAny">
      <li v-for="p in shownPodcasts" :key="p.feed_id" class="lp-rail-item">
        <ShowTile :show="p" followable data-testid="trending-show-card" />
      </li>
    </CardRail>
  </section>
</template>
