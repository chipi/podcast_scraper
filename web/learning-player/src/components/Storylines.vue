<script setup lang="ts">
/**
 * Storylines (option B) — theme clusters (topics discussed together) as a browsable Home rail,
 * sibling of TrendingTopics. Each chip opens the storyline's anchor topic card (whose "discussed
 * together" set is the whole storyline) and carries a one-tap follow (＋/✓) that adds the theme
 * cluster (thc:…) to your interests — the same store the entity-card + trending follows use, so a
 * storyline re-ranks discovery. Reads /api/app/theme-clusters; hides when the corpus has none.
 */
import { computed, ref, watch } from 'vue'
import { useSectionState } from '../composables/useSectionState'
import SectionStatus from './SectionStatus.vue'
import { storeToRefs } from 'pinia'
import { useI18n } from 'vue-i18n'
import { getStorylines, getTrending } from '../services/api'
import { useTrendingScope } from '../composables/useTrendingScope'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import type { Storyline } from '../services/types'
import TrendMomentum from './TrendMomentum.vue'

// #9 — emit the whole storyline (not just the anchor topic id) so the opener can title the sheet
// with the storyline and list its member topics, rather than opening one member's topic card.
const emit = defineEmits<{ (e: 'open', storyline: Storyline): void }>()
/** Suppress the internal heading when a parent (Home discovery tabs) already labels it. */
withDefaults(defineProps<{ hideHeading?: boolean }>(), { hideHeading: false })
const { t } = useI18n()

const auth = useAuthStore()
const interests = useInterestsStore()
const { ids: followedIds } = storeToRefs(interests)
const canFollow = computed(() => auth.isAuthenticated)
if (auth.isAuthenticated) void interests.ensureLoaded().catch(() => {})
function isFollowed(id: string): boolean {
  return followedIds.value.includes(id)
}
function onFollow(id: string): void {
  void interests.toggle(id)
}

const section = useSectionState<Storyline[]>([], { cacheKey: 'home.storylines' })
const storylines = computed(() => section.data.value)
/** #1591 — a rejection lands in the error phase instead of collapsing into empty. */
function load(): Promise<void> {
  return section.load(() => getStorylines(12))
}
void load()

// Storyline momentum (BT.4): a storyline is a theme cluster (thc:…), and /trending?kind=storyline
// keys the same id, so a plain by-id join lights up the cards that are currently trending. Velocity
// is Σ of the cluster's member-topic series (an aggregate), so a storyline missing from the trending
// set simply shows no badge — best-effort, never blocks the rail.
const { scope } = useTrendingScope()
const momentum = ref<Record<string, { v: number; series: number[] }>>({})
function loadMomentum(): void {
  // #2030 — follow the app-level lens (Corpus ⇄ My listening), same as the other trending rails.
  void getTrending('storyline', scope.value, 50)
    .then((rows) => {
      const m: Record<string, { v: number; series: number[] }> = {}
      for (const r of rows) m[r.entity_id] = { v: r.velocity, series: r.series }
      momentum.value = m
    })
    .catch(() => {
      /* momentum is decoration; the rail renders without it */
    })
}
loadMomentum()
watch(scope, loadMomentum)
function momentumOf(id: string): { v: number; series: number[] } | null {
  return momentum.value[id] ?? null
}
const hasAny = computed(() => storylines.value.length > 0)

// #3 — the rail wraps one storyline per row on phones (labels need the width), so a dozen of them
// made Home very tall. Cap to the top few with a show-more, same as the Rising/Trending rails.
const COLLAPSED = 5
const expanded = ref(false)
const visible = computed(() =>
  expanded.value ? storylines.value : storylines.value.slice(0, COLLAPSED),
)
const hiddenCount = computed(() => Math.max(0, storylines.value.length - COLLAPSED))
</script>

<template>
  <section v-if="hasAny || !section.isReady.value" class="mt-7" data-testid="home-storylines">
    <h2 v-if="!hideHeading" class="lp-section">{{ t('home.storylines') }}</h2>
    <SectionStatus :phase="section.phase.value" :rows="2" @retry="load" />
    <template v-if="hasAny">
    <p class="mb-2 text-sm text-muted">{{ t('home.storylinesHint') }}</p>
    <div class="flex flex-wrap gap-1.5">
      <!--
        No 50% cap on phones. The cap existed to fit two chips per row, but a chip also carries its
        "· N topics" count and a follow control, and those are `shrink-0` — so the label got
        whatever was left. Measured at 390px: "Managing risk across domains" needed 204px and was
        given 61px, rendering as "Manag…". A chip whose label is 30% visible is not a chip.

        A storyline's label is the only thing that identifies it, so it gets the row it needs and
        wraps to the next line when there is no room. Two-up returns from `sm`, where labels fit.
      -->
      <div
        v-for="s in visible"
        :key="s.id"
        class="lp-theme-chip inline-flex min-w-0 max-w-full items-center rounded-full text-sm text-surface-foreground transition sm:max-w-none"
        data-testid="storyline-chip"
      >
        <button
          type="button"
          class="inline-flex min-w-0 items-center gap-1.5 py-1.5 pl-3"
          :class="canFollow ? 'pr-1.5' : 'rounded-full pr-3'"
          :aria-label="t('home.storylineOpen', { label: s.label, count: s.size })"
          @click="emit('open', s)"
        >
          <span class="truncate font-semibold">{{ s.label }}</span>
          <span class="shrink-0 text-xs opacity-80">{{
            t('home.storylineSize', s.size, { named: { count: s.size } })
          }}</span>
          <TrendMomentum
            v-if="momentumOf(s.id)"
            variant="rail"
            :velocity="momentumOf(s.id)!.v"
            :series="momentumOf(s.id)!.series"
            class="shrink-0"
          />
        </button>
        <button
          v-if="canFollow"
          type="button"
          class="rounded-r-full py-1.5 pl-1 pr-3 text-base leading-none transition"
          :class="isFollowed(s.id) ? 'opacity-100' : 'opacity-60 hover:opacity-100'"
          data-testid="storyline-follow"
          :aria-pressed="isFollowed(s.id)"
          :aria-label="
            isFollowed(s.id)
              ? t('home.storylineFollowing', { label: s.label })
              : t('home.storylineFollow', { label: s.label })
          "
          @click="onFollow(s.id)"
        >{{ isFollowed(s.id) ? '✓' : '＋' }}</button>
      </div>
    </div>
    <button
      v-if="hiddenCount > 0"
      type="button"
      class="mt-2 px-1 py-1 text-xs font-semibold text-accent transition hover:opacity-80"
      data-testid="storyline-expand"
      :aria-expanded="expanded"
      @click="expanded = !expanded"
    >
      {{ expanded ? t('home.showLess') : t('home.showMore', { count: hiddenCount }) }}
    </button>
    </template>
  </section>
</template>
