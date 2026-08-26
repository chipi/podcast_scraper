<script setup lang="ts">
/**
 * Momentum rail (RFC-103) — one generic "trending now" rail for any entity kind, powered by the
 * read-time momentum endpoint (GET /api/app/trending). A compact, aligned list: each row is the
 * entity's label, a direction-coloured ×velocity, and a weekly sparkline in the SAME hue (rising
 * green / cooling red / steady amber — so the pulse itself carries the trend). Interest-token kinds
 * (topic / cluster / storyline / person) get a one-tap follow. Collapsed to the top few (mobile
 * vertical space is precious) with an expand toggle. Emits `open` with the entity.
 */
import { computed, ref, watch } from 'vue'
import { useSectionState } from '../composables/useSectionState'
import SectionStatus from './SectionStatus.vue'
import TrendWindowTabs from './TrendWindowTabs.vue'
import { useI18n } from 'vue-i18n'
import { storeToRefs } from 'pinia'
import { getTrending, type TrendWindow } from '../services/api'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import type { TrendingEntity } from '../services/types'
import Sparkline from './Sparkline.vue'
import { trendArrow, trendColor, trendDirection } from './trending'

const { t } = useI18n()

const props = withDefaults(
  defineProps<{
    kind: string
    title: string
    scope?: 'corpus' | 'mine'
    limit?: number
    /** Suppress the internal heading when a parent (e.g. the Home discovery tabs) already labels it. */
    hideHeading?: boolean
  }>(),
  { scope: 'corpus', limit: 12, hideHeading: false }
)
const emit = defineEmits<{ (e: 'open', entity: TrendingEntity): void }>()

const auth = useAuthStore()
const interests = useInterestsStore()
const { ids: followedIds } = storeToRefs(interests)
if (auth.isAuthenticated) void interests.ensureLoaded().catch(() => {})

// Only interest tokens are followable (topic: / tc: / thc: / person:); episodes/shows/insights aren't.
const _FOLLOWABLE = /^(topic:|tc:|thc:|person:)/
function isFollowable(id: string): boolean {
  return auth.isAuthenticated && _FOLLOWABLE.test(id)
}
function isFollowed(id: string): boolean {
  return followedIds.value.includes(id)
}
function onFollow(id: string): void {
  void interests.toggle(id)
}

// RFC-103 R2 — the trend window (1m/3m/6m/1y); default 3m. Changing it reloads the rail.
const window = ref<TrendWindow>('3m')
// #1591 — a rejection lands in the error phase rather than collapsing into empty.
const section = useSectionState<TrendingEntity[]>([])
function load(): Promise<void> {
  return section.load(() => getTrending(props.kind, props.scope, props.limit, window.value))
}
void load()
watch(window, load)
const items = computed(() => section.data.value)
const hasAny = computed(() => items.value.length > 0)

const COLLAPSED = 5
const expanded = ref(false)
const visible = computed(() => (expanded.value ? items.value : items.value.slice(0, COLLAPSED)))
const hiddenCount = computed(() => Math.max(0, items.value.length - COLLAPSED))

function vFmt(v: number): number {
  return Math.round(v * 10) / 10
}
function titleOf(e: TrendingEntity): string {
  const dir = trendDirection(e.velocity)
  const word = dir === 'up' ? 'rising' : dir === 'down' ? 'cooling' : 'steady'
  return `${e.label} — ${vFmt(e.velocity)}× (${word})`
}
</script>

<template>
  <section
    v-if="hasAny || !section.isReady.value"
    class="mt-7"
    :data-testid="`momentum-rail-${kind}`"
  >
    <div class="mb-2 flex items-center justify-between gap-2">
      <h2 v-if="!hideHeading" class="lp-section mb-0">{{ title }}</h2>
      <span v-else class="sr-only">{{ title }}</span>
      <TrendWindowTabs v-model="window" />
    </div>
    <!-- #1595: the × metric was explained only in a `title` attribute, which does not exist on
         touch — so on the primary platform this rail showed an undecoded number. -->
    <p v-if="hasAny" class="mb-2 text-xs text-muted">{{ t('home.momentumHint') }}</p>
    <SectionStatus :phase="section.phase.value" :rows="2" @retry="load" />
    <ul v-if="hasAny" class="flex flex-col">
      <li
        v-for="e in visible"
        :key="e.entity_id"
        class="flex items-center gap-1 rounded-lg transition hover:bg-overlay"
        data-testid="momentum-chip"
      >
        <button
          type="button"
          class="flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-2 py-1 text-left"
          :title="titleOf(e)"
          :aria-label="titleOf(e)"
          @click="emit('open', e)"
        >
          <span class="min-w-0 flex-1 truncate text-sm">{{ e.label }}</span>
          <span
            class="w-12 shrink-0 text-right text-xs font-semibold tabular-nums"
            :style="{ color: trendColor(e.velocity) }"
            >{{ trendArrow(e.velocity) }} {{ vFmt(e.velocity) }}×</span
          >
          <Sparkline
            :values="e.series"
            :width="56"
            :height="20"
            class="shrink-0"
            :style="{ color: trendColor(e.velocity) }"
          />
        </button>
        <button
          v-if="isFollowable(e.entity_id)"
          type="button"
          class="shrink-0 rounded-full px-2 py-1 text-base leading-none transition"
          :class="isFollowed(e.entity_id) ? 'text-accent' : 'text-muted hover:text-accent'"
          data-testid="momentum-follow"
          :aria-pressed="isFollowed(e.entity_id)"
          @click="onFollow(e.entity_id)"
        >
          {{ isFollowed(e.entity_id) ? '✓' : '＋' }}
        </button>
      </li>
    </ul>

    <button
      v-if="hiddenCount > 0"
      type="button"
      class="mt-1 px-2 py-1 text-xs font-semibold text-accent transition hover:opacity-80"
      data-testid="momentum-expand"
      :aria-expanded="expanded"
      @click="expanded = !expanded"
    >
      {{ expanded ? t('home.showLess') : t('home.showMore', { count: hiddenCount }) }}
    </button>
  </section>
</template>
