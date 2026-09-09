<script setup lang="ts">
/** Trending view "Sparklines" — each rising topic as a compact row: a theme-colour swatch,
 *  the label, its ×velocity, and a mini sparkline of its monthly shape (ramp vs spike vs climb).
 *  Topics in the same co-occurrence theme ("storyline") share a hue and are grouped together;
 *  unclustered topics use a neutral hue and sort last. Collapsed to the top few on mobile with an
 *  expand toggle — vertical space is precious. */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import Sparkline from './Sparkline.vue'
import { THEME_NEUTRAL, type RisingTopic, type TopicTheme } from './trending'

const { t } = useI18n()

const props = withDefaults(
  defineProps<{
    topics: RisingTopic[]
    /** topic id → { colour, theme label, group } (see TrendingTopics). */
    topicTheme?: Record<string, TopicTheme>
    neutralColor?: string
    followedIds?: string[]
    canFollow?: boolean
    /** How many rows to show before the "show more" toggle. Home uses 5 (tight); a browse index
     *  passes a larger number so it actually shows the trending list, not just the top 5. */
    collapseAt?: number
    /** When >0, "show more" reveals this many MORE rows per tap (progressive), instead of expanding
     *  to the whole list at once. Browse indexes pass 10 (top-10, then +10). Unset = expand-all. */
    step?: number
  }>(),
  { collapseAt: 5 },
)
const emit = defineEmits<{ (e: 'open', id: string): void; (e: 'follow', id: string): void }>()

const expanded = ref(false)

const neutral = computed(() => props.neutralColor ?? THEME_NEUTRAL)
function themeOf(id: string): TopicTheme | undefined {
  return props.topicTheme?.[id]
}
function colorOf(id: string): string {
  return themeOf(id)?.color ?? neutral.value
}
function groupOf(id: string): number {
  return themeOf(id)?.group ?? Number.MAX_SAFE_INTEGER
}
function isFollowed(id: string): boolean {
  return props.followedIds?.includes(id) ?? false
}
function rowTitle(tp: RisingTopic): string {
  const theme = themeOf(tp.id)?.label
  const base = `${tp.label} — ${tp.v}× vs recent average · ${tp.total} mentions`
  return theme ? `${base} · ${theme}` : base
}

/** The rail's ranking signal (#1931): ``trend_score`` when the artifact carries one, else the
 *  velocity ratio. Grouping the chips by theme RE-ORDERS the server's list, so it has to re-order
 *  on the SAME key the server sorted by — ranking colour blocks by peak *velocity* would float a
 *  quiet-but-accelerating storyline above the one actually dominating the corpus. */
function heat(tp: RisingTopic): number {
  return typeof tp.score === 'number' ? tp.score : tp.v
}

// Peak heat per theme group — lets the hottest storyline's colour block lead the list.
const groupPeak = computed(() => {
  const peak: Record<number, number> = {}
  for (const tp of props.topics) {
    const g = groupOf(tp.id)
    peak[g] = Math.max(peak[g] ?? -Infinity, heat(tp))
  }
  return peak
})
// Group by theme into contiguous colour blocks, clusters ordered by peak heat (hottest storyline
// first), unclustered topics last; within a group, hottest first. Stable + meaningful.
const ordered = computed(() => {
  const gp = groupPeak.value
  const UNCLUSTERED = Number.MAX_SAFE_INTEGER
  const rank = (id: string): number =>
    groupOf(id) === UNCLUSTERED ? Number.POSITIVE_INFINITY : -(gp[groupOf(id)] ?? 0)
  return [...props.topics].sort(
    (a, b) => rank(a.id) - rank(b.id) || groupOf(a.id) - groupOf(b.id) || heat(b) - heat(a),
  )
})
// Progressive reveal (`step` set) vs expand-all (legacy). In step mode a running `shownCount` grows
// by `step` per tap; otherwise the boolean `expanded` shows everything at once.
const stepMode = computed(() => (props.step ?? 0) > 0)
const shownCount = ref(props.collapseAt)
// Reset the window when the underlying list changes (e.g. the trend-window switch reloads it).
watch(
  () => ordered.value.length,
  () => {
    shownCount.value = props.collapseAt
  },
)
const visible = computed(() => {
  if (stepMode.value) return ordered.value.slice(0, shownCount.value)
  return expanded.value ? ordered.value : ordered.value.slice(0, props.collapseAt)
})
const remaining = computed(() => Math.max(0, ordered.value.length - shownCount.value))
const hiddenCount = computed(() => Math.max(0, ordered.value.length - props.collapseAt))
// One control, both modes. Step: "show more (+N)" while rows remain, else "show less" once expanded
// past the initial window. Legacy: the expand/collapse toggle.
const canShowMore = computed(() =>
  stepMode.value ? remaining.value > 0 : !expanded.value && hiddenCount.value > 0,
)
const canShowLess = computed(() =>
  stepMode.value ? remaining.value === 0 && shownCount.value > props.collapseAt : expanded.value,
)
const moreCount = computed(() =>
  stepMode.value ? Math.min(props.step ?? 0, remaining.value) : hiddenCount.value,
)
function toggleShown(): void {
  if (!stepMode.value) {
    expanded.value = !expanded.value
    return
  }
  if (remaining.value > 0) {
    shownCount.value = Math.min(ordered.value.length, shownCount.value + (props.step ?? 0))
  } else {
    shownCount.value = props.collapseAt // collapse back to the top-N
  }
}
</script>

<template>
  <div data-testid="trend-sparks">
    <ul class="flex flex-col">
      <li
        v-for="tp in visible"
        :key="tp.id"
        class="flex items-center gap-1 rounded-lg transition hover:bg-overlay"
      >
        <button
          type="button"
          class="flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-2 py-1 text-left"
          data-testid="trend-spark-row"
          :title="rowTitle(tp)"
          :aria-label="`${tp.label}, trending at ${tp.v} times its recent average`"
          @click="emit('open', tp.id)"
        >
          <!-- Theme swatch: same hue as the sparkline, so the colour's meaning is explicit. -->
          <span
            class="h-2.5 w-2.5 shrink-0 rounded-full"
            :style="{ backgroundColor: colorOf(tp.id) }"
            aria-hidden="true"
          />
          <span class="min-w-0 flex-1 truncate text-sm">{{ tp.label }}</span>
          <!-- Role badge (people): says WHY someone trends — a busy host vs a recurring guest vs a
               much-mentioned figure. Absent for topics and for people with no KG role. -->
          <span
            v-if="tp.role"
            class="shrink-0 rounded-full border border-border px-1.5 py-px text-[9px] font-bold uppercase tracking-wide text-muted"
            data-testid="trend-spark-role"
          >{{ tp.role }}</span>
          <span class="w-10 shrink-0 text-right text-xs font-semibold tabular-nums text-muted"
            >{{ tp.v }}×</span
          >
          <Sparkline
            :values="tp.series"
            :width="56"
            :height="20"
            class="shrink-0"
            :style="{ color: colorOf(tp.id) }"
          />
        </button>
        <button
          v-if="canFollow"
          type="button"
          class="shrink-0 rounded-full px-2 py-1 text-base leading-none transition"
          :class="isFollowed(tp.id) ? 'text-accent' : 'text-muted hover:text-accent'"
          data-testid="trend-spark-follow"
          :aria-pressed="isFollowed(tp.id)"
          :aria-label="isFollowed(tp.id) ? `Following ${tp.label}` : `Add ${tp.label} to my interests`"
          @click="emit('follow', tp.id)"
        >{{ isFollowed(tp.id) ? '✓' : '＋' }}</button>
      </li>
    </ul>

    <button
      v-if="canShowMore || canShowLess"
      type="button"
      class="mt-1 px-2 py-1 text-xs font-semibold text-accent transition hover:opacity-80"
      data-testid="trend-spark-expand"
      :aria-expanded="stepMode ? remaining === 0 : expanded"
      @click="toggleShown"
    >
      {{ canShowMore ? t('home.showMore', { count: moreCount }) : t('home.showLess') }}
    </button>
  </div>
</template>
