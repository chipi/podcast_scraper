<script setup lang="ts">
/** Trending view "Sparklines" — each rising topic as a compact row: a theme-colour swatch,
 *  the label, its ×velocity, and a mini sparkline of its monthly shape (ramp vs spike vs climb).
 *  Topics in the same co-occurrence theme ("storyline") share a hue and are grouped together;
 *  unclustered topics use a neutral hue and sort last. Collapsed to the top few on mobile with an
 *  expand toggle — vertical space is precious. */
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import Sparkline from './Sparkline.vue'
import { THEME_NEUTRAL, type RisingTopic, type TopicTheme } from './trending'

const { t } = useI18n()

const props = defineProps<{
  topics: RisingTopic[]
  /** topic id → { colour, theme label, group } (see TrendingTopics). */
  topicTheme?: Record<string, TopicTheme>
  neutralColor?: string
  followedIds?: string[]
  canFollow?: boolean
}>()
const emit = defineEmits<{ (e: 'open', id: string): void; (e: 'follow', id: string): void }>()

const COLLAPSED = 5
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

// Peak velocity per theme group — lets the hottest storyline's colour block lead the list.
const groupPeak = computed(() => {
  const peak: Record<number, number> = {}
  for (const tp of props.topics) {
    const g = groupOf(tp.id)
    peak[g] = Math.max(peak[g] ?? -Infinity, tp.v)
  }
  return peak
})
// Group by theme into contiguous colour blocks, clusters ordered by peak velocity (hottest
// storyline first), unclustered topics last; within a group, hottest first. Stable + meaningful.
const ordered = computed(() => {
  const gp = groupPeak.value
  const UNCLUSTERED = Number.MAX_SAFE_INTEGER
  const rank = (id: string): number =>
    groupOf(id) === UNCLUSTERED ? Number.POSITIVE_INFINITY : -(gp[groupOf(id)] ?? 0)
  return [...props.topics].sort(
    (a, b) => rank(a.id) - rank(b.id) || groupOf(a.id) - groupOf(b.id) || b.v - a.v,
  )
})
const visible = computed(() => (expanded.value ? ordered.value : ordered.value.slice(0, COLLAPSED)))
const hiddenCount = computed(() => Math.max(0, ordered.value.length - COLLAPSED))
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
      v-if="hiddenCount > 0"
      type="button"
      class="mt-1 px-2 py-1 text-xs font-semibold text-accent transition hover:opacity-80"
      data-testid="trend-spark-expand"
      :aria-expanded="expanded"
      @click="expanded = !expanded"
    >
      {{ expanded ? t('home.showLess') : t('home.showMore', { count: hiddenCount }) }}
    </button>
  </div>
</template>
