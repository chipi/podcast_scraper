<script setup lang="ts">
/** Trending view "Pills" — rising topics as pills "• topic ↑N×", with a one-tap follow (＋/✓) to
 *  add the topic to your profile interests (#12). A storyline swatch + hue-matched ×velocity colour
 *  the pill by theme cluster (unclustered → neutral), consistent with the Sparklines view. */
import { THEME_NEUTRAL, trendArrow, type RisingTopic, type TopicTheme } from './trending'

const props = defineProps<{
  topics: RisingTopic[]
  /** topic id → { colour, theme label, group } (see TrendingTopics). */
  topicTheme?: Record<string, TopicTheme>
  neutralColor?: string
  followedIds?: string[]
  canFollow?: boolean
  /** Topic ids in a co-occurrence theme cluster — kept for compatibility; colour comes from topicTheme. */
  themeMemberIds?: Set<string>
}>()
const emit = defineEmits<{ (e: 'open', id: string): void; (e: 'follow', id: string): void }>()

function isFollowed(id: string): boolean {
  return props.followedIds?.includes(id) ?? false
}
function colorOf(id: string): string {
  return props.topicTheme?.[id]?.color ?? props.neutralColor ?? THEME_NEUTRAL
}
function titleOf(tp: RisingTopic): string {
  const theme = props.topicTheme?.[tp.id]?.label
  const base = `${tp.label} — ${tp.v}× vs recent average · ${tp.total} mentions`
  return theme ? `${base} · ${theme}` : base
}
</script>

<template>
  <div class="flex flex-wrap gap-1.5" data-testid="trend-chips">
    <div
      v-for="tp in topics"
      :key="tp.id"
      class="inline-flex min-w-0 max-w-[calc(50%-0.375rem)] items-center rounded-full bg-overlay text-sm text-topic transition hover:bg-elevated sm:max-w-none"
      data-testid="trend-chip"
    >
      <button
        type="button"
        class="inline-flex min-w-0 items-center gap-1.5 py-1.5 pl-2.5"
        :class="canFollow ? 'pr-1.5' : 'rounded-full pr-3'"
        :title="titleOf(tp)"
        :aria-label="`${tp.label}, trending at ${tp.v} times its recent average`"
        @click="emit('open', tp.id)"
      >
        <!-- Storyline swatch: same hue as this topic's cluster (neutral when unclustered). -->
        <span
          class="h-2 w-2 shrink-0 rounded-full"
          :style="{ backgroundColor: colorOf(tp.id) }"
          aria-hidden="true"
        />
        <span class="truncate">{{ tp.label }}</span>
        <span class="shrink-0 text-xs font-semibold" :style="{ color: colorOf(tp.id) }"
          >{{ trendArrow(tp.v) }} {{ tp.v }}×</span
        >
      </button>
      <button
        v-if="canFollow"
        type="button"
        class="rounded-r-full py-1.5 pl-1 pr-3 text-base leading-none transition"
        :class="isFollowed(tp.id) ? 'text-accent' : 'text-muted hover:text-accent'"
        data-testid="trend-chip-follow"
        :aria-pressed="isFollowed(tp.id)"
        :aria-label="isFollowed(tp.id) ? `Following ${tp.label}` : `Add ${tp.label} to my interests`"
        @click="emit('follow', tp.id)"
      >{{ isFollowed(tp.id) ? '✓' : '＋' }}</button>
    </div>
  </div>
</template>
