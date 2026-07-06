<script setup lang="ts">
/** Trending view 1 — each rising topic as a row: label + ↑factor + a mini
 *  sparkline of its monthly shape (ramp vs spike vs steady climb). */
import { trendArrow, trendColor, type RisingTopic } from './trending'
import Sparkline from './Sparkline.vue'

const props = defineProps<{
  topics: RisingTopic[]
  followedIds?: string[]
  canFollow?: boolean
}>()
const emit = defineEmits<{ (e: 'open', id: string): void; (e: 'follow', id: string): void }>()

function isFollowed(id: string): boolean {
  return props.followedIds?.includes(id) ?? false
}
</script>

<template>
  <ul class="flex flex-col gap-0.5" data-testid="trend-sparks">
    <li
      v-for="tp in topics.slice(0, 8)"
      :key="tp.id"
      class="flex items-center gap-1 rounded-lg pr-1 transition hover:bg-overlay"
    >
      <button
        type="button"
        class="flex min-w-0 flex-1 items-center gap-3 rounded-lg px-2 py-1.5 text-left"
        data-testid="trend-spark-row"
        :aria-label="`${tp.label}, trending at ${tp.v} times its recent average`"
        @click="emit('open', tp.id)"
      >
        <span class="min-w-0 flex-1 truncate text-sm text-topic">{{ tp.label }}</span>
        <span class="shrink-0 text-xs font-semibold" :style="{ color: trendColor(tp.v) }"
          >{{ trendArrow(tp.v) }} {{ tp.v }}×</span
        >
        <Sparkline
          :values="tp.series"
          :width="72"
          :height="22"
          class="shrink-0"
          :style="{ color: trendColor(tp.v) }"
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
</template>
