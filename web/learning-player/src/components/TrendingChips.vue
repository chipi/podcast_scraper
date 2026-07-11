<script setup lang="ts">
/** Trending view 0 (the original) — rising topics as number pills "topic ↑N×", with a one-tap
 *  follow (＋/✓) to add the topic to your profile interests (#12). Simplest / most familiar. */
import { trendArrow, trendColor, type RisingTopic } from './trending'

const props = defineProps<{
  topics: RisingTopic[]
  followedIds?: string[]
  canFollow?: boolean
  /** Topic ids in a co-occurrence theme cluster — marked with the standard teal theme chrome. */
  themeMemberIds?: Set<string>
}>()
const emit = defineEmits<{ (e: 'open', id: string): void; (e: 'follow', id: string): void }>()

function isFollowed(id: string): boolean {
  return props.followedIds?.includes(id) ?? false
}
function isTheme(id: string): boolean {
  return props.themeMemberIds?.has(id) ?? false
}
</script>

<template>
  <div class="flex flex-wrap gap-1.5" data-testid="trend-chips">
    <div
      v-for="tp in topics"
      :key="tp.id"
      class="inline-flex min-w-0 max-w-[calc(50%-0.375rem)] items-center rounded-full text-sm transition sm:max-w-none"
      :class="isTheme(tp.id) ? 'lp-theme-chip text-surface-foreground' : 'bg-overlay text-topic hover:bg-elevated'"
      :data-theme-member="isTheme(tp.id) ? '' : undefined"
      data-testid="trend-chip"
    >
      <button
        type="button"
        class="inline-flex min-w-0 items-center gap-1.5 py-1.5 pl-3"
        :class="canFollow ? 'pr-1.5' : 'rounded-full pr-3'"
        :aria-label="`${tp.label}, trending at ${tp.v} times its recent average`"
        @click="emit('open', tp.id)"
      >
        <span class="truncate">{{ tp.label }}</span>
        <span class="shrink-0 text-xs font-semibold" :style="{ color: trendColor(tp.v) }"
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
