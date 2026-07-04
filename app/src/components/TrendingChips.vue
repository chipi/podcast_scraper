<script setup lang="ts">
/** Trending view 0 (the original) — rising topics as plain number pills:
 *  "topic ↑N×". Simplest / most familiar; the baseline the other views riff on. */
import { trendArrow, trendColor, type RisingTopic } from './trending'

defineProps<{ topics: RisingTopic[] }>()
const emit = defineEmits<{ (e: 'open', id: string): void }>()
</script>

<template>
  <div class="flex flex-wrap gap-1.5" data-testid="trend-chips">
    <button
      v-for="tp in topics"
      :key="tp.id"
      type="button"
      class="inline-flex items-center gap-1.5 rounded-full bg-overlay px-3 py-1.5 text-sm text-topic transition hover:bg-elevated"
      data-testid="trend-chip"
      :aria-label="`${tp.label}, trending at ${tp.v} times its recent average`"
      @click="emit('open', tp.id)"
    >
      {{ tp.label }}
      <span class="text-xs font-semibold" :style="{ color: trendColor(tp.v) }"
        >{{ trendArrow(tp.v) }} {{ tp.v }}×</span
      >
    </button>
  </div>
</template>
