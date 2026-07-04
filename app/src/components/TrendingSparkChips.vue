<script setup lang="ts">
/** Trending view 1 — each rising topic as a row: label + ↑factor + a mini
 *  sparkline of its monthly shape (ramp vs spike vs steady climb). */
import type { RisingTopic } from './trending'
import Sparkline from './Sparkline.vue'

defineProps<{ topics: RisingTopic[] }>()
const emit = defineEmits<{ (e: 'open', id: string): void }>()
</script>

<template>
  <ul class="flex flex-col gap-0.5" data-testid="trend-sparks">
    <li v-for="tp in topics.slice(0, 8)" :key="tp.id">
      <button
        type="button"
        class="flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left transition hover:bg-overlay"
        data-testid="trend-spark-row"
        :aria-label="`${tp.label}, trending at ${tp.v} times its recent average`"
        @click="emit('open', tp.id)"
      >
        <span class="min-w-0 flex-1 truncate text-sm text-topic">{{ tp.label }}</span>
        <span class="shrink-0 text-xs font-semibold text-accent">↑ {{ tp.v }}×</span>
        <Sparkline :values="tp.series" :width="72" :height="22" class="shrink-0 text-accent" />
      </button>
    </li>
  </ul>
</template>
