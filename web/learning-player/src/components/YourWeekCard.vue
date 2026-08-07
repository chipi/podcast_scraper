<script setup lang="ts">
/**
 * One "Your Week" highlight card. Quote-forward for revisits (the captured line is the hook, the
 * episode title becomes the attribution); title-forward for new-in-follows / trending. The whole
 * card links into the player — at the captured timestamp when the item carries one.
 */
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import type { YourWeekItem } from '../services/types'

const props = defineProps<{ item: YourWeekItem }>()

const to = computed(() => ({
  name: 'player' as const,
  params: { slug: props.item.episode_slug },
  query: props.item.t_ms ? { t: String(Math.floor(props.item.t_ms / 1000)) } : {},
}))

const chips = computed(() => (props.item.graph_refs ?? []).slice(0, 2))
</script>

<template>
  <RouterLink
    :to="to"
    class="flex h-full flex-col rounded-xl border border-border bg-surface p-4 no-underline text-canvas-foreground transition hover:border-accent"
  >
    <p v-if="item.quote" class="line-clamp-4 font-display text-sm font-semibold leading-snug">
      “{{ item.quote }}”
    </p>
    <div :class="item.quote ? 'mt-auto pt-3' : ''">
      <div class="line-clamp-2 text-sm font-bold leading-tight">{{ item.episode_title }}</div>
      <ul v-if="chips.length" class="mt-2 flex flex-wrap gap-1.5">
        <li
          v-for="c in chips"
          :key="c.id"
          class="rounded-full bg-overlay px-2 py-0.5 text-xs font-semibold text-muted"
        >
          {{ c.label }}
        </li>
      </ul>
    </div>
  </RouterLink>
</template>
