<script setup lang="ts">
/**
 * One "Your Week" highlight card. When the item carries episode/show artwork it becomes the card's
 * backdrop under a dark gradient scrim (brings the corpus's colour into the home) with the content
 * — quote / title / graph chips — layered on top in legible white; without art it falls back to a
 * flat surface card. Quote-forward for revisits (the captured line is the hook, the episode title
 * the attribution); title-forward otherwise. The whole card links into the player — at the
 * captured timestamp when the item has one.
 */
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import type { YourWeekItem } from '../services/types'

const props = defineProps<{ item: YourWeekItem }>()

const hasImage = computed(() => !!props.item.image_url)

// `revisit` (when the item is one of the user's own captures) advances that highlight's spaced
// ladder once the player is reached — see the player's onMounted. Auto-picks carry no
// highlight_id: there is no ladder behind them. Without this, consuming revisit through Your Week
// never progressed anything, so the same cards came back every week (#35).
const to = computed(() => ({
  name: 'player' as const,
  params: { slug: props.item.episode_slug },
  query: {
    ...(props.item.t_ms ? { t: String(Math.floor(props.item.t_ms / 1000)) } : {}),
    ...(props.item.highlight_id ? { revisit: props.item.highlight_id } : {}),
  },
}))

const chips = computed(() => (props.item.graph_refs ?? []).slice(0, 2))

// The route backfills episode_title for every resolvable item; fall back to the lead graph label
// so an unresolvable slug (e.g. a stale reference) never renders a blank card headline.
const title = computed(() => props.item.episode_title || props.item.graph_refs?.[0]?.label || '')
</script>

<template>
  <RouterLink
    :to="to"
    class="relative flex h-full flex-col overflow-hidden rounded-xl border border-border no-underline transition hover:border-accent"
    :class="hasImage ? 'text-white' : 'bg-surface text-canvas-foreground'"
  >
    <template v-if="hasImage">
      <img :src="item.image_url!" alt="" class="absolute inset-0 h-full w-full object-cover" />
      <!-- Scrim: darkest at the bottom (title) but keeping colour up top (quote). -->
      <div class="absolute inset-0 bg-gradient-to-t from-black/90 via-black/65 to-black/40" />
    </template>
    <div
      class="relative flex h-full flex-col p-4"
      :class="hasImage ? '[text-shadow:0_1px_3px_rgba(0,0,0,0.65)]' : ''"
    >
      <p v-if="item.quote" class="line-clamp-4 font-display text-sm font-semibold leading-snug">
        “{{ item.quote }}”
      </p>
      <div :class="item.quote ? 'mt-auto pt-3' : ''">
        <div class="line-clamp-2 text-sm font-bold leading-tight">{{ title }}</div>
        <ul v-if="chips.length" class="mt-2 flex flex-wrap gap-1.5">
          <li
            v-for="c in chips"
            :key="c.id"
            class="rounded-full px-2 py-0.5 text-xs font-semibold"
            :class="hasImage ? 'bg-white/25 text-white' : 'bg-overlay text-muted'"
          >
            {{ c.label }}
          </li>
        </ul>
      </div>
    </div>
  </RouterLink>
</template>
