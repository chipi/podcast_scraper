<script setup lang="ts">
/**
 * One "Your Week" highlight card. When the item carries episode/show artwork it becomes the card's
 * backdrop under a dark gradient scrim (brings the corpus's colour into the home) with the content
 * — quote / title / graph chips — layered on top in legible white; without art it falls back to a
 * flat surface card. The whole card links into the player.
 *
 * Title-forward only. It used to be quote-forward for REVISIT items and carried their timestamp and
 * `?revisit=<highlight_id>` into the player — but Home stopped showing the digest's revisit section
 * (2026-09-30; see YourWeek.vue `sections`), and revisit items were the only ones with a quote, a
 * timestamp or a highlight id. Due highlights are revisited through RevisitRail / the Revisit tab,
 * and the digest email's links still carry `?revisit=` straight to the player.
 */
import { computed } from 'vue'
import { resolveMediaUrl } from '../services/tier'
import { RouterLink } from 'vue-router'
import type { YourWeekItem } from '../services/types'

const props = defineProps<{ item: YourWeekItem }>()

const hasImage = computed(() => !!props.item.image_url)
// The digest payload carries the same relative artwork url the catalog does.
const cardImage = computed(() => resolveMediaUrl(props.item.image_url))

const to = computed(() => ({ name: 'player' as const, params: { slug: props.item.episode_slug } }))

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
      <img :src="cardImage!" alt="" class="absolute inset-0 h-full w-full object-cover" />
      <!-- Scrim: darkest at the bottom, where the title sits, keeping the artwork's colour up top. -->
      <div class="absolute inset-0 bg-gradient-to-t from-black/90 via-black/65 to-black/40" />
    </template>
    <div
      class="relative flex h-full flex-col p-4"
      :class="hasImage ? '[text-shadow:0_1px_3px_rgba(0,0,0,0.65)]' : ''"
    >
      <!-- Title + chips sit at the BOTTOM of the artwork, always (operator 2026-09-19).

           `mt-auto` used to be conditional on there being a quote, so a card without one — the
           common case on this rail — pinned its title to the TOP, over the brightest part of the
           image and furthest from the scrim that makes it legible. Two cards side by side then
           disagreed about where their text lived. The scrim is already darkest at the bottom; this
           puts the words where it was built to carry them. -->
      <div class="mt-auto pt-3">
        <!-- Not clamped (#2004 item 3b): same rule as the other cards — the tile keeps rows even, the
           title is allowed to be as long as it is. -->
      <!-- Shaped like Continue listening (operator 2026-10-08): the episode title large in the display
           face, the show it is from under it. The card named no show at all before. -->
      <div class="font-display text-lg font-extrabold leading-tight tracking-tight" data-testid="yourweek-card-title">{{ title }}</div>
      <div
        v-if="item.podcast_title"
        class="lp-show-name mt-1 text-sm"
        :class="hasImage ? 'text-white/80' : 'text-muted'"
        data-testid="yourweek-card-show"
      >{{ item.podcast_title }}</div>
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
