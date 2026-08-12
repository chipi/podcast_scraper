<script setup lang="ts">
/**
 * One show as a square-artwork tile with a **fixed-height label box** (#1584).
 *
 * The label box is the whole point. A CSS grid row is as tall as its tallest cell, so an unclamped
 * label makes row height a function of title length: one three-line show name inflates the gap for
 * that entire row while single-word names sit tight, and the grid reads as inconsistently spaced.
 *
 * Clamping alone does NOT fix it — a one-line title next to a two-line title still differs by one
 * line. Uniformity needs the clamp **and** a reserved minimum height, so the box is the same size
 * whether or not it is filled. Both live here so no caller can get it half-right, which is exactly
 * how the four drifted call sites happened: wherever a component owned the tile it was correct,
 * wherever markup was hand-rolled inline it drifted.
 */
import { RouterLink } from 'vue-router'
import type { Podcast } from '../services/types'
import { showArtwork } from '../utils/episode'

const props = withDefaults(
  defineProps<{
    show: Podcast
    /** Lines the label box reserves. 2 suits the grid; 1 suits a dense rail. */
    lines?: 1 | 2
  }>(),
  { lines: 2 },
)

const art = (): string | null => showArtwork(props.show)
</script>

<template>
  <RouterLink
    :to="{ name: 'podcast', params: { feedId: show.feed_id } }"
    class="block no-underline text-canvas-foreground"
  >
    <img
      v-if="art()"
      :src="art()!"
      alt=""
      loading="lazy"
      class="aspect-square w-full rounded-xl bg-elevated object-cover"
    />
    <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
    <div
      class="mt-1 text-xs font-bold leading-tight"
      :class="lines === 2 ? 'line-clamp-2 min-h-[2.25rem]' : 'truncate'"
      :title="show.title ?? show.feed_id"
    >
      {{ show.title ?? show.feed_id }}
    </div>
  </RouterLink>
</template>
