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
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import FollowButton from './FollowButton.vue'
import type { Podcast } from '../services/types'
import { showArtwork } from '../utils/episode'
import { useLibraryStore } from '../stores/library'
import { useSignInGate } from '../composables/useSignInGate'

const props = withDefaults(
  defineProps<{
    show: Podcast
    /** Lines the label box reserves. 2 suits the grid; 1 suits a dense rail. */
    /**
     * Render a follow toggle over the artwork. Used where following IS the point of showing the
     * tile — e.g. the empty "Your shows" state, where the user must be able to complete the action
     * in place rather than be sent somewhere else to do it.
     */
    followable?: boolean
  }>(),
  { followable: false },
)

const library = useLibraryStore()
const { isGated, gated } = useSignInGate()
const following = computed(() => library.has(props.show.feed_id))
const busy = ref(false)

/**
 * Signed-out follows must route to sign-in, not call the API (#1590).
 *
 * The store swallows failures and reverts optimistically, so an ungated click here flipped the
 * button, fired a 401, and flipped back — a control that appears to work for one frame and then
 * silently undoes itself, which is worse than the hidden control #1590 replaced.
 */
const toggleFollow = gated(async () => {
  busy.value = true
  try {
    await library.toggle(props.show.feed_id, { title: props.show.title })
  } finally {
    busy.value = false
  }
})

const art = (): string | null => showArtwork(props.show)
</script>

<template>
  <RouterLink
    :to="{ name: 'podcast', params: { feedId: show.feed_id } }"
    class="relative flex h-full flex-col no-underline text-canvas-foreground"
  >
    <img
      v-if="art()"
      :src="art()!"
      alt=""
      loading="lazy"
      class="aspect-square w-full rounded-xl bg-elevated object-cover"
    />
    <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
    <!-- The shared show-follow pill (F2.4), overlay variant. `.prevent.stop` (inside FollowButton)
         so following does not also navigate: the whole tile is a link, and the point is to follow
         without leaving Home. -->
    <FollowButton
      v-if="followable"
      variant="overlay"
      :following="following"
      :busy="busy"
      :gated="isGated"
      @toggle="toggleFollow"
    />
    <!--
      The NAME IS NOT CLIPPED (#2004 items 3/3c).

      This clamped at two lines with a reserved `min-h`, so any show whose name runs longer lost the
      end of it — visible across Browse → Shows, both Home rails and Library, since they all render
      this tile. The clamp was there to keep grid rows even (#1584), which is a real requirement:
      a one-line name beside a two-line one leaves the row ragged.

      Rows are now even because the TILE is even, not because the text is cut. The tile is a flex
      column that fills its grid cell (`h-full`), the artwork is fixed, and the name takes the
      remaining space and wraps as far as it needs. Alignment is paid for by the layout instead of
      by the content.
    -->
    <div class="mt-1 flex-1 text-xs font-bold leading-tight">
      {{ show.title ?? show.feed_id }}
    </div>
  </RouterLink>
</template>
