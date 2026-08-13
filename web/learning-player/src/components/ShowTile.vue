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
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import type { Podcast } from '../services/types'
import { showArtwork } from '../utils/episode'
import { useLibraryStore } from '../stores/library'
import { useSignInGate } from '../composables/useSignInGate'

const props = withDefaults(
  defineProps<{
    show: Podcast
    /** Lines the label box reserves. 2 suits the grid; 1 suits a dense rail. */
    lines?: 1 | 2
    /**
     * Render a follow toggle over the artwork. Used where following IS the point of showing the
     * tile — e.g. the empty "Your shows" state, where the user must be able to complete the action
     * in place rather than be sent somewhere else to do it.
     */
    followable?: boolean
  }>(),
  { lines: 2, followable: false },
)

const { t } = useI18n()
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
    class="relative block no-underline text-canvas-foreground"
  >
    <img
      v-if="art()"
      :src="art()!"
      alt=""
      loading="lazy"
      class="aspect-square w-full rounded-xl bg-elevated object-cover"
    />
    <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
    <!-- `.prevent.stop` so following does not also navigate to the show page: the whole tile is a
         link, and the point of this control is to complete the action without leaving Home. -->
    <button
      v-if="followable"
      type="button"
      class="absolute right-1.5 top-1.5 inline-flex h-7 items-center gap-1 rounded-full px-2 text-[0.65rem] font-bold shadow-lg backdrop-blur transition disabled:opacity-60"
      :class="following ? 'bg-accent text-accent-foreground' : 'bg-canvas/80 text-canvas-foreground hover:bg-canvas'"
      :aria-pressed="isGated ? undefined : following"
      :aria-label="isGated ? t('auth.signInToFollow') : following ? t('podcast.following') : t('podcast.follow')"
      :disabled="busy"
      @click.prevent.stop="toggleFollow"
    >
      <span aria-hidden="true">{{ following ? '✓' : '+' }}</span>
      {{ following ? t('podcast.following') : t('podcast.follow') }}
    </button>
    <div
      class="mt-1 text-xs font-bold leading-tight"
      :class="lines === 2 ? 'line-clamp-2 min-h-[2.25rem]' : 'truncate'"
      :title="show.title ?? show.feed_id"
    >
      {{ show.title ?? show.feed_id }}
    </div>
  </RouterLink>
</template>
