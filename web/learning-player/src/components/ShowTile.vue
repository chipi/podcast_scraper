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
import FavoriteButton from './FavoriteButton.vue'
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
  <!-- The Follow and Save controls are SIBLINGS of the link, not children of it (2026-09-26).

       They used to sit inside the RouterLink, which put a <button> inside an <a> — interactive
       content nested in interactive content, which HTML forbids. Chromium's accessibility mapping
       then failed to compute a name for the heart, and `AccessibleNameAuditTests` reported
       `[Discover] <NO NAME> ToggleButton Rect(267, 674 - 354, 761) near=[]` — a control TalkBack
       announces as an unnamed toggle. The same `FavoriteButton` names itself correctly on a topic
       card ("Save to favorites"), where nothing wraps it, which is what localised the bug here
       rather than in the component.

       FollowButton survived only because it carries VISIBLE text; the heart's name is an `sr-only`
       span, and that is what the nesting dropped.

       The link now covers the tile via `absolute inset-0` and the controls layer above it, so the
       whole tile still navigates and the buttons still act without navigating. -->
  <div class="relative flex h-full flex-col text-canvas-foreground">
    <img
      v-if="art()"
      :src="art()!"
      alt=""
      loading="lazy"
      class="aspect-square w-full rounded-xl bg-elevated object-cover"
    />
    <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
    <!-- Follow + Save over the artwork, as ONE right-aligned column in the top-right corner: the
         pill on top, the heart directly under it, both flush right — the L the episode tile's icon
         cluster makes when it wraps (operator 2026-09-17).

         Stacked rather than side by side because they do not fit side by side: "+ Follow show" is
         91px of a 108px phone tile. A column needs no width negotiation at any viewport, and
         right-alignment is what keeps the two edges reading as one object instead of two floating
         controls.

         Save is not Follow — following surfaces new episodes in Your Week, saving puts the show in
         the Library. `PodcastView` already offers both; the tile offered only one.

         Acting must not also navigate: the tile is a link, and the point is to act without leaving
         the page. Both controls stop their own click — FollowButton internally, FavoriteButton in
         `onGatedClick` — so no wrapper has to do it for them.

         The plate classes match EpisodeActions' `overlay`, so contrast never depends on whatever
         artwork happens to be underneath. -->
    <!-- `z-10` keeps these above the stretched link below, so a tap on Follow or the heart reaches
         the button instead of navigating. -->
    <div class="absolute right-1.5 top-1.5 z-10 flex flex-col items-end gap-1.5">
      <FollowButton
        v-if="followable"
        variant="overlay"
        :following="following"
        :busy="busy"
        :gated="isGated"
        @toggle="toggleFollow"
      />
      <!-- NO `@click` ON THIS WRAPPER (2026-09-26). It is a plate for the styling below, nothing
           more.

           With a click handler the span becomes an interactive node in its own right, and Chromium
           collapses it with its single child: the reported node keeps the button's ToggleButton
           role but takes the span's name, which is none. `AccessibleNameAuditTests` flagged exactly
           that — `[Discover] <NO NAME> ToggleButton Rect(267, 674 - 354, 761) near=[]`, the empty
           neighbours being this wrapper isolating the heart from the Follow pill beside it. The
           same FavoriteButton announces "Save to favorites" on a topic card, where nothing wraps
           it.

           It was redundant anyway: `FavoriteButton.onGatedClick` already calls `preventDefault()`
           and `stopPropagation()`, so the tile does not navigate when the heart is tapped. -->
      <span
        class="[&>button]:border-white/25 [&>button]:bg-black/55 [&>button]:shadow-lg [&>button]:backdrop-blur-sm"
      >
        <FavoriteButton
          :item="{ kind: 'show', ref: show.feed_id, label: show.title ?? show.feed_id }"
        />
      </span>
    </div>
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
    <!-- Stretched link, LAST so it does not cover the controls above it in the stacking order.
         It carries the show name as its accessible name, because it has no text of its own. -->
    <RouterLink
      :to="{ name: 'podcast', params: { feedId: show.feed_id } }"
      class="absolute inset-0 z-0 no-underline"
      :aria-label="show.title ?? show.feed_id"
    >
      <span class="sr-only">{{ show.title ?? show.feed_id }}</span>
    </RouterLink>
  </div>
</template>
