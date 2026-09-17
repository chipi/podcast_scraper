<script setup lang="ts">
/**
 * A show as a LIST ROW, built to the same proportions as {@link EpisodeCard} (operator 2026-09-17).
 *
 * ## Why this exists
 *
 * Shows had two list representations and neither matched the episode list they sit beside: Discover's
 * Shows tab used a 44px thumbnail with a title and an episode count, and Library's Saved tab had a
 * bare line of text. Both read as a different KIND of thing from the episode rows directly above
 * them, when a show and an episode are the same kind of thing to a reader — cover art, a name, a
 * line about it, something to open.
 *
 * So this is EpisodeCard's shape with a show's content: 128px artwork in the left column with the
 * facts and the actions under it, the name and description filling the right. Defined once and used
 * by both surfaces, so they cannot drift apart again — which is exactly how they got here.
 *
 * ## The description clamp
 *
 * Uses the shared `lp-media-*` window (see style.css): the aside sets the row's height and the prose
 * fills it and clips, rather than a fixed `line-clamp-N` that leaves dead space beside the artwork on
 * one row and overshoots on the next.
 */
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import type { Podcast } from '../services/types'
import { showArtwork } from '../utils/episode'

const props = defineProps<{ show: Podcast }>()
const { t } = useI18n()

const art = computed(() => showArtwork(props.show))
const title = computed(() => props.show.title ?? props.show.feed_id)
const description = computed(() => props.show.description?.trim() ?? '')

// Read more/less — the same measurement EpisodeCard makes, line for line: the PROSE against the
// WINDOW (measuring the window against itself always matches, because it stretches to fit), a safe
// `true` default so a needed toggle is never hidden before layout, and a ResizeObserver to re-read
// when the column changes.
const descExpanded = ref(false)
const descEl = ref<HTMLElement | null>(null)
const descClipped = ref(true)

function measureDesc(): void {
  const el = descEl.value
  if (!el || descExpanded.value) return
  const prose = el.firstElementChild
  if (!prose || el.clientHeight === 0) return
  descClipped.value = prose.scrollHeight - el.clientHeight > 1
}

/**
 * Observe the window WHENEVER IT APPEARS, not once at mount.
 *
 * `onMounted` was wrong here, and subtly: the `show` prop arrives in two phases. `useFollowedShows`
 * maps `library.items` through a fallback record with `description: null` whenever the catalogue join
 * has not landed, and `library.items` is assigned inside `library.load()` while `catalogue` is
 * assigned only after the `Promise.all`. Vue's flush is queued at the first assignment and wins the
 * microtask race, so the first render has the show but no description — `v-if="description"` is
 * false, the window element does not exist, and the mount-time guard skipped observer creation
 * entirely. For the life of that instance there was then NO observer, and the single `watch` on the
 * description was the only re-measure: fine if the row was visible at that instant, permanently
 * stuck at the `true` default if it was in a hidden tab panel.
 *
 * Watching the ref covers both cases — `immediate: true` makes it a strict superset of `onMounted`,
 * `flush: 'post'` guarantees the DOM exists. ResizeObserver's own initial callback delivers the first
 * size, so no explicit measure on attach is needed, and it fires again across
 * `display: none` → visible (verified in-browser), which is what makes a hidden mount harmless.
 *
 * `onBeforeUnmount` stays at setup top level: Vue does not set `currentInstance` for watcher
 * callbacks, so registering it inside would warn and not bind.
 */
let ro: ResizeObserver | null = null
watch(
  descEl,
  (el) => {
    ro?.disconnect()
    ro = null
    if (!el || typeof ResizeObserver === 'undefined') return
    ro = new ResizeObserver(() => measureDesc())
    ro.observe(el)
  },
  { flush: 'post', immediate: true },
)
onBeforeUnmount(() => ro?.disconnect())

const canExpand = computed(() => !!description.value && (descClipped.value || descExpanded.value))
</script>

<template>
  <article
    class="lp-media-row group relative -mx-3 gap-4 rounded-xl border-b border-border px-3 py-5 transition-colors sm:gap-5"
    data-testid="show-row"
  >
    <!-- LEFT: artwork, the episode count, then the surface's own controls — the same stack the
         episode card puts under its artwork. -->
    <div class="lp-media-aside">
      <img
        v-if="art"
        :src="art"
        :alt="title"
        loading="lazy"
        class="h-32 w-32 rounded-lg bg-elevated object-cover"
      />
      <!-- Without a fixed-width child the column collapses and squeezes what sits beneath it. -->
      <div v-else class="h-32 w-32 rounded-lg bg-elevated" aria-hidden="true" />
      <div v-if="show.episode_count" class="text-xs font-medium text-muted">
        {{ t('podcast.episodeCount', { count: show.episode_count }, show.episode_count) }}
      </div>
      <!-- Under the artwork, where EpisodeCard puts its action row — this row is built to the
           episode card's proportions, so its controls belong in the same place (operator
           2026-09-17). `w-32` matches the artwork; `relative z-30` keeps them tappable above the
           title's stretched card-link overlay. -->
      <div v-if="$slots.actions" class="relative z-30 w-32">
        <slot name="actions" />
      </div>
    </div>

    <div class="lp-media-body">
      <!-- The name WRAPS (UXS-014:70) — the width is elastic here, so there is no reserved-height
           row for a clamp to protect. The stretched ::after makes the whole row open the show. -->
      <RouterLink
        :to="{ name: 'podcast', params: { feedId: show.feed_id } }"
        class="block font-display text-lg font-bold leading-snug text-canvas-foreground no-underline after:absolute after:inset-0 sm:text-xl"
        >{{ title }}</RouterLink
      >
      <div
        v-if="description"
        ref="descEl"
        class="lp-media-clip mt-2"
        :class="descExpanded ? 'lp-media-clip--open' : ''"
      >
        <p class="text-sm leading-relaxed text-muted">{{ description }}</p>
      </div>
      <button
        v-if="canExpand"
        type="button"
        class="lp-media-foot relative z-30 mt-1 w-fit text-xs font-bold text-accent transition hover:opacity-80"
        data-testid="show-row-read-more"
        :aria-expanded="descExpanded"
        @click="descExpanded = !descExpanded"
      >
        {{ descExpanded ? t('card.readLess') : t('card.readMore') }}
      </button>
    </div>

  </article>
</template>
