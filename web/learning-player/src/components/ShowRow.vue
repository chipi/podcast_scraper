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
import { formatPublishDate } from '../utils/format'

const props = defineProps<{ show: Podcast }>()
const { t, locale } = useI18n()

const art = computed(() => showArtwork(props.show))
const title = computed(() => props.show.title ?? props.show.feed_id)
const description = computed(() => props.show.description?.trim() ?? '')

/**
 * Show metadata, split by WHERE it belongs rather than dumped in one line.
 *
 * Under the artwork go the two facts about the FEED as an object — how many episodes, when it last
 * moved. Beside the title goes what the show IS — who makes it, what it is filed under — because
 * that reads as part of the identity, above the description it introduces.
 *
 * Cadence and typical length, which the show PAGE also shows, are deliberately absent: both are
 * derived from the episode list, and a list row holds only the catalogue record. Faking them from
 * `episode_count` would be inventing data.
 */
const updated = computed(() => formatPublishDate(props.show.last_updated ?? null, locale.value))
const identity = computed<string[]>(() => {
  const out: string[] = []
  if (props.show.authors?.length)
    out.push(t('podcast.byline', { authors: props.show.authors.join(', ') }))
  if (props.show.category) out.push(props.show.category)
  return out
})

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
    <!-- LEFT: artwork with the controls OVER it, the episode count beneath. -->
    <div class="lp-media-aside">
      <div class="relative">
        <img
          v-if="art"
          :src="art"
          :alt="title"
          loading="lazy"
          class="h-32 w-32 rounded-lg bg-elevated object-cover"
        />
        <!-- Without a fixed-width child the column collapses and squeezes what sits beneath it. -->
        <div v-else class="h-32 w-32 rounded-lg bg-elevated" aria-hidden="true" />
        <!-- The surface's controls OVER the artwork as one right-aligned column — the same L the
             grid tile makes (operator 2026-09-17). Under the artwork they cost a row of vertical
             space on every entry and, at 128px, a labelled Follow plus the heart did not fit side by
             side anyway.
             Stacked and flush right so the two edges read as one object. The plate classes match
             ShowTile's and EpisodeActions' `overlay`, so contrast never depends on whatever artwork
             happens to be underneath. `z-30` keeps them above the title's stretched card-link
             overlay; `.prevent.stop` so acting never also opens the show. -->
        <div
          v-if="$slots.actions"
          class="absolute right-1.5 top-1.5 z-30 flex flex-col items-end gap-1.5 [&>button]:border-white/25 [&>button]:bg-black/55 [&>button]:shadow-lg [&>button]:backdrop-blur-sm"
          @click.prevent.stop
        >
          <slot name="actions" />
        </div>
      </div>
      <!-- Facts about the feed as an object, under the artwork — the slot the episode card gives its
           date and duration. Stacked rather than joined with separators: the column is 128px, so one
           line would wrap anyway and wrap in the wrong places. -->
      <div v-if="show.episode_count || updated" class="text-xs font-medium leading-snug text-muted">
        <div v-if="show.episode_count">
          {{ t('podcast.episodeCount', { count: show.episode_count }, show.episode_count) }}
        </div>
        <div v-if="updated">{{ t('podcast.updated', { date: updated }) }}</div>
      </div>
      <!-- Surface-specific INFORMATION under the artwork (Discover's trending sparkline), kept apart
           from `#actions` so a readout never lands in the overlay's control column. -->
      <div v-if="$slots.meta" class="relative z-30">
        <slot name="meta" />
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
      <!-- Who makes it and what it is filed under, between the title and the description: it belongs
           to the show's identity, so it reads above the blurb it introduces rather than below it
           (operator 2026-09-17). Joined, so separators fall only between values actually PRESENT —
           a template-level "· unless first" has to know what preceded it and gets it wrong the
           moment a field is missing. The row has the width for one line here; the artwork column
           does not, which is why the feed facts are stacked over there instead. -->
      <p v-if="identity.length" class="lp-kicker mt-0.5" data-testid="show-row-identity">
        {{ identity.join(' · ') }}
      </p>
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
