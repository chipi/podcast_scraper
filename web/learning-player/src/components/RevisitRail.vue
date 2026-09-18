<script setup lang="ts">
/**
 * "Worth revisiting" — a few due captures on Home (operator 2026-09-18).
 *
 * ## Why Home, and why this is the intervention rather than more email
 *
 * Simulated over a year of captures, what moves the resurfacing loop is the NUMBER of reviews
 * answered per week, and it saturates hard: 7/week reaches 50% of captures, 14/week reaches 99%,
 * and past that nothing improves. Cadence alone is worth nothing — spreading the same budget
 * across seven days instead of one scores identically (0 points, at every budget tested).
 *
 * So the only thing that helps is causing reviews that would not otherwise happen. The user is
 * ALREADY here every day, because this is where they come to listen; Revisit sat two taps away
 * behind a Library tab whose only signal was a number on an icon. This rail asks for two answers a
 * day from someone already standing here — which is the whole 14/week — and costs no notification,
 * so it cannot provoke the pause switch that suppresses everything.
 *
 * ## The card
 *
 * The quote leads, with the episode as a small square thumbnail beside it and the capture's colour
 * on the left edge. Two overlay treatments were tried first — artwork dimmed behind the text, then
 * a cropped strip fading into it — and both looked forced (operator 2026-09-18). The episode is a
 * fact ABOUT the quote rather than a backdrop for it, so it sits beside the text in the same row
 * idiom `EpisodeRow` and the downloads list already use.
 *
 * Tapping goes to the REVISIT TAB, scrolled to this capture (`?focus=<id>`), not to the player
 * (operator 2026-09-18). From Home the user is deciding what to do with a capture, and the three
 * outcomes — reviewed, stop resurfacing, unsave — live on that card. Jumping to the player would
 * also have marked it reviewed on arrival (#35), deciding on their behalf the one thing they came
 * to decide.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getEpisode, getMyStats } from '../services/api'
import type { EpisodeDetail } from '../services/types'
import BellOffIcon from './BellOffIcon.vue'
import CheckIcon from './CheckIcon.vue'
import SectionHeading from './SectionHeading.vue'
import { useIsDesktop } from '../composables/useMediaQuery'
import { useResurfacingStore } from '../stores/resurfacing'
import { borderClass } from '../utils/highlightColors'

const { t } = useI18n()
const resurfacing = useResurfacingStore()

/**
 * Three cards on a phone, four on a desktop (operator 2026-09-18).
 *
 * The rail sits beside Trends on `lg`, where a fourth card earns its place; on a phone it is a
 * full-width stack above the fold and a fourth pushes the next section off screen. The store hands
 * over every one-per-episode candidate and the view takes what fits — how many fit is a layout
 * question, not a data one.
 */
const isDesktop = useIsDesktop()
const items = computed(() => resurfacing.railItems.slice(0, isDesktop.value ? 4 : 3))
/**
 * The loop in one line: what you have kept, and what you have done with it.
 *
 * Its own small fetch rather than another field on the resurfacing response — these are PROFILE
 * stats, and `/me/stats` already computes them. Failing softly to null hides the line: a header
 * that said "0 kept" while four captures sat beneath it would be worse than saying nothing.
 *
 * Deliberately NOT the due count. The Library nav badge already carries that, and two places
 * showing the same number is two places to disagree; kept-and-answered is progress rather than
 * workload, so it adds instead of repeating.
 */
const loop = ref<{ kept: number; reviewed: number } | null>(null)

onMounted(async () => {
  const s = await getMyStats().catch(() => null)
  const kept = s?.captures ?? 0
  // `captures_reviewed`, NOT `reviews_total`: the latter counts review EVENTS, so a header saying
  // "8 reviewed" would claim eight captures when it was eight passes over four of them.
  if (s && kept > 0) loop.value = { kept, reviewed: s.captures_reviewed ?? 0 }
})


/**
 * Home is now one of the events that refreshes the count.
 *
 * The store loaded on sign-in, on visiting the Library, and after a capture — the three things
 * that could move the BADGE. This rail renders the items themselves on a fourth surface, so
 * without this it showed nothing until the user had been to the Library, which is the very trip
 * the rail exists to save. Still no polling: the ladder is measured in days.
 */
onMounted(() => {
  void resurfacing.load()
})

/**
 * Episode detail per slug, for the artwork and the title.
 *
 * `ResurfacingItem` carries the highlight and a prompt, not its episode, so this hydrates the same
 * way `ResurfacingInbox` does — at most four slugs, each failing softly. A card whose episode never
 * arrives still renders: it loses its picture and its show name, not the quote, which is the part
 * being asked about.
 */
const details = ref<Record<string, EpisodeDetail>>({})

watch(
  items,
  async (list) => {
    const slugs = [...new Set(list.map((i) => i.highlight.episode_slug))].filter(
      (s) => s && !details.value[s],
    )
    await Promise.all(
      slugs.map(async (slug) => {
        const d = await getEpisode(slug).catch(() => null)
        if (d) details.value[slug] = d
      }),
    )
  },
  { immediate: true },
)

/** The captured words; empty for a moment saved before quote text was stored. */
function quoteOf(h: { quote_text?: string | null }): string {
  return (h.quote_text ?? '').trim()
}

/** Stop resurfacing one, in place. Not a delete — it stays in Saved, reversible there. */
function mute(id: string): void {
  void resurfacing.mute(id)
}

/** Answer one, in place. The store drops it and `railItems` pulls the next one in. */
function review(id: string): void {
  void resurfacing.review(id)
}

const artOf = (slug: string): string | null => details.value[slug]?.artwork_url ?? null
const titleOf = (slug: string): string => details.value[slug]?.title ?? ''
</script>

<template>
  <section v-if="items.length" class="mt-7" data-testid="home-revisit-rail">
    <SectionHeading
      :title="t('home.revisitTitle')"
      :kicker="loop ? t('home.revisitLoop', { kept: loop.kept, reviewed: loop.reviewed }) : null"
    >
      <template #action>
        <!-- "See all" rather than a count: the kicker already carries the numbers, and the Library
             nav badge carries the due count. -->
        <RouterLink
          :to="{ name: 'library', query: { tab: 'revisit' } }"
          class="text-xs font-semibold text-accent no-underline"
          data-testid="home-revisit-see-all"
        >{{ t('home.revisitSeeAll') }}</RouterLink>
      </template>
    </SectionHeading>

    <ul class="flex flex-col gap-2">
      <li v-for="item in items" :key="item.highlight.id" class="flex items-center gap-2">
        <RouterLink
          :to="{
            name: 'library',
            query: { tab: 'revisit', focus: item.highlight.id },
          }"
          class="flex min-w-0 flex-1 items-center gap-3 rounded-xl border border-l-4 border-border bg-elevated p-3 no-underline"
          :class="borderClass(item.highlight.color)"
          data-testid="home-revisit-card"
        >
          <div class="min-w-0 flex-1">
            <p class="line-clamp-2 text-sm font-semibold italic leading-snug text-canvas-foreground">
              {{ quoteOf(item.highlight) }}
            </p>
            <p class="lp-kicker mt-1 truncate">
              <span v-if="item.highlight.speaker">{{ item.highlight.speaker }} · </span>
              <span>{{ titleOf(item.highlight.episode_slug) }}</span>
            </p>
          </div>
          <!-- A plain square thumbnail, the same one `EpisodeRow` and the downloads rows use.
               The artwork was a dimmed background behind the quote, then a cropped strip fading
               into it — both read as forced (operator 2026-09-18). The episode is a fact ABOUT the
               quote, not a backdrop for it, so it sits beside the text like every other row in the
               app rather than inventing a treatment for this one surface. -->
          <img
            v-if="artOf(item.highlight.episode_slug)"
            :src="artOf(item.highlight.episode_slug)!"
            alt=""
            aria-hidden="true"
            loading="lazy"
            class="h-11 w-11 shrink-0 rounded-lg bg-overlay object-cover"
          />
        </RouterLink>
        <!-- TWO actions, stacked and right-aligned (operator 2026-09-18).
             Reviews-answered-per-week is the only number that moves this loop, so the common
             answer is worth a tap in place — and mute earns the second slot because "stop asking
             me about this" is the other thing a glance produces, and it is NOT destructive: the
             capture stays in Saved, where the bell marker makes it reversible.

             Unsave is the one that stays on the Revisit card. It destroys authored content and is
             confirm-gated there; a confirmation dialog raised from a homepage rail would be the
             app stopping you mid-scroll.

             The SAME 32px circles the Revisit card uses, quiet until hovered. They were one
             full-height accent-outlined rectangle, which failed twice over: strong colour on a
             tick reads as "this IS checked" rather than "check this" — the mistake already
             rejected as a filled accent disc on the Revisit card — and a tall rectangle is simply
             not the control that lives in the tab. -->
        <div class="flex shrink-0 flex-col gap-1.5">
          <button
            type="button"
            class="lp-tap flex h-8 w-8 items-center justify-center rounded-full border border-border text-muted transition hover:border-accent hover:text-accent"
            :aria-label="t('home.revisitMarkReviewed')"
            :title="t('home.revisitMarkReviewed')"
            data-testid="home-revisit-reviewed"
            @click="review(item.highlight.id)"
          ><CheckIcon /></button>
          <button
            type="button"
            class="lp-tap flex h-8 w-8 items-center justify-center rounded-full border border-border text-muted transition hover:border-accent hover:text-accent"
            :aria-label="t('home.revisitMute')"
            :title="t('home.revisitMute')"
            data-testid="home-revisit-mute"
            @click="mute(item.highlight.id)"
          ><BellOffIcon /></button>
        </div>
      </li>
    </ul>
  </section>
</template>
