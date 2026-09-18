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
import { getEpisode } from '../services/api'
import type { EpisodeDetail } from '../services/types'
import CheckIcon from './CheckIcon.vue'
import { useResurfacingStore } from '../stores/resurfacing'
import { borderClass } from '../utils/highlightColors'

const { t } = useI18n()
const resurfacing = useResurfacingStore()

const items = computed(() => resurfacing.railItems)

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

/** Answer one, in place. The store drops it and `railItems` pulls the next one in. */
function review(id: string): void {
  void resurfacing.review(id)
}

const artOf = (slug: string): string | null => details.value[slug]?.artwork_url ?? null
const titleOf = (slug: string): string => details.value[slug]?.title ?? ''
</script>

<template>
  <section v-if="items.length" class="mt-7" data-testid="home-revisit-rail">
    <div class="mb-3 flex items-baseline justify-between gap-3">
      <h2 class="lp-section">{{ t('home.revisitTitle') }}</h2>
      <!-- "See all" rather than a count: the badge on the Library icon already carries the number,
           and two places saying "7" is two places to disagree. -->
      <RouterLink
        :to="{ name: 'library', query: { tab: 'revisit' } }"
        class="shrink-0 text-xs font-semibold text-accent no-underline"
        data-testid="home-revisit-see-all"
      >{{ t('home.revisitSeeAll') }}</RouterLink>
    </div>

    <ul class="flex flex-col gap-2">
      <li v-for="item in items" :key="item.highlight.id" class="flex items-stretch gap-2">
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
        <!-- ONE action, not three (operator 2026-09-18).
             Reviews-answered-per-week is the only number that moves this loop, so the common
             answer — "seen it, done" — is worth a tap in place. The other two outcomes, stop
             resurfacing and unsave, are consequential and keep the context of the Revisit card;
             four cards x three controls would also put twelve buttons on Home.
             Same accent-outlined tick as that card, so it is a control already learned. -->
        <button
          type="button"
          class="lp-tap flex w-10 shrink-0 items-center justify-center rounded-xl border border-accent text-accent transition hover:bg-accent/10"
          :aria-label="t('home.revisitMarkReviewed')"
          :title="t('home.revisitMarkReviewed')"
          data-testid="home-revisit-reviewed"
          @click="review(item.highlight.id)"
        ><CheckIcon /></button>
      </li>
    </ul>
  </section>
</template>
