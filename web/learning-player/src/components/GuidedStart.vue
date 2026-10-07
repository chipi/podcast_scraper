<script setup lang="ts">
/**
 * The guided start on Home (operator 2026-10-07) — a short flow of cards for a new listener, in
 * place of the one welcome card: choose a few interests, follow a few shows, done. Home on a first
 * visit was two screens of not much; this gives it a job, and once the minimum is set (3 interests,
 * 1 show — or each step skipped) it closes and Home rebuilds from the choices.
 *
 * The steps follow the listener's real state, not a counter: step 1 while fewer than 3 interests,
 * step 2 while no show is followed, then done. A skip moves past a step for this run. Home decides
 * when the flow is shown and stores that it ran (see `GUIDED_START_PREF`).
 */
import { computed, onMounted, ref } from "vue"
import { useI18n } from "vue-i18n"
import { getPodcasts } from "../services/api"
import type { Podcast } from "../services/types"
import { useInterestsStore } from "../stores/interests"
import { useLibraryStore } from "../stores/library"
import BrandGlyph from "./BrandGlyph.vue"
import CardRail from "./CardRail.vue"
import ShowTile from "./ShowTile.vue"

const MIN_INTERESTS = 3
const MIN_SHOWS = 1

defineProps<{ welcomeName: string | null }>()
const emit = defineEmits<{
  (e: "choose-interests"): void
  (e: "dismiss"): void
  (e: "finish"): void
}>()

const { t } = useI18n()
const interests = useInterestsStore()
const library = useLibraryStore()

const skippedInterests = ref(false)
const skippedShows = ref(false)
const interestCount = computed(() => interests.ids.length)
const showCount = computed(() => library.feedIds.length)

const step = computed<1 | 2 | 3>(() => {
  if (interestCount.value < MIN_INTERESTS && !skippedInterests.value) return 1
  if (showCount.value < MIN_SHOWS && !skippedShows.value) return 2
  return 3
})
/** "Skip step" and "Not now" — the two ways out, quieter than the step's action. */
const EXIT_BUTTON =
  "inline-flex h-11 shrink-0 items-center whitespace-nowrap rounded-full px-2.5 text-sm font-semibold text-muted transition hover:text-canvas-foreground"

const interestsToGo = computed(() => Math.max(0, MIN_INTERESTS - interestCount.value))

/** Shows to follow in step 2 — the first 8 of `/podcasts`, which is sorted by feed id: unranked. */
const shows = ref<Podcast[]>([])
onMounted(async () => {
  try {
    shows.value = (await getPodcasts()).slice(0, 8)
  } catch {
    shows.value = []
  }
})
</script>

<template>
  <section
    class="relative mt-4 overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-accent/20 via-elevated to-surface p-5 sm:p-6"
    data-testid="interests-welcome"
    :data-step="step"
  >
    <BrandGlyph
      class="pointer-events-none absolute -right-4 -top-4 h-28 w-28 opacity-15"
      aria-hidden="true"
    />
    <!-- Where you are in the flow: three dots, the current one filled. -->
    <div class="mb-3 flex items-center gap-1.5" :aria-label="t('guided.progress', { step, total: 3 })" role="img">
      <span
        v-for="n in 3"
        :key="n"
        class="h-1.5 rounded-full transition-all"
        :class="n === step ? 'w-6 bg-accent' : n < step ? 'w-1.5 bg-accent/60' : 'w-1.5 bg-border'"
      />
    </div>

    <template v-if="step === 1">
      <p class="lp-kicker mb-2">{{ t("interests.cardTitle") }}</p>
      <h2 class="font-display text-2xl font-extrabold tracking-tight text-canvas-foreground">
        {{ welcomeName ? t("interests.welcome", { name: welcomeName }) : t("interests.welcomeNoName") }}
      </h2>
      <p class="mt-2 max-w-prose text-sm leading-relaxed text-muted">{{ t("guided.step1Body") }}</p>
      <p v-if="interestCount > 0" class="mt-2 text-sm font-semibold text-canvas-foreground" data-testid="guided-interests-to-go">
        {{ t("guided.interestsToGo", interestsToGo, { named: { count: interestsToGo } }) }}
      </p>
      <!-- One row (operator 2026-10-08): the action, then "Skip step" (this step only) and "Not now"
           (closes the whole guide), in that order. Text buttons for the two ways out, so the row
           fits a phone without wrapping. -->
      <div class="mt-4 flex flex-nowrap items-center gap-1">
        <button
          type="button"
          class="inline-flex h-11 shrink-0 items-center whitespace-nowrap rounded-full bg-accent px-4 text-sm font-bold text-accent-foreground shadow-sm transition hover:opacity-90"
          data-testid="interests-choose"
          @click="emit('choose-interests')"
        >
          {{ t("interests.cardCta") }}
        </button>
        <button type="button" :class="EXIT_BUTTON" data-testid="guided-skip" @click="skippedInterests = true">
          {{ t("guided.skip") }}
        </button>
        <button type="button" :class="EXIT_BUTTON" data-testid="interests-not-now" @click="emit('dismiss')">
          {{ t("interests.dismiss") }}
        </button>
      </div>
    </template>

    <template v-else-if="step === 2">
      <p class="lp-kicker mb-2">{{ t("guided.step2Kicker") }}</p>
      <h2 class="font-display text-2xl font-extrabold tracking-tight text-canvas-foreground">
        {{ t("guided.step2Title") }}
      </h2>
      <p class="mt-2 max-w-prose text-sm leading-relaxed text-muted">{{ t("guided.step2Body") }}</p>
      <!-- The same followable tiles as Discover's Shows: follow in place, no trip away. -->
      <CardRail v-if="shows.length" class="mt-4" data-testid="guided-shows">
        <li v-for="p in shows" :key="p.feed_id" class="lp-rail-item">
          <ShowTile :show="p" followable />
        </li>
      </CardRail>
      <div class="mt-4 flex flex-nowrap items-center gap-1">
        <RouterLink
          :to="{ name: 'browse', query: { tab: 'shows' } }"
          class="inline-flex h-11 shrink-0 items-center whitespace-nowrap rounded-full border border-border px-4 text-sm font-semibold text-canvas-foreground no-underline transition hover:bg-overlay"
          data-testid="interests-follow-shows"
        >
          {{ t("guided.allShows") }}
        </RouterLink>
        <button type="button" :class="EXIT_BUTTON" data-testid="guided-skip" @click="skippedShows = true">
          {{ t("guided.skip") }}
        </button>
        <button type="button" :class="EXIT_BUTTON" data-testid="interests-not-now" @click="emit('dismiss')">
          {{ t("interests.dismiss") }}
        </button>
      </div>
    </template>

    <template v-else>
      <p class="lp-kicker mb-2">{{ t("guided.doneKicker") }}</p>
      <h2 class="font-display text-2xl font-extrabold tracking-tight text-canvas-foreground">
        {{ t("guided.doneTitle") }}
      </h2>
      <p class="mt-2 max-w-prose text-sm leading-relaxed text-muted" data-testid="guided-summary">
        {{
          t("guided.doneBody", {
            interests: t("guided.interestCount", interestCount, { named: { count: interestCount } }),
            shows: t("guided.showCount", showCount, { named: { count: showCount } }),
          })
        }}
      </p>
      <div class="mt-4">
        <button
          type="button"
          class="inline-flex h-11 items-center rounded-full bg-accent px-5 text-sm font-bold text-accent-foreground shadow-sm transition hover:opacity-90"
          data-testid="guided-finish"
          @click="emit('finish')"
        >
          {{ t("guided.finish") }}
        </button>
      </div>
    </template>
  </section>
</template>
