<script setup lang="ts">
/**
 * A one-line look back, on Home (#1914 slice 5).
 *
 * Deliberately NOT the recap. The recap lives in Profile and stays there — that is the place you
 * can always go to look. This is the periodic prompt that reminds you it exists, because a recap
 * nobody sees is not a feature, and Profile is a screen people visit on purpose rather than by
 * accident.
 *
 * So it says the least it can while still being worth reading: how long, how many, and the one
 * thing that kept coming up. Everything else is one tap away.
 *
 * Self-hides when signed out, while loading, or when there is nothing recorded — the same
 * contract YourWeek has directly above it. A row that says "0h, 0 episodes" is worse than no row:
 * it takes space to tell you nothing happened, which you already knew.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getRecap } from '../services/api'
import type { RecapResponse } from '../services/types'
import { useAuthStore } from '../stores/auth'

const { t } = useI18n()
const auth = useAuthStore()
const recap = ref<RecapResponse | null>(null)

onMounted(async () => {
  if (!auth.isAuthenticated) return
  recap.value = await getRecap('week')
})

const hours = computed(() => (recap.value?.listening_seconds ?? 0) / 3600)
const hoursLabel = computed(() =>
  hours.value >= 10 ? String(Math.round(hours.value)) : hours.value.toFixed(1),
)
/** The single strongest theme — a rail is not the place for a list. */
const headline = computed(() => recap.value?.topics[0]?.label ?? null)
/** Nothing listened to means nothing to look back on. */
const worthShowing = computed(() => (recap.value?.listening_seconds ?? 0) > 0)

/**
 * The week's own shape, as the card's backdrop (operator 2026-09-19: "not just another empty-ish
 * card").
 *
 * Drawn from `by_day` rather than invented: a gradient or a stock flourish would be decoration
 * that says nothing, and this row's whole argument is that it is worth a tap. The bars ARE the
 * week — a heavy Tuesday looks like a heavy Tuesday — so the card gets a face without gaining a
 * single word, which is the constraint the component was written under.
 *
 * Normalised to the busiest day so the silhouette reads at any absolute volume; days with nothing
 * recorded keep a hairline so the week stays seven columns wide rather than collapsing.
 */
const dayBars = computed<number[]>(() => {
  const days = Object.keys(recap.value?.by_day ?? {}).sort()
  const values = days.map((d) => recap.value?.by_day[d] ?? 0)
  const peak = Math.max(...values, 0)
  if (!peak) return []
  return values.map((v) => Math.max(0.04, v / peak))
})
</script>

<template>
  <RouterLink
    v-if="worthShowing"
    :to="{ name: 'profile', query: { tab: 'stats' } }"
    class="lp-recap-prompt relative mt-7 flex items-center justify-between gap-3 overflow-hidden rounded-2xl border border-border bg-surface px-4 py-3 transition hover:border-accent"
  >
    <!-- The week, behind the words. Bars are bottom-anchored and run the full width of the card,
         fading out under the text so the line stays the thing you read. `aria-hidden`: it restates
         the hours the text already gives, and a screen reader does not need the silhouette. -->
    <div
      v-if="dayBars.length"
      class="pointer-events-none absolute inset-0 flex items-end gap-px opacity-[0.18]"
      aria-hidden="true"
    >
      <span
        v-for="(h, i) in dayBars"
        :key="i"
        class="flex-1 rounded-t-sm bg-accent"
        :style="{ height: `${Math.round(h * 100)}%` }"
      />
    </div>
    <div class="relative min-w-0">
      <span class="lp-kicker">{{ t('recap.promptTitle') }}</span>
      <p class="mt-1 truncate text-sm">
        <span class="font-bold">{{ hoursLabel }}h</span>
        <span class="text-muted"> · </span>
        <span>{{ t('recap.promptEpisodes', recap!.distinct_episodes, { named: { count: recap!.distinct_episodes } }) }}</span>
        <template v-if="headline">
          <span class="text-muted"> · </span><span class="text-muted">{{ headline }}</span>
        </template>
      </p>
    </div>
    <span class="relative shrink-0 text-sm font-medium text-muted" aria-hidden="true">→</span>
  </RouterLink>
</template>
