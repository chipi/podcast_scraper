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
</script>

<template>
  <RouterLink
    v-if="worthShowing"
    :to="{ name: 'profile' }"
    class="mt-7 flex items-center justify-between gap-3 rounded-2xl border border-border bg-surface px-4 py-3 transition hover:border-accent"
  >
    <div class="min-w-0">
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
    <span class="shrink-0 text-sm font-medium text-accent" aria-hidden="true">→</span>
  </RouterLink>
</template>
