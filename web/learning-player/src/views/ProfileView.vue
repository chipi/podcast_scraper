<script setup lang="ts">
/**
 * Profile / account — where the signed-in user sees who they are and edits their personalization,
 * starting with their interest topics (chosen at sign-in via the onboarding card). Auth-gated.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getMyStats, getTopClusters, getUserInterests } from '../services/api'
import type { InterestCluster, UserStats } from '../services/types'
import { useAuthStore } from '../stores/auth'
import InterestsPicker from '../components/InterestsPicker.vue'
import Sparkline from '../components/Sparkline.vue'

const { t } = useI18n()
const auth = useAuthStore()

const interests = ref<string[]>([])
const clusters = ref<InterestCluster[]>([])
const pickerOpen = ref(false)

// Listening analytics (UXS-014) — the user's own play history, summarized.
const stats = ref<UserStats | null>(null)
const hours = computed(() => (stats.value ? stats.value.listening_seconds / 3600 : 0))
const hoursLabel = computed(() => (hours.value >= 10 ? Math.round(hours.value) : hours.value.toFixed(1)))
const series = computed(() => stats.value?.daily.map((d) => d.count) ?? [])
const hasStats = computed(() => !!stats.value && stats.value.episodes > 0)

// Map saved interest tokens → human labels. Clusters resolve via the top-cluster set; topics and
// people (followed from entity cards) de-slug from their id (`topic:personal-growth` → "personal
// growth"). `kind` drives the chip hue so people read distinct from topics.
const interestLabels = computed(() => {
  const byId = new Map(clusters.value.map((c) => [c.id, c.label]))
  return interests.value.map((id) => ({
    id,
    kind: id.startsWith('person:') ? 'person' : 'topic',
    label: byId.get(id) ?? id.replace(/^(tc|topic|person):/, '').replace(/-/g, ' '),
  }))
})

async function load(): Promise<void> {
  const [ints, tops, st] = await Promise.all([
    getUserInterests().catch(() => [] as string[]),
    getTopClusters(50).catch(() => [] as InterestCluster[]),
    getMyStats().catch(() => null),
  ])
  interests.value = ints
  clusters.value = tops
  stats.value = st
}

function onSaved(ids: string[]): void {
  interests.value = ids
}

onMounted(load)
</script>

<template>
  <section class="max-w-2xl">
    <h1 class="mb-1 font-display text-3xl font-extrabold tracking-tight">{{ t('profile.title') }}</h1>
    <p class="mb-6 text-muted">{{ auth.user?.name }}<span v-if="auth.user?.email"> · {{ auth.user?.email }}</span></p>

    <section class="rounded-2xl border border-border p-5">
      <div class="mb-3 flex items-center justify-between gap-2">
        <h2 class="lp-section">{{ t('profile.interests') }}</h2>
        <button type="button" class="text-sm font-bold text-accent" @click="pickerOpen = true">
          {{ t('profile.editInterests') }}
        </button>
      </div>
      <p class="mb-3 text-sm text-muted">{{ t('profile.interestsHelp') }}</p>
      <div v-if="interestLabels.length" class="flex flex-wrap gap-1.5">
        <span
          v-for="i in interestLabels"
          :key="i.id"
          class="rounded-full bg-overlay px-2.5 py-1 text-xs"
          :class="i.kind === 'person' ? 'text-person' : 'text-topic'"
        >{{ i.label }}</span>
      </div>
      <p v-else class="text-sm text-muted">{{ t('profile.noInterests') }}</p>
    </section>

    <!-- Listening analytics (UXS-014) — derived entirely from this user's own play history. -->
    <section class="mt-6 rounded-2xl border border-border p-5">
      <h2 class="lp-section mb-4">{{ t('stats.title') }}</h2>
      <template v-if="hasStats">
        <div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div class="rounded-xl bg-overlay p-4">
            <div class="flex items-baseline gap-1">
              <span class="font-display text-3xl font-extrabold leading-none text-accent">{{ stats!.day_streak }}</span>
              <span v-if="stats!.day_streak > 0" aria-hidden="true">🔥</span>
            </div>
            <div class="mt-2 text-xs font-medium text-muted">{{ t('stats.streak') }}</div>
          </div>
          <div class="rounded-xl bg-overlay p-4">
            <span class="font-display text-3xl font-extrabold leading-none">{{ stats!.episodes }}</span>
            <div class="mt-2 text-xs font-medium text-muted">{{ t('stats.episodes') }}</div>
          </div>
          <div class="rounded-xl bg-overlay p-4">
            <span class="font-display text-3xl font-extrabold leading-none">{{ stats!.shows }}</span>
            <div class="mt-2 text-xs font-medium text-muted">{{ t('stats.shows') }}</div>
          </div>
          <div class="rounded-xl bg-overlay p-4">
            <span class="font-display text-3xl font-extrabold leading-none">{{ hoursLabel }}<span class="text-lg">h</span></span>
            <div class="mt-2 text-xs font-medium text-muted">{{ t('stats.hours') }}</div>
          </div>
        </div>
        <div class="mt-3 rounded-xl bg-overlay p-4">
          <div class="mb-2 flex items-baseline justify-between">
            <span class="text-xs font-medium text-muted">{{ t('stats.overTime') }}</span>
            <span class="text-xs text-muted">{{ t('stats.activeDays', stats!.active_days, { named: { count: stats!.active_days } }) }}</span>
          </div>
          <Sparkline :values="series" :width="320" :height="44" class="block w-full text-accent" />
        </div>
      </template>
      <p v-else class="text-sm text-muted">{{ t('stats.empty') }}</p>
    </section>

    <InterestsPicker v-if="pickerOpen" @close="pickerOpen = false" @saved="onSaved" />
  </section>
</template>
