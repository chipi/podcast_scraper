<script setup lang="ts">
/**
 * "Your listening" — the recap panel in Profile (#1914).
 *
 * This REPLACES a number that was not true. The tile here used to read `listening_seconds / 3600`
 * from `/me/stats`, which is `sum(position_seconds)` — a lifetime snapshot of furthest position
 * reached. It goes up when you seek forward without hearing anything, does not move when you
 * re-listen, and cannot be windowed at all. The figure shown now is time actually accrued,
 * recorded per save and clamped so a seek cannot inflate it.
 *
 * Because that recording only started with Phase 0, the panel ALWAYS states its coverage
 * ("recorded 5 of 7 days"). Showing a partial total silently would repeat the original sin in a
 * new form; the caveat costs one line and disappears on its own as the window fills in.
 *
 * Renders nothing at all when there is no recording yet — an empty recap is worse than no recap.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getRecap } from '../services/api'
import type { RecapResponse, RecapWindow } from '../services/types'

const { t } = useI18n()

const WINDOWS: RecapWindow[] = ['week', 'month']
const window_ = ref<RecapWindow>('week')
const recap = ref<RecapResponse | null>(null)
const loading = ref(true)

async function load(): Promise<void> {
  loading.value = true
  recap.value = await getRecap(window_.value)
  loading.value = false
}
onMounted(load)
watch(window_, load)

/** Days are the listener's own, so the bars line up with the days they remember. */
const bars = computed(() => Object.entries(recap.value?.by_day ?? {}))
const peak = computed(() => Math.max(1, ...bars.value.map(([, v]) => v)))

const hours = computed(() => (recap.value?.listening_seconds ?? 0) / 3600)
/** Under ten hours a single decimal carries real information; above it, it is noise. */
const hoursLabel = computed(() =>
  hours.value >= 10 ? String(Math.round(hours.value)) : hours.value.toFixed(1),
)

/** Nothing recorded yet means nothing honest to say. */
const hasAnything = computed(
  () => !!recap.value && (recap.value.days_recorded > 0 || recap.value.episodes_started > 0),
)
/** Only worth saying while it is actually partial. */
const partial = computed(
  () => !!recap.value && recap.value.days_recorded < recap.value.days_in_window,
)

/** A saved line opens its episode AT the moment it came from, not at the beginning. */
const lineTarget = computed(() => {
  const line = recap.value?.best_line
  if (!line?.episode_slug) return null
  const t_ = line.start_ms != null ? Math.floor(line.start_ms / 1000) : null
  return { name: 'player', params: { slug: line.episode_slug }, query: t_ ? { t: String(t_) } : {} }
})
</script>

<template>
  <section v-if="!loading && hasAnything" class="mt-6">
    <div class="mb-3 flex items-baseline justify-between">
      <h2 class="lp-section">{{ t('recap.title') }}</h2>
      <div class="flex gap-1" role="group" :aria-label="t('recap.window')">
        <button
          v-for="w in WINDOWS"
          :key="w"
          type="button"
          class="rounded-full px-3 py-1 text-xs font-bold transition"
          :class="window_ === w ? 'bg-accent text-accent-foreground' : 'bg-overlay text-muted'"
          :aria-pressed="window_ === w"
          @click="window_ = w"
        >{{ t(`recap.${w}`) }}</button>
      </div>
    </div>

    <div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <div class="rounded-xl bg-overlay p-4">
        <span class="font-display text-3xl font-extrabold leading-none">{{ hoursLabel }}<span class="text-lg">h</span></span>
        <div class="mt-2 text-xs font-medium text-muted">{{ t('recap.listened') }}</div>
      </div>
      <div class="rounded-xl bg-overlay p-4">
        <span class="font-display text-3xl font-extrabold leading-none">{{ recap!.distinct_episodes }}</span>
        <div class="mt-2 text-xs font-medium text-muted">{{ t('recap.episodes') }}</div>
      </div>
      <div class="rounded-xl bg-overlay p-4">
        <span class="font-display text-3xl font-extrabold leading-none">{{ recap!.episodes_finished }}</span>
        <div class="mt-2 text-xs font-medium text-muted">{{ t('recap.finished') }}</div>
      </div>
      <div class="rounded-xl bg-overlay p-4">
        <span class="font-display text-3xl font-extrabold leading-none">{{ recap!.episodes_started }}</span>
        <div class="mt-2 text-xs font-medium text-muted">{{ t('recap.starts') }}</div>
      </div>
    </div>

    <!-- The day series. Bars rather than a sparkline: these are discrete days, and a listener
         reads "which day did I listen" off bars far more easily than off a line. -->
    <div class="mt-3 rounded-xl bg-overlay p-4">
      <div class="flex h-16 items-end gap-1" role="img" :aria-label="t('recap.byDay')">
        <div
          v-for="[day, value] in bars"
          :key="day"
          class="flex-1 rounded-sm bg-accent"
          :style="{ height: `${Math.max(2, (value / peak) * 100)}%`, opacity: value > 0 ? 1 : 0.25 }"
          :title="`${day}: ${Math.round(value / 60)}m`"
        />
      </div>
      <!-- Always visible while partial: the number above is real, and this says how much of the
           window produced it. -->
      <p v-if="partial" class="mt-2 text-xs text-muted">
        {{ t('recap.coverage', { recorded: recap!.days_recorded, total: recap!.days_in_window }) }}
      </p>
    </div>

    <div v-if="recap!.topics.length || recap!.people.length" class="mt-3">
      <h3 class="lp-kicker mb-2">{{ t('recap.recurring') }}</h3>
      <div class="flex flex-wrap gap-2">
        <span
          v-for="theme in [...recap!.topics, ...recap!.people]"
          :key="theme.token"
          class="rounded-full bg-overlay px-3 py-1 text-sm"
        >{{ theme.label }}<span class="ml-1 text-xs text-muted">{{ theme.episodes }}</span></span>
      </div>
    </div>

    <!-- The one part of this that is an artifact rather than a statistic. -->
    <figure v-if="recap!.best_line" class="mt-3 rounded-xl bg-overlay p-4">
      <h3 class="lp-kicker mb-2">{{ t('recap.line') }}</h3>
      <blockquote class="text-sm italic">“{{ recap!.best_line.quote_text }}”</blockquote>
      <RouterLink
        v-if="lineTarget"
        :to="lineTarget"
        class="mt-2 inline-block text-xs font-medium text-accent"
      >{{ t('recap.openLine') }}</RouterLink>
    </figure>
  </section>
</template>
