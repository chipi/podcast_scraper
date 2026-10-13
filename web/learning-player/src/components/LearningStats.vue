<script setup lang="ts">
/**
 * Profile › Stats › Learning (operator 2026-10-12): the paths you are on — each with its kind, its
 * day and how many days are done — and how your paths have ended. A path left or swapped out used
 * to vanish with its days; the server now keeps it, so "3 finished" can be said at all.
 *
 * Hidden with nothing to say: no path on, none ended, no day done.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import type { UserStats } from '../services/types'
import KindWord from './KindWord.vue'

const props = defineProps<{ stats: UserStats }>()
const { t } = useI18n()

const paths = computed(() => props.stats.learning_now ?? [])
const ended = computed(() => (props.stats.paths_finished ?? 0) + (props.stats.paths_left ?? 0))
/** Only the counts that are not zero: "0 finished · 0 left" was noise. */
const totals = computed(() => {
  const s = props.stats
  const parts: string[] = []
  const n = (k: string, c: number) => t(k, c, { named: { count: c } })
  if (s.paths_finished) parts.push(n('stats.learningFinished', s.paths_finished))
  if (s.paths_left) parts.push(n('stats.learningLeft', s.paths_left))
  if (s.learning_days_total) parts.push(n('stats.learningDaysAll', s.learning_days_total))
  return parts.join(' · ')
})
const show = computed(
  () => paths.value.length > 0 || ended.value > 0 || (props.stats.learning_days_total ?? 0) > 0,
)
</script>

<template>
  <section v-if="show" class="rounded-2xl border border-border p-5" data-testid="stats-learning">
    <h2 class="lp-section mb-3">{{ t('stats.learningTitle') }}</h2>
    <ul v-if="paths.length" class="flex flex-col gap-2">
      <li v-for="p in paths" :key="p.subject_id">
        <RouterLink
          :to="{ name: 'learn', query: { subject: p.subject_id } }"
          class="block rounded-xl bg-overlay p-3 no-underline"
          data-testid="stats-learning-path"
        >
          <span class="block text-sm font-semibold text-learn"><KindWord :kind="p.subject_kind" />{{ p.subject_label }}</span>
          <span class="mt-0.5 block text-xs text-muted">{{
            t('stats.learningDay', { day: p.day, days: p.days, done: p.done })
          }}</span>
          <!-- Days done of the path's days: the bar is the path, not the calendar. -->
          <span class="mt-2 block h-1.5 overflow-hidden rounded-full bg-canvas" aria-hidden="true">
            <span
              class="block h-full rounded-full bg-learn"
              :style="{ width: `${p.days ? Math.round((p.done / p.days) * 100) : 0}%` }"
            />
          </span>
        </RouterLink>
      </li>
    </ul>
    <p v-if="totals" class="mt-3 text-sm text-muted" data-testid="stats-learning-totals">{{ totals }}</p>
  </section>
</template>
