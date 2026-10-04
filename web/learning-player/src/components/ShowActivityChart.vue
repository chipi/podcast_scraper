<script setup lang="ts">
/**
 * Show activity — episodes published per month as a compact bar chart ("how active / consistent is
 * this show"). Built client-side from the loaded episodes' publish dates; fills gaps as zero-height
 * bars, caps the window so it stays small, and hides itself when there aren't enough dated episodes
 * to be meaningful.
 *
 * Readable and navigable (operator 2026-10-04): the only labels used to be a "YYYY-MM – YYYY-MM"
 * range in the right corner, so neither the span nor what one bar stood for was clear. Each bar now
 * carries its month underneath (the year at the first bar and at every January) and its count above,
 * the heading says "episodes per month", and a bar with episodes is a button that jumps to that
 * month in the list below (`select`, handled by the page that owns the list).
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { EpisodeSummary } from '../services/types'

const props = defineProps<{ episodes: EpisodeSummary[] }>()
const emit = defineEmits<{ (e: 'select', month: string): void }>()
const { t, locale } = useI18n()

const MAX_MONTHS = 24
/** Above this many bars a label under every one would overlap, so every third is named. */
const LABEL_EVERY_BAR_UP_TO = 12

interface Bar {
  key: string
  count: number
  month: string
  /** The year, shown only at the first bar and at each January. */
  year: string | null
  labelled: boolean
  /** Spoken / hover name of the whole bar, e.g. "April 2026: 3 episodes". */
  name: string
}

const bars = computed<Bar[]>(() => {
  const byMonth = new Map<string, number>()
  for (const e of props.episodes) {
    const key = (e.publish_date ?? '').slice(0, 7) // YYYY-MM
    if (key.length === 7) byMonth.set(key, (byMonth.get(key) ?? 0) + 1)
  }
  if (byMonth.size < 2) return []
  const months = [...byMonth.keys()].sort()
  const [ey, em] = months[months.length - 1].split('-').map(Number)
  let [y, m] = months[0].split('-').map(Number)
  const keys: Array<[string, number, number]> = []
  while ((y < ey || (y === ey && m <= em)) && keys.length < MAX_MONTHS) {
    keys.push([`${y}-${String(m).padStart(2, '0')}`, y, m])
    m += 1
    if (m > 12) {
      m = 1
      y += 1
    }
  }
  const short = new Intl.DateTimeFormat(locale.value, { month: 'short', timeZone: 'UTC' })
  const long = new Intl.DateTimeFormat(locale.value, { month: 'long', year: 'numeric', timeZone: 'UTC' })
  const every = keys.length > LABEL_EVERY_BAR_UP_TO ? 3 : 1
  return keys.map(([key, yy, mm], i) => {
    const date = new Date(Date.UTC(yy, mm - 1, 1))
    const count = byMonth.get(key) ?? 0
    const labelled = i % every === 0 || i === keys.length - 1
    return {
      key,
      count,
      month: short.format(date),
      year: labelled && (i === 0 || mm === 1) ? String(yy) : null,
      labelled,
      name: t('podcast.activityBar', count, { named: { month: long.format(date), count } }),
    }
  })
})

const maxCount = computed(() => Math.max(1, ...bars.value.map((b) => b.count)))
const showCounts = computed(() => bars.value.length <= LABEL_EVERY_BAR_UP_TO)
</script>

<template>
  <section v-if="bars.length" class="mb-6" data-testid="show-activity">
    <div class="mb-1.5 flex items-baseline gap-2">
      <h3 class="lp-kicker">{{ t('podcast.activity') }}</h3>
      <span class="text-[11px] text-muted" data-testid="show-activity-unit">{{ t('podcast.activityPerMonth') }}</span>
    </div>
    <!--
      Bars are DATA, not a kind, so they take neither the accent (it means "you can act on this" and
      sat right under its own heading as decoration) nor the topic hue (since 2026-10-04 that is the
      violet every topic pill wears, and a violet bar would read as "topics"). Published months are a
      solid neutral, silent months a faint baseline tick, so the gaps read as clearly as the bursts.
    -->
    <div class="flex items-end gap-px" style="height: 56px">
      <template v-for="b in bars" :key="b.key">
        <button
          v-if="b.count > 0"
          type="button"
          class="lp-tap group flex h-full min-w-[3px] flex-1 flex-col items-center justify-end"
          :title="b.name"
          :aria-label="b.name"
          :data-testid="`show-activity-bar-${b.key}`"
          @click="emit('select', b.key)"
        >
          <span
            v-if="showCounts"
            class="mb-0.5 text-[9px] font-semibold tabular-nums text-muted"
            aria-hidden="true"
            >{{ b.count }}</span
          >
          <span
            class="block w-full rounded-sm bg-muted/60 transition group-hover:bg-muted"
            :style="{ height: Math.round((b.count / maxCount) * 40) + 4 + 'px' }"
          />
        </button>
        <div
          v-else
          class="flex h-full min-w-[3px] flex-1 items-end"
          :title="b.name"
          :data-testid="`show-activity-bar-${b.key}`"
        >
          <span class="block h-1 w-full rounded-sm bg-muted/20" />
        </div>
      </template>
    </div>
    <!-- The axis: what each bar is. Same flex columns as the bars, so a label sits under its bar. -->
    <div class="mt-1 flex gap-px" aria-hidden="true" data-testid="show-activity-axis">
      <div v-for="b in bars" :key="b.key" class="min-w-[3px] flex-1 overflow-visible text-center">
        <span v-if="b.labelled" class="block whitespace-nowrap text-[10px] leading-tight text-muted">{{ b.month }}</span>
        <span v-if="b.year" class="block whitespace-nowrap text-[9px] leading-tight text-muted/70">{{ b.year }}</span>
      </div>
    </div>
  </section>
</template>
