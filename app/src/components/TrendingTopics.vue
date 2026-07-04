<script setup lang="ts">
/**
 * Trending topics (Plan B #4 — temporal_velocity on Home). Topics "heating up"
 * across the corpus: last month running >= 1.5x their 6-month average, with a
 * floor on total mentions to cut sample noise. Three views the operator can flip
 * between to decide what to keep: Sparklines (shape), Over time (stacked stream),
 * Momentum (velocity-vs-volume map). Reads the shared, memoized
 * /api/app/corpus/enrichment; hides when nothing is rising. Chips/points open the
 * topic entity card (whose Signals show the same momentum).
 */
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getCorpusEnrichment } from '../services/api'
import type { RisingTopic } from './trending'
import TrendingSparkChips from './TrendingSparkChips.vue'
import TrendingStream from './TrendingStream.vue'
import TrendingMomentum from './TrendingMomentum.vue'

const emit = defineEmits<{ (e: 'open', id: string): void }>()
const { t } = useI18n()

const RISING = 1.5
const MIN_TOTAL = 3
const MAX = 12

const months = ref<string[]>([])
const topics = ref<RisingTopic[]>([])

void getCorpusEnrichment()
  .then((s) => {
    const tv = s.temporal_velocity
    const rows = tv?.topics ?? []
    // Month axis: the envelope's window_months, else the union of keys seen.
    const axis =
      tv?.window_months && tv.window_months.length
        ? [...tv.window_months]
        : [...new Set(rows.flatMap((r) => Object.keys(r.monthly_counts ?? {})))].sort()
    months.value = axis
    topics.value = rows
      .filter((x) => (x.velocity_last_over_6mo ?? 0) >= RISING && (x.total ?? 0) >= MIN_TOTAL)
      .sort((a, b) => (b.velocity_last_over_6mo ?? 0) - (a.velocity_last_over_6mo ?? 0))
      .slice(0, MAX)
      .map((x) => ({
        id: x.topic_id,
        label: x.topic_label?.trim() || x.topic_id.replace(/^topic:/, '').replace(/[-_]+/g, ' '),
        v: Math.round((x.velocity_last_over_6mo ?? 0) * 10) / 10,
        total: x.total ?? 0,
        series: axis.map((m) => x.monthly_counts?.[m] ?? 0),
      }))
  })
  .catch(() => {
    topics.value = []
  })

type View = 'sparks' | 'stream' | 'momentum'
const view = ref<View>('sparks')
const VIEWS: Array<{ key: View; label: string }> = [
  { key: 'sparks', label: 'trendViewSparks' },
  { key: 'stream', label: 'trendViewStream' },
  { key: 'momentum', label: 'trendViewMomentum' },
]

const hasAny = computed(() => topics.value.length > 0)
</script>

<template>
  <section v-if="hasAny" class="mt-7" data-testid="home-trending">
    <div class="mb-1 flex items-baseline justify-between gap-2">
      <h2 class="lp-section">{{ t('home.trending') }}</h2>
      <div
        role="tablist"
        :aria-label="t('home.trendViewLabel')"
        class="inline-flex gap-0.5 rounded-full border border-border p-0.5 text-xs"
      >
        <button
          v-for="opt in VIEWS"
          :key="opt.key"
          type="button"
          role="tab"
          :aria-selected="view === opt.key"
          :data-testid="`trend-view-${opt.key}`"
          class="rounded-full px-2.5 py-0.5 font-semibold transition"
          :class="view === opt.key ? 'bg-accent text-accent-foreground' : 'text-muted hover:text-canvas-foreground'"
          @click="view = opt.key"
        >
          {{ t(`home.${opt.label}`) }}
        </button>
      </div>
    </div>
    <p class="mb-2 text-sm text-muted">{{ t('home.trendingHint') }}</p>

    <TrendingSparkChips v-if="view === 'sparks'" :topics="topics" @open="emit('open', $event)" />
    <TrendingStream
      v-else-if="view === 'stream'"
      :topics="topics"
      :months="months"
      @open="emit('open', $event)"
    />
    <TrendingMomentum v-else :topics="topics" @open="emit('open', $event)" />
  </section>
</template>
