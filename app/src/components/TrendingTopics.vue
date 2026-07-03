<script setup lang="ts">
/**
 * Trending topics (Plan B #4 — temporal_velocity on Home). Topics "heating up"
 * across the corpus: last month running >= 1.5x their 6-month average, with a
 * floor on total mentions to cut sample noise. Chips open the topic entity card
 * (whose Signals now show the same momentum). Reads the shared, memoized
 * /api/app/corpus/enrichment. Hides itself when nothing is rising.
 */
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getCorpusEnrichment } from '../services/api'

const emit = defineEmits<{ (e: 'open', id: string): void }>()
const { t } = useI18n()

const RISING = 1.5
const MIN_TOTAL = 3
const MAX = 10

const trending = ref<Array<{ id: string; label: string; v: number }>>([])
void getCorpusEnrichment()
  .then((s) => {
    trending.value = (s.temporal_velocity?.topics ?? [])
      .filter((x) => (x.velocity_last_over_6mo ?? 0) >= RISING && (x.total ?? 0) >= MIN_TOTAL)
      .sort((a, b) => (b.velocity_last_over_6mo ?? 0) - (a.velocity_last_over_6mo ?? 0))
      .slice(0, MAX)
      .map((x) => ({
        id: x.topic_id,
        label: x.topic_label?.trim() || x.topic_id.replace(/^topic:/, '').replace(/[-_]+/g, ' '),
        v: Math.round((x.velocity_last_over_6mo ?? 0) * 10) / 10,
      }))
  })
  .catch(() => {
    trending.value = []
  })
</script>

<template>
  <section v-if="trending.length" class="mt-7" data-testid="home-trending">
    <h2 class="lp-section mb-1">{{ t('home.trending') }}</h2>
    <p class="mb-2 text-sm text-muted">{{ t('home.trendingHint') }}</p>
    <div class="flex flex-wrap gap-1.5">
      <button
        v-for="tp in trending"
        :key="tp.id"
        type="button"
        class="inline-flex items-center gap-1.5 rounded-full bg-overlay px-3 py-1.5 text-sm text-topic transition hover:bg-elevated"
        data-testid="home-trending-chip"
        :aria-label="t('home.trendingChip', { topic: tp.label, factor: tp.v })"
        @click="emit('open', tp.id)"
      >
        {{ tp.label }}
        <span class="text-xs font-semibold text-accent">↑ {{ tp.v }}×</span>
      </button>
    </div>
  </section>
</template>
